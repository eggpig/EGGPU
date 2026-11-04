#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

enum class PartitionedHostPlanAlgorithm {
    Bfs,
    Wcc,
    Sssp,
    Ego,
};

const char* partitioned_host_plan_algorithm_name(
    PartitionedHostPlanAlgorithm algorithm);

// This key identifies immutable host-side execution metadata derived from a
// validated partition manifest.  The device group is canonicalized for cache
// identity only; callers retain their requested order for execution and
// result diagnostics.
struct PartitionedHostPlanCacheKey {
    PartitionedHostPlanAlgorithm algorithm;
    std::string graph_id;
    std::uint64_t generation;
    std::string view_signature;
    std::string topology_signature;
    bool weighted;
    std::string weight_signature;
    std::uint64_t vertex_count;
    std::uint64_t adjacency_count;
    std::vector<int> device_group;
    // Host representation choice; independent of the validated graph identity.
    std::string plan_variant;

    bool operator==(const PartitionedHostPlanCacheKey& other) const;
};

struct PartitionedHostPlanCacheKeyHash {
    std::size_t operator()(const PartitionedHostPlanCacheKey& key) const;
};

// Constructs a fail-closed key from a validated native graph identity.  All
// manifest signatures are required to be canonical lower-case SHA-256 values;
// weighted and unweighted views have distinct weight-signature rules.
PartitionedHostPlanCacheKey make_partitioned_host_plan_cache_key(
    PartitionedHostPlanAlgorithm algorithm,
    const std::string& graph_id,
    std::uint64_t generation,
    const std::string& view_signature,
    const std::string& topology_signature,
    bool weighted,
    const std::string& weight_signature,
    std::uint64_t vertex_count,
    std::uint64_t adjacency_count,
    const std::vector<int>& requested_device_group,
    const std::string& plan_variant = std::string());

struct PartitionedHostPlanCacheStats {
    std::uint64_t hits;
    std::uint64_t builders;
    std::uint64_t waiters;
    std::uint64_t build_failures;
    std::uint64_t evictions;
    std::uint64_t generation_invalidations;
    std::size_t entries;
    std::size_t capacity;

    PartitionedHostPlanCacheStats()
        : hits(0),
          builders(0),
          waiters(0),
          build_failures(0),
          evictions(0),
          generation_invalidations(0),
          entries(0),
          capacity(0) {}
};

template <typename Plan>
class PartitionedHostPlanCache {
public:
    typedef std::shared_ptr<const Plan> PlanPtr;

    struct AcquireResult {
        PlanPtr plan;
        bool cache_hit;
        bool builder;
        bool waiter;
        bool build_seconds_measured;
        double build_seconds;
        std::size_t entry_count;

        AcquireResult()
            : cache_hit(false),
              builder(false),
              waiter(false),
              build_seconds_measured(false),
              build_seconds(0.0),
              entry_count(0) {}
    };

    explicit PartitionedHostPlanCache(std::size_t capacity)
        : capacity_(capacity), next_token_(1) {
        if (capacity_ == 0) {
            throw std::invalid_argument(
                "partitioned host-plan cache capacity must be positive");
        }
        stats_.capacity = capacity_;
    }

    template <typename Builder>
    AcquireResult acquire(
        const PartitionedHostPlanCacheKey& key, Builder builder) {
        typedef std::promise<PlanPtr> Promise;
        typedef std::shared_future<PlanPtr> Future;

        Future future;
        std::shared_ptr<Promise> promise;
        std::uint64_t token = 0;
        bool is_builder = false;
        bool is_waiter = false;
        bool is_hit = false;
        bool cache_builder = false;

        {
            std::lock_guard<std::mutex> guard(mutex_);
            const std::string logical_graph = logical_graph_key(key);
            typename LatestGenerationMap::iterator latest =
                latest_generation_.find(logical_graph);
            const bool stale_snapshot =
                latest != latest_generation_.end() &&
                key.generation < latest->second;
            if (latest == latest_generation_.end() ||
                key.generation > latest->second) {
                invalidate_older_generations_locked(key);
                latest_generation_[logical_graph] = key.generation;
            }

            typename EntryMap::iterator found = stale_snapshot
                ? entries_.end()
                : entries_.find(key);
            if (found != entries_.end()) {
                touch_locked(found);
                future = found->second.future;
                if (found->second.building) {
                    is_waiter = true;
                    ++stats_.waiters;
                } else {
                    is_hit = true;
                    ++stats_.hits;
                }
            } else {
                is_builder = true;
                ++stats_.builders;
                // A request already holding an older immutable snapshot may
                // arrive after a newer generation.  It remains executable,
                // but it must neither evict nor replace the newer plan.
                if (!stale_snapshot) {
                    promise.reset(new Promise());
                    future = promise->get_future().share();
                    token = next_token_++;
                    lru_.push_front(key);
                    Entry entry;
                    entry.future = future;
                    entry.building = true;
                    entry.token = token;
                    entry.lru = lru_.begin();
                    entries_.insert(std::make_pair(key, entry));
                    cache_builder = true;
                    prune_locked();
                }
            }
            stats_.entries = entries_.size();
        }

        AcquireResult result;
        result.cache_hit = is_hit;
        result.builder = is_builder;
        result.waiter = is_waiter;

        if (!is_builder) {
            result.plan = future.get();
            std::lock_guard<std::mutex> guard(mutex_);
            result.entry_count = entries_.size();
            return result;
        }

        const std::chrono::steady_clock::time_point build_begin =
            std::chrono::steady_clock::now();
        PlanPtr built;
        try {
            built = builder();
            if (!built) {
                throw std::runtime_error(
                    "partitioned host-plan builder returned a null plan");
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            if (cache_builder) promise->set_exception(error);
            {
                std::lock_guard<std::mutex> guard(mutex_);
                if (cache_builder) {
                    typename EntryMap::iterator found = entries_.find(key);
                    if (found != entries_.end() &&
                        found->second.token == token) {
                        erase_locked(found);
                    }
                }
                ++stats_.build_failures;
                stats_.entries = entries_.size();
            }
            std::rethrow_exception(error);
        }
        const std::chrono::steady_clock::time_point build_end =
            std::chrono::steady_clock::now();

        if (cache_builder) promise->set_value(built);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (cache_builder) {
                typename EntryMap::iterator found = entries_.find(key);
                if (found != entries_.end() && found->second.token == token) {
                    found->second.building = false;
                    touch_locked(found);
                }
            }
            stats_.entries = entries_.size();
            result.entry_count = entries_.size();
        }
        result.plan = built;
        result.build_seconds_measured = true;
        result.build_seconds =
            std::chrono::duration<double>(build_end - build_begin).count();
        return result;
    }

    // Borrow a completed entry without inserting an index, changing LRU order,
    // or waiting for a builder. The predicate must not re-enter this cache.
    // The caller decides its semantic compatibility and bounds the lease.
    template <typename Predicate>
    PlanPtr find_completed(Predicate predicate) const {
        std::lock_guard<std::mutex> guard(mutex_);
        for (const auto& key : lru_) {
            const auto found = entries_.find(key);
            if (found == entries_.end() || found->second.building) continue;
            const PlanPtr& plan = found->second.future.get();
            if (plan && predicate(key, *plan)) return plan;
        }
        return PlanPtr();
    }

    // Clearing drops only cache ownership.  Plans already leased by active
    // calls, and futures already observed by waiters, remain valid.
    std::size_t clear() {
        std::lock_guard<std::mutex> guard(mutex_);
        const std::size_t released = entries_.size();
        entries_.clear();
        lru_.clear();
        latest_generation_.clear();
        stats_.entries = 0;
        return released;
    }

    PartitionedHostPlanCacheStats stats() const {
        std::lock_guard<std::mutex> guard(mutex_);
        PartitionedHostPlanCacheStats result = stats_;
        result.entries = entries_.size();
        result.capacity = capacity_;
        return result;
    }

private:
    struct Entry {
        std::shared_future<PlanPtr> future;
        bool building;
        std::uint64_t token;
        typename std::list<PartitionedHostPlanCacheKey>::iterator lru;

        Entry() : building(false), token(0) {}
    };

    typedef std::unordered_map<
        PartitionedHostPlanCacheKey,
        Entry,
        PartitionedHostPlanCacheKeyHash> EntryMap;
    typedef std::unordered_map<std::string, std::uint64_t>
        LatestGenerationMap;

    static std::string logical_graph_key(
        const PartitionedHostPlanCacheKey& key) {
        std::string result(partitioned_host_plan_algorithm_name(key.algorithm));
        result.push_back('\0');
        result.append(key.graph_id);
        return result;
    }

    void touch_locked(typename EntryMap::iterator entry) {
        lru_.splice(lru_.begin(), lru_, entry->second.lru);
        entry->second.lru = lru_.begin();
    }

    void erase_locked(typename EntryMap::iterator entry) {
        lru_.erase(entry->second.lru);
        entries_.erase(entry);
    }

    void invalidate_older_generations_locked(
        const PartitionedHostPlanCacheKey& key) {
        typename EntryMap::iterator current = entries_.begin();
        while (current != entries_.end()) {
            if (current->first.algorithm == key.algorithm &&
                current->first.graph_id == key.graph_id &&
                current->first.generation != key.generation) {
                typename EntryMap::iterator stale = current++;
                erase_locked(stale);
                ++stats_.generation_invalidations;
            } else {
                ++current;
            }
        }
    }

    void prune_locked() {
        while (entries_.size() > capacity_) {
            const PartitionedHostPlanCacheKey victim_key = lru_.back();
            typename EntryMap::iterator victim = entries_.find(victim_key);
            if (victim == entries_.end()) {
                throw std::logic_error(
                    "partitioned host-plan cache LRU is inconsistent");
            }
            erase_locked(victim);
            ++stats_.evictions;
        }
    }

    const std::size_t capacity_;
    mutable std::mutex mutex_;
    EntryMap entries_;
    std::list<PartitionedHostPlanCacheKey> lru_;
    LatestGenerationMap latest_generation_;
    std::uint64_t next_token_;
    PartitionedHostPlanCacheStats stats_;
};

}  // namespace runtime
}  // namespace gpu_easygraph
