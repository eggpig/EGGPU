#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace gpu_easygraph { namespace runtime {

// Storage only: no topology, label values, or public result belongs to the pool.
// A caller must hold lease_mutex through all gang transfer completion.
// Buffer is PinnedHostBuffer in production; a CPU allocator can test the exact
// capacity/accounting logic without creating a CUDA context.
template <typename Buffer>
struct WccPinnedRoutePool {
    std::mutex lease_mutex;
    std::vector<std::shared_ptr<Buffer> > buffers;
    std::size_t allocation_count = 0;
    std::size_t allocated_bytes = 0;
    std::size_t retained_bytes = 0;
    double allocation_seconds = 0.0;
    // Protected by the caller's lease. If a submitted query throws before its
    // successful completion boundary, keep the storage resident but poisoned.
    // A failed CUDA wait does not prove asynchronous DMA has stopped.
    bool unfinished_query = false;

    void require_reusable() const {
        if (unfinished_query)
            throw std::runtime_error("WCC pinned route pool has an unfinished/failed query; runtime teardown and restart required");
    }
    void begin_query() {
        require_reusable();
        unfinished_query = true;
    }
    void complete_query() {
        if (!unfinished_query)
            throw std::logic_error("WCC pinned route pool completion without a submitted query");
        unfinished_query = false;
    }

    template <typename Plan>
    std::size_t ensure(const Plan& plan) {
        require_reusable(); // Before any resize, allocation, or pointer change.
        const std::size_t ranks = plan.partitions.size();
        if (ranks && ranks > std::numeric_limits<std::size_t>::max() / ranks)
            throw std::overflow_error("WCC pinned route slot count overflow");
        const std::size_t slots = ranks * ranks;
        // The runtime key includes the ordered device group, so cardinality is
        // invariant. Cross-graph requests may change capacities, not slot IDs.
        if (!buffers.empty() && buffers.size() != slots)
            throw std::invalid_argument("WCC pinned pool device cardinality changed");
        std::vector<std::size_t> required(slots, 0);
        std::size_t required_bytes = 0;
        for (std::size_t source = 0; source < ranks; ++source) {
            for (std::size_t destination = 0; destination < ranks; ++destination) {
                const std::size_t count = plan.route(source, destination).size();
                if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t))
                    throw std::overflow_error("WCC pinned route byte count overflow");
                const std::size_t bytes = count * sizeof(std::uint64_t);
                if (bytes > std::numeric_limits<std::size_t>::max() - required_bytes)
                    throw std::overflow_error("WCC pinned required byte total overflow");
                required[source * ranks + destination] = bytes;
                required_bytes += bytes;
            }
        }
        buffers.resize(slots);
        for (std::size_t slot = 0; slot < slots; ++slot) {
            const std::size_t old = buffers[slot] ? buffers[slot]->size() : 0;
            const std::size_t bytes = required[slot];
            if (bytes <= old) continue;
            if (bytes - old > std::numeric_limits<std::size_t>::max() - retained_bytes)
                throw std::overflow_error("WCC pinned retained byte total overflow");
            const auto begin = std::chrono::steady_clock::now();
            std::shared_ptr<Buffer> replacement(new Buffer(bytes));
            buffers[slot] = std::move(replacement);
            allocation_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count();
            retained_bytes += bytes - old;
            allocated_bytes += bytes;
            ++allocation_count;
        }
        return required_bytes;
    }
};

}} // namespace gpu_easygraph::runtime
