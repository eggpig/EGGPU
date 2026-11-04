#pragma once
#include "managed_state_inventory.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace gpu_easygraph {
namespace runtime {

enum class SharedArrayKind {
    Offsets,
    Indices,
    LocalToGlobal,
    Owners,
    Weights,
};

struct SharedArrayKey {
    int device_id;
    SharedArrayKind kind;
    std::string sha256;
    std::string compatibility;
    std::size_t bytes;

    SharedArrayKey();
    bool operator==(const SharedArrayKey& other) const;
};

struct SharedArrayKeyHash {
    std::size_t operator()(const SharedArrayKey& key) const;
};

class SharedDeviceArray {
public:
    SharedDeviceArray(
        const SharedArrayKey& key,
        void* data,
        const std::shared_ptr<void>& owner);

    const SharedArrayKey& key() const;
    void* data() const;
    std::size_t bytes() const;

private:
    friend class SharedDeviceArrayCatalog;
    SharedArrayKey key_;
    void* data_;
    std::shared_ptr<void> owner_;
};

struct SharedArrayAcquireResult {
    std::shared_ptr<SharedDeviceArray> array;
    bool cache_hit;
    bool retained;
    std::size_t new_bytes;
    std::size_t retained_bytes;

    SharedArrayAcquireResult();
};

struct SharedArrayCallStats {
    std::size_t hits;
    std::size_t new_bytes;
    std::size_t retained_bytes;
    bool all_retained;

    SharedArrayCallStats();
    void add(const SharedArrayAcquireResult& result);
};

struct SharedArrayCatalogStats {
    std::size_t budget_bytes;
    std::size_t retained_bytes;
    std::size_t retained_entries;
    std::size_t active_entries;
    std::uint64_t hits;
    std::uint64_t misses;
    std::uint64_t evictions;
    std::uint64_t allocated_bytes;
};

class SharedDeviceArrayCatalog {
public:
    typedef std::function<std::shared_ptr<SharedDeviceArray>(
        const SharedArrayKey&)> Factory;

    explicit SharedDeviceArrayCatalog(std::size_t budget_bytes);

    SharedArrayAcquireResult acquire(
        const SharedArrayKey& key, const Factory& factory);
    // Rekey only a completed array held exclusively by this catalog. A failed
    // mutation removes the old entry, so partially written bytes are unpublished.
    SharedArrayAcquireResult reuse_idle(
        const SharedArrayKey& previous, const SharedArrayKey& next,
        const std::function<void(void*)>& patch);
    SharedArrayAcquireResult find_existing(const SharedArrayKey& key);
    void set_budget_bytes(std::size_t budget_bytes);
    SharedArrayCatalogStats stats() const;
    std::size_t clear();
    void append_managed_inventory(ManagedStateInventory& inventory) const;
    void release_managed_idle(ManagedEvictRecord& receipt);

private:
    struct Entry {
        std::shared_ptr<SharedDeviceArray> array;
        std::uint64_t last_use;
    };

    typedef std::unordered_map<SharedArrayKey, Entry, SharedArrayKeyHash> Map;

    void validate_key(const SharedArrayKey& key) const;
    bool evict_one_idle();
    void trim_idle_to_budget();

    std::size_t budget_bytes_;
    std::size_t retained_bytes_;
    std::uint64_t clock_;
    std::uint64_t hits_;
    std::uint64_t misses_;
    std::uint64_t evictions_;
    std::uint64_t allocated_bytes_;
    Map entries_;
};

const char* shared_array_kind_name(SharedArrayKind kind);

// One catalog belongs to one fixed CUDA worker thread.  Callers acquire
// shared_ptr leases before using an array, so an array cannot be destroyed
// while a kernel or transfer still references it.
SharedDeviceArrayCatalog& shared_device_array_catalog();

// Bounds each algorithm-specific resident view/workspace registry.  The
// shared read-only arrays have their independent byte budget above.
std::size_t partition_state_cache_capacity();

}  // namespace runtime
}  // namespace gpu_easygraph
