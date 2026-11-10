#pragma once
#include "device_allocation_observer.h"

// Request-boundary inventory for retained BFS/PR state and the shared-array
// catalog. This is NOT a CUDA process allocator or a construction-peak bound.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_easygraph { namespace runtime {

inline std::string new_managed_owner_id(const char* algorithm, const char* kind,
                                         int device) {
    static std::atomic<std::uint64_t> sequence(0);
    std::ostringstream value;
    value << algorithm << ':' << kind << ':' << device << ':' << ++sequence;
    return value.str();
}

inline bool managed_workspace_per_worker() {
    const char* raw = std::getenv("EGMGPU_WORKSPACE_POLICY");
    const std::string policy = raw ? raw : "per_view";
    if (policy == "per_view") return false;
    if (policy == "per_worker") return true;
    throw std::invalid_argument("EGMGPU_WORKSPACE_POLICY must be per_view or per_worker");
}

struct ManagedOwnerRecord {
    std::string owner_id, kind, algorithm, graph_id, parent_owner_id, policy;
    int device_id = -1;
    int partition_id = -1;
    std::uint64_t generation = 0;
    std::size_t active_leases = 0;
};

struct ManagedAllocationRecord {
    int device_id = -1;
    std::uintptr_t pointer = 0;
    std::uint64_t allocation_generation = 0;
    std::size_t bytes = 0;
    std::set<std::string> owners, components;
    bool catalog_retained = false;
    std::size_t catalog_external_references = 0;
    std::size_t inventoried_shared_references = 0;
};

struct ManagedEvictRecord {
    std::vector<std::string> removed, skipped_active, released_workspaces;
    std::size_t idle_catalog_released_bytes = 0;
    std::size_t idle_catalog_released_entries = 0;
};

class ManagedStateInventory {
public:
    explicit ManagedStateInventory(int device = -1) : device_id(device) {}
    int device_id;
    std::map<std::string, ManagedOwnerRecord> owners;
    std::map<std::uintptr_t, ManagedAllocationRecord> allocations;
    std::map<std::string, std::uint64_t> counters;

    void owner(const ManagedOwnerRecord& value) {
        if (value.device_id != device_id || value.owner_id.empty())
            throw std::runtime_error("Invalid managed owner identity/device");
        if (!owners.insert(std::make_pair(value.owner_id, value)).second)
            throw std::runtime_error("Duplicate managed owner identity");
    }

    void allocation(const void* pointer, std::size_t bytes,
                    const std::string& owner_id, const char* component,
                    bool shared_reference = false, bool catalog = false,
                    std::size_t external_references = 0) {
        if (!pointer && !bytes) return;
        if (!pointer || !bytes || owners.count(owner_id) != 1)
            throw std::runtime_error("Invalid managed allocation/owner");
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        ManagedAllocationRecord& value = allocations[address];
        if (value.bytes && value.bytes != bytes)
            throw std::runtime_error("Aliased device pointer has inconsistent allocation bytes");
        value.device_id = device_id;
        value.pointer = address;
        value.allocation_generation = observed_allocation_generation(device_id, pointer, bytes);
        value.bytes = bytes;
        value.owners.insert(owner_id);
        value.components.insert(component);
        value.inventoried_shared_references += shared_reference ? 1U : 0U;
        if (catalog) {
            if (value.catalog_retained)
                throw std::runtime_error("A shared allocation is catalogued twice");
            value.catalog_retained = true;
            value.catalog_external_references = external_references;
        }
    }

    template <typename T>
    void add(const std::shared_ptr<T>& value, const std::string& owner_id,
             const char* component, bool shared_reference = false) {
        if (value) allocation(value->data(), value->bytes(), owner_id,
                              component, shared_reference);
    }

    std::size_t bytes() const {
        std::size_t total = 0;
        for (const auto& value : allocations) total += value.second.bytes;
        return total;
    }
};

} }  // namespace gpu_easygraph::runtime
