#pragma once

// Observes DeviceAllocation objects only, including temporary construction
// allocations. CUDA context, external libraries and raw cudaMalloc calls are
// outside this ledger. Observation is not an allocator budget enforcement.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace gpu_easygraph { namespace runtime {
struct DeviceAllocationObservation {
    std::uint64_t live_bytes = 0, peak_bytes = 0, window_peak_bytes = 0;
    std::uint64_t live_count = 0, allocation_count = 0, free_count = 0;
    std::uint64_t allocated_bytes = 0, freed_bytes = 0, free_errors = 0;
    std::uint64_t window_id = 0, window_start_bytes = 0;
};
struct ObservedAllocation {
    std::uint64_t generation = 0;
    std::size_t bytes = 0;
};
struct DeviceAllocationObservationStore {
    std::mutex mutex;
    std::uint64_t generation = 0;
    std::map<int, DeviceAllocationObservation> devices;
    std::map<std::pair<int, std::uintptr_t>, ObservedAllocation> live;
};
inline DeviceAllocationObservationStore& allocation_observation_store() {
    // Intentionally process-lifetime: CUDA/catalog destructors can run late.
    static auto* value = new DeviceAllocationObservationStore();
    return *value;
}
inline std::uint64_t observe_device_allocation(int device, const void* pointer,
                                             std::size_t bytes) {
    if (device < 0 || !pointer || !bytes)
        throw std::invalid_argument("Invalid observed device allocation");
    auto& store = allocation_observation_store();
    std::lock_guard<std::mutex> lock(store.mutex);
    const auto key = std::make_pair(device, reinterpret_cast<std::uintptr_t>(pointer));
    if (store.live.count(key)) throw std::runtime_error("Live allocation address reused");
    ObservedAllocation record;
    record.generation = ++store.generation;
    record.bytes = bytes;
    store.live.emplace(key, record);
    auto& value = store.devices[device];
    value.live_bytes += bytes;
    value.allocated_bytes += bytes;
    ++value.live_count;
    ++value.allocation_count;
    value.peak_bytes = std::max(value.peak_bytes, value.live_bytes);
    value.window_peak_bytes = std::max(value.window_peak_bytes, value.live_bytes);
    return record.generation;
}
inline void observe_device_free(int device, const void* pointer, bool succeeded) {
    auto& store = allocation_observation_store();
    std::lock_guard<std::mutex> lock(store.mutex);
    auto& value = store.devices[device];
    if (!succeeded) { ++value.free_errors; return; }
    const auto key = std::make_pair(device, reinterpret_cast<std::uintptr_t>(pointer));
    const auto found = store.live.find(key);
    if (found == store.live.end()) { ++value.free_errors; return; }
    value.live_bytes -= found->second.bytes;
    value.freed_bytes += found->second.bytes;
    --value.live_count;
    ++value.free_count;
    store.live.erase(found);
}
inline std::uint64_t observed_allocation_generation(int device,
                                                  const void* pointer,
                                                  std::size_t bytes) {
    auto& store = allocation_observation_store();
    std::lock_guard<std::mutex> lock(store.mutex);
    const auto found = store.live.find(std::make_pair(device, reinterpret_cast<std::uintptr_t>(pointer)));
    if (found == store.live.end()) return 0;
    if (found->second.bytes != bytes)
        throw std::runtime_error("Allocation observation byte extent mismatch");
    return found->second.generation;
}
inline DeviceAllocationObservation device_allocation_observation(int device) {
    auto& store = allocation_observation_store();
    std::lock_guard<std::mutex> lock(store.mutex);
    return store.devices[device];
}
inline void reset_device_allocation_observation(int device) {
    auto& store = allocation_observation_store();
    std::lock_guard<std::mutex> lock(store.mutex);
    auto& value = store.devices[device];
    ++value.window_id;
    value.window_start_bytes = value.live_bytes;
    value.window_peak_bytes = value.live_bytes;
}
} }  // namespace gpu_easygraph::runtime
