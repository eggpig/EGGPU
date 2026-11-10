#pragma once

#include "device_topology.h"
#include "device_worker.h"
#include "runtime_types.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct RuntimeStatus {
    bool running;
    std::vector<int> device_ids;
    CommunicationBackend backend;
    std::vector<DeviceWorkerStatus> workers;
};

struct GangSubmission {
    std::uint64_t request_id;
    std::vector<std::future<void> > completions;
};

// NativeRuntime provides ordered gang dispatch and fixed-device workers.  It
// is an execution foundation, not a partitioned graph algorithm: algorithms
// must explicitly provide one native task per participating device.
class NativeRuntime {
public:
    typedef DeviceWorker::Task Task;

    NativeRuntime(
        std::vector<int> device_ids,
        CommunicationBackend requested_backend);
    static std::unique_ptr<NativeRuntime> create_for_testing(
        std::vector<int> device_ids,
        DeviceWorker::Binder binder);
    ~NativeRuntime();

    NativeRuntime(const NativeRuntime&) = delete;
    NativeRuntime& operator=(const NativeRuntime&) = delete;

    const DeviceTopology& topology() const;
    const DeviceGroup& device_group() const;
    RuntimeStatus status() const;
    // Resources with a runtime lifetime (e.g. group communicators and pinned
    // pools). The factory runs once per key. Mutable users provide their own
    // query lease; the cache only controls construction and ownership.
    std::shared_ptr<void> acquire_resident_resource(
        const std::string& key,
        const std::function<std::shared_ptr<void>()>& factory,
        bool* cache_hit = nullptr);

    GangSubmission submit_gang(
        const DeviceGroup& group,
        std::vector<Task> tasks);
    std::vector<int> ping();
    void shutdown(bool drain);

private:
    NativeRuntime(
        std::vector<int> device_ids,
        CommunicationBackend requested_backend,
        DeviceWorker::Binder binder,
        bool validate_topology);
    DeviceWorker& worker_for(int device_id) const;

    DeviceTopology topology_;
    DeviceGroup device_group_;
    std::vector<std::unique_ptr<DeviceWorker> > workers_;
    mutable std::mutex dispatch_mutex_;
    std::mutex resource_mutex_;
    std::map<std::string, std::shared_ptr<void> > resident_resources_;
    std::atomic<std::uint64_t> next_request_id_;
    std::atomic<bool> stopped_;
};

// RuntimeManager is the process-level native owner used by the thin pybind
// controls.  It does not retain any Python object or callback.
class RuntimeManager {
public:
    RuntimeManager();
    ~RuntimeManager();

    static std::unique_ptr<RuntimeManager> create_for_testing(
        std::shared_ptr<NativeRuntime> runtime);

    RuntimeStatus start(
        const std::vector<int>& requested_devices,
        CommunicationBackend requested_backend);
    RuntimeStatus status() const;
    DeviceTopology topology() const;
    DeviceGroup device_group() const;
    std::shared_ptr<void> acquire_resident_resource(
        const std::string& key,
        const std::function<std::shared_ptr<void>()>& factory,
        bool* cache_hit = nullptr);
    void execute_gang(
        const std::vector<int>& device_ids,
        std::vector<NativeRuntime::Task> tasks);
    std::vector<int> ping();
    void shutdown(bool drain);

private:
    explicit RuntimeManager(std::shared_ptr<NativeRuntime> runtime);
    mutable std::mutex mutex_;
    mutable std::condition_variable shutdown_condition_;
    std::shared_ptr<NativeRuntime> runtime_;
    bool shutting_down_;
};

RuntimeManager& runtime_manager();

}  // namespace runtime
}  // namespace gpu_easygraph
