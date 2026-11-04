#include "native_runtime.h"
#include "gang_completion.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <stdexcept>
#include <utility>

namespace gpu_easygraph {
namespace runtime {
namespace {

class GangBarrier {
public:
    explicit GangBarrier(std::size_t participants)
        : participants_(participants),
          arrived_(0),
          released_(false),
          cancelled_(false) {
        if (participants == 0) {
            throw std::invalid_argument("a gang barrier requires participants");
        }
    }

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cancelled_) {
            throw std::runtime_error("EGMGPU gang dispatch was cancelled");
        }
        ++arrived_;
        if (arrived_ == participants_) {
            released_ = true;
            condition_.notify_all();
            return;
        }
        condition_.wait(lock, [this]() { return released_ || cancelled_; });
        if (cancelled_) {
            throw std::runtime_error("EGMGPU gang dispatch was cancelled");
        }
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        condition_.notify_all();
    }

private:
    const std::size_t participants_;
    std::size_t arrived_;
    bool released_;
    bool cancelled_;
    std::mutex mutex_;
    std::condition_variable condition_;
};

std::vector<int> all_device_ids(const DeviceTopology& topology) {
    std::vector<int> ids;
    const std::vector<DeviceInfo>& devices = topology.devices();
    ids.reserve(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
        ids.push_back(devices[i].device_id);
    }
    return ids;
}

RuntimeStatus stopped_status() {
    RuntimeStatus out;
    out.running = false;
    out.backend = CommunicationBackend::None;
    return out;
}

}  // namespace

NativeRuntime::NativeRuntime(
    std::vector<int> device_ids,
    CommunicationBackend requested_backend)
    : NativeRuntime(
          std::move(device_ids),
          requested_backend,
          DeviceWorker::Binder(),
          true) {}

NativeRuntime::NativeRuntime(
    std::vector<int> device_ids,
    CommunicationBackend requested_backend,
    DeviceWorker::Binder binder,
    bool validate_topology)
    : topology_(validate_topology ? DeviceTopology::probe() : DeviceTopology()),
      device_group_(std::move(device_ids), requested_backend),
      next_request_id_(1),
      stopped_(false) {
    if (validate_topology && !topology_.runtime_available()) {
        throw std::runtime_error(
            "CUDA runtime is unavailable: " + topology_.diagnostic());
    }
    const std::vector<int>& ids = device_group_.device_ids();
    if (validate_topology) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (!topology_.has_device(ids[i])) {
                throw std::out_of_range(
                    "requested CUDA device is not present in the probed topology");
            }
        }
    }

    CommunicationBackend resolved = requested_backend;
    if (validate_topology) {
        resolved = topology_.resolve_backend(device_group_, requested_backend);
    } else if (ids.size() == 1) {
        resolved = CommunicationBackend::None;
    } else if (resolved == CommunicationBackend::Auto) {
        resolved = CommunicationBackend::HostStaging;
    } else if (resolved == CommunicationBackend::None) {
        throw std::invalid_argument(
            "communication backend 'none' is invalid for a multi-device group");
    }
    device_group_ = DeviceGroup(ids, resolved);

    workers_.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (validate_topology) {
            workers_.push_back(
                std::unique_ptr<DeviceWorker>(new DeviceWorker(ids[i])));
        } else {
            workers_.push_back(
                std::unique_ptr<DeviceWorker>(new DeviceWorker(ids[i], binder)));
        }
    }
}

std::unique_ptr<NativeRuntime> NativeRuntime::create_for_testing(
    std::vector<int> device_ids,
    DeviceWorker::Binder binder) {
    return std::unique_ptr<NativeRuntime>(new NativeRuntime(
        std::move(device_ids),
        CommunicationBackend::Auto,
        std::move(binder),
        false));
}

NativeRuntime::~NativeRuntime() { shutdown(true); }

const DeviceTopology& NativeRuntime::topology() const { return topology_; }
const DeviceGroup& NativeRuntime::device_group() const { return device_group_; }

RuntimeStatus NativeRuntime::status() const {
    RuntimeStatus out;
    out.running = !stopped_.load();
    out.device_ids = device_group_.device_ids();
    out.backend = device_group_.backend();
    out.workers.reserve(workers_.size());
    for (std::size_t i = 0; i < workers_.size(); ++i) {
        out.workers.push_back(workers_[i]->status());
    }
    return out;
}

std::shared_ptr<void> NativeRuntime::acquire_resident_resource(
    const std::string& key,
    const std::function<std::shared_ptr<void>()>& factory,
    bool* cache_hit) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    if (stopped_.load())
        throw std::runtime_error("the EGMGPU runtime is stopped");
    const auto found = resident_resources_.find(key);
    if (found != resident_resources_.end()) {
        if (cache_hit) *cache_hit = true;
        return found->second;
    }
    auto resource = factory();
    resident_resources_.emplace(key, resource);
    if (cache_hit) *cache_hit = false;
    return resource;
}

DeviceWorker& NativeRuntime::worker_for(int device_id) const {
    for (std::size_t i = 0; i < workers_.size(); ++i) {
        if (workers_[i]->device_id() == device_id) {
            return *workers_[i];
        }
    }
    throw std::out_of_range("device is outside the active EGMGPU runtime");
}

GangSubmission NativeRuntime::submit_gang(
    const DeviceGroup& group,
    std::vector<Task> tasks) {
    if (stopped_.load()) {
        throw std::runtime_error("the EGMGPU runtime is stopped");
    }
    if (tasks.size() != group.size()) {
        throw std::invalid_argument(
            "gang submission requires exactly one task per device");
    }
    const std::vector<int>& ids = group.device_ids();
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!device_group_.contains(ids[i])) {
            throw std::out_of_range(
                "gang submission references a device outside the active runtime");
        }
    }

    const std::uint64_t request_id = next_request_id_.fetch_add(1);
    const std::shared_ptr<GangBarrier> barrier(new GangBarrier(ids.size()));

    GangSubmission submission;
    submission.request_id = request_id;
    submission.completions.reserve(ids.size());

    // Serializing queue insertion gives every pair of overlapping gangs the
    // same order on every worker.  The start barrier can therefore reserve a
    // complete device group without cyclic queue dependencies.
    std::lock_guard<std::mutex> dispatch_lock(dispatch_mutex_);
    // The optimistic check above gives fast rejection.  This synchronized
    // check defines the exact shutdown boundary: either the complete gang is
    // inserted before shutdown, or no task from it is accepted.
    if (stopped_.load()) {
        throw std::runtime_error("the EGMGPU runtime is stopped");
    }
    try {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const Task task = std::move(tasks[i]);
            submission.completions.push_back(worker_for(ids[i]).submit(
                request_id,
                [barrier, task](int device_id) {
                    barrier->arrive_and_wait();
                    task(device_id);
                }));
        }
    } catch (...) {
        barrier->cancel();
        throw;
    }
    return submission;
}

std::vector<int> NativeRuntime::ping() {
    const std::vector<int> ids = device_group_.device_ids();
    const std::shared_ptr<std::vector<int> > observed(
        new std::vector<int>(ids.size(), -1));
    std::vector<Task> tasks;
    tasks.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        tasks.push_back([observed, i](int worker_device) {
            int current_device = -1;
            const cudaError_t status = cudaGetDevice(&current_device);
            if (status != cudaSuccess) {
                throw std::runtime_error("cudaGetDevice failed in EGMGPU worker ping");
            }
            if (current_device != worker_device) {
                throw std::runtime_error(
                    "EGMGPU worker changed CUDA device after initialization");
            }
            (*observed)[i] = current_device;
        });
    }
    GangSubmission submission = submit_gang(device_group_, std::move(tasks));
    for (std::size_t i = 0; i < submission.completions.size(); ++i) {
        submission.completions[i].get();
    }
    return *observed;
}

void NativeRuntime::shutdown(bool drain) {
    // Serialize the state transition with queue insertion, but never hold the
    // dispatch lock while joining workers.  This both defines the submission
    // boundary for shutdown and prevents a worker-side failure path from
    // deadlocking behind a long-lived manager lock.
    {
        std::lock_guard<std::mutex> dispatch_lock(dispatch_mutex_);
        if (stopped_.exchange(true)) {
            return;
        }
    }
    for (std::size_t i = 0; i < workers_.size(); ++i) {
        workers_[i]->stop(drain);
    }
    // All accepted work has finished before communicators or staging pools
    // lose their runtime owner. A restarted runtime receives a fresh cache.
    std::lock_guard<std::mutex> resource_lock(resource_mutex_);
    resident_resources_.clear();
}

RuntimeManager::RuntimeManager() : shutting_down_(false) {}
RuntimeManager::RuntimeManager(std::shared_ptr<NativeRuntime> runtime)
    : runtime_(std::move(runtime)), shutting_down_(false) {}
RuntimeManager::~RuntimeManager() { shutdown(true); }

std::unique_ptr<RuntimeManager> RuntimeManager::create_for_testing(
    std::shared_ptr<NativeRuntime> runtime) {
    if (!runtime) {
        throw std::invalid_argument("test RuntimeManager requires a runtime");
    }
    return std::unique_ptr<RuntimeManager>(
        new RuntimeManager(std::move(runtime)));
}

RuntimeStatus RuntimeManager::start(
    const std::vector<int>& requested_devices,
    CommunicationBackend requested_backend) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutting_down_) {
        throw std::runtime_error(
            "EGMGPU runtime shutdown is in progress; wait before restarting");
    }
    if (runtime_) {
        const RuntimeStatus existing = runtime_->status();
        const std::vector<int> devices =
            requested_devices.empty() ? existing.device_ids : requested_devices;
        const DeviceGroup requested(devices, requested_backend);
        if (existing.device_ids == requested.device_ids() &&
            (requested_backend == CommunicationBackend::Auto ||
             existing.backend == requested_backend)) {
            return existing;
        }
        throw std::runtime_error(
            "EGMGPU runtime is already active with a different configuration; "
            "shut it down before reconfiguration");
    }

    std::vector<int> devices = requested_devices;
    if (devices.empty()) {
        const DeviceTopology topology = DeviceTopology::probe();
        if (!topology.runtime_available()) {
            throw std::runtime_error(
                "CUDA runtime is unavailable: " + topology.diagnostic());
        }
        devices = all_device_ids(topology);
        if (devices.empty()) {
            throw std::runtime_error("no CUDA devices are available to EGMGPU");
        }
    }
    runtime_ = std::make_shared<NativeRuntime>(devices, requested_backend);
    return runtime_->status();
}

RuntimeStatus RuntimeManager::status() const {
    std::shared_ptr<NativeRuntime> runtime;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runtime = runtime_;
    }
    return runtime ? runtime->status() : stopped_status();
}

DeviceTopology RuntimeManager::topology() const {
    std::shared_ptr<NativeRuntime> runtime;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runtime = runtime_;
    }
    if (!runtime) {
        throw std::runtime_error("EGMGPU runtime has not been started");
    }
    return runtime->topology();
}

DeviceGroup RuntimeManager::device_group() const {
    std::shared_ptr<NativeRuntime> runtime;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runtime = runtime_;
    }
    if (!runtime) {
        throw std::runtime_error("EGMGPU runtime has not been started");
    }
    return runtime->device_group();
}

std::shared_ptr<void> RuntimeManager::acquire_resident_resource(
    const std::string& key,
    const std::function<std::shared_ptr<void>()>& factory,
    bool* cache_hit) {
    std::shared_ptr<NativeRuntime> runtime;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runtime = runtime_;
    }
    if (!runtime)
        throw std::runtime_error("EGMGPU runtime has not been started");
    return runtime->acquire_resident_resource(key, factory, cache_hit);
}

void RuntimeManager::execute_gang(
    const std::vector<int>& device_ids,
    std::vector<NativeRuntime::Task> tasks) {
    std::shared_ptr<NativeRuntime> runtime;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runtime = runtime_;
    }
    if (!runtime) {
        throw std::runtime_error("EGMGPU runtime has not been started");
    }
    if (device_ids.empty()) {
        throw std::invalid_argument("EGMGPU gang execution requires devices");
    }
    DeviceGroup group(device_ids, runtime->device_group().backend());
    GangSubmission submission = runtime->submit_gang(group, std::move(tasks));
    await_all_or_rethrow(&submission.completions);
}

std::vector<int> RuntimeManager::ping() {
    std::shared_ptr<NativeRuntime> runtime;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runtime = runtime_;
    }
    if (!runtime) {
        throw std::runtime_error("EGMGPU runtime has not been started");
    }
    return runtime->ping();
}

void RuntimeManager::shutdown(bool drain) {
    std::shared_ptr<NativeRuntime> runtime;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (shutting_down_) {
            shutdown_condition_.wait(
                lock, [this]() { return !shutting_down_; });
            return;
        }
        if (!runtime_) {
            return;
        }
        shutting_down_ = true;
        runtime.swap(runtime_);
    }
    std::exception_ptr failure;
    try {
        runtime->shutdown(drain);
    } catch (...) {
        failure = std::current_exception();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutting_down_ = false;
    }
    shutdown_condition_.notify_all();
    if (failure) {
        std::rethrow_exception(failure);
    }
}

RuntimeManager& runtime_manager() {
    static RuntimeManager manager;
    return manager;
}

}  // namespace runtime
}  // namespace gpu_easygraph
