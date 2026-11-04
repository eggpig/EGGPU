#include "device_worker.h"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpu_easygraph {
namespace runtime {
namespace {

std::runtime_error cuda_worker_error(
    const char* operation,
    int device_id,
    cudaError_t error) {
    std::ostringstream out;
    out << operation << " failed for CUDA device " << device_id << ": "
        << cudaGetErrorName(error) << " (" << cudaGetErrorString(error) << ")";
    return std::runtime_error(out.str());
}

}  // namespace

void bind_cuda_device(int device_id) {
    const cudaError_t bind_status = cudaSetDevice(device_id);
    if (bind_status != cudaSuccess) {
        throw cuda_worker_error("cudaSetDevice", device_id, bind_status);
    }
}

DeviceWorker::WorkItem::WorkItem(
    std::uint64_t request_id_value,
    std::packaged_task<void(int)> task_value)
    : request_id(request_id_value), task(std::move(task_value)) {}

DeviceWorker::DeviceWorker(int device_id)
    : DeviceWorker(device_id, &bind_cuda_device) {}

DeviceWorker::DeviceWorker(int device_id, Binder binder)
    : device_id_(device_id),
      binder_(std::move(binder)),
      stopping_(false),
      drain_on_stop_(true),
      running_(false),
      active_(false),
      completed_tasks_(0) {
    if (device_id < 0) {
        throw std::invalid_argument("DeviceWorker requires a non-negative device id");
    }
    if (!binder_) {
        throw std::invalid_argument("DeviceWorker requires a device binder");
    }

    const std::shared_ptr<std::promise<void> > initialized(
        new std::promise<void>());
    std::future<void> initialized_future = initialized->get_future();
    thread_ = std::thread(&DeviceWorker::run, this, initialized);
    try {
        initialized_future.get();
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            drain_on_stop_ = false;
        }
        condition_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
        throw;
    }
}

DeviceWorker::~DeviceWorker() { stop(true); }

std::future<void> DeviceWorker::submit(
    std::uint64_t request_id,
    Task task) {
    if (!task) {
        throw std::invalid_argument("cannot submit an empty EGMGPU worker task");
    }
    std::packaged_task<void(int)> packaged(std::move(task));
    std::future<void> future = packaged.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || !running_) {
            throw std::runtime_error("cannot submit work to a stopped EGMGPU worker");
        }
        queue_.push_back(std::unique_ptr<WorkItem>(
            new WorkItem(request_id, std::move(packaged))));
    }
    condition_.notify_one();
    return future;
}

DeviceWorkerStatus DeviceWorker::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    DeviceWorkerStatus out;
    out.device_id = device_id_;
    out.running = running_ && !stopping_;
    out.active = active_;
    out.queued_tasks = queue_.size();
    out.completed_tasks = completed_tasks_;
    return out;
}

int DeviceWorker::device_id() const { return device_id_; }

void DeviceWorker::stop(bool drain) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!thread_.joinable()) {
            return;
        }
        stopping_ = true;
        drain_on_stop_ = drain;
        if (!drain) {
            queue_.clear();
        }
    }
    condition_.notify_all();
    thread_.join();
}

void DeviceWorker::run(
    const std::shared_ptr<std::promise<void> >& initialized) {
    try {
        binder_(device_id_);
    } catch (...) {
        initialized->set_exception(std::current_exception());
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = true;
    }
    initialized->set_value();

    for (;;) {
        std::unique_ptr<WorkItem> item;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
            if (stopping_ && (!drain_on_stop_ || queue_.empty())) {
                running_ = false;
                active_ = false;
                return;
            }
            item = std::move(queue_.front());
            queue_.pop_front();
            active_ = true;
        }

        item->task(device_id_);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_ = false;
            ++completed_tasks_;
        }
    }
}

}  // namespace runtime
}  // namespace gpu_easygraph
