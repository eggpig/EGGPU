#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

namespace gpu_easygraph {
namespace runtime {

struct DeviceWorkerStatus {
    int device_id;
    bool running;
    bool active;
    std::size_t queued_tasks;
    std::uint64_t completed_tasks;
};

// A DeviceWorker owns one native thread.  run() binds that thread to its CUDA
// device exactly once before accepting work; individual tasks must not call
// cudaSetDevice or carry Python objects.
class DeviceWorker {
public:
    typedef std::function<void(int)> Task;
    typedef std::function<void(int)> Binder;

    explicit DeviceWorker(int device_id);
    DeviceWorker(int device_id, Binder binder);
    ~DeviceWorker();

    DeviceWorker(const DeviceWorker&) = delete;
    DeviceWorker& operator=(const DeviceWorker&) = delete;

    std::future<void> submit(std::uint64_t request_id, Task task);
    DeviceWorkerStatus status() const;
    int device_id() const;
    void stop(bool drain);

private:
    struct WorkItem {
        WorkItem(std::uint64_t request_id_value, std::packaged_task<void(int)> task_value);

        std::uint64_t request_id;
        std::packaged_task<void(int)> task;
    };

    void run(const std::shared_ptr<std::promise<void> >& initialized);

    int device_id_;
    Binder binder_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::unique_ptr<WorkItem> > queue_;
    std::thread thread_;
    bool stopping_;
    bool drain_on_stop_;
    bool running_;
    bool active_;
    std::uint64_t completed_tasks_;
};

}  // namespace runtime
}  // namespace gpu_easygraph
