#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>

namespace gpu_easygraph {
namespace runtime {

// Reusable, cancellation-aware barrier for iterative multi-device operators.
// If one rank fails, cancel() wakes every peer so a cooperative request cannot
// deadlock while waiting at its next communication or termination phase.
class CooperativeBarrier {
public:
    explicit CooperativeBarrier(std::size_t participants);

    CooperativeBarrier(const CooperativeBarrier&) = delete;
    CooperativeBarrier& operator=(const CooperativeBarrier&) = delete;

    void arrive_and_wait();
    // Runs exactly once, on the last arriving rank, before publishing this
    // generation. Completion must be host-only and must not re-enter this
    // barrier. A thrown exception cancels/wakes every participant before release.
    // The callback/context remain on the arriving rank's stack: no allocation.
    template <typename Completion>
    void arrive_and_wait_then(Completion completion) {
        arrive_and_wait_impl([](void* context) {
            (*static_cast<Completion*>(context))();
        }, &completion);
    }
    void cancel(std::exception_ptr failure = std::exception_ptr());
    bool cancelled() const;
    std::size_t generation() const;

private:
    void arrive_and_wait_impl(void (*completion)(void*), void* context);
    const std::size_t participants_;
    std::size_t arrived_;
    std::size_t generation_;
    bool cancelled_;
    std::exception_ptr failure_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};

template <typename Function>
void run_cooperative_rank(CooperativeBarrier& barrier, Function function) {
    try {
        function();
    } catch (...) {
        barrier.cancel(std::current_exception());
        throw;
    }
}

}  // namespace runtime
}  // namespace gpu_easygraph
