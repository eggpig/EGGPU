#include "cooperative_barrier.h"

#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {

CooperativeBarrier::CooperativeBarrier(std::size_t participants)
    : participants_(participants),
      arrived_(0),
      generation_(0),
      cancelled_(false) {
    if (participants_ == 0) {
        throw std::invalid_argument(
            "cooperative barrier requires at least one participant");
    }
}

void CooperativeBarrier::arrive_and_wait() {
    arrive_and_wait_impl(nullptr, nullptr);
}

void CooperativeBarrier::arrive_and_wait_impl(
    void (*completion)(void*), void* context) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cancelled_) {
        if (failure_) std::rethrow_exception(failure_);
        throw std::runtime_error("EGMGPU cooperative request was cancelled");
    }
    const std::size_t observed_generation = generation_;
    ++arrived_;
    if (arrived_ == participants_) {
        try {
            if (completion) completion(context);
        } catch (...) {
            // Still hold the generation mutex: no successful waiter can
            // observe a partial reduction or enter the next generation.
            cancelled_ = true;
            failure_ = std::current_exception();
            arrived_ = 0;
            condition_.notify_all();
            throw;
        }
        arrived_ = 0;
        ++generation_;
        condition_.notify_all();
        return;
    }
    condition_.wait(lock, [this, observed_generation]() {
        return cancelled_ || generation_ != observed_generation;
    });
    if (cancelled_) {
        if (failure_) std::rethrow_exception(failure_);
        throw std::runtime_error("EGMGPU cooperative request was cancelled");
    }
}

void CooperativeBarrier::cancel(std::exception_ptr failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cancelled_) {
        cancelled_ = true;
        failure_ = failure;
    }
    condition_.notify_all();
}

bool CooperativeBarrier::cancelled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelled_;
}

std::size_t CooperativeBarrier::generation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return generation_;
}

}  // namespace runtime
}  // namespace gpu_easygraph
