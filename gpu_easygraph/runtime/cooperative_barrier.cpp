#include "cooperative_barrier.h"

#include <stdexcept>
#include <thread>

namespace gpu_easygraph {
namespace runtime {

CooperativeBarrier::CooperativeBarrier(std::size_t participants)
    : participants_(participants), arrived_(0), generation_(0), cancelled_(false) {
    if (participants_ == 0)
        throw std::invalid_argument("cooperative barrier requires at least one participant");
}

void CooperativeBarrier::arrive_and_wait() {
    arrive_and_wait_impl(nullptr, nullptr);
}

void CooperativeBarrier::arrive_and_wait_impl(void (*completion)(void*), void* context) {
    if (cancelled_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_) std::rethrow_exception(failure_);
        throw std::runtime_error("EGMGPU cooperative request was cancelled");
    }
    const std::size_t observed = generation_.load(std::memory_order_acquire);
    // The RMW chain publishes every rank's preceding host writes to the last
    // arrival. Only that rank runs the completion and advances the generation.
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1U == participants_) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cancelled_.load(std::memory_order_acquire)) {
            if (failure_) std::rethrow_exception(failure_);
            throw std::runtime_error("EGMGPU cooperative request was cancelled");
        }
        try {
            if (completion) completion(context);
        } catch (...) {
            failure_ = std::current_exception();
            cancelled_.store(true, std::memory_order_release);
            lock.unlock();
            condition_.notify_all();
            throw;
        }
        arrived_.store(0, std::memory_order_relaxed);
        // Publication holds the same mutex as the sleeping path's predicate
        // check. A notification therefore cannot be lost at the sleep boundary.
        generation_.store(observed + 1U, std::memory_order_release);
        lock.unlock();
        condition_.notify_all();
        return;
    }
    for (unsigned spin=0; spin<4096U; ++spin) {
        if (cancelled_.load(std::memory_order_acquire)) break;
        if (generation_.load(std::memory_order_acquire) != observed) return;
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ __volatile__("yield");
#else
        std::this_thread::yield();
#endif
    }
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this,observed] {
        return cancelled_.load(std::memory_order_acquire) ||
               generation_.load(std::memory_order_acquire) != observed;
    });
    if (cancelled_.load(std::memory_order_acquire)) {
        if (failure_) std::rethrow_exception(failure_);
        throw std::runtime_error("EGMGPU cooperative request was cancelled");
    }
}

void CooperativeBarrier::cancel(std::exception_ptr failure) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!cancelled_.load(std::memory_order_relaxed)) {
            failure_=failure;
            cancelled_.store(true,std::memory_order_release);
        }
    }
    condition_.notify_all();
}

bool CooperativeBarrier::cancelled() const {
    return cancelled_.load(std::memory_order_acquire);
}

std::size_t CooperativeBarrier::generation() const {
    return generation_.load(std::memory_order_acquire);
}

}  // namespace runtime
}  // namespace gpu_easygraph
