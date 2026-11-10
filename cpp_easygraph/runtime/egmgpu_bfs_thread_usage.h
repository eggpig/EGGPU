#pragma once

#include <chrono>
#include <cstdint>
#include <cerrno>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <system_error>
#include <time.h>
#include <unistd.h>

namespace easygraph { namespace egmgpu {

struct BfsThreadUsageSample {
    std::int64_t steady_ns = 0, monotonic_ns = 0, thread_cpu_ns = 0;
    std::int64_t user_cpu_us = 0, system_cpu_us = 0;
    std::int64_t minor_faults = 0, major_faults = 0;
    std::int64_t voluntary_switches = 0, involuntary_switches = 0;
    long tid = 0;
    int cpu = -1;
};

inline BfsThreadUsageSample sample_bfs_thread_usage() {
    BfsThreadUsageSample result;
    result.steady_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    timespec monotonic, thread_cpu;
    if (clock_gettime(CLOCK_MONOTONIC, &monotonic) != 0 ||
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &thread_cpu) != 0)
        throw std::system_error(errno, std::generic_category(), "BFS usage clock_gettime");
    rusage usage;
    if (getrusage(RUSAGE_THREAD, &usage) != 0)
        throw std::system_error(errno, std::generic_category(), "BFS RUSAGE_THREAD");
    result.monotonic_ns = monotonic.tv_sec * std::int64_t(1000000000) + monotonic.tv_nsec;
    result.thread_cpu_ns = thread_cpu.tv_sec * std::int64_t(1000000000) + thread_cpu.tv_nsec;
    result.user_cpu_us = usage.ru_utime.tv_sec * std::int64_t(1000000) + usage.ru_utime.tv_usec;
    result.system_cpu_us = usage.ru_stime.tv_sec * std::int64_t(1000000) + usage.ru_stime.tv_usec;
    result.minor_faults = usage.ru_minflt; result.major_faults = usage.ru_majflt;
    result.voluntary_switches = usage.ru_nvcsw; result.involuntary_switches = usage.ru_nivcsw;
    result.tid = syscall(SYS_gettid); result.cpu = sched_getcpu();
    return result;
}

struct BfsThreadUsageWindow {
    bool measured = false;
    BfsThreadUsageSample begin, end;
    void start() { begin = sample_bfs_thread_usage(); }
    void finish() { end = sample_bfs_thread_usage(); measured = true; }
};

} }  // namespace easygraph::egmgpu
