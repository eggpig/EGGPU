#pragma once
// Observation only: call after the existing complete owning host result and
// its synchronization, before output archival. No numerical/execution change.
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <time.h>
#include <unistd.h>

inline void eggpu_sg_complete_clock(long long sequence, unsigned long long count) {
    struct timespec value;
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        throw std::runtime_error("CLOCK_MONOTONIC observation failed");
    const unsigned long long ns = static_cast<unsigned long long>(value.tv_sec) *
        1000000000ULL + static_cast<unsigned long long>(value.tv_nsec);
    std::printf("SG_COMPLETE_MONOTONIC_JSON {\"sequence\":%lld,\"values\":%llu,"
                "\"pid\":%lld,\"t_ns\":%llu,\"clock\":\"CLOCK_MONOTONIC\"}\n",
                sequence, count, static_cast<long long>(::getpid()), ns);
    std::fflush(stdout);
}
