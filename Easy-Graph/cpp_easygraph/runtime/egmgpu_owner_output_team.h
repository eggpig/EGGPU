#pragma once

#include <cerrno>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <sched.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if EASYGRAPH_USE_OPENMP
#include <omp.h>
#endif

namespace easygraph { namespace egmgpu { namespace owner_output {

// One immutable decision is resolved before dispatch. This experimental switch
// is process configuration, not a service-level per-request policy interface.
inline bool enabled(bool supports_explicit_placement = false) {
    const char* placement = std::getenv("EGMGPU_CORE_OUTPUT_CPU_PLACEMENT");
    if (placement && std::string(placement) != "inherited" && !supports_explicit_placement)
        throw std::invalid_argument("this core output entry has no explicit CPU placement contract");
    const char* raw = std::getenv("EGMGPU_CORE_OUTPUT_TEAM");
    const std::string policy = raw ? raw : "original";
    if (policy == "original") return false;
    if (policy != "team16")
        throw std::invalid_argument("EGMGPU_CORE_OUTPUT_TEAM must be original or team16");
#if EASYGRAPH_USE_OPENMP
    return true;
#else
    throw std::invalid_argument("core output team16 requires an OpenMP build");
#endif
}

inline int requested_width(std::size_t workers) {
    if (!workers || workers > 16U || 16U % workers)
        throw std::invalid_argument("core output worker count must divide 16");
#if !EASYGRAPH_USE_OPENMP
    throw std::invalid_argument("core output team16 requires an OpenMP build");
#endif
    return static_cast<int>(16U / workers);
}

struct CpuPlacement {
    bool explicit_binding = false;
    std::vector<std::vector<int> > workers;
    const std::vector<int>* for_worker(std::size_t rank) const {
        return explicit_binding ? &workers.at(rank) : nullptr;
    }
};

// Called once by the request caller before dispatch; workers never read env.
// The resulting object is held as shared_ptr<const CpuPlacement> in the query.
inline CpuPlacement parse_placement(bool team16, std::size_t physical_workers) {
    CpuPlacement result;
    const char* raw = std::getenv("EGMGPU_CORE_OUTPUT_CPU_PLACEMENT");
    const std::string mode = raw ? raw : "inherited";
    if (mode == "inherited") return result;
    if (mode != "explicit")
        throw std::invalid_argument("core output CPU placement must be inherited or explicit");
    if (!team16) throw std::invalid_argument("explicit output CPU placement requires team16");
    const int width = requested_width(physical_workers);
    raw = std::getenv("EGMGPU_CORE_OUTPUT_CPUS");
    if (!raw || !*raw) throw std::invalid_argument("explicit output placement requires worker CPU lists");
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        throw std::runtime_error("cannot observe caller CPU budget for output placement");
    std::set<int> assigned;
    std::stringstream groups(raw);
    std::string group;
    while (std::getline(groups, group, ';')) {
        if (group.empty()) throw std::invalid_argument("empty output worker CPU list");
        std::stringstream fields(group);
        std::string field;
        std::vector<int> cpus;
        while (std::getline(fields, field, ',')) {
            char* end = nullptr;
            errno = 0;
            const long cpu = std::strtol(field.c_str(), &end, 10);
            if (field.empty() || end == field.c_str() || *end || errno ||
                cpu < 0 || cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed))
                throw std::invalid_argument("output CPU is outside the caller's allowed budget");
            if (!assigned.insert(static_cast<int>(cpu)).second)
                throw std::invalid_argument("output worker CPU lists must be disjoint");
            cpus.push_back(static_cast<int>(cpu));
        }
        if (group.back() == ',' || cpus.size() != static_cast<std::size_t>(width))
            throw std::invalid_argument("output worker CPU count differs from its team width");
        result.workers.push_back(std::move(cpus));
    }
    if (std::string(raw).back() == ';' || result.workers.size() != physical_workers)
        throw std::invalid_argument("output CPU list count differs from actual GPU worker count");
    result.explicit_binding = true;
    return result;
}

struct ThreadObservation {
    cpu_set_t mask;
    cpu_set_t before_mask;
    cpu_set_t restored_mask;
    int cpu = -1;
    int error = 0;
    int requested_cpu = -1;
    int bind_error = 0;
    int restore_error = 0;
    bool before_valid = false;
    bool restored_valid = false;
    bool affinity_changed = false;
    double bind_wall_seconds = 0.0;
    double team_start_wait_wall_seconds = 0.0;
    double body_wall_seconds = 0.0;
    double restore_wall_seconds = 0.0;
    std::exception_ptr callback_error;
    std::chrono::steady_clock::time_point body_begin;
    ThreadObservation() { CPU_ZERO(&mask); CPU_ZERO(&before_mask); CPU_ZERO(&restored_mask); }
    void capture() noexcept {
        cpu = sched_getcpu();
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0) error = errno;
    }
    void enter(int target) noexcept {
        const auto begin = std::chrono::steady_clock::now();
        requested_cpu = target;
        before_valid = sched_getaffinity(0, sizeof(before_mask), &before_mask) == 0;
        if (!before_valid) bind_error = errno;
        else if (target >= 0) {
            if (target >= CPU_SETSIZE || !CPU_ISSET(target, &before_mask)) bind_error = EINVAL;
            else {
                cpu_set_t selected; CPU_ZERO(&selected); CPU_SET(target, &selected);
                if (sched_setaffinity(0, sizeof(selected), &selected) != 0) bind_error = errno;
                else affinity_changed = true;
            }
        }
        capture();
        if (!error && target >= 0 && !bind_error &&
            (cpu != target || CPU_COUNT(&mask) != 1 || !CPU_ISSET(target, &mask))) bind_error = EINVAL;
        body_begin = std::chrono::steady_clock::now();
        bind_wall_seconds = std::chrono::duration<double>(body_begin-begin).count();
    }
    void leave() noexcept {
        const auto begin = std::chrono::steady_clock::now();
        body_wall_seconds = std::chrono::duration<double>(begin-body_begin).count();
        // Restore explicit attempts even when set/capture failed; entry is the
        // thread's actual mask for this invocation, never a process-wide guess.
        if (requested_cpu >= 0 && before_valid && sched_setaffinity(0, sizeof(before_mask), &before_mask) != 0)
            restore_error = errno;
        restored_valid = sched_getaffinity(0, sizeof(restored_mask), &restored_mask) == 0;
        if (!restored_valid && !restore_error) restore_error = errno;
        if (before_valid && restored_valid && !CPU_EQUAL(&before_mask, &restored_mask) && !restore_error)
            restore_error = EINVAL;
        restore_wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    }
    void begin_body() noexcept {
        const auto now = std::chrono::steady_clock::now();
        team_start_wait_wall_seconds = std::chrono::duration<double>(now-body_begin).count();
        body_begin = now;
    }
};

struct Observation {
    int actual = 0;
    std::vector<ThreadObservation> threads;
};

inline void check_observation(const Observation& observation, int width) {
    if (observation.actual != width)
        throw std::runtime_error("core output actual thread width differs from requested budget");
    for (const auto& thread : observation.threads)
        if (thread.error || thread.cpu < 0 || thread.bind_error || thread.restore_error)
            throw std::runtime_error("core output affinity/observation failed: requested_cpu=" +
                std::to_string(thread.requested_cpu) + ", bind_errno=" + std::to_string(thread.bind_error) +
                ", restore_errno=" + std::to_string(thread.restore_error) + ", observation_errno=" +
                std::to_string(thread.error));
    for (const auto& thread : observation.threads)
        if (thread.callback_error) std::rethrow_exception(thread.callback_error);
}

inline void observe_original(Observation& observation) {
    observation.actual = 1;
    observation.threads.resize(1U);
    observation.threads[0].capture();
    check_observation(observation, 1);
}

inline std::pair<std::size_t, std::size_t> bounds(
    std::size_t count, std::size_t thread, std::size_t width) noexcept {
    // Quotient/remainder arithmetic avoids count*thread overflow.
    const std::size_t quotient = count / width, remainder = count % width;
    const auto offset = [&](std::size_t t) {
        return quotient * t + (t < remainder ? t : remainder);
    };
    return std::make_pair(offset(thread), offset(thread + 1U));
}

// The callback writes a disjoint [begin,end) owner-local range and must not
// throw. All callbacks finish before returning, preserving pinned-source leases
// and the owning output's publication boundary. No Python/CUDA work occurs here.
template <class Function>
void run(std::size_t count, int width, Observation& observation, Function function,
         const std::vector<int>* cpu_ids = nullptr) {
    if (width < 1 || width > 16)
        throw std::invalid_argument("invalid core output team width");
    if (cpu_ids && cpu_ids->size() != static_cast<std::size_t>(width))
        throw std::invalid_argument("output CPU list length differs from team width");
    // A reused observation must not retain prior affinity or error state.
    observation.threads.assign(static_cast<std::size_t>(width), ThreadObservation());
    std::atomic<bool> binding_failed(false);
#if EASYGRAPH_USE_OPENMP
#pragma omp parallel num_threads(width)
    {
        const int thread = omp_get_thread_num();
        auto& observed = observation.threads[static_cast<std::size_t>(thread)];
        observed.enter(cpu_ids ? (*cpu_ids)[static_cast<std::size_t>(thread)] : -1);
        if (observed.bind_error || observed.error) binding_failed.store(true);
#pragma omp single
        observation.actual = omp_get_num_threads();
        // The existing single's implicit barrier publishes all binding results.
        // Failed binding skips all writes; restoration still runs on every thread.
        observed.begin_body();
        const auto range = bounds(count, static_cast<std::size_t>(thread),
            static_cast<std::size_t>(omp_get_num_threads()));
        if (!binding_failed.load()) {
            try { function(range.first, range.second); }
            catch (...) { observed.callback_error = std::current_exception(); }
        }
        observed.leave();
    }
#else
    if (width != 1) throw std::invalid_argument("core output team requires OpenMP");
    observation.actual = 1;
    auto& observed = observation.threads[0];
    observed.enter(cpu_ids ? (*cpu_ids)[0] : -1);
    observed.begin_body();
    if (!observed.bind_error && !observed.error) {
        try { function(0U, count); }
        catch (...) { observed.callback_error = std::current_exception(); }
    }
    observed.leave();
#endif
    check_observation(observation, width);
}

} } }
