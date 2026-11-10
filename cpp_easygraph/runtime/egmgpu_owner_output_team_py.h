#pragma once
#include "egmgpu_owner_output_team.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace easygraph { namespace egmgpu { namespace owner_output {
inline std::vector<int> cpu_list(const cpu_set_t& mask) {
    std::vector<int> result;
    for (int cpu=0; cpu<CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &mask)) result.push_back(cpu);
    return result;
}
inline pybind11::dict placement_to_python(const CpuPlacement& placement) {
    pybind11::dict out;
    out["mode"]=placement.explicit_binding ? "explicit" : "inherited";
    out["worker_cpus"]=placement.workers;
    out["scope"]="output body only; each thread restores its own entry mask before team join";
    return out;
}
inline pybind11::dict to_python(const Observation& observation) {
    pybind11::dict out;
    out["actual"] = observation.actual;
    pybind11::list threads;
    for (const auto& thread : observation.threads) {
        pybind11::dict item;
        item["cpu"] = thread.cpu;
        item["allowed_cpus"] = cpu_list(thread.mask);
        item["requested_cpu"] = thread.requested_cpu;
        item["entry_allowed_cpus"] = thread.before_valid ? pybind11::cast(cpu_list(thread.before_mask)) : pybind11::none();
        item["restored_allowed_cpus"] = thread.restored_valid ? pybind11::cast(cpu_list(thread.restored_mask)) : pybind11::none();
        item["bind_error"] = thread.bind_error;
        item["restore_error"] = thread.restore_error;
        item["bind_wall_seconds"] = thread.before_valid ? pybind11::cast(thread.bind_wall_seconds) : pybind11::none();
        item["team_start_wait_wall_seconds"] = thread.before_valid ? pybind11::cast(thread.team_start_wait_wall_seconds) : pybind11::none();
        item["body_wall_seconds"] = thread.before_valid ? pybind11::cast(thread.body_wall_seconds) : pybind11::none();
        item["restore_wall_seconds"] = thread.restored_valid ? pybind11::cast(thread.restore_wall_seconds) : pybind11::none();
        threads.append(item);
    }
    out["threads"] = threads;
    return out;
}
} } }
