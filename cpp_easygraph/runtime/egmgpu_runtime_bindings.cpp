#include "egmgpu_runtime_bindings.h"

#ifdef EASYGRAPH_ENABLE_GPU
#include <runtime/native_runtime.h>
#include <runtime/transport_benchmark.h>

#include <string>
#include <vector>
#endif

namespace {

#ifdef EASYGRAPH_ENABLE_GPU
py::dict worker_status_dict(
    const gpu_easygraph::runtime::DeviceWorkerStatus& status) {
    py::dict out;
    out["device_id"] = status.device_id;
    out["running"] = status.running;
    out["active"] = status.active;
    out["queued_tasks"] = status.queued_tasks;
    out["completed_tasks"] = status.completed_tasks;
    return out;
}

py::dict runtime_status_dict(
    const gpu_easygraph::runtime::RuntimeStatus& status) {
    py::dict out;
    out["running"] = status.running;
    out["device_ids"] = status.device_ids;
    out["communication_backend"] =
        gpu_easygraph::runtime::communication_backend_name(status.backend);
    py::list workers;
    for (std::size_t i = 0; i < status.workers.size(); ++i) {
        workers.append(worker_status_dict(status.workers[i]));
    }
    out["workers"] = workers;
    return out;
}

py::dict topology_dict(const gpu_easygraph::runtime::DeviceTopology& topology) {
    py::dict out;
    out["runtime_available"] = topology.runtime_available();
    out["diagnostic"] = topology.diagnostic();
    out["nccl_compiled"] = topology.nccl_compiled();
    out["device_count"] = topology.devices().size();

    py::list devices;
    const std::vector<gpu_easygraph::runtime::DeviceInfo>& device_info =
        topology.devices();
    for (std::size_t i = 0; i < device_info.size(); ++i) {
        const gpu_easygraph::runtime::DeviceInfo& info = device_info[i];
        py::dict item;
        item["device_id"] = info.device_id;
        item["name"] = info.name;
        item["total_global_memory"] = info.total_global_memory;
        item["compute_capability"] =
            std::to_string(info.compute_major) + "." +
            std::to_string(info.compute_minor);
        item["multiprocessor_count"] = info.multiprocessor_count;
        item["warp_size"] = info.warp_size;
        item["pci_bus_id"] = info.pci_bus_id;
        item["unified_addressing"] = info.unified_addressing;
        devices.append(item);
    }
    out["devices"] = devices;

    py::list peer_matrix;
    for (std::size_t i = 0; i < device_info.size(); ++i) {
        py::list row;
        for (std::size_t j = 0; j < device_info.size(); ++j) {
            row.append(topology.can_access_peer(
                device_info[i].device_id,
                device_info[j].device_id));
        }
        peer_matrix.append(row);
    }
    out["peer_access"] = peer_matrix;
    return out;
}

std::vector<int> parse_devices(const py::object& devices) {
    if (devices.is_none()) {
        return std::vector<int>();
    }
    return devices.cast<std::vector<int> >();
}

py::dict transport_benchmark_dict(
    const gpu_easygraph::runtime::TransportBenchmarkResult& result) {
    py::dict out;
    py::dict config;
    config["bandwidth_bytes"] = result.config.bandwidth_bytes;
    config["bandwidth_iterations"] = result.config.bandwidth_iterations;
    config["latency_iterations"] = result.config.latency_iterations;
    config["warmup_iterations"] = result.config.warmup_iterations;
    out["config"] = config;

    py::list peer_links;
    for (std::size_t index = 0; index < result.peer_links.size(); ++index) {
        const gpu_easygraph::runtime::PeerTransportMeasurement& value =
            result.peer_links[index];
        py::dict item;
        item["from_device"] = value.from_device;
        item["to_device"] = value.to_device;
        item["supported"] = value.supported;
        item["bandwidth_gbps"] = value.bandwidth_gbps;
        item["latency_microseconds"] = value.latency_microseconds;
        item["diagnostic"] = value.diagnostic;
        peer_links.append(item);
    }
    out["peer_links"] = peer_links;

    py::list host_links;
    for (std::size_t index = 0; index < result.host_links.size(); ++index) {
        const gpu_easygraph::runtime::HostTransportMeasurement& value =
            result.host_links[index];
        py::dict item;
        item["device_id"] = value.device_id;
        item["host_to_device_gbps"] = value.host_to_device_gbps;
        item["device_to_host_gbps"] = value.device_to_host_gbps;
        item["diagnostic"] = value.diagnostic;
        host_links.append(item);
    }
    out["host_links"] = host_links;
    return out;
}
#endif

}  // namespace

void bind_egmgpu_runtime(py::module_& module) {
#ifdef EASYGRAPH_ENABLE_GPU
    module.def("cpp_egmgpu_probe_topology", []() {
        const gpu_easygraph::runtime::DeviceTopology topology =
            gpu_easygraph::runtime::DeviceTopology::probe();
        return topology_dict(topology);
    });

    module.def(
        "cpp_egmgpu_runtime_start",
        [](const py::object& devices, const std::string& backend) {
            const std::vector<int> device_ids = parse_devices(devices);
            const gpu_easygraph::runtime::CommunicationBackend requested =
                gpu_easygraph::runtime::parse_communication_backend(backend);
            gpu_easygraph::runtime::RuntimeStatus status;
            {
                py::gil_scoped_release release;
                status = gpu_easygraph::runtime::runtime_manager().start(
                    device_ids, requested);
            }
            return runtime_status_dict(status);
        },
        py::arg("devices") = py::none(),
        py::arg("backend") = "auto");

    module.def("cpp_egmgpu_runtime_status", []() {
        return runtime_status_dict(
            gpu_easygraph::runtime::runtime_manager().status());
    });

    module.def("cpp_egmgpu_runtime_ping", []() {
        std::vector<int> devices;
        {
            py::gil_scoped_release release;
            devices = gpu_easygraph::runtime::runtime_manager().ping();
        }
        return devices;
    });

    module.def(
        "cpp_egmgpu_benchmark_transports",
        [](const py::object& devices,
           std::size_t bandwidth_bytes,
           int bandwidth_iterations,
           int latency_iterations,
           int warmup_iterations) {
            std::vector<int> device_ids = parse_devices(devices);
            if (device_ids.empty()) {
                const gpu_easygraph::runtime::DeviceTopology topology =
                    gpu_easygraph::runtime::DeviceTopology::probe();
                if (!topology.runtime_available()) {
                    throw std::runtime_error(
                        "CUDA runtime is unavailable: " + topology.diagnostic());
                }
                const std::vector<gpu_easygraph::runtime::DeviceInfo>& info =
                    topology.devices();
                for (std::size_t index = 0; index < info.size(); ++index) {
                    device_ids.push_back(info[index].device_id);
                }
            }
            gpu_easygraph::runtime::TransportBenchmarkConfig config;
            config.bandwidth_bytes = bandwidth_bytes;
            config.bandwidth_iterations = bandwidth_iterations;
            config.latency_iterations = latency_iterations;
            config.warmup_iterations = warmup_iterations;
            gpu_easygraph::runtime::TransportBenchmarkResult result;
            {
                py::gil_scoped_release release;
                result = gpu_easygraph::runtime::benchmark_transports(
                    device_ids, config);
            }
            return transport_benchmark_dict(result);
        },
        py::arg("devices") = py::none(),
        py::arg("bandwidth_bytes") = std::size_t(64) << 20,
        py::arg("bandwidth_iterations") = 20,
        py::arg("latency_iterations") = 1000,
        py::arg("warmup_iterations") = 3,
        "Measure directional CUDA P2P and pinned-host transport paths.");

    module.def(
        "cpp_egmgpu_runtime_shutdown",
        [](bool drain) {
            py::gil_scoped_release release;
            gpu_easygraph::runtime::runtime_manager().shutdown(drain);
        },
        py::arg("drain") = true);
#else
    (void)module;
#endif
}
