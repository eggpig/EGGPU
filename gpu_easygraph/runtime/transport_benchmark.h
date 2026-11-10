#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct TransportBenchmarkConfig {
    std::size_t bandwidth_bytes;
    int bandwidth_iterations;
    int latency_iterations;
    int warmup_iterations;

    TransportBenchmarkConfig();
};

struct PeerTransportMeasurement {
    int from_device;
    int to_device;
    bool supported;
    double bandwidth_gbps;
    double latency_microseconds;
    std::string diagnostic;
};

struct HostTransportMeasurement {
    int device_id;
    double host_to_device_gbps;
    double device_to_host_gbps;
    std::string diagnostic;
};

struct TransportBenchmarkResult {
    TransportBenchmarkConfig config;
    std::vector<PeerTransportMeasurement> peer_links;
    std::vector<HostTransportMeasurement> host_links;
};

void validate_transport_benchmark_config(
    const TransportBenchmarkConfig& config);

// This is an explicit, stateful benchmark. It allocates CUDA buffers, enables
// peer access when supported, and synchronizes the measured streams. Merely
// probing DeviceTopology remains allocation-free and side-effect-free.
TransportBenchmarkResult benchmark_transports(
    const std::vector<int>& device_ids,
    const TransportBenchmarkConfig& config = TransportBenchmarkConfig());

}  // namespace runtime
}  // namespace gpu_easygraph
