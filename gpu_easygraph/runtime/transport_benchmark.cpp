#include "transport_benchmark.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {
namespace {

std::string cuda_message(const char* operation, cudaError_t status) {
    std::ostringstream output;
    output << operation << " failed: " << cudaGetErrorName(status)
           << " (" << cudaGetErrorString(status) << ")";
    return output.str();
}

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_message(operation, status));
    }
}

class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() : device_(-1), valid_(false) {
        valid_ = cudaGetDevice(&device_) == cudaSuccess;
    }

    ~CurrentDeviceGuard() {
        if (valid_) {
            (void)cudaSetDevice(device_);
        }
    }

private:
    int device_;
    bool valid_;
};

class DeviceAllocation {
public:
    DeviceAllocation(int device_id, std::size_t bytes)
        : device_id_(device_id), pointer_(nullptr) {
        require_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
        require_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }

    ~DeviceAllocation() {
        if (pointer_ != nullptr) {
            (void)cudaSetDevice(device_id_);
            (void)cudaFree(pointer_);
        }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    void* get() const { return pointer_; }

private:
    int device_id_;
    void* pointer_;
};

class PinnedAllocation {
public:
    explicit PinnedAllocation(std::size_t bytes) : pointer_(nullptr) {
        require_cuda(cudaMallocHost(&pointer_, bytes), "cudaMallocHost");
    }

    ~PinnedAllocation() {
        if (pointer_ != nullptr) {
            (void)cudaFreeHost(pointer_);
        }
    }

    PinnedAllocation(const PinnedAllocation&) = delete;
    PinnedAllocation& operator=(const PinnedAllocation&) = delete;

    void* get() const { return pointer_; }

private:
    void* pointer_;
};

class StreamAndEvents {
public:
    explicit StreamAndEvents(int device_id)
        : device_id_(device_id), stream_(nullptr), start_(nullptr), stop_(nullptr) {
        require_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
        require_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
        try {
            require_cuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
            require_cuda(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
        } catch (...) {
            if (start_ != nullptr) (void)cudaEventDestroy(start_);
            (void)cudaStreamDestroy(stream_);
            throw;
        }
    }

    ~StreamAndEvents() {
        (void)cudaSetDevice(device_id_);
        if (stop_ != nullptr) (void)cudaEventDestroy(stop_);
        if (start_ != nullptr) (void)cudaEventDestroy(start_);
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }

    cudaStream_t stream() const { return stream_; }
    cudaEvent_t start() const { return start_; }
    cudaEvent_t stop() const { return stop_; }

private:
    int device_id_;
    cudaStream_t stream_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

template <typename Enqueue>
double timed_iterations(
    StreamAndEvents& timing,
    int warmups,
    int iterations,
    Enqueue enqueue) {
    for (int index = 0; index < warmups; ++index) {
        require_cuda(enqueue(timing.stream()), "transport warmup copy");
    }
    require_cuda(cudaStreamSynchronize(timing.stream()), "transport warmup sync");
    require_cuda(cudaEventRecord(timing.start(), timing.stream()), "event record start");
    for (int index = 0; index < iterations; ++index) {
        require_cuda(enqueue(timing.stream()), "transport measured copy");
    }
    require_cuda(cudaEventRecord(timing.stop(), timing.stream()), "event record stop");
    require_cuda(cudaEventSynchronize(timing.stop()), "event synchronize stop");
    float milliseconds = 0.0F;
    require_cuda(
        cudaEventElapsedTime(&milliseconds, timing.start(), timing.stop()),
        "cudaEventElapsedTime");
    if (!(milliseconds > 0.0F)) {
        throw std::runtime_error("CUDA event timer returned a non-positive interval");
    }
    return static_cast<double>(milliseconds) * 1.0e-3;
}

double gigabytes_per_second(
    std::size_t bytes,
    int iterations,
    double seconds) {
    return static_cast<double>(bytes) * static_cast<double>(iterations) /
           seconds / 1.0e9;
}

void enable_peer_if_needed(int accessing_device, int peer_device) {
    require_cuda(cudaSetDevice(accessing_device), "cudaSetDevice(peer enable)");
    const cudaError_t status = cudaDeviceEnablePeerAccess(peer_device, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
        return;
    }
    require_cuda(status, "cudaDeviceEnablePeerAccess");
}

PeerTransportMeasurement benchmark_peer(
    int from_device,
    int to_device,
    const TransportBenchmarkConfig& config) {
    PeerTransportMeasurement result;
    result.from_device = from_device;
    result.to_device = to_device;
    result.supported = false;
    result.bandwidth_gbps = 0.0;
    result.latency_microseconds = 0.0;

    int accessible = 0;
    const cudaError_t access_status =
        cudaDeviceCanAccessPeer(&accessible, to_device, from_device);
    if (access_status != cudaSuccess) {
        result.diagnostic = cuda_message("cudaDeviceCanAccessPeer", access_status);
        return result;
    }
    if (accessible == 0) {
        result.diagnostic = "destination device cannot directly access source memory";
        return result;
    }

    try {
        enable_peer_if_needed(to_device, from_device);
        DeviceAllocation source(from_device, config.bandwidth_bytes);
        DeviceAllocation destination(to_device, config.bandwidth_bytes);
        require_cuda(cudaSetDevice(to_device), "cudaSetDevice(peer benchmark)");
        StreamAndEvents timing(to_device);
        const double bandwidth_seconds = timed_iterations(
            timing,
            config.warmup_iterations,
            config.bandwidth_iterations,
            [&](cudaStream_t stream) {
                return cudaMemcpyPeerAsync(
                    destination.get(), to_device,
                    source.get(), from_device,
                    config.bandwidth_bytes, stream);
            });
        const double latency_seconds = timed_iterations(
            timing,
            config.warmup_iterations,
            config.latency_iterations,
            [&](cudaStream_t stream) {
                return cudaMemcpyPeerAsync(
                    destination.get(), to_device,
                    source.get(), from_device,
                    4, stream);
            });
        result.supported = true;
        result.bandwidth_gbps = gigabytes_per_second(
            config.bandwidth_bytes,
            config.bandwidth_iterations,
            bandwidth_seconds);
        result.latency_microseconds =
            latency_seconds * 1.0e6 /
            static_cast<double>(config.latency_iterations);
    } catch (const std::exception& error) {
        result.diagnostic = error.what();
    }
    return result;
}

HostTransportMeasurement benchmark_host(
    int device_id,
    const TransportBenchmarkConfig& config) {
    HostTransportMeasurement result;
    result.device_id = device_id;
    result.host_to_device_gbps = 0.0;
    result.device_to_host_gbps = 0.0;
    try {
        DeviceAllocation device(device_id, config.bandwidth_bytes);
        PinnedAllocation host(config.bandwidth_bytes);
        require_cuda(cudaSetDevice(device_id), "cudaSetDevice(host benchmark)");
        StreamAndEvents timing(device_id);
        const double h2d_seconds = timed_iterations(
            timing,
            config.warmup_iterations,
            config.bandwidth_iterations,
            [&](cudaStream_t stream) {
                return cudaMemcpyAsync(
                    device.get(), host.get(), config.bandwidth_bytes,
                    cudaMemcpyHostToDevice, stream);
            });
        const double d2h_seconds = timed_iterations(
            timing,
            config.warmup_iterations,
            config.bandwidth_iterations,
            [&](cudaStream_t stream) {
                return cudaMemcpyAsync(
                    host.get(), device.get(), config.bandwidth_bytes,
                    cudaMemcpyDeviceToHost, stream);
            });
        result.host_to_device_gbps = gigabytes_per_second(
            config.bandwidth_bytes,
            config.bandwidth_iterations,
            h2d_seconds);
        result.device_to_host_gbps = gigabytes_per_second(
            config.bandwidth_bytes,
            config.bandwidth_iterations,
            d2h_seconds);
    } catch (const std::exception& error) {
        result.diagnostic = error.what();
    }
    return result;
}

}  // namespace

TransportBenchmarkConfig::TransportBenchmarkConfig()
    : bandwidth_bytes(std::size_t(64) << 20),
      bandwidth_iterations(20),
      latency_iterations(1000),
      warmup_iterations(3) {}

void validate_transport_benchmark_config(
    const TransportBenchmarkConfig& config) {
    if (config.bandwidth_bytes < 4) {
        throw std::invalid_argument("bandwidth_bytes must be at least four");
    }
    if (config.bandwidth_bytes >
        static_cast<std::size_t>(std::numeric_limits<int>::max()) * 1024U) {
        throw std::invalid_argument("bandwidth_bytes is unreasonably large");
    }
    if (config.bandwidth_iterations <= 0 ||
        config.latency_iterations <= 0 ||
        config.warmup_iterations < 0) {
        throw std::invalid_argument(
            "transport iterations must be positive and warmups non-negative");
    }
}

TransportBenchmarkResult benchmark_transports(
    const std::vector<int>& device_ids,
    const TransportBenchmarkConfig& config) {
    validate_transport_benchmark_config(config);
    if (device_ids.empty()) {
        throw std::invalid_argument("transport benchmark requires CUDA devices");
    }
    std::vector<int> devices = device_ids;
    std::sort(devices.begin(), devices.end());
    if (std::adjacent_find(devices.begin(), devices.end()) != devices.end() ||
        devices.front() < 0) {
        throw std::invalid_argument(
            "transport benchmark devices must be unique and non-negative");
    }

    int device_count = 0;
    require_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (devices.back() >= device_count) {
        throw std::out_of_range(
            "transport benchmark references an unavailable CUDA device");
    }

    CurrentDeviceGuard restore_device;
    TransportBenchmarkResult result;
    result.config = config;
    for (std::size_t index = 0; index < devices.size(); ++index) {
        result.host_links.push_back(benchmark_host(devices[index], config));
    }
    for (std::size_t source = 0; source < devices.size(); ++source) {
        for (std::size_t target = 0; target < devices.size(); ++target) {
            if (source == target) continue;
            result.peer_links.push_back(
                benchmark_peer(devices[source], devices[target], config));
        }
    }
    return result;
}

}  // namespace runtime
}  // namespace gpu_easygraph
