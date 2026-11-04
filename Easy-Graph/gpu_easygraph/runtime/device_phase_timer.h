#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace gpu_easygraph {
namespace runtime {

// Optional diagnostics for any stream operator. Recording never synchronizes;
// the caller reads elapsed time after its existing algorithmic completion.
// Timing events are reused for the full call, rather than allocated per step.
class DevicePhaseTimer {
public:
    explicit DevicePhaseTimer(bool enabled) : enabled_(enabled) {
        if (enabled_) {
            check(cudaEventCreate(&begin_));
            check(cudaEventCreate(&end_));
        }
    }
    ~DevicePhaseTimer() {
        if (end_) cudaEventDestroy(end_);
        if (begin_) cudaEventDestroy(begin_);
    }
    DevicePhaseTimer(const DevicePhaseTimer&) = delete;
    DevicePhaseTimer& operator=(const DevicePhaseTimer&) = delete;
    void begin(cudaStream_t stream) {
        if (enabled_) check(cudaEventRecord(begin_, stream));
    }
    void end(cudaStream_t stream) {
        if (enabled_) check(cudaEventRecord(end_, stream));
    }
    double seconds() const {
        if (!enabled_) return 0.0;
        float ms = 0.0F;
        check(cudaEventElapsedTime(&ms, begin_, end_));
        return static_cast<double>(ms) / 1000.0;
    }
private:
    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("device phase timer: ") + cudaGetErrorString(status));
    }
    bool enabled_;
    cudaEvent_t begin_ = nullptr;
    cudaEvent_t end_ = nullptr;
};

}  // namespace runtime
}  // namespace gpu_easygraph
