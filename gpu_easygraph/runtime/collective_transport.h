#pragma once
#include "variable_gather_layout.h"
#include "sparse_halo_layout.h"

#include "device_topology.h"
#include "runtime_types.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

enum class CollectiveDataType {
    Int32,
    Int64,
    Float32,
    Float64,
};

enum class CollectiveReduce {
    Sum,
    Minimum,
    Maximum,
};

// Streams are thread-local because one fixed worker owns one CUDA device.
// New partitioned kernels can overlap packing, transfer, and local compute
// without changing the legacy EGGPU default-stream kernels.
class WorkerStreams {
public:
    WorkerStreams();
    ~WorkerStreams();

    WorkerStreams(const WorkerStreams&) = delete;
    WorkerStreams& operator=(const WorkerStreams&) = delete;

    int device_id() const;
    cudaStream_t compute() const;
    cudaStream_t pack() const;
    cudaStream_t transfer() const;
    cudaEvent_t pack_ready() const;
    cudaEvent_t transfer_ready() const;

private:
    int device_id_;
    cudaStream_t compute_;
    cudaStream_t pack_;
    cudaStream_t transfer_;
    cudaEvent_t pack_ready_;
    cudaEvent_t transfer_ready_;
};

WorkerStreams& worker_streams();

// One communicator per device rank.  Construction is explicit and occurs
// only when a measured topology selects NCCL; no static initializer creates
// CUDA state.
class CollectiveContext {
public:
    explicit CollectiveContext(const DeviceGroup& group);
    ~CollectiveContext();

    CollectiveContext(const CollectiveContext&) = delete;
    CollectiveContext& operator=(const CollectiveContext&) = delete;

    static bool compiled();
    const DeviceGroup& group() const;

    void all_reduce(
        int device_id,
        const void* send_buffer,
        void* receive_buffer,
        std::size_t count,
        CollectiveDataType type,
        CollectiveReduce operation,
        cudaStream_t stream) const;

    void all_gather(
        int device_id,
        const void* send_buffer,
        void* receive_buffer,
        std::size_t count_per_rank,
        CollectiveDataType type,
        cudaStream_t stream) const;

    // Each fixed GPU worker publishes its packed local slot before entering.
    // All workers use identical counts/root ordering; this is not sparse exchange.
    void all_gather_variable_f64(int device_id, double* receive_buffer,
        const VariableGatherLayout& layout, cudaStream_t stream) const;

    // Query-local send packing, direct receives into immutable validated halo
    // slots. Nonempty directions are fused in this fixed worker's NCCL group.
    void exchange_sparse_halo_f64(int device_id,const double* send_buffer,
        double* local_contribution,const SparseHaloLayout& layout,cudaStream_t stream) const;

private:
    struct Impl;
    DeviceGroup group_;
    std::unique_ptr<Impl> impl_;
};

// Enable every valid directed peer edge from the current fixed worker.  The
// function is idempotent and never changes the current CUDA device.
void enable_worker_peer_access(
    const DeviceGroup& group,
    const DeviceTopology& topology);

// Issue a destination-owned peer copy on the current fixed worker stream.
void peer_copy_async(
    void* destination,
    int destination_device,
    const void* source,
    int source_device,
    std::size_t bytes,
    cudaStream_t stream);

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    void* data();
    const void* data() const;
    std::size_t size() const;

private:
    void* data_;
    std::size_t size_;
};

const char* collective_data_type_name(CollectiveDataType type);
const char* collective_reduce_name(CollectiveReduce operation);

}  // namespace runtime
}  // namespace gpu_easygraph
