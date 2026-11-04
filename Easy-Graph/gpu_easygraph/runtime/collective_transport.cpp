#include "collective_transport.h"

#ifdef EGMGPU_HAS_NCCL
#include <nccl.h>
#endif

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ")";
    throw std::runtime_error(message.str());
}

#ifdef EGMGPU_HAS_NCCL
void check_nccl(ncclResult_t status, const char* operation) {
    if (status == ncclSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << ncclGetErrorString(status);
    throw std::runtime_error(message.str());
}

ncclDataType_t nccl_type(CollectiveDataType type) {
    switch (type) {
        case CollectiveDataType::Int32:
            return ncclInt32;
        case CollectiveDataType::Int64:
            return ncclInt64;
        case CollectiveDataType::Float32:
            return ncclFloat32;
        case CollectiveDataType::Float64:
            return ncclFloat64;
    }
    throw std::invalid_argument("unsupported collective data type");
}

ncclRedOp_t nccl_operation(CollectiveReduce operation) {
    switch (operation) {
        case CollectiveReduce::Sum:
            return ncclSum;
        case CollectiveReduce::Minimum:
            return ncclMin;
        case CollectiveReduce::Maximum:
            return ncclMax;
    }
    throw std::invalid_argument("unsupported collective reduction");
}
#endif

}  // namespace

WorkerStreams::WorkerStreams()
    : device_id_(-1),
      compute_(nullptr),
      pack_(nullptr),
      transfer_(nullptr),
      pack_ready_(nullptr),
      transfer_ready_(nullptr) {
    check_cuda(cudaGetDevice(&device_id_), "cudaGetDevice for worker streams");
    try {
        check_cuda(
            cudaStreamCreateWithFlags(&compute_, cudaStreamNonBlocking),
            "cudaStreamCreate compute");
        check_cuda(
            cudaStreamCreateWithFlags(&pack_, cudaStreamNonBlocking),
            "cudaStreamCreate pack");
        check_cuda(
            cudaStreamCreateWithFlags(&transfer_, cudaStreamNonBlocking),
            "cudaStreamCreate transfer");
        check_cuda(
            cudaEventCreateWithFlags(&pack_ready_, cudaEventDisableTiming),
            "cudaEventCreate pack_ready");
        check_cuda(
            cudaEventCreateWithFlags(&transfer_ready_, cudaEventDisableTiming),
            "cudaEventCreate transfer_ready");
    } catch (...) {
        if (pack_ready_) cudaEventDestroy(pack_ready_);
        if (transfer_ready_) cudaEventDestroy(transfer_ready_);
        if (compute_) cudaStreamDestroy(compute_);
        if (pack_) cudaStreamDestroy(pack_);
        if (transfer_) cudaStreamDestroy(transfer_);
        throw;
    }
}

WorkerStreams::~WorkerStreams() {
    if (pack_ready_) cudaEventDestroy(pack_ready_);
    if (transfer_ready_) cudaEventDestroy(transfer_ready_);
    if (compute_) cudaStreamDestroy(compute_);
    if (pack_) cudaStreamDestroy(pack_);
    if (transfer_) cudaStreamDestroy(transfer_);
}

int WorkerStreams::device_id() const { return device_id_; }
cudaStream_t WorkerStreams::compute() const { return compute_; }
cudaStream_t WorkerStreams::pack() const { return pack_; }
cudaStream_t WorkerStreams::transfer() const { return transfer_; }
cudaEvent_t WorkerStreams::pack_ready() const { return pack_ready_; }
cudaEvent_t WorkerStreams::transfer_ready() const { return transfer_ready_; }

WorkerStreams& worker_streams() {
    thread_local WorkerStreams streams;
    int current = -1;
    check_cuda(cudaGetDevice(&current), "cudaGetDevice for fixed worker validation");
    if (current != streams.device_id()) {
        throw std::runtime_error(
            "EGMGPU fixed worker changed CUDA device after stream creation");
    }
    return streams;
}

struct CollectiveContext::Impl {
#ifdef EGMGPU_HAS_NCCL
    std::vector<ncclComm_t> communicators;
#endif
};

CollectiveContext::CollectiveContext(const DeviceGroup& group)
    : group_(group), impl_(new Impl()) {
#ifdef EGMGPU_HAS_NCCL
    if (group_.size() < 2) {
        throw std::invalid_argument("NCCL context requires at least two devices");
    }
    if (group_.backend() != CommunicationBackend::Nccl) {
        throw std::invalid_argument(
            "NCCL context requires a device group resolved to NCCL");
    }
    impl_->communicators.assign(group_.size(), nullptr);
    std::vector<int> devices = group_.device_ids();
    const ncclResult_t status = ncclCommInitAll(
        impl_->communicators.data(),
        static_cast<int>(devices.size()),
        devices.data());
    if (status != ncclSuccess) {
        for (std::size_t i = 0; i < impl_->communicators.size(); ++i) {
            if (impl_->communicators[i]) {
                ncclCommAbort(impl_->communicators[i]);
                impl_->communicators[i] = nullptr;
            }
        }
        check_nccl(status, "ncclCommInitAll");
    }
#else
    (void)group_;
    throw std::runtime_error(
        "NCCL collective context requested from an EGMGPU build without NCCL");
#endif
}

CollectiveContext::~CollectiveContext() {
#ifdef EGMGPU_HAS_NCCL
    for (std::size_t i = 0; i < impl_->communicators.size(); ++i) {
        if (impl_->communicators[i]) {
            ncclCommDestroy(impl_->communicators[i]);
        }
    }
#endif
}

bool CollectiveContext::compiled() {
#ifdef EGMGPU_HAS_NCCL
    return true;
#else
    return false;
#endif
}

const DeviceGroup& CollectiveContext::group() const { return group_; }

void CollectiveContext::all_reduce(
    int device_id,
    const void* send_buffer,
    void* receive_buffer,
    std::size_t count,
    CollectiveDataType type,
    CollectiveReduce operation,
    cudaStream_t stream) const {
#ifdef EGMGPU_HAS_NCCL
    int current = -1;
    check_cuda(cudaGetDevice(&current), "cudaGetDevice for ncclAllReduce");
    if (current != device_id) {
        throw std::runtime_error(
            "NCCL collective must execute on its fixed device worker");
    }
    const std::vector<int>& devices = group_.device_ids();
    const std::vector<int>::const_iterator found =
        std::lower_bound(devices.begin(), devices.end(), device_id);
    if (found == devices.end() || *found != device_id) {
        throw std::out_of_range("device is outside the NCCL context");
    }
    const std::size_t rank = static_cast<std::size_t>(found - devices.begin());
    check_nccl(
        ncclAllReduce(
            send_buffer,
            receive_buffer,
            count,
            nccl_type(type),
            nccl_operation(operation),
            impl_->communicators[rank],
            stream),
        "ncclAllReduce");
#else
    (void)device_id;
    (void)send_buffer;
    (void)receive_buffer;
    (void)count;
    (void)type;
    (void)operation;
    (void)stream;
    throw std::runtime_error("NCCL is not compiled into EGMGPU");
#endif
}

void CollectiveContext::all_gather(
    int device_id,
    const void* send_buffer,
    void* receive_buffer,
    std::size_t count_per_rank,
    CollectiveDataType type,
    cudaStream_t stream) const {
#ifdef EGMGPU_HAS_NCCL
    int current = -1;
    check_cuda(cudaGetDevice(&current), "cudaGetDevice for ncclAllGather");
    if (current != device_id) {
        throw std::runtime_error(
            "NCCL collective must execute on its fixed device worker");
    }
    const std::vector<int>& devices = group_.device_ids();
    const std::vector<int>::const_iterator found =
        std::lower_bound(devices.begin(), devices.end(), device_id);
    if (found == devices.end() || *found != device_id) {
        throw std::out_of_range("device is outside the NCCL context");
    }
    const std::size_t rank = static_cast<std::size_t>(found - devices.begin());
    check_nccl(
        ncclAllGather(
            send_buffer,
            receive_buffer,
            count_per_rank,
            nccl_type(type),
            impl_->communicators[rank],
            stream),
        "ncclAllGather");
#else
    (void)device_id;
    (void)send_buffer;
    (void)receive_buffer;
    (void)count_per_rank;
    (void)type;
    (void)stream;
    throw std::runtime_error("NCCL is not compiled into EGMGPU");
#endif
}

void CollectiveContext::all_gather_variable_f64(
    int device_id, double* receive_buffer,
    const VariableGatherLayout& layout, cudaStream_t stream) const {
#ifdef EGMGPU_HAS_NCCL
    int current=-1;
    check_cuda(cudaGetDevice(&current), "cudaGetDevice for variable gather");
    if (current!=device_id) throw std::invalid_argument("Variable gather must run on owning fixed worker");
    const auto& devices=group_.device_ids();
    auto found=std::lower_bound(devices.begin(),devices.end(),device_id);
    if (found==devices.end() || *found!=device_id || layout.counts.size()!=devices.size())
        throw std::invalid_argument("Variable gather device/rank mismatch");
    const auto expected=variable_gather_layout(layout.counts);
    if (expected.offsets!=layout.offsets || expected.total_values!=layout.total_values ||
        expected.positive_ranks!=layout.positive_ranks || (!receive_buffer && layout.total_values))
        throw std::invalid_argument("Invalid variable gather receive layout");
    const auto rank=static_cast<std::size_t>(found-devices.begin());
    // Groups belong to this host thread. Every communicator enters the same
    // ordered broadcasts, including ranks that consume no halo values.
    check_nccl(ncclGroupStart(), "ncclGroupStart variable gather");
    ncclResult_t first_error=ncclSuccess;
    for (std::size_t root=0;root<devices.size();++root) {
        if (!layout.counts[root]) continue;
        double* segment=receive_buffer+layout.offsets[root];
        const auto status=ncclBroadcast(segment,segment,layout.counts[root],ncclFloat64,
            static_cast<int>(root),impl_->communicators[rank],stream);
        if (status!=ncclSuccess) {first_error=status;break;}
    }
    const auto ended=ncclGroupEnd();
    check_nccl(first_error, "ncclBroadcast variable gather");
    check_nccl(ended, "ncclGroupEnd variable gather");
#else
    (void)device_id;(void)receive_buffer;(void)layout;(void)stream;
    throw std::runtime_error("Variable gather requires compiled NCCL support");
#endif
}

void CollectiveContext::exchange_sparse_halo_f64(
    int device_id,const double* send_buffer,double* local_contribution,
    const SparseHaloLayout& layout,cudaStream_t stream) const {
#ifdef EGMGPU_HAS_NCCL
    int current=-1;check_cuda(cudaGetDevice(&current),"cudaGetDevice for sparse halo");
    const auto& devices=group_.device_ids();
    const auto found=std::lower_bound(devices.begin(),devices.end(),device_id);
    if (current!=device_id || found==devices.end() || *found!=device_id || layout.ranks.size()!=devices.size())
        throw std::invalid_argument("sparse halo fixed worker/rank mismatch");
    const auto rank=static_cast<std::size_t>(found-devices.begin());
    const auto& r=layout.ranks[rank];
    if ((!send_buffer && !r.source_indices.empty()) || (!local_contribution && r.local))
        throw std::invalid_argument("missing sparse halo device buffer");
    if (r.sends.empty() && r.receives.empty()) return;
    // The producer validates the whole immutable transpose before any worker
    // launches. These cheap per-run guards do not rescan O(H) indices each round.
    for (const auto& run:r.sends)
        if (run.peer>=devices.size() || run.peer==rank || !run.count || run.send_offset>r.source_indices.size() || run.count>r.source_indices.size()-run.send_offset)
            throw std::invalid_argument("invalid sparse halo send run");
    for (const auto& run:r.receives)
        if (run.peer>=devices.size() || run.peer==rank || !run.count || run.destination<r.owned || run.destination>r.local || run.count>r.local-run.destination)
            throw std::invalid_argument("invalid sparse halo receive run");
    check_nccl(ncclGroupStart(),"ncclGroupStart sparse halo");
    ncclResult_t first_error=ncclSuccess;
    for (std::size_t peer=0;peer<devices.size();++peer) {
        for (const auto& run:r.sends) if (run.peer==peer) {
            const auto status=ncclSend(send_buffer+run.send_offset,run.count,ncclFloat64,static_cast<int>(peer),impl_->communicators[rank],stream);
            if (status!=ncclSuccess && first_error==ncclSuccess) first_error=status;
        }
        for (const auto& run:r.receives) if (run.peer==peer) {
            const auto status=ncclRecv(local_contribution+run.destination,run.count,ncclFloat64,static_cast<int>(peer),impl_->communicators[rank],stream);
            if (status!=ncclSuccess && first_error==ncclSuccess) first_error=status;
        }
    }
    const auto ended=ncclGroupEnd();
    check_nccl(first_error,"ncclSend/ncclRecv sparse halo");check_nccl(ended,"ncclGroupEnd sparse halo");
#else
    (void)device_id;(void)send_buffer;(void)local_contribution;(void)layout;(void)stream;
    throw std::runtime_error("sparse halo requires compiled NCCL support");
#endif
}

void enable_worker_peer_access(
    const DeviceGroup& group,
    const DeviceTopology& topology) {
    int current = -1;
    check_cuda(cudaGetDevice(&current), "cudaGetDevice for peer access");
    if (!group.contains(current)) {
        throw std::out_of_range("current worker is outside the device group");
    }
    const std::vector<int>& devices = group.device_ids();
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const int peer = devices[i];
        if (peer == current || !topology.can_access_peer(current, peer)) {
            continue;
        }
        const cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
            continue;
        }
        check_cuda(status, "cudaDeviceEnablePeerAccess");
    }
}

void peer_copy_async(
    void* destination,
    int destination_device,
    const void* source,
    int source_device,
    std::size_t bytes,
    cudaStream_t stream) {
    if (bytes == 0) {
        return;
    }
    if (!destination || !source) {
        throw std::invalid_argument("nonempty peer copy requires valid buffers");
    }
    int current = -1;
    check_cuda(cudaGetDevice(&current), "cudaGetDevice for peer copy");
    if (current != destination_device) {
        throw std::runtime_error(
            "peer copy must be issued by the destination fixed worker");
    }
    check_cuda(
        cudaMemcpyPeerAsync(
            destination,
            destination_device,
            source,
            source_device,
            bytes,
            stream),
        "cudaMemcpyPeerAsync");
}

PinnedHostBuffer::PinnedHostBuffer(std::size_t bytes)
    : data_(nullptr), size_(bytes) {
    if (size_ != 0) {
        check_cuda(
            cudaHostAlloc(&data_, size_, cudaHostAllocPortable),
            "cudaHostAlloc staging buffer");
    }
}

PinnedHostBuffer::~PinnedHostBuffer() {
    if (data_) cudaFreeHost(data_);
}

void* PinnedHostBuffer::data() { return data_; }
const void* PinnedHostBuffer::data() const { return data_; }
std::size_t PinnedHostBuffer::size() const { return size_; }

const char* collective_data_type_name(CollectiveDataType type) {
    switch (type) {
        case CollectiveDataType::Int32:
            return "int32";
        case CollectiveDataType::Int64:
            return "int64";
        case CollectiveDataType::Float32:
            return "float32";
        case CollectiveDataType::Float64:
            return "float64";
    }
    return "unknown";
}

const char* collective_reduce_name(CollectiveReduce operation) {
    switch (operation) {
        case CollectiveReduce::Sum:
            return "sum";
        case CollectiveReduce::Minimum:
            return "min";
        case CollectiveReduce::Maximum:
            return "max";
    }
    return "unknown";
}

}  // namespace runtime
}  // namespace gpu_easygraph
