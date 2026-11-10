#include "pagerank_cusparse_transpose.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <chrono>
#include <stdexcept>
#include <string>

namespace gpu_easygraph { namespace runtime {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}
void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void sparse_check(cusparseStatus_t code) {
    if (code != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string("PageRank cuSPARSE transpose: ") + cusparseGetErrorString(code));
}
struct TemporaryState {
    int previous_device = -1;
    cudaStream_t stream = nullptr;
    cusparseHandle_t handle = nullptr;
    std::vector<void*> buffers;
    void* allocate(std::size_t bytes) {
        if (!bytes) return nullptr;
        void* pointer = nullptr; cuda_check(cudaMalloc(&pointer, bytes));
        buffers.push_back(pointer); return pointer;
    }
    void release() {
        for (auto& pointer : buffers) {
            cuda_check(cudaFree(pointer)); pointer = nullptr;
        }
        if (handle) { sparse_check(cusparseDestroy(handle)); handle = nullptr; }
        if (stream) { cuda_check(cudaStreamDestroy(stream)); stream = nullptr; }
        if (previous_device >= 0) {
            cuda_check(cudaSetDevice(previous_device)); previous_device = -1;
        }
    }
    ~TemporaryState() {
        // Exception cleanup must also restore the caller's thread-local device.
        for (void* pointer : buffers) if (pointer) cudaFree(pointer);
        if (handle) cusparseDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
        if (previous_device >= 0) cudaSetDevice(previous_device);
    }
};
}

std::vector<PageRankHostPartition> build_whole_contiguous_page_rank_cusparse(
    const std::vector<PageRankOutgoingPartition>& outgoing,
    std::uint64_t vertex_count, bool materialize_unit_weights, int logical_device,
    PageRankCuSparseTransposeMetrics* metrics) {
    const auto begin = Clock::now();
    PageRankCuSparseTransposeMetrics measured;
    measured.logical_device = logical_device;
    auto result = make_whole_contiguous_page_rank_metadata(outgoing, vertex_count);
    if (result.empty()) { if (metrics) *metrics = measured; return result; }
    const auto& source = outgoing.front(); auto& target = result.front();
    if (!source.outgoing_weights.empty())
        throw std::invalid_argument("symbolic cuSPARSE PageRank transpose requires unweighted CSR");
    const auto n = static_cast<int>(vertex_count);
    const auto m = static_cast<int>(source.outgoing_destinations.size());
    // Keep the old denominator and endpoint validity contract; no random CPU
    // incoming-degree updates or scatter remain on this selected path.
    for (int row=0; row<n; ++row)
        target.owned_out_weight[row] = source.outgoing_offsets[row+1]-source.outgoing_offsets[row];
    for (int destination : source.outgoing_destinations)
        if (destination < 0 || destination >= n)
            throw std::invalid_argument("whole contiguous edge has invalid endpoint");
    auto step = Clock::now(); measured.metadata_and_denominator_seconds = seconds(begin, step);
    target.incoming_sources.resize(static_cast<std::size_t>(m));
    if (materialize_unit_weights) target.incoming_weights.assign(static_cast<std::size_t>(m), 1.0);
    auto now = Clock::now(); measured.host_csc_allocation_seconds = seconds(step, now); step = now;
    if (!m) {
        measured.total_seconds = seconds(begin, now);
        if (metrics) *metrics = measured;
        return result;
    }
    TemporaryState temporary;
    cuda_check(cudaGetDevice(&temporary.previous_device));
    cuda_check(cudaSetDevice(logical_device));
    cuda_check(cudaStreamCreateWithFlags(&temporary.stream, cudaStreamNonBlocking));
    sparse_check(cusparseCreate(&temporary.handle));
    sparse_check(cusparseSetStream(temporary.handle, temporary.stream));
    sparse_check(cusparseGetVersion(temporary.handle, &measured.cusparse_version));
    now = Clock::now(); measured.context_handle_stream_seconds = seconds(step, now); step = now;
    const std::size_t offset_bytes = (static_cast<std::size_t>(n)+1)*sizeof(int);
    const std::size_t index_bytes = static_cast<std::size_t>(m)*sizeof(int);
    auto* in_offsets = static_cast<int*>(temporary.allocate(offset_bytes));
    auto* in_indices = static_cast<int*>(temporary.allocate(index_bytes));
    auto* out_offsets = static_cast<int*>(temporary.allocate(offset_bytes));
    auto* out_indices = static_cast<int*>(temporary.allocate(index_bytes));
    now = Clock::now(); measured.device_array_allocation_seconds = seconds(step, now); step = now;
    cuda_check(cudaMemcpyAsync(in_offsets, source.outgoing_offsets.data(), offset_bytes,
                              cudaMemcpyHostToDevice, temporary.stream));
    cuda_check(cudaMemcpyAsync(in_indices, source.outgoing_destinations.data(), index_bytes,
                              cudaMemcpyHostToDevice, temporary.stream));
    cuda_check(cudaStreamSynchronize(temporary.stream));
    now = Clock::now(); measured.h2d_wall_seconds = seconds(step, now); step = now;
    std::size_t workspace_bytes = 0;
    sparse_check(cusparseCsr2cscEx2_bufferSize(temporary.handle, n, n, m, nullptr,
        in_offsets, in_indices, nullptr, out_offsets, out_indices, CUDA_R_32F,
        CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &workspace_bytes));
    void* workspace = temporary.allocate(workspace_bytes);
    measured.workspace_bytes = workspace_bytes;
    measured.temporary_explicit_device_peak_bytes = 2*(offset_bytes+index_bytes)+workspace_bytes;
    now = Clock::now(); measured.workspace_plan_allocation_seconds = seconds(step, now); step = now;
    sparse_check(cusparseCsr2cscEx2(temporary.handle, n, n, m, nullptr,
        in_offsets, in_indices, nullptr, out_offsets, out_indices, CUDA_R_32F,
        CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, workspace));
    cuda_check(cudaStreamSynchronize(temporary.stream));
    now = Clock::now(); measured.conversion_synchronized_wall_seconds = seconds(step, now); step = now;
    cuda_check(cudaMemcpyAsync(target.incoming_offsets.data(), out_offsets, offset_bytes,
                              cudaMemcpyDeviceToHost, temporary.stream));
    cuda_check(cudaMemcpyAsync(target.incoming_sources.data(), out_indices, index_bytes,
                              cudaMemcpyDeviceToHost, temporary.stream));
    cuda_check(cudaStreamSynchronize(temporary.stream));
    now = Clock::now(); measured.d2h_wall_seconds = seconds(step, now); step = now;
    measured.h2d_bytes = measured.d2h_bytes = offset_bytes+index_bytes;
    temporary.release();
    now = Clock::now(); measured.device_release_seconds = seconds(step, now);
    measured.total_seconds = seconds(begin, now); measured.conversion_performed = true;
    if (metrics) *metrics = measured;
    return result;
}
}}
