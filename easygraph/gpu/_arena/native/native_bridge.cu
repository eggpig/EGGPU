// The generated include is the byte-for-byte original EGGPU kernel prefix.
// native_bridge.py records both complete source and prefix SHA256 at build time.
#include "native_kernel_prefix.inc"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
std::size_t padded(std::size_t bytes) { return (bytes + 255u) & ~std::size_t(255u); }
std::size_t required_bytes(int n, int source_count) {
    if (n < 0 || source_count < 0 ||
        std::uint64_t(n) * std::uint64_t(source_count) > std::uint64_t(INT32_MAX)) return 0;
    return padded(std::size_t(n) * source_count * sizeof(int)) +
           2 * padded(std::size_t(n) * sizeof(int)) + 3 * 256;
}
}

extern "C" std::size_t eggpu_native_bfs_workspace_bytes(int n, int sources) {
    return required_bytes(n, sources);
}

extern "C" int eggpu_native_bfs_prepared(
    int* row, int* col, int* sources, int n, int m, int source_count,
    int target, int scan_mode, void* workspace, std::size_t workspace_capacity,
    double* output, cudaStream_t stream, std::uint64_t* control_d2h_bytes,
    int* expansion_rounds) {
    // All device allocations are borrowed. This entry contains no allocation,
    // graph conversion, graph cache lookup, or full-result host materialization.
    if (control_d2h_bytes) *control_d2h_bytes = 0;
    if (expansion_rounds) *expansion_rounds = 0;
    if (n < 0 || m < 0 || source_count < 0 || target >= n || target < -1)
        return int(cudaErrorInvalidValue);
    if (n == 0 || source_count == 0) return int(cudaSuccess);
    const std::size_t need = required_bytes(n, source_count);
    if (!need || !row || !col || !sources || !workspace || !output ||
        workspace_capacity < need || (reinterpret_cast<std::uintptr_t>(workspace) % 256))
        return int(cudaErrorInvalidValue);
    auto* bytes = static_cast<unsigned char*>(workspace);
    int* distance = reinterpret_cast<int*>(bytes);
    bytes += padded(std::size_t(n) * source_count * sizeof(int));
    int* frontier_a = reinterpret_cast<int*>(bytes);
    bytes += padded(std::size_t(n) * sizeof(int));
    int* frontier_b = reinterpret_cast<int*>(bytes);
    bytes += padded(std::size_t(n) * sizeof(int));
    int* frontier_length = reinterpret_cast<int*>(bytes); bytes += 256;
    int* next_length = reinterpret_cast<int*>(bytes); bytes += 256;
    int* target_found = reinterpret_cast<int*>(bytes);
    const bool scan = scan_mode < 0
        ? (n <= 20000 && m <= 250000 && source_count <= 65535)
        : scan_mode != 0;
    const int block = 256;
    const int grid = std::min(65535, std::max(1, (n + block - 1) / block));
    cudaError_t err;
#define CHECK_CUDA(statement) do { err = (statement); if (err != cudaSuccess) return int(err); } while (0)
    if (scan) {
        gpu_easygraph::d_sssp_unweighted_bfs<<<source_count, block, 0, stream>>>(
            row, col, sources, distance, n, source_count, target);
        CHECK_CUDA(cudaGetLastError());
    } else {
        for (int r = 0; r < source_count; ++r) {
            int* dist = distance + std::size_t(r) * n;
            gpu_easygraph::d_bfs_frontier_init<<<grid, block, 0, stream>>>(
                sources, dist, frontier_a, frontier_length, n, r);
            CHECK_CUDA(cudaGetLastError());
            int active = 1, found = 0, level = 0;
            int* current = frontier_a;
            int* next = frontier_b;
            while (active > 0 && found == 0) {
                CHECK_CUDA(cudaMemsetAsync(next_length, 0, sizeof(int), stream));
                CHECK_CUDA(cudaMemsetAsync(target_found, 0, sizeof(int), stream));
                const int expand_grid = std::min(65535, std::max(1,
                    int((std::int64_t(active) * 32 + block - 1) / block)));
                gpu_easygraph::d_bfs_expand_frontier<<<expand_grid, block, 0, stream>>>(
                    row, col, current, active, dist, next, next_length, target_found,
                    target, level, 32);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaMemcpyAsync(&active, next_length, sizeof(int), cudaMemcpyDeviceToHost, stream));
                if (target >= 0)
                    CHECK_CUDA(cudaMemcpyAsync(&found, target_found, sizeof(int), cudaMemcpyDeviceToHost, stream));
                CHECK_CUDA(cudaStreamSynchronize(stream));
                if (control_d2h_bytes) *control_d2h_bytes += sizeof(int) * (target >= 0 ? 2 : 1);
                if (expansion_rounds) ++*expansion_rounds;
                std::swap(current, next);
                if (++level > n) return int(cudaErrorUnknown);
            }
        }
    }
    const int total = n * source_count;
    const int convert_grid = std::min(65535, std::max(1,
        int((std::int64_t(total) + block - 1) / block)));
    gpu_easygraph::d_bfs_int_to_double<<<convert_grid, block, 0, stream>>>(distance, output, total);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
#undef CHECK_CUDA
    return int(cudaSuccess);
}

namespace {
int reduction_blocks(int n) {
    return std::min(4096, std::max(1, int((std::int64_t(n) + 255) / 256)));
}
std::size_t sssp_required_bytes(int n, int sources) {
    if (n < 0 || sources < 0 ||
        std::uint64_t(n) * std::uint64_t(sources) > std::uint64_t(INT32_MAX)) return 0;
    return 2 * padded(std::size_t(n) * sizeof(int)) +
           padded(std::size_t(n) * sizeof(double)) +
           padded(std::size_t(reduction_blocks(n)) * sizeof(double)) + 256;
}
}

extern "C" std::size_t eggpu_native_sssp_workspace_bytes(int n, int sources) {
    return sssp_required_bytes(n, sources);
}

extern "C" int eggpu_native_sssp_prepared(
    int* row, int* col, double* weight, int* sources,
    int n, int m, int source_count, int warp_size,
    void* workspace, std::size_t workspace_capacity, double* output,
    double* host_partial, std::size_t host_partial_capacity,
    cudaStream_t stream, std::uint64_t* control_d2h_bytes, int* expansion_rounds) {
    // The unchanged native parallel-delta kernels operate on nonnegative finite
    // outgoing edge weights. All device and host scratch is caller-owned.
    // The min-edge derived vector is rebuilt each call, so graph/weight updates
    // require no implicit cache validity state inside this entry point.
    if (control_d2h_bytes) *control_d2h_bytes = 0;
    if (expansion_rounds) *expansion_rounds = 0;
    if (n < 0 || m < 0 || source_count < 0 || warp_size < 1 || warp_size > 32 ||
        (warp_size & (warp_size - 1))) return int(cudaErrorInvalidValue);
    if (n == 0 || source_count == 0) return int(cudaSuccess);
    const int block = 256;
    const int grid = std::min(65535, std::max(1, int((std::int64_t(n) + block - 1) / block)));
    const int reduce_grid = reduction_blocks(n);
    const std::size_t need = sssp_required_bytes(n, source_count);
    if (!need || !row || (m && (!col || !weight)) || !sources || !workspace || !output ||
        !host_partial || host_partial_capacity < std::size_t(reduce_grid) * sizeof(double) ||
        workspace_capacity < need || (reinterpret_cast<std::uintptr_t>(workspace) % 256))
        return int(cudaErrorInvalidValue);
    auto* bytes = static_cast<unsigned char*>(workspace);
    int* unsettled = reinterpret_cast<int*>(bytes); bytes += padded(std::size_t(n) * sizeof(int));
    int* frontier = reinterpret_cast<int*>(bytes); bytes += padded(std::size_t(n) * sizeof(int));
    double* min_edge = reinterpret_cast<double*>(bytes); bytes += padded(std::size_t(n) * sizeof(double));
    double* partial = reinterpret_cast<double*>(bytes); bytes += padded(std::size_t(reduce_grid) * sizeof(double));
    int* frontier_length = reinterpret_cast<int*>(bytes);
    cudaError_t err;
#define CHECK_CUDA(statement) do { err = (statement); if (err != cudaSuccess) return int(err); } while (0)
    gpu_easygraph::d_calc_min_edge<<<grid, block, 0, stream>>>(row, col, weight, n, m, min_edge);
    CHECK_CUDA(cudaGetLastError());
    for (int r = 0; r < source_count; ++r) {
        double* dist = output + std::size_t(r) * n;
        gpu_easygraph::d_sssp_delta_init<<<grid, block, 0, stream>>>(
            sources, dist, unsettled, frontier, frontier_length, n, r);
        CHECK_CUDA(cudaGetLastError());
        int active = 1, iterations = 0;
        while (active > 0) {
            const int relax_grid = std::min(65535, std::max(1,
                int((std::int64_t(active) * warp_size + block - 1) / block)));
            gpu_easygraph::d_sssp_delta_relax<<<relax_grid, block, 0, stream>>>(
                row, col, weight, frontier, active, dist, warp_size);
            CHECK_CUDA(cudaGetLastError());
            gpu_easygraph::d_sssp_delta_reduce_partial<<<reduce_grid, block, 0, stream>>>(
                dist, unsettled, min_edge, partial, n);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaMemcpyAsync(host_partial, partial, std::size_t(reduce_grid) * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            if (control_d2h_bytes) *control_d2h_bytes += std::size_t(reduce_grid) * sizeof(double);
            if (expansion_rounds) ++*expansion_rounds;
            double delta = EG_DOUBLE_INF;
            for (int i = 0; i < reduce_grid; ++i) delta = std::min(delta, host_partial[i]);
            if (delta >= EG_DOUBLE_INF) break;
            CHECK_CUDA(cudaMemsetAsync(frontier_length, 0, sizeof(int), stream));
            gpu_easygraph::d_sssp_delta_build_frontier<<<grid, block, 0, stream>>>(
                unsettled, dist, delta, frontier, frontier_length, n);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaMemcpyAsync(&active, frontier_length, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            if (control_d2h_bytes) *control_d2h_bytes += sizeof(int);
            if (++iterations > n) return int(cudaErrorUnknown);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
#undef CHECK_CUDA
    return int(cudaSuccess);
}
