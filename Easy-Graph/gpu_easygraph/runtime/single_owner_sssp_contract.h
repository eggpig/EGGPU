#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gpu_easygraph { namespace runtime {
// Pure CPU capability and byte-layout contract. Neither helper touches CUDA.
inline bool single_owner_sssp_eligible(bool enabled, bool serial,
    std::size_t devices, std::size_t partitions, std::uint64_t owned,
    std::uint64_t local, std::uint64_t edges, std::uint64_t remote_routes,
    bool f64_weights, bool f64_wire, bool profile,
    bool default_round_limit, bool supported_execution_controls) {
    return enabled && !serial && devices == 1 && partitions == 1 &&
        owned == local && owned > 0 && remote_routes == 0 &&
        // The borrowed kernels use signed int loop counters/strides. A half
        // range cap leaves room for their positive stride increments.
        owned <= static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 2) &&
        edges <= static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 2) &&
        f64_weights && f64_wire && !profile && default_round_limit &&
        supported_execution_controls;
}
inline std::size_t single_owner_sssp_padded(std::size_t bytes) {
    return (bytes + 255U) / 256U * 256U;
}
inline std::size_t single_owner_sssp_payload_bytes(std::uint64_t vertices) {
    if (vertices > static_cast<std::uint64_t>(std::numeric_limits<int>::max() - 256) ||
        std::numeric_limits<std::size_t>::max() < (UINT64_C(1) << 36)) return 0;
    const auto n = static_cast<std::size_t>(vertices);
    const auto reduction = std::min<std::size_t>(4096, std::max<std::size_t>(1, (n + 255) / 256));
    // Exact frozen IEEE delta layout. It also covers 3*n ints + 4 controls
    // for the sparse frontier implementation, without overlapping distances.
    return 2 * single_owner_sssp_padded(n * sizeof(int)) +
        single_owner_sssp_padded(n * sizeof(double)) +
        single_owner_sssp_padded(reduction * sizeof(double)) + 256;
}
inline std::size_t single_owner_sssp_workspace_bytes(std::uint64_t vertices) {
    const auto payload = single_owner_sssp_payload_bytes(vertices);
    // Separate final 256-byte aligned slot contains the current source index.
    return payload ? payload + 256 : 0;
}
struct SingleOwnerSsspStats {
    bool used = false;
    bool sparse_frontier = false;
    int maximum_degree = -1;
    int rounds = 0;
    int queued_rounds = 0;
    int checkpoints = 0;
    int sm_count = 0;
    std::uint64_t kernel_launches = 0;
    std::uint64_t stream_synchronizations = 0;
    double execution_wall_seconds = 0.;
    std::uint64_t control_d2h_bytes = 0;
    std::uint64_t source_h2d_bytes = 0;
    std::size_t scratch_capacity_bytes = 0;
};
}}  // namespace gpu_easygraph::runtime
