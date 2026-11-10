#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <utility>

namespace easygraph { namespace egmgpu {

// The validated immutable owner sets are sorted, disjoint and cover [0,N).
// Forest convergence has projected the component's minimum GLOBAL ID to every
// owned label. This only materializes those labels; it does not relabel roots.
// Each worker writes its own IDs into a different, per-query owning array.
inline void assemble_wcc_owned_host(
    const std::uint64_t* global_ids, std::size_t count,
    const std::uint64_t* observed, std::uint64_t* destination) {
    if (!count) return;
    if (global_ids[count - 1U] - global_ids[0] == count - 1U) {
        std::memcpy(destination + global_ids[0], observed, count * sizeof(std::uint64_t));
    } else {
        for (std::size_t local = 0; local < count; ++local)
            destination[global_ids[local]] = observed[local];
    }
}

// Split physical destination cache lines, not merely groups of eight IDs:
// new[] need not return a 64-byte-aligned base. Only the first/last line may
// be partial. Every interior boundary is physically cache-line aligned.
template <typename Value>
inline std::pair<std::size_t, std::size_t> wcc_output_tile_bounds(
    const Value* destination, std::size_t count,
    std::size_t rank, std::size_t workers) {
    const std::size_t prefix = (reinterpret_cast<std::uintptr_t>(destination) % 64U) / 8U;
    const std::size_t lines = (prefix + count + 7U) / 8U;
    auto boundary = [&](std::size_t worker) {
        const std::size_t slot = (lines * worker / workers) * 8U;
        return slot <= prefix ? std::size_t(0) : std::min(count, slot-prefix);
    };
    return std::make_pair(rank == 0U ? 0U : boundary(rank),
        rank + 1U == workers ? count : boundary(rank+1U));
}

inline void assemble_wcc_tiled_host(
    const std::uint64_t* source_for_global,
    const std::uint64_t* const* sources, std::uint64_t* destination,
    std::size_t begin, std::size_t end) {
    for (std::size_t global = begin; global < end; ++global) {
        const std::uint64_t source = source_for_global[global];
        destination[global] = sources[source >> 32][static_cast<std::uint32_t>(source)];
    }
}

} }  // namespace easygraph::egmgpu
