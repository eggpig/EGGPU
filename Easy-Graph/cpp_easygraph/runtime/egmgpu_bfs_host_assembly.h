#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace easygraph {
namespace egmgpu {

// The immutable plan proves sorted unique owner IDs and a disjoint global
// cover. Each caller writes one owner's IDs in a separate owning result array.
// reached_count comes from the source plus completed BFS next-frontier counts;
// it is not estimated from graph connectivity or computed by another scan.
inline void assemble_bfs_owned_host(
    const std::uint64_t* global_ids, std::size_t count,
    const std::int64_t* observed, std::int64_t* destination,
    std::uint64_t reached_count) {
    if (!count) return;
    const bool contiguous = global_ids[count - 1] - global_ids[0] == count - 1;
    const bool all_reached = reached_count == count;
    if (contiguous && all_reached) {
        std::memcpy(destination + global_ids[0], observed, count * sizeof(std::int64_t));
    } else if (contiguous) {
        destination += global_ids[0];
        for (std::size_t local = 0; local < count; ++local) {
            const std::int64_t value = observed[local];
            destination[local] = value == std::numeric_limits<std::int64_t>::max() ? -1 : value;
        }
    } else if (all_reached) {
        for (std::size_t local = 0; local < count; ++local)
            destination[global_ids[local]] = observed[local];
    } else {
        for (std::size_t local = 0; local < count; ++local) {
            const std::int64_t value = observed[local];
            destination[global_ids[local]] = value == std::numeric_limits<std::int64_t>::max() ? -1 : value;
        }
    }
}

}  // namespace egmgpu
}  // namespace easygraph
