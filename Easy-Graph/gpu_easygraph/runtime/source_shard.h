#pragma once

#include <cstddef>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct ContiguousShard {
    std::size_t begin;
    std::size_t end;

    std::size_t size() const { return end - begin; }
};

// Return at most max_shards non-empty, balanced, contiguous shards.  This
// preserves source order during concatenation and avoids dispatching empty GPU
// work when the device count exceeds the source count.
std::vector<ContiguousShard> balanced_contiguous_shards(
    std::size_t item_count,
    std::size_t max_shards);

}  // namespace runtime
}  // namespace gpu_easygraph
