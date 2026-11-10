#pragma once
// CPU-only, immutable layout for Float64 variable-count collective publication.
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
namespace gpu_easygraph { namespace runtime {
struct VariableGatherLayout {
    std::vector<std::size_t> counts;
    std::vector<std::size_t> offsets;
    std::size_t total_values = 0;
    std::size_t positive_ranks = 0;
};
inline VariableGatherLayout variable_gather_layout(const std::vector<std::size_t>& counts) {
    if (counts.empty()) throw std::invalid_argument("Variable gather requires ranks");
    VariableGatherLayout out; out.counts=counts; out.offsets.push_back(0);
    for (auto count : counts) {
        if (count > std::numeric_limits<std::size_t>::max()/sizeof(double)-out.total_values)
            throw std::overflow_error("Variable gather Float64 buffer overflow");
        out.total_values += count;
        out.offsets.push_back(out.total_values);
        out.positive_ranks += count != 0;
    }
    return out;
}
inline std::size_t variable_gather_index(const VariableGatherLayout& layout,
                                       std::size_t owner, std::size_t local) {
    if (owner>=layout.counts.size() || local>=layout.counts[owner])
        throw std::out_of_range("Variable gather source outside actual owner");
    return layout.offsets[owner]+local;
}
}} // namespace
