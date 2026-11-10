#pragma once

#include <cstddef>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

std::vector<double> deterministic_dense_sum(
    const std::vector<std::vector<double> >& partials,
    std::size_t expected_width);

std::vector<double> concatenate_in_order(
    const std::vector<std::vector<double> >& partials,
    std::size_t expected_size);

}  // namespace runtime
}  // namespace gpu_easygraph
