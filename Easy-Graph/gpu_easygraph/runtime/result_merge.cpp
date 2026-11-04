#include "result_merge.h"

#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {

std::vector<double> deterministic_dense_sum(
    const std::vector<std::vector<double> >& partials,
    std::size_t expected_width) {
    std::vector<double> merged(expected_width, 0.0);
    for (std::size_t device = 0; device < partials.size(); ++device) {
        if (partials[device].size() != expected_width) {
            throw std::invalid_argument(
                "dense shard result width does not match the graph");
        }
        for (std::size_t item = 0; item < expected_width; ++item) {
            merged[item] += partials[device][item];
        }
    }
    return merged;
}

std::vector<double> concatenate_in_order(
    const std::vector<std::vector<double> >& partials,
    std::size_t expected_size) {
    std::vector<double> merged;
    merged.reserve(expected_size);
    for (std::size_t device = 0; device < partials.size(); ++device) {
        merged.insert(
            merged.end(), partials[device].begin(), partials[device].end());
    }
    if (merged.size() != expected_size) {
        throw std::invalid_argument(
            "ordered shard results do not match the source cardinality");
    }
    return merged;
}

}  // namespace runtime
}  // namespace gpu_easygraph
