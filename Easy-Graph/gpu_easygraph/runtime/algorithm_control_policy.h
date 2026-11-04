#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "sssp_numeric_domain.h"

namespace gpu_easygraph { namespace runtime {

inline bool algorithm_candidate_flag(const char* name) {
    const char* value = std::getenv(name);
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

// This decision consumes the CURRENT immutable plan's exact maximum, not a
// dtype/dataset/frontier heuristic. The domain already validated nonnegative,
// finite weights and requested/effective delta. Fail closed on inconsistency.
inline bool sssp_skip_empty_heavy_eligible(
    bool partitioned_concurrent_route, bool current_statistics_available,
    double current_maximum_weight, const SsspNumericDomain& domain) {
    return partitioned_concurrent_route && current_statistics_available &&
        domain.all_edges_light && std::isfinite(current_maximum_weight) &&
        current_maximum_weight >= 0. && std::isfinite(domain.effective_delta) &&
        domain.effective_delta > 0. &&
        current_maximum_weight <= domain.effective_delta;
}

}}  // namespace gpu_easygraph::runtime
