#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace gpu_easygraph { namespace runtime {
struct SsspNumericDomain {
    double requested_delta = 0.;
    double effective_delta = 0.;
    bool integer_bound_used = false;
    bool all_edges_light = false;
    const char* route = "";
    const char* reason = "";
};

// The maximum and noninteger count must describe the CURRENT immutable plan.
// No additional edge scan, graph identity shortcut or floating approximation.
inline SsspNumericDomain select_sssp_numeric_domain(std::uint64_t vertices,
    double maximum_weight, std::uint64_t noninteger_count,
    bool statistics_available, double requested_delta) {
    if (!vertices || !statistics_available || !std::isfinite(maximum_weight) ||
        maximum_weight < 0. || !std::isfinite(requested_delta) || requested_delta <= 0.)
        throw std::invalid_argument("SSSP numerical domain requires current exact weight statistics and finite positive delta");
    SsspNumericDomain result;
    result.requested_delta = requested_delta;
    result.effective_delta = requested_delta;
    if (maximum_weight <= requested_delta) {
        result.all_edges_light = true;
        result.route = "requested_all_light";
        result.reason = "current_maximum_weight_at_most_requested_delta";
        return result;
    }
    const std::uint64_t exact_limit = UINT64_C(9007199254740992);
    // Reject large/fractional M before casting. t has at most64 bits and M
    // at most53 bits, so their exact integer product fits unsigned128 bits.
    if (requested_delta == 16. && noninteger_count == 0 &&
        maximum_weight < static_cast<double>(exact_limit) &&
        std::floor(maximum_weight) == maximum_weight &&
        static_cast<unsigned __int128>(vertices - 1U) *
            static_cast<std::uint64_t>(maximum_weight) < exact_limit) {
        result.integer_bound_used = true;
        result.route = "requested_exact_integer_delta16";
        result.reason = "integer_simple_path_bound_below_2p53_with_exact_binary_bucket_scaling";
        return result;
    }
    // M>requested_delta here. Every edge is now light, including zero and
    // DBL_MAX. The existing globally closed light worklist precedes settlement;
    // no reliance on an unsaturated quotient or an approximate n*M bound.
    result.effective_delta = maximum_weight;
    result.all_edges_light = true;
    result.route = "certified_maximum_all_light";
    result.reason = "fixed_requested_bucket_domain_unproved_use_all_edges_light";
    return result;
}
}}  // namespace gpu_easygraph::runtime
