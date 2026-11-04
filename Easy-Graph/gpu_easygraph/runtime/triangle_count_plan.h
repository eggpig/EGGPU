#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace gpu_easygraph { namespace runtime {

struct TriangleOwnerRange {
    int begin = 0;
    int end = 0;
    std::uint64_t estimated_scan_work = 0;
};

// Same degree/ID orientation and sorted forward rows as the LCC view.
// Only contiguous owner root ranges differ between the two scheduling policies.
struct TriangleHostPlan {
    std::uint64_t identity = 0;
    int vertex_count = 0;
    int input_arcs = 0;
    std::vector<int> offsets;
    std::vector<int> indices;
    std::vector<int> edge_sources; // each device uploads only its owned edge slice
    std::vector<TriangleOwnerRange> owners;
    std::string owner_policy;
    std::uint64_t triangle_upper_bound = 0;
    double validation_seconds = 0.;
    double orientation_seconds = 0.;
    double owner_partition_seconds = 0.;
};

TriangleHostPlan make_triangle_plan(const int* offsets, std::size_t offset_count,
    const int* indices, std::size_t arc_count, int parts, const std::string& policy);

// CPU-only exact reference for the small plan unit tests; GPU does not call it.
std::uint64_t triangle_plan_cpu_count(const TriangleHostPlan& plan,
                                     const TriangleOwnerRange& owner);

} }
