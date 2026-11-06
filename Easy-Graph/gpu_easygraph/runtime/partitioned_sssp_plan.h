#pragma once

#include "partitioned_sssp_types.h"
#include "../../cpp_easygraph/runtime/immutable_host_array.h"
#include "partitioned_host_plan_cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <limits>

namespace gpu_easygraph {
namespace runtime {

using easygraph::egmgpu::ImmutableHostArray;

struct SsspHostPartition {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    ImmutableHostArray<int> offsets;
    ImmutableHostArray<int> indices;
    ImmutableHostArray<double> weights;
    // Original FP64 values remain available to the host reference. Only this
    // certified exact view is uploaded when U8Exact was explicitly requested.
    ImmutableHostArray<std::uint8_t> weights_u8;
    ImmutableHostArray<std::uint64_t> local_to_global;
    ImmutableHostArray<int> owners;
};

// One source partition has one slot per distinct remote halo.  A relaxation
// kernel takes the minimum candidate for that slot before any transfer.
struct SsspOwnerRoute {
    int source_rank;
    int destination_rank;
    ImmutableHostArray<int> source_halo_local_ids;
    ImmutableHostArray<int> destination_owner_local_ids;

    std::size_t capacity() const;
};

struct SsspPartitionUpload;

struct SsspUploadBuildStats {
    double copy_seconds = 0.0;
    double weight_digest_seconds = 0.0;
    double routes_seconds = 0.0;
    double other_seconds = 0.0;
    std::size_t descriptors = 0;
    std::size_t shared_payload_bytes = 0;
    std::size_t weight_hash_bytes = 0;
    std::size_t weight_hash_descriptors = 0;
    std::size_t weight_manifest_bytes = 0;
    std::size_t weight_manifest_descriptors = 0;
    std::size_t weight_normalized_float32_descriptors = 0;
};

// Internal immutable native-commit packet; no public dirty-mask API exists.
// Native reconstruction copies the certified parent and applies these values,
// so an unrelated current-weight vector cannot bypass full validation.
struct SsspNativeWeightDelta {
    std::string base_weight_sha256;
    std::string target_weight_sha256;
    std::vector<int> positions;
    std::vector<double> values;
};

// Verify the producer packet against exact parent descriptors/current identity.
// Empty ranks must retain their parent hash; packet coverage is producer-owned.
void validate_sssp_native_delta_identity(
    const std::vector<std::string>& parent_hashes,
    const std::vector<std::string>& current_hashes,
    const std::vector<SsspNativeWeightDelta>& deltas,
    std::uint64_t producer_changed_positions);

// Integer threshold avoids floating-point ambiguity at the density boundary.
// This chooses a certification route only; it is not a numerical approximation.
bool sssp_native_delta_density_eligible(std::uint64_t changed_positions,
    std::uint64_t source_edges, std::uint64_t maximum_density_ppm);

struct PartitionedSsspPlan {
    std::uint64_t vertex_count;
    SsspWeightEncoding weight_encoding = SsspWeightEncoding::F64;
    SsspWireEncoding wire_encoding = SsspWireEncoding::F64;
    bool wire_integer_bound_certified = false;
    double maximum_integer_weight = 0.;
    double wire_distance_bound = std::numeric_limits<double>::infinity();
    double weight_validation_and_encoding_seconds = 0.0;
    std::vector<SsspHostPartition> partitions;
    // Row-major [source_rank * partition_count + destination_rank].
    std::vector<SsspOwnerRoute> routes;
    ImmutableHostArray<int> owner_by_global;
    std::uint64_t route_capacity;
    // Immutable descriptor lifetime follows this plan, including in-flight tasks.
    std::vector<std::shared_ptr<const SsspPartitionUpload> > prepared_uploads;
    SsspUploadBuildStats prepared_upload_build;
    // Numeric vector allocated capacity, excluding metadata/allocator overhead.
    std::size_t host_plan_array_capacity_bytes = 0;
    std::size_t prepared_upload_array_capacity_bytes = 0;
    // Arrays own shared immutable payloads independently of ancestor plans.
    // Releasing an ancestor cannot invalidate this plan or its descriptors.
    std::string weight_certification_selected_route = "full";
    std::string weight_certification_selection_reason = "initial_full_builder";
    std::uint64_t weight_certification_candidate_positions = 0;
    bool weight_statistics_available = false;
    std::uint64_t maximum_weight_count = 0;
    std::uint64_t noninteger_weight_count = 0;
    std::uint64_t weight_validation_edges = 0;
    std::uint64_t weight_delta_positions = 0;
    bool weight_delta_certification_used = false;
    bool weight_delta_rescan_required = false;
    bool parent_topology_reused = false;
    std::size_t parent_topology_copied_bytes = 0;
    std::size_t parent_topology_shared_bytes = 0;

    const SsspOwnerRoute& route(
        std::size_t source_rank, std::size_t destination_rank) const;
    std::size_t owner_rank(std::uint64_t global_vertex) const;
    std::size_t owner_local(std::uint64_t global_vertex) const;
};

// Validates exact ownership, strict empty partitions, local CSR, and finite
// nonnegative weights before deriving owner-directed relaxation routes.
PartitionedSsspPlan build_partitioned_sssp_plan(
    const std::vector<SsspHostPartition>& partitions,
    std::uint64_t vertex_count,
    SsspWeightEncoding weight_encoding = SsspWeightEncoding::F64,
    SsspWireEncoding wire_encoding = SsspWireEncoding::F64);

// Internal eligibility check for an immutable native-memory commit. The caller
// supplies the native producer's exact parent identity, never a user dirty mask.
bool matches_sssp_direct_parent(
    const PartitionedHostPlanCacheKey& cached_key,
    const PartitionedSsspPlan& cached_plan,
    const PartitionedHostPlanCacheKey& target_key,
    const std::string& base_graph_id,
    const std::string& base_view_signature,
    const std::vector<std::pair<int, int> >& target_partition_devices);

// The caller must hold the completed plan selected by the exact-parent check.
// Share validated immutable topology/owner/routes; move current weights into a fresh
// plan; a complete native delta permits exact incremental certification,
// otherwise every weight is checked. No old descriptor/packet is copied.
PartitionedSsspPlan rebuild_sssp_f64_from_parent(
    const PartitionedSsspPlan& parent,
    std::vector<std::vector<double> > current_weights);

// Weights are reconstructed exclusively from this validated parent and packet.
// The binding verifies producer identity/coverage before invoking this helper.
PartitionedSsspPlan rebuild_sssp_f64_from_native_delta(
    const PartitionedSsspPlan& parent,
    const std::vector<SsspNativeWeightDelta>& native_deltas);

// Deterministic host Dijkstra oracle used only for correctness tests.
std::vector<double> partitioned_sssp_reference(
    const PartitionedSsspPlan& plan,
    std::uint64_t source);

std::vector<double> merge_partitioned_sssp_owner_distances(
    const PartitionedSsspPlan& plan,
    const std::vector<std::vector<double> >& owned_distances);

}  // namespace runtime
}  // namespace gpu_easygraph
