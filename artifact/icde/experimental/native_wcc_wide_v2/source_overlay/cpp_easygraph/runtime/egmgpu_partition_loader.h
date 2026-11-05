#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>
#include <utility>
#include <string>
#include <vector>
#include "immutable_host_array.h"

namespace easygraph {
namespace egmgpu {

enum class PartitionWeightType {
    kNone,
    kFloat32,
    kFloat64,
};

struct PartitionLoadSpec {
    std::int32_t partition_id;
    std::int32_t device_id;
    std::string ownership_encoding;
    std::uint64_t owned_vertex_count;
    std::uint64_t halo_vertex_count;
    std::uint64_t adjacency_count;
    std::uint64_t cut_edge_count;
    bool has_vertex_range;
    std::uint64_t vertex_begin;
    std::uint64_t vertex_end;
    // The vertex-ID domain remains int32; only adjacency positions may widen.
    bool wide_offsets = false;
    std::string offsets_path;
    std::string offsets_sha256;
    std::string indices_path;
    std::string indices_sha256;
    std::string local_to_global_path;
    std::string local_to_global_sha256;
    std::string owners_path;
    std::string owners_sha256;
    std::string owned_global_ids_path;
    std::string owned_global_ids_sha256;
    std::string weights_path;
    std::string weights_sha256;
    PartitionWeightType weight_type;

    PartitionLoadSpec();
};

struct MultiPartitionLoadSpec {
    bool manifest_validated;
    // full: verify array content digests; structural: prepared immutable input,
    // still read every byte and enforce every native content constraint.
    std::string import_policy;
    double manifest_metadata_seconds = 0.0;
    double manifest_projection_seconds = 0.0;
    std::string graph_id;
    std::uint64_t generation;
    bool directed;
    bool weighted;
    std::string view_orientation;
    std::string weight_key;
    std::string view_signature;
    std::string topology_signature;
    std::string weight_signature;
    std::uint64_t vertex_count;
    std::uint64_t edge_count;
    std::uint64_t adjacency_count;
    std::string ownership_encoding;
    std::vector<PartitionLoadSpec> partitions;

    MultiPartitionLoadSpec();
};

struct AbsoluteWeightSet {
    std::int32_t partition_id;
    std::int64_t position;
    double value;
};

// Constructed exclusively by native commit from an immutable registered base.
// Positions are in this partition's outgoing layout, never PR incoming order.
struct NativeWeightPatch {
    std::string base_weight_sha256;
    std::string target_weight_sha256;
    std::vector<int> positions;
    std::vector<double> values;
};

struct NativeWeightCommitProfile {
    std::string base_graph_id;
    std::string base_view_signature;
    std::uint64_t requested_positions = 0;
    std::uint64_t changed_positions = 0;
    std::uint64_t copied_array_payload_bytes = 0;
    std::uint64_t shared_topology_payload_bytes = 0;
    std::uint64_t shared_unchanged_weight_payload_bytes = 0;
    std::string host_storage_policy = "deep";
    std::uint64_t weight_bytes_hashed = 0;
    std::uint64_t packet_payload_bytes = 0;
    double validation_seconds = 0.0;
    double clone_seconds = 0.0;
    double apply_seconds = 0.0;
    double weight_hash_seconds = 0.0;
    double signature_seconds = 0.0;
    double total_seconds = 0.0;
};

struct PartitionCSR {
    std::int32_t partition_id;
    std::int32_t device_id;
    std::uint64_t owned_vertex_count;
    std::uint64_t halo_vertex_count;
    std::uint64_t adjacency_count;
    std::uint64_t cut_edge_count;
    bool has_vertex_range;
    std::uint64_t vertex_begin;
    std::uint64_t vertex_end;
    bool wide_offsets = false;
    ImmutableHostArray<std::int32_t> offsets;
    ImmutableHostArray<std::int64_t> offsets64;
    std::size_t offset_count() const { return wide_offsets ? offsets64.size() : offsets.size(); }
    std::int64_t offset(std::size_t i) const { return wide_offsets ? offsets64[i] : offsets[i]; }
    std::size_t offset_payload_bytes() const { return wide_offsets ? offsets64.payload_bytes() : offsets.payload_bytes(); }
    void require_narrow_offsets(const char* algorithm) const;
    std::string offsets_sha256;
    ImmutableHostArray<std::int32_t> indices;
    std::string indices_sha256;
    ImmutableHostArray<std::uint64_t> local_to_global;
    std::string local_to_global_sha256;
    ImmutableHostArray<std::int32_t> owners;
    std::string owners_sha256;
    std::string ownership_encoding;
    ImmutableHostArray<std::uint64_t> owned_global_ids;
    std::string owned_global_ids_sha256;
    PartitionWeightType weight_type;
    std::string weights_sha256;
    ImmutableHostArray<float> weights_float32;
    ImmutableHostArray<double> weights_float64;
    std::shared_ptr<const NativeWeightPatch> native_weight_patch;

    std::uint64_t local_vertex_count() const;
    bool weighted() const;
};

struct ArrayImportProfile {
    std::int32_t partition_id = -1;
    std::string name;
    std::string path;
    std::uint64_t count = 0;
    std::uint64_t item_bytes = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_hashed = 0;
    std::uint64_t bytes_swapped = 0;
    // Mutually exclusive host wall intervals. Allocation includes vector
    // value initialization; read is stream.read plus the complete-read check.
    double open_size_seconds = 0.0;
    double allocation_seconds = 0.0;
    double read_seconds = 0.0;
    double sha256_seconds = 0.0;
    double endian_seconds = 0.0;
    double close_seconds = 0.0;
    double total_seconds = 0.0;  // Envelope, not an extra additive phase.
};

struct PartitionImportProfile {
    std::int32_t partition_id = -1;
    double metadata_seconds = 0.0;
    double structural_seconds = 0.0;  // CSR, owners, owned prefix, finite weights.
    double total_seconds = 0.0;      // Includes arrays; do not add twice.
    std::vector<ArrayImportProfile> arrays;
};

struct GraphImportProfile {
    std::string policy;
    double manifest_metadata_seconds = 0.0;
    double manifest_projection_seconds = 0.0;
    double global_metadata_seconds = 0.0;
    double global_counts_seconds = 0.0;
    double global_cover_seconds = 0.0;
    double global_halo_seconds = 0.0;
    double native_total_seconds = 0.0;  // Native constructor envelope only.
    std::vector<PartitionImportProfile> partitions;
};

class MultiPartitionGraph {
public:
    explicit MultiPartitionGraph(const MultiPartitionLoadSpec& spec);
    // The internal binding supplies the canonical signer; user input supplies
    // only base, new ID and absolute sets, never a claimed complete dirty mask.
    // An optional internal hasher synchronously hashes the complete read-only
    // payload as standard SHA-256. The owner permits retained buffer aliases;
    // neither the callback nor its exporter is retained by the target graph.
    // Pure C++ callers retain the original scalar implementation by default.
    static std::shared_ptr<MultiPartitionGraph> commit_absolute_weights(
        const std::shared_ptr<MultiPartitionGraph>& base,
        const std::string& new_id, const std::vector<AbsoluteWeightSet>& updates,
        const std::function<std::pair<std::string, std::string>(const MultiPartitionGraph&)>& signer,
        const std::function<std::string(const std::shared_ptr<MultiPartitionGraph>&,
                                        const unsigned char*, std::size_t)>& payload_hasher = {});
    const NativeWeightCommitProfile& native_commit_profile() const;
    const std::string& snapshot_origin() const;

    const std::string& graph_id() const;
    std::uint64_t generation() const;
    bool directed() const;
    bool weighted() const;
    const std::string& view_orientation() const;
    const std::string& weight_key() const;
    const std::string& view_signature() const;
    const std::string& topology_signature() const;
    const std::string& weight_signature() const;
    std::uint64_t vertex_count() const;
    std::uint64_t edge_count() const;
    std::uint64_t adjacency_count() const;
    const std::string& ownership_encoding() const;
    const std::vector<PartitionCSR>& partitions() const;
    const GraphImportProfile& import_profile() const;

private:
    std::string graph_id_;
    std::uint64_t generation_;
    bool directed_;
    bool weighted_;
    std::string view_orientation_;
    std::string weight_key_;
    std::string view_signature_;
    std::string topology_signature_;
    std::string weight_signature_;
    std::uint64_t vertex_count_;
    std::uint64_t edge_count_;
    std::uint64_t adjacency_count_;
    std::string ownership_encoding_;
    std::vector<PartitionCSR> partitions_;
    GraphImportProfile import_profile_;
    NativeWeightCommitProfile native_commit_profile_;
    std::string snapshot_origin_ = "manifest";
};

// The caller sets manifest_validated after manifest-v1 schema and path/array
// metadata/catalog-signature validation. Both policies completely read arrays
// and verify content constraints before publishing the immutable graph. full
// additionally verifies actual array SHA-256; structural requires immutable
// prepared files, and any change must be published as a new graph generation.
std::shared_ptr<MultiPartitionGraph> load_multi_partition_graph(
    const MultiPartitionLoadSpec& spec);

const char* partition_weight_type_name(PartitionWeightType type);

}  // namespace egmgpu
}  // namespace easygraph
