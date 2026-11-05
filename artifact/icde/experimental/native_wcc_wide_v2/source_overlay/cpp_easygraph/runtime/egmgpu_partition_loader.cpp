#include "egmgpu_partition_loader.h"
#include "egmgpu_sha256.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace easygraph {
namespace egmgpu {
namespace {

typedef std::chrono::steady_clock ImportClock;

double elapsed(const ImportClock::time_point& start) {
    return std::chrono::duration<double>(ImportClock::now() - start).count();
}

const std::uint64_t kMaxLocalEntries = (std::uint64_t(1) << 31) - 1;
const std::uint64_t kMaxLocalVertices = std::uint64_t(1) << 31;

bool host_is_little_endian() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

bool is_sha256(const std::string& value) {
    return value.size() == 64 &&
           value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

template <typename T>
T byte_swap(T value) {
    unsigned char input[sizeof(T)];
    unsigned char output[sizeof(T)];
    std::memcpy(input, &value, sizeof(T));
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        output[index] = input[sizeof(T) - 1 - index];
    }
    std::memcpy(&value, output, sizeof(T));
    return value;
}

std::size_t checked_size(std::uint64_t count, std::size_t item_size,
                         const std::string& context) {
    const std::uint64_t maximum =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (item_size == 0 || count > maximum / item_size) {
        throw std::overflow_error(context + " does not fit host address space");
    }
    return static_cast<std::size_t>(count);
}

template <typename T>
std::vector<T> read_array(const std::string& path, std::uint64_t count,
                          const std::string& expected_sha256,
                          const std::string& context, bool verify_sha256,
                          PartitionImportProfile* partition_profile) {
    partition_profile->arrays.emplace_back();
    ArrayImportProfile& profile = partition_profile->arrays.back();
    profile.partition_id = partition_profile->partition_id;
    profile.name = context;
    profile.path = path;
    profile.count = count;
    profile.item_bytes = sizeof(T);
    const ImportClock::time_point total_start = ImportClock::now();
    ImportClock::time_point phase_start = total_start;
    if (!is_sha256(expected_sha256)) {
        throw std::invalid_argument(context + " has an invalid expected SHA-256");
    }
    const std::size_t host_count = checked_size(count, sizeof(T), context);
    std::ifstream stream(path.c_str(), std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open " + context + ": " + path);
    }
    const std::streamoff observed = stream.tellg();
    const std::uint64_t expected = count * static_cast<std::uint64_t>(sizeof(T));
    if (observed < 0 || static_cast<std::uint64_t>(observed) != expected) {
        throw std::runtime_error(context + " byte size changed after validation");
    }
    stream.seekg(0, std::ios::beg);
    profile.open_size_seconds = elapsed(phase_start);
    phase_start = ImportClock::now();
    std::vector<T> values(host_count);
    profile.allocation_seconds = elapsed(phase_start);
    phase_start = ImportClock::now();
    if (expected != 0) {
        stream.read(reinterpret_cast<char*>(&values[0]),
                    static_cast<std::streamsize>(expected));
        if (!stream || static_cast<std::uint64_t>(stream.gcount()) != expected) {
            throw std::runtime_error("short read from " + context);
        }
    }
    profile.read_seconds = elapsed(phase_start);
    profile.bytes_read = expected;
    if (verify_sha256) {
        phase_start = ImportClock::now();
        Sha256 digest;
        if (expected != 0) {
            digest.update(reinterpret_cast<const unsigned char*>(&values[0]),
                          static_cast<std::size_t>(expected));
        }
        if (digest.final_hex() != expected_sha256) {
            throw std::runtime_error(context + " SHA-256 changed after validation");
        }
        profile.sha256_seconds = elapsed(phase_start);
        profile.bytes_hashed = expected;
    }
    phase_start = ImportClock::now();
    if (!host_is_little_endian()) {
        profile.bytes_swapped = expected;
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = byte_swap(values[index]);
        }
    }
    profile.endian_seconds = elapsed(phase_start);
    phase_start = ImportClock::now();
    stream.close();
    profile.close_seconds = elapsed(phase_start);
    profile.total_seconds = elapsed(total_start);
    return values;
}

void validate_orientation(const MultiPartitionLoadSpec& spec) {
    const bool directed_orientation =
        spec.view_orientation == "outgoing" ||
        spec.view_orientation == "incoming";
    if (!directed_orientation && spec.view_orientation != "undirected") {
        throw std::invalid_argument("unsupported graph-view orientation");
    }
    if (spec.directed != directed_orientation) {
        throw std::invalid_argument(
            "graph directedness disagrees with graph-view orientation");
    }
    if (!spec.weighted && !spec.weight_key.empty()) {
        throw std::invalid_argument("unweighted graph view has a weight key");
    }
    if (!is_sha256(spec.view_signature) ||
        !is_sha256(spec.topology_signature)) {
        throw std::invalid_argument("graph and view signatures must be SHA-256");
    }
    if ((spec.weighted && !is_sha256(spec.weight_signature)) ||
        (!spec.weighted && !spec.weight_signature.empty())) {
        throw std::invalid_argument(
            "weight signature is inconsistent with graph weightedness");
    }
}

PartitionCSR load_partition(const PartitionLoadSpec& spec,
                            std::size_t partition_count, bool graph_weighted,
                            bool verify_sha256, PartitionImportProfile* profile) {
    const ImportClock::time_point total_start = ImportClock::now();
    profile->partition_id = spec.partition_id;
    profile->arrays.reserve(6);
    if (spec.partition_id < 0 || spec.device_id < 0) {
        throw std::invalid_argument("partition and device IDs must be nonnegative");
    }
    if (spec.owned_vertex_count == 0 &&
        (spec.halo_vertex_count != 0 || spec.adjacency_count != 0 ||
         spec.cut_edge_count != 0)) {
        throw std::invalid_argument(
            "an empty partition must have zero halo, adjacency, and cut-edge counts");
    }
    if (spec.adjacency_count > (spec.wide_offsets ?
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) : kMaxLocalEntries)) {
        throw std::invalid_argument("partition adjacency count exceeds signed CSR offset dtype");
    }
    if (spec.owned_vertex_count > kMaxLocalVertices ||
        spec.halo_vertex_count >
            kMaxLocalVertices - spec.owned_vertex_count) {
        throw std::invalid_argument("partition local vertex count exceeds int32 IDs");
    }
    if (spec.cut_edge_count > spec.adjacency_count) {
        throw std::invalid_argument("partition cut-edge count exceeds adjacency count");
    }
    if (spec.has_vertex_range &&
        spec.vertex_end - spec.vertex_begin != spec.owned_vertex_count) {
        throw std::invalid_argument("partition range size is inconsistent");
    }

    PartitionCSR result;
    result.partition_id = spec.partition_id;
    result.device_id = spec.device_id;
    result.owned_vertex_count = spec.owned_vertex_count;
    result.halo_vertex_count = spec.halo_vertex_count;
    result.adjacency_count = spec.adjacency_count;
    result.cut_edge_count = spec.cut_edge_count;
    result.ownership_encoding = spec.ownership_encoding;
    result.has_vertex_range = spec.has_vertex_range;
    result.vertex_begin = spec.vertex_begin;
    result.vertex_end = spec.vertex_end;
    profile->metadata_seconds = elapsed(total_start);
    result.wide_offsets = spec.wide_offsets;
    if (spec.wide_offsets) {
        result.offsets64 = read_array<std::int64_t>(
            spec.offsets_path, spec.owned_vertex_count + 1, spec.offsets_sha256,
            "partition offsets", verify_sha256, profile);
    } else {
        result.offsets = read_array<std::int32_t>(
            spec.offsets_path, spec.owned_vertex_count + 1, spec.offsets_sha256,
            "partition offsets", verify_sha256, profile);
    }
    result.offsets_sha256 = spec.offsets_sha256;
    result.indices = read_array<std::int32_t>(
        spec.indices_path, spec.adjacency_count, spec.indices_sha256,
        "partition indices", verify_sha256, profile);
    result.indices_sha256 = spec.indices_sha256;
    const std::uint64_t local_count =
        spec.owned_vertex_count + spec.halo_vertex_count;
    result.local_to_global = read_array<std::uint64_t>(
        spec.local_to_global_path, local_count, spec.local_to_global_sha256,
        "partition local-to-global map", verify_sha256, profile);
    result.local_to_global_sha256 = spec.local_to_global_sha256;
    result.owners = read_array<std::int32_t>(
        spec.owners_path, local_count, spec.owners_sha256, "partition owner map", verify_sha256, profile);
    result.owners_sha256 = spec.owners_sha256;
    if (spec.ownership_encoding == "explicit") {
        result.owned_global_ids = read_array<std::uint64_t>(
            spec.owned_global_ids_path, spec.owned_vertex_count,
            spec.owned_global_ids_sha256, "partition explicit owner IDs", verify_sha256, profile);
        result.owned_global_ids_sha256 = spec.owned_global_ids_sha256;
    } else if (spec.ownership_encoding != "range") {
        throw std::invalid_argument("partition has an unsupported ownership encoding");
    }
    result.weight_type = spec.weight_type;
    result.weights_sha256 = spec.weights_sha256;
    if (spec.weight_type == PartitionWeightType::kNone &&
        (!spec.weights_path.empty() || !spec.weights_sha256.empty())) {
        throw std::invalid_argument(
            "unweighted partition carries stale weight-array metadata");
    }
    if (spec.weight_type == PartitionWeightType::kFloat32) {
        result.weights_float32 = read_array<float>(
            spec.weights_path, spec.adjacency_count, spec.weights_sha256,
            "partition float32 weights", verify_sha256, profile);
        const ImportClock::time_point weights_start = ImportClock::now();
        for (float weight : result.weights_float32) {
            if (!std::isfinite(weight)) {
                throw std::invalid_argument("partition weights must be finite");
            }
        }
        profile->structural_seconds += elapsed(weights_start);
    } else if (spec.weight_type == PartitionWeightType::kFloat64) {
        result.weights_float64 = read_array<double>(
            spec.weights_path, spec.adjacency_count, spec.weights_sha256,
            "partition float64 weights", verify_sha256, profile);
        const ImportClock::time_point weights_start = ImportClock::now();
        for (double weight : result.weights_float64) {
            if (!std::isfinite(weight)) {
                throw std::invalid_argument("partition weights must be finite");
            }
        }
        profile->structural_seconds += elapsed(weights_start);
    } else if (graph_weighted) {
        throw std::invalid_argument("weighted graph partition has no weight array");
    }
    const ImportClock::time_point structural_start = ImportClock::now();
    if (!graph_weighted && spec.weight_type != PartitionWeightType::kNone) {
        throw std::invalid_argument("unweighted graph partition declares weights");
    }

    if (result.offset_count() == 0 || result.offset(0) != 0) {
        throw std::invalid_argument("partition offsets must start at zero");
    }
    for (std::size_t index = 1; index < result.offset_count(); ++index) {
        if (result.offset(index) < result.offset(index - 1)) {
            throw std::invalid_argument("partition offsets must be monotone");
        }
    }
    if (static_cast<std::uint64_t>(result.offset(result.offset_count() - 1)) !=
        result.adjacency_count) {
        throw std::invalid_argument("partition final offset is inconsistent");
    }
    for (std::size_t index = 0; index < result.indices.size(); ++index) {
        if (result.indices[index] < 0 ||
            static_cast<std::uint64_t>(result.indices[index]) >= local_count) {
            throw std::invalid_argument("partition local index is out of range");
        }
    }
    for (std::size_t index = 0; index < result.owners.size(); ++index) {
        if (result.owners[index] < 0 ||
            static_cast<std::size_t>(result.owners[index]) >= partition_count) {
            throw std::invalid_argument("partition owner ID is out of range");
        }
        if (index < spec.owned_vertex_count &&
            result.owners[index] != spec.partition_id) {
            throw std::invalid_argument("owned vertex names a different owner");
        }
    }
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(spec.owned_vertex_count); ++index) {
        const std::uint64_t expected_global =
            spec.ownership_encoding == "range"
                ? spec.vertex_begin + static_cast<std::uint64_t>(index)
                : result.owned_global_ids[index];
        if (result.local_to_global[index] != expected_global) {
            throw std::invalid_argument(
                "partition owned prefix disagrees with ownership metadata");
        }
        if (index != 0 && spec.ownership_encoding == "explicit" &&
            result.owned_global_ids[index] <= result.owned_global_ids[index - 1]) {
            throw std::invalid_argument(
                "explicit partition owner IDs must be strictly increasing");
        }
    }
    profile->structural_seconds += elapsed(structural_start);
    profile->total_seconds = elapsed(total_start);
    return result;
}

}  // namespace

PartitionLoadSpec::PartitionLoadSpec()
    : partition_id(-1),
      device_id(-1),
      owned_vertex_count(0),
      halo_vertex_count(0),
      adjacency_count(0),
      cut_edge_count(0),
      has_vertex_range(false),
      vertex_begin(0),
      vertex_end(0),
      weight_type(PartitionWeightType::kNone) {}

MultiPartitionLoadSpec::MultiPartitionLoadSpec()
    : manifest_validated(false),
      import_policy("full"),
      generation(0),
      directed(false),
      weighted(false),
      vertex_count(0),
      edge_count(0),
      adjacency_count(0) {}

void PartitionCSR::require_narrow_offsets(const char* algorithm) const {
    if (wide_offsets) throw std::invalid_argument(std::string(algorithm) +
        " INDEX_UNSUPPORTED: this native operator requires int32 offsets; no narrowing performed");
}

std::uint64_t PartitionCSR::local_vertex_count() const {
    return owned_vertex_count + halo_vertex_count;
}

bool PartitionCSR::weighted() const {
    return weight_type != PartitionWeightType::kNone;
}

MultiPartitionGraph::MultiPartitionGraph(const MultiPartitionLoadSpec& spec)
    : graph_id_(spec.graph_id),
      generation_(spec.generation),
      directed_(spec.directed),
      weighted_(spec.weighted),
      view_orientation_(spec.view_orientation),
      weight_key_(spec.weight_key),
      view_signature_(spec.view_signature),
      topology_signature_(spec.topology_signature),
      weight_signature_(spec.weight_signature),
      vertex_count_(spec.vertex_count),
      edge_count_(spec.edge_count),
      adjacency_count_(spec.adjacency_count),
      ownership_encoding_(spec.ownership_encoding) {
    const ImportClock::time_point native_start = ImportClock::now();
    const ImportClock::time_point metadata_start = native_start;
    if (spec.import_policy != "full" && spec.import_policy != "structural") {
        throw std::invalid_argument("import_policy must be full or structural");
    }
    import_profile_.policy = spec.import_policy;
    import_profile_.manifest_metadata_seconds = spec.manifest_metadata_seconds;
    import_profile_.manifest_projection_seconds = spec.manifest_projection_seconds;
    import_profile_.partitions.reserve(spec.partitions.size());
    const bool verify_sha256 = spec.import_policy == "full";
    if (!spec.manifest_validated) {
        throw std::invalid_argument(
            "native partition loader requires a validated manifest");
    }
    if (graph_id_.empty()) {
        throw std::invalid_argument("graph ID must be nonempty");
    }
    if (edge_count_ > adjacency_count_) {
        throw std::invalid_argument("edge count exceeds materialized adjacency count");
    }
    if (spec.partitions.empty()) {
        throw std::invalid_argument("at least one partition is required");
    }
    if (ownership_encoding_ != "range" && ownership_encoding_ != "explicit") {
        throw std::invalid_argument("unsupported ownership encoding");
    }
    validate_orientation(spec);

    std::uint64_t owned_total = 0;
    std::uint64_t adjacency_total = 0;
    std::set<std::int32_t> devices;
    std::uint64_t expected_range_begin = 0;
    partitions_.reserve(spec.partitions.size());
    import_profile_.global_metadata_seconds = elapsed(metadata_start);
    for (std::size_t index = 0; index < spec.partitions.size(); ++index) {
        ImportClock::time_point counts_start = ImportClock::now();
        const PartitionLoadSpec& partition_spec = spec.partitions[index];
        if (partition_spec.partition_id != static_cast<std::int32_t>(index)) {
            throw std::invalid_argument("partition IDs must be contiguous from zero");
        }
        if (!devices.insert(partition_spec.device_id).second) {
            throw std::invalid_argument("each partition must target a distinct device");
        }
        if (partition_spec.ownership_encoding != ownership_encoding_) {
            throw std::invalid_argument(
                "partition ownership encoding disagrees with the manifest");
        }
        if (ownership_encoding_ == "range") {
            if (!partition_spec.has_vertex_range ||
                partition_spec.vertex_begin != expected_range_begin) {
                throw std::invalid_argument("range partitions do not form a cover");
            }
            expected_range_begin = partition_spec.vertex_end;
        }
        import_profile_.partitions.emplace_back();
        import_profile_.global_counts_seconds += elapsed(counts_start);
        PartitionCSR partition = load_partition(
            partition_spec, spec.partitions.size(), weighted_, verify_sha256,
            &import_profile_.partitions.back());
        counts_start = ImportClock::now();
        if (partition.owned_vertex_count > vertex_count_ - owned_total ||
            partition.adjacency_count > adjacency_count_ - adjacency_total) {
            throw std::invalid_argument("partition counts exceed global counts");
        }
        owned_total += partition.owned_vertex_count;
        adjacency_total += partition.adjacency_count;
        partitions_.push_back(std::move(partition));
        import_profile_.global_counts_seconds += elapsed(counts_start);
    }
    const ImportClock::time_point cover_start = ImportClock::now();
    if (owned_total != vertex_count_ || adjacency_total != adjacency_count_) {
        throw std::invalid_argument("partition counts do not match global counts");
    }
    if (ownership_encoding_ == "range" && expected_range_begin != vertex_count_) {
        throw std::invalid_argument("range partitions do not cover all vertices");
    }

    if (ownership_encoding_ == "explicit") {
        typedef std::pair<std::uint64_t, std::size_t> OwnedCursor;
        std::priority_queue<OwnedCursor, std::vector<OwnedCursor>,
                            std::greater<OwnedCursor> > queue;
        std::vector<std::size_t> positions(partitions_.size(), 0);
        for (std::size_t partition = 0; partition < partitions_.size(); ++partition) {
            if (!partitions_[partition].owned_global_ids.empty()) {
                queue.push(OwnedCursor(
                    partitions_[partition].owned_global_ids[0], partition));
            }
        }
        std::uint64_t expected_global = 0;
        while (!queue.empty()) {
            const OwnedCursor current = queue.top();
            queue.pop();
            if (current.first != expected_global) {
                throw std::invalid_argument(
                    "explicit ownership does not exactly cover global vertices");
            }
            ++expected_global;
            const std::size_t partition = current.second;
            ++positions[partition];
            if (positions[partition] <
                partitions_[partition].owned_global_ids.size()) {
                queue.push(OwnedCursor(
                    partitions_[partition].owned_global_ids[positions[partition]],
                    partition));
            }
        }
        if (expected_global != vertex_count_) {
            throw std::invalid_argument(
                "explicit ownership does not cover every global vertex");
        }
    }

    import_profile_.global_cover_seconds = elapsed(cover_start);
    const ImportClock::time_point halo_start = ImportClock::now();
    for (std::size_t partition_index = 0;
         partition_index < partitions_.size(); ++partition_index) {
        const PartitionCSR& partition = partitions_[partition_index];
        std::uint64_t previous_halo = 0;
        bool have_halo = false;
        for (std::size_t local_id = 0;
             local_id < partition.local_to_global.size(); ++local_id) {
            const std::uint64_t global_id = partition.local_to_global[local_id];
            if (global_id >= vertex_count_) {
                throw std::invalid_argument(
                    "partition local-to-global ID exceeds the graph domain");
            }
            // load_partition already checked the owned prefix and its rank;
            // the disjoint cover above establishes its unique global owner.
            if (local_id < partition.owned_vertex_count) continue;
            // A halo only needs membership in its declared owner. Disjointness
            // makes searching every other partition redundant.
            const std::int32_t actual_owner = partition.owners[local_id];
            const PartitionCSR& owner = partitions_[static_cast<std::size_t>(actual_owner)];
            const bool member = ownership_encoding_ == "range"
                ? owner.vertex_begin <= global_id && global_id < owner.vertex_end
                : std::binary_search(owner.owned_global_ids.begin(),
                                     owner.owned_global_ids.end(), global_id);
            if (!member) {
                throw std::invalid_argument(
                    "partition owner map disagrees with global ownership");
            }
            if (local_id >= partition.owned_vertex_count) {
                if (actual_owner == partition.partition_id ||
                    (have_halo && global_id <= previous_halo)) {
                    throw std::invalid_argument(
                        "partition halo must be remote and strictly increasing");
                }
                previous_halo = global_id;
                have_halo = true;
            }
        }
    }
    import_profile_.global_halo_seconds = elapsed(halo_start);
    import_profile_.native_total_seconds = elapsed(native_start);
}

const std::string& MultiPartitionGraph::graph_id() const { return graph_id_; }
std::uint64_t MultiPartitionGraph::generation() const { return generation_; }
bool MultiPartitionGraph::directed() const { return directed_; }
bool MultiPartitionGraph::weighted() const { return weighted_; }
const std::string& MultiPartitionGraph::view_orientation() const {
    return view_orientation_;
}
const std::string& MultiPartitionGraph::weight_key() const { return weight_key_; }
const std::string& MultiPartitionGraph::view_signature() const {
    return view_signature_;
}
const std::string& MultiPartitionGraph::topology_signature() const {
    return topology_signature_;
}
const std::string& MultiPartitionGraph::weight_signature() const {
    return weight_signature_;
}
std::uint64_t MultiPartitionGraph::vertex_count() const { return vertex_count_; }
std::uint64_t MultiPartitionGraph::edge_count() const { return edge_count_; }
std::uint64_t MultiPartitionGraph::adjacency_count() const {
    return adjacency_count_;
}
const std::string& MultiPartitionGraph::ownership_encoding() const {
    return ownership_encoding_;
}
const std::vector<PartitionCSR>& MultiPartitionGraph::partitions() const {
    return partitions_;
}

const GraphImportProfile& MultiPartitionGraph::import_profile() const {
    return import_profile_;
}

const NativeWeightCommitProfile& MultiPartitionGraph::native_commit_profile() const {
    return native_commit_profile_;
}

const std::string& MultiPartitionGraph::snapshot_origin() const {
    return snapshot_origin_;
}

std::shared_ptr<MultiPartitionGraph> MultiPartitionGraph::commit_absolute_weights(
    const std::shared_ptr<MultiPartitionGraph>& base,
    const std::string& new_id, const std::vector<AbsoluteWeightSet>& updates,
    const std::function<std::pair<std::string, std::string>(const MultiPartitionGraph&)>& signer,
    const std::function<std::string(const std::shared_ptr<MultiPartitionGraph>&,
                                    const unsigned char*, std::size_t)>& payload_hasher) {
    const auto begin = ImportClock::now();
    if (!base || !base->weighted_ || !signer)
        throw std::invalid_argument("memory commit requires a registered weighted base and internal signer");
    if (new_id.empty() || new_id == base->graph_id_ || new_id.find('\0') != std::string::npos)
        throw std::invalid_argument("memory commit requires a distinct nonempty graph ID");
    if (base->generation_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("snapshot generation overflow");
    if (!host_is_little_endian() || sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559)
        throw std::invalid_argument("memory commit currently requires little-endian IEEE f64");
    for (std::size_t rank = 0; rank < base->partitions_.size(); ++rank) {
        const auto& part = base->partitions_[rank];
        if (part.partition_id != static_cast<std::int32_t>(rank) ||
            part.weight_type != PartitionWeightType::kFloat64 ||
            part.weights_float64.size() != part.adjacency_count)
            throw std::invalid_argument("memory commit preserves f64 partition geometry only");
    }
    std::set<std::pair<std::int32_t, std::int64_t> > seen;
    for (const auto& update : updates) {
        if (update.partition_id < 0 || static_cast<std::size_t>(update.partition_id) >= base->partitions_.size() ||
            update.position < 0 || static_cast<std::uint64_t>(update.position) >= base->partitions_[update.partition_id].adjacency_count ||
            update.position > std::numeric_limits<int>::max())
            throw std::invalid_argument("memory commit position outside partition");
        if (!std::isfinite(update.value) || update.value < 0.0)
            throw std::invalid_argument("memory commit requires finite nonnegative f64 values");
        if (!seen.insert({update.partition_id, update.position}).second)
            throw std::invalid_argument("duplicate absolute-set position");
    }
    const double validation_seconds = elapsed(begin);
    const char* policy=std::getenv("EGGPU_HOST_SNAPSHOT_STORAGE");
    const std::string storage_policy=policy?policy:"deep";
    if(storage_policy!="deep" && storage_policy!="shared")
        throw std::invalid_argument("EGGPU_HOST_SNAPSHOT_STORAGE requires deep or shared");
    const bool shared=storage_policy=="shared";
    auto phase = ImportClock::now();
    // Copy metadata and immutable array handles; no whole-parent graph chain.
    auto target = std::make_shared<MultiPartitionGraph>(*base);
    target->native_commit_profile_ = NativeWeightCommitProfile();
    auto& profile = target->native_commit_profile_;
    profile.host_storage_policy=storage_policy;
    profile.validation_seconds=validation_seconds;
    profile.base_graph_id=base->graph_id_;
    profile.base_view_signature=base->view_signature_;
    profile.requested_positions=updates.size();
    target->graph_id_=new_id;
    target->generation_=base->generation_+1;
    target->snapshot_origin_="native_memory";
    target->import_profile_=GraphImportProfile();
    target->import_profile_.policy="native_memory_absolute_set";
    std::vector<std::shared_ptr<NativeWeightPatch> > packets(target->partitions_.size());
    for(auto& part:target->partitions_) {
        part.native_weight_patch.reset();
        if(!shared) {
            // Same-binary control: reproduce the original full payload copy.
            part.offsets=part.offsets.deep_clone();
            part.offsets64=part.offsets64.deep_clone();
            part.indices=part.indices.deep_clone();
            part.local_to_global=part.local_to_global.deep_clone();
            part.owners=part.owners.deep_clone();
            part.owned_global_ids=part.owned_global_ids.deep_clone();
            part.weights_float32=part.weights_float32.deep_clone();
            part.weights_float64=part.weights_float64.deep_clone();
            profile.copied_array_payload_bytes+=
                part.offset_payload_bytes()+part.indices.payload_bytes()+part.local_to_global.payload_bytes()+
                part.owners.payload_bytes()+part.owned_global_ids.payload_bytes()+part.weights_float32.payload_bytes()+part.weights_float64.payload_bytes();
        } else {
            profile.shared_topology_payload_bytes+=part.offset_payload_bytes()+part.indices.payload_bytes()+
                part.local_to_global.payload_bytes()+part.owners.payload_bytes()+part.owned_global_ids.payload_bytes();
        }
    }
    profile.clone_seconds=elapsed(phase);
    phase=ImportClock::now();
    for(const auto& update:updates) {
        const auto& part=target->partitions_[update.partition_id];
        const double& value=part.weights_float64[static_cast<std::size_t>(update.position)];
        if(std::memcmp(&value,&update.value,sizeof(double))==0)continue;
        auto& packet=packets[update.partition_id];
        if(!packet) {
            packet=std::make_shared<NativeWeightPatch>();
            packet->base_weight_sha256=base->partitions_[update.partition_id].weights_sha256;
        }
        packet->positions.push_back(static_cast<int>(update.position));
        packet->values.push_back(update.value);
        ++profile.changed_positions;
    }
    profile.apply_seconds=elapsed(phase);
    for(std::size_t rank=0;rank<packets.size();++rank) {
        auto& part=target->partitions_[rank];
        if(!packets[rank]) {
            if(shared)profile.shared_unchanged_weight_payload_bytes+=part.weights_float32.payload_bytes()+part.weights_float64.payload_bytes();
            continue;
        }
        phase=ImportClock::now();
        profile.copied_array_payload_bytes+=part.weights_float64.detach();
        profile.clone_seconds+=elapsed(phase);
        phase=ImportClock::now();
        for(std::size_t i=0;i<packets[rank]->positions.size();++i)
            part.weights_float64.set(packets[rank]->positions[i],packets[rank]->values[i]);
        profile.apply_seconds+=elapsed(phase);
        phase=ImportClock::now();
        const std::size_t bytes=part.weights_float64.size()*sizeof(double);
        if (payload_hasher) {
            part.weights_sha256 = payload_hasher(
                target, reinterpret_cast<const unsigned char*>(part.weights_float64.data()), bytes);
            if (!is_sha256(part.weights_sha256))
                throw std::invalid_argument("internal memory commit hasher returned invalid SHA-256");
        } else {
            Sha256 digest;
            if (bytes) digest.update(reinterpret_cast<const unsigned char*>(part.weights_float64.data()), bytes);
            part.weights_sha256 = digest.final_hex();
        }
        packets[rank]->target_weight_sha256=part.weights_sha256;
        part.native_weight_patch=packets[rank];
        profile.weight_bytes_hashed+=bytes;
        profile.weight_hash_seconds+=elapsed(phase);
    }
    profile.packet_payload_bytes = profile.changed_positions * (sizeof(int) + sizeof(double));
    phase = ImportClock::now();
    const auto signatures = signer(*target);
    if (!is_sha256(signatures.first) || !is_sha256(signatures.second))
        throw std::invalid_argument("internal memory commit signer returned invalid content identity");
    target->weight_signature_ = signatures.first;
    target->view_signature_ = signatures.second;
    profile.signature_seconds = elapsed(phase);
    profile.total_seconds = elapsed(begin);
    return target;
}

std::shared_ptr<MultiPartitionGraph> load_multi_partition_graph(
    const MultiPartitionLoadSpec& spec) {
    return std::shared_ptr<MultiPartitionGraph>(new MultiPartitionGraph(spec));
}

const char* partition_weight_type_name(PartitionWeightType type) {
    switch (type) {
        case PartitionWeightType::kNone:
            return "none";
        case PartitionWeightType::kFloat32:
            return "float32";
        case PartitionWeightType::kFloat64:
            return "float64";
    }
    return "unknown";
}

}  // namespace egmgpu
}  // namespace easygraph
