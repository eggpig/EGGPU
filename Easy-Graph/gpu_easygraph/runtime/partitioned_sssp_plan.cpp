#include "partitioned_sssp_plan.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <cstring>
#include <unordered_set>

namespace gpu_easygraph {
namespace runtime {
namespace {

std::size_t checked_size(std::uint64_t value, const char* context) {
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(context);
    }
    return static_cast<std::size_t>(value);
}

std::size_t find_owned_local(
    const SsspHostPartition& partition, std::uint64_t global_id) {
    const std::vector<std::uint64_t>::const_iterator owned_end =
        partition.local_to_global.begin() +
        static_cast<std::ptrdiff_t>(partition.owned_vertex_count);
    const std::vector<std::uint64_t>::const_iterator found = std::lower_bound(
        partition.local_to_global.begin(), owned_end, global_id);
    if (found == owned_end || *found != global_id) {
        throw std::invalid_argument(
            "SSSP halo global ID is absent from its declared owner");
    }
    return static_cast<std::size_t>(
        found - partition.local_to_global.begin());
}

}  // namespace

std::size_t SsspOwnerRoute::capacity() const {
    return source_halo_local_ids.size();
}

const SsspOwnerRoute& PartitionedSsspPlan::route(
    std::size_t source_rank, std::size_t destination_rank) const {
    const std::size_t count = partitions.size();
    if (source_rank >= count || destination_rank >= count) {
        throw std::out_of_range("SSSP route rank is out of range");
    }
    return routes[source_rank * count + destination_rank];
}

std::size_t PartitionedSsspPlan::owner_rank(
    std::uint64_t global_vertex) const {
    if (global_vertex >= owner_by_global.size()) {
        throw std::out_of_range("SSSP source vertex is outside the graph");
    }
    return static_cast<std::size_t>(
        owner_by_global[static_cast<std::size_t>(global_vertex)]);
}

std::size_t PartitionedSsspPlan::owner_local(
    std::uint64_t global_vertex) const {
    const std::size_t rank = owner_rank(global_vertex);
    return find_owned_local(partitions[rank], global_vertex);
}

PartitionedSsspPlan build_partitioned_sssp_plan(
    const std::vector<SsspHostPartition>& partitions,
    std::uint64_t vertex_count,
    SsspWeightEncoding weight_encoding,
    SsspWireEncoding wire_encoding) {
    if (vertex_count == 0 || partitions.empty()) {
        throw std::invalid_argument(
            "partitioned SSSP requires a nonempty graph and partitions");
    }
    const std::size_t global_count = checked_size(
        vertex_count, "SSSP vertex count exceeds host address space");
    std::vector<int> owner(global_count, -1);
    std::vector<int> devices;
    devices.reserve(partitions.size());
    if (weight_encoding != SsspWeightEncoding::F64 &&
        weight_encoding != SsspWeightEncoding::U8Exact) {
        throw std::invalid_argument("unknown SSSP weight encoding");
    }
    const bool compact_weights = weight_encoding == SsspWeightEncoding::U8Exact;
    std::vector<std::vector<std::uint8_t> > encoded(partitions.size());
    double fused_validation_encoding_seconds = 0.0;
    bool all_integer_weights = true;
    double max_weight = 0.;
    std::uint64_t maximum_count = 0, noninteger_count = 0, scanned_edges = 0;
    if (wire_encoding!=SsspWireEncoding::F64 && wire_encoding!=SsspWireEncoding::U32Exact)
        throw std::invalid_argument("unknown SSSP wire encoding");

    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        const SsspHostPartition& partition = partitions[rank];
        if (partition.partition_id != static_cast<int>(rank) ||
            partition.device_id < 0 ||
            std::find(devices.begin(), devices.end(), partition.device_id) !=
                devices.end()) {
            throw std::invalid_argument(
                "SSSP partition IDs must be contiguous and devices distinct");
        }
        devices.push_back(partition.device_id);
        const std::size_t local_count = partition.local_to_global.size();
        if (partition.owned_vertex_count >
                static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            partition.owned_vertex_count > local_count ||
            partition.owners.size() != local_count ||
            partition.offsets.size() != partition.owned_vertex_count + 1U ||
            partition.offsets.empty() || partition.offsets.front() != 0 ||
            partition.offsets.back() < 0 ||
            static_cast<std::size_t>(partition.offsets.back()) !=
                partition.indices.size() ||
            partition.weights.size() != partition.indices.size()) {
            throw std::invalid_argument(
                "SSSP partition cardinalities are inconsistent");
        }
        if (partition.owned_vertex_count == 0 &&
            (local_count != 0 || !partition.indices.empty() ||
             !partition.weights.empty() ||
             partition.offsets != std::vector<int>(1U, 0))) {
            throw std::invalid_argument(
                "empty SSSP partition must contain only the zero CSR offset");
        }
        for (std::size_t row = 1; row < partition.offsets.size(); ++row) {
            if (partition.offsets[row] < partition.offsets[row - 1U]) {
                throw std::invalid_argument("SSSP offsets must be monotone");
            }
        }
        if (!std::is_sorted(
                partition.local_to_global.begin(),
                partition.local_to_global.begin() +
                    static_cast<std::ptrdiff_t>(
                        partition.owned_vertex_count))) {
            throw std::invalid_argument(
                "SSSP owned global IDs must be sorted");
        }
        std::uint64_t previous_halo = 0;
        bool have_halo = false;
        for (std::size_t local = 0; local < local_count; ++local) {
            const std::uint64_t global = partition.local_to_global[local];
            const int declared_owner = partition.owners[local];
            if (global >= vertex_count || declared_owner < 0 ||
                static_cast<std::size_t>(declared_owner) >=
                    partitions.size()) {
                throw std::invalid_argument(
                    "SSSP local vertex metadata is invalid");
            }
            if (local < partition.owned_vertex_count) {
                if (declared_owner != partition.partition_id ||
                    owner[static_cast<std::size_t>(global)] != -1) {
                    throw std::invalid_argument(
                        "SSSP owned vertices are not a disjoint cover");
                }
                owner[static_cast<std::size_t>(global)] =
                    partition.partition_id;
            } else {
                if (declared_owner == partition.partition_id ||
                    (have_halo && global <= previous_halo)) {
                    throw std::invalid_argument(
                        "SSSP halo IDs must be remote and strictly increasing");
                }
                previous_halo = global;
                have_halo = true;
            }
        }
        // Certification and conversion share the already-required edge scan.
        // This interval includes endpoint and finite/nonnegative validation;
        // it is not presented as a pure conversion-only time.
        const auto weight_scan_begin = std::chrono::steady_clock::now();
        if (compact_weights) encoded[rank].resize(partition.weights.size());
        for (std::size_t edge = 0; edge < partition.indices.size(); ++edge) {
            const int target = partition.indices[edge];
            if (target < 0 || static_cast<std::size_t>(target) >= local_count) {
                throw std::invalid_argument(
                    "SSSP local edge endpoint is out of range");
            }
            if (!std::isfinite(partition.weights[edge]) ||
                partition.weights[edge] < 0.0) {
                throw std::invalid_argument(
                    "delta-stepping SSSP requires finite nonnegative weights");
            }
            if (compact_weights) {
                const double value = partition.weights[edge];
                if (value > 255.0 || std::floor(value) != value) {
                    throw std::invalid_argument(
                        "u8_exact SSSP requires every weight to be an integer in [0,255]");
                }
                encoded[rank][edge] = static_cast<std::uint8_t>(value);
            }
            const bool integer = std::floor(partition.weights[edge]) == partition.weights[edge];
            noninteger_count += !integer;
            all_integer_weights = all_integer_weights && integer;
            if (partition.weights[edge] > max_weight) maximum_count = 1;
            else if (partition.weights[edge] == max_weight) ++maximum_count;
            ++scanned_edges;
            max_weight = std::max(max_weight,partition.weights[edge]);
        }
        fused_validation_encoding_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - weight_scan_begin).count();
    }
    if (std::find(owner.begin(), owner.end(), -1) != owner.end()) {
        throw std::invalid_argument(
            "SSSP partitions do not own every global vertex");
    }

    PartitionedSsspPlan plan;
    plan.vertex_count = vertex_count;
    plan.partitions = partitions;
    plan.weight_encoding = weight_encoding;
    plan.wire_encoding = wire_encoding;
    plan.maximum_integer_weight = max_weight;
    plan.maximum_weight_count = maximum_count;
    plan.noninteger_weight_count = noninteger_count;
    plan.weight_statistics_available = true;
    plan.weight_validation_edges = scanned_edges;
    const long double bound = static_cast<long double>(vertex_count-1U) * max_weight;
    plan.wire_integer_bound_certified = all_integer_weights && bound < static_cast<long double>(UINT32_MAX);
    if (wire_encoding==SsspWireEncoding::U32Exact && !plan.wire_integer_bound_certified)
        throw std::invalid_argument("u32_exact wire requires integer nonnegative weights and (N-1)*maxW < UINT32_MAX");
    // Common pruning for both representations when the same exact bound is certified.
    // Nonnegative shortest paths have a simple path of <=N-1 edges; every
    // prefix of an optimal path is <=bound. Discarding larger messages is exact.
    if (plan.wire_integer_bound_certified) plan.wire_distance_bound = static_cast<double>(bound);
    plan.weight_validation_and_encoding_seconds = fused_validation_encoding_seconds;
    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        plan.partitions[rank].weights_u8 = std::move(encoded[rank]);
    }
    plan.owner_by_global = owner;
    plan.route_capacity = 0;
    const std::size_t count = partitions.size();
    plan.routes.resize(count * count);
    for (std::size_t source = 0; source < count; ++source) {
        for (std::size_t destination = 0; destination < count;
             ++destination) {
            SsspOwnerRoute& route =
                plan.routes[source * count + destination];
            route.source_rank = static_cast<int>(source);
            route.destination_rank = static_cast<int>(destination);
        }
    }
    for (std::size_t source = 0; source < count; ++source) {
        const SsspHostPartition& partition = partitions[source];
        for (std::size_t local = partition.owned_vertex_count;
             local < partition.local_to_global.size(); ++local) {
            const std::uint64_t global = partition.local_to_global[local];
            const std::size_t destination = static_cast<std::size_t>(
                partition.owners[local]);
            if (owner[static_cast<std::size_t>(global)] !=
                static_cast<int>(destination)) {
                throw std::invalid_argument(
                    "SSSP halo owner disagrees with global ownership");
            }
            SsspOwnerRoute& route =
                plan.routes[source * count + destination];
            route.source_halo_local_ids.push_back(static_cast<int>(local));
            route.destination_owner_local_ids.push_back(static_cast<int>(
                find_owned_local(partitions[destination], global)));
            ++plan.route_capacity;
        }
    }
    return plan;
}

bool sssp_native_delta_density_eligible(std::uint64_t changed_positions,
    std::uint64_t source_edges, std::uint64_t maximum_density_ppm) {
    const std::uint64_t scale = 1000000;
    if (maximum_density_ppm > scale || changed_positions > source_edges)
        throw std::invalid_argument("Invalid SSSP delta density domain");
    // floor(m*rho/scale), without overflowing uint64_t even at its maximum.
    const std::uint64_t limit = (source_edges / scale) * maximum_density_ppm +
        ((source_edges % scale) * maximum_density_ppm) / scale;
    return changed_positions <= limit;
}

bool matches_sssp_direct_parent(
    const PartitionedHostPlanCacheKey& cached_key,
    const PartitionedSsspPlan& cached_plan,
    const PartitionedHostPlanCacheKey& target_key,
    const std::string& base_graph_id,
    const std::string& base_view_signature,
    const std::vector<std::pair<int, int> >& target_partition_devices) {
    if (target_key.algorithm != PartitionedHostPlanAlgorithm::Sssp ||
        cached_key.algorithm != target_key.algorithm ||
        base_graph_id.empty() || base_view_signature.empty() ||
        cached_key.graph_id != base_graph_id ||
        cached_key.view_signature != base_view_signature ||
        target_key.generation == 0 ||
        cached_key.generation != target_key.generation - 1 ||
        cached_key.topology_signature != target_key.topology_signature ||
        !cached_key.weighted || !target_key.weighted ||
        cached_key.vertex_count != target_key.vertex_count ||
        cached_key.adjacency_count != target_key.adjacency_count ||
        cached_key.device_group != target_key.device_group ||
        cached_key.plan_variant != target_key.plan_variant ||
        cached_plan.vertex_count != target_key.vertex_count ||
        cached_plan.weight_encoding != SsspWeightEncoding::F64 ||
        cached_plan.wire_encoding != SsspWireEncoding::F64 ||
        cached_plan.partitions.size() != target_partition_devices.size()) return false;
    // Topology digests and sorted device_group do not encode rank placement.
    for (std::size_t rank = 0; rank < cached_plan.partitions.size(); ++rank) {
        const auto& part = cached_plan.partitions[rank];
        if (part.partition_id != target_partition_devices[rank].first ||
            part.device_id != target_partition_devices[rank].second) return false;
    }
    return true;
}

static PartitionedSsspPlan rebuild_sssp_f64_from_parent_impl(
    const PartitionedSsspPlan& parent,
    std::vector<std::vector<double> > current_weights,
    const std::vector<SsspNativeWeightDelta>* native_deltas) {
    if (parent.vertex_count == 0 || parent.partitions.empty() ||
        parent.weight_encoding != SsspWeightEncoding::F64 ||
        parent.wire_encoding != SsspWireEncoding::F64 ||
        current_weights.size() != parent.partitions.size())
        throw std::invalid_argument("SSSP parent topology requires a validated F64 plan and matching weights");
    const auto delta_validation_begin = std::chrono::steady_clock::now();
    bool delta_certified = false;
    double delta_maximum = parent.maximum_integer_weight;
    std::uint64_t delta_maximum_count = parent.maximum_weight_count;
    std::uint64_t delta_noninteger_count = parent.noninteger_weight_count;
    std::uint64_t delta_positions = 0;
    if (native_deltas) {
        if (!parent.weight_statistics_available || native_deltas->size() != parent.partitions.size())
            throw std::invalid_argument("SSSP native delta requires certified parent statistics and matching ranks");
        std::uint64_t removed_maximum = 0, inserted_at_maximum = 0;
        std::uint64_t removed_noninteger = 0, inserted_noninteger = 0, total_edges = 0;
        double inserted_maximum = 0.;
        std::uint64_t inserted_maximum_count = 0;
        for (std::size_t rank = 0; rank < parent.partitions.size(); ++rank) {
            const auto& old = parent.partitions[rank].weights;
            auto& current = current_weights[rank];
            const auto& delta = (*native_deltas)[rank];
            if (current.size() != old.size() || delta.positions.size() != delta.values.size())
                throw std::invalid_argument("SSSP native delta geometry mismatch");
            total_edges += old.size();
            std::unordered_set<int> seen;
            seen.reserve(delta.positions.size());
            for (std::size_t i = 0; i < delta.positions.size(); ++i) {
                const int position = delta.positions[i];
                const double value = delta.values[i];
                if (position < 0 || static_cast<std::size_t>(position) >= old.size() ||
                    !seen.insert(position).second || !std::isfinite(value) || value < 0.)
                    throw std::invalid_argument("SSSP native delta position/value mismatch");
                const double previous = old[position];
                current[position] = value;
                removed_maximum += previous == parent.maximum_integer_weight;
                inserted_at_maximum += value == parent.maximum_integer_weight;
                removed_noninteger += std::floor(previous) != previous;
                inserted_noninteger += std::floor(value) != value;
                if (value > inserted_maximum) inserted_maximum_count = 1;
                else if (value == inserted_maximum) ++inserted_maximum_count;
                inserted_maximum = std::max(inserted_maximum, value);
                ++delta_positions;
            }
        }
        if (removed_maximum > parent.maximum_weight_count || removed_noninteger > parent.noninteger_weight_count)
            throw std::logic_error("SSSP parent statistics do not match the complete native delta");
        delta_noninteger_count = parent.noninteger_weight_count - removed_noninteger + inserted_noninteger;
        const auto remaining_maximum = parent.maximum_weight_count - removed_maximum + inserted_at_maximum;
        if (inserted_maximum > parent.maximum_integer_weight) {
            delta_maximum = inserted_maximum;
            delta_maximum_count = inserted_maximum_count;
            delta_certified = true;
        } else if (remaining_maximum || total_edges == 0) {
            delta_maximum_count = remaining_maximum;
            delta_certified = true;
        }
        // Removing the last maximum can expose any unchanged edge as the new
        // maximum. Fall back to the unchanged full scan instead of guessing a
        // bound, adding an index, or retaining an ancestor graph/plan.
    }

    PartitionedSsspPlan plan;
    plan.vertex_count = parent.vertex_count;
    plan.weight_validation_and_encoding_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - delta_validation_begin).count();
    plan.partitions.resize(parent.partitions.size());
    plan.owner_by_global = parent.owner_by_global;
    plan.routes = parent.routes;
    plan.route_capacity = parent.route_capacity;
    plan.parent_topology_copied_bytes = parent.owner_by_global.size() * sizeof(int);
    for (const auto& route : parent.routes) {
        plan.parent_topology_copied_bytes +=
            (route.source_halo_local_ids.size() + route.destination_owner_local_ids.size()) * sizeof(int);
    }
    bool all_integer_weights = true;
    double max_weight = 0.;
    std::uint64_t maximum_count = 0, noninteger_count = 0, scanned_edges = 0;
    for (std::size_t rank = 0; rank < parent.partitions.size(); ++rank) {
        const auto& source = parent.partitions[rank];
        auto& target = plan.partitions[rank];
        if (current_weights[rank].size() != source.indices.size())
            throw std::invalid_argument("SSSP current weights do not match parent topology");
        target.partition_id = source.partition_id;
        target.device_id = source.device_id;
        target.owned_vertex_count = source.owned_vertex_count;
        target.offsets = source.offsets;
        target.indices = source.indices;
        target.local_to_global = source.local_to_global;
        target.owners = source.owners;
        target.weights = std::move(current_weights[rank]);
        plan.parent_topology_copied_bytes +=
            (source.offsets.size() + source.indices.size() + source.owners.size()) * sizeof(int) +
            source.local_to_global.size() * sizeof(std::uint64_t);
        if (delta_certified) continue;
        // Preserve the existing rank/edge scan, endpoint checks, FP64 tests,
        // integer predicate and std::max order. No incremental bound update.
        const auto weight_scan_begin = std::chrono::steady_clock::now();
        for (std::size_t edge = 0; edge < target.indices.size(); ++edge) {
            const int endpoint = target.indices[edge];
            if (endpoint < 0 || static_cast<std::size_t>(endpoint) >= target.local_to_global.size())
                throw std::invalid_argument("SSSP local edge endpoint is out of range");
            if (!std::isfinite(target.weights[edge]) || target.weights[edge] < 0.0)
                throw std::invalid_argument("delta-stepping SSSP requires finite nonnegative weights");
            const bool integer = std::floor(target.weights[edge]) == target.weights[edge];
            noninteger_count += !integer;
            all_integer_weights = all_integer_weights && integer;
            if (target.weights[edge] > max_weight) maximum_count = 1;
            else if (target.weights[edge] == max_weight) ++maximum_count;
            ++scanned_edges;
            max_weight = std::max(max_weight, target.weights[edge]);
        }
        plan.weight_validation_and_encoding_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - weight_scan_begin).count();
    }
    if (delta_certified) {
        max_weight = delta_maximum;
        maximum_count = delta_maximum_count;
        noninteger_count = delta_noninteger_count;
        all_integer_weights = noninteger_count == 0;
    }
    plan.weight_delta_positions = delta_positions;
    plan.weight_delta_certification_used = delta_certified;
    plan.weight_delta_rescan_required = native_deltas && !delta_certified;
    plan.maximum_integer_weight = max_weight;
    plan.maximum_weight_count = maximum_count;
    plan.noninteger_weight_count = noninteger_count;
    plan.weight_statistics_available = true;
    plan.weight_validation_edges = scanned_edges;
    const long double bound = static_cast<long double>(plan.vertex_count - 1U) * max_weight;
    plan.wire_integer_bound_certified = all_integer_weights && bound < static_cast<long double>(UINT32_MAX);
    if (plan.wire_integer_bound_certified) plan.wire_distance_bound = static_cast<double>(bound);
    plan.parent_topology_reused = true;
    return plan;
}

void validate_sssp_native_delta_identity(
    const std::vector<std::string>& parent_hashes,
    const std::vector<std::string>& current_hashes,
    const std::vector<SsspNativeWeightDelta>& deltas,
    std::uint64_t producer_changed_positions) {
    if (parent_hashes.size() != deltas.size() || current_hashes.size() != deltas.size())
        throw std::invalid_argument("Native SSSP certification identity rank mismatch");
    std::uint64_t changed_positions = 0;
    for (std::size_t rank = 0; rank < deltas.size(); ++rank) {
        const auto& delta = deltas[rank];
        if (parent_hashes[rank].size() != 64 || current_hashes[rank].size() != 64 ||
            delta.positions.size() != delta.values.size())
            throw std::invalid_argument("Native SSSP certification packet geometry mismatch");
        if (delta.positions.empty()) {
            if (current_hashes[rank] != parent_hashes[rank] ||
                !delta.base_weight_sha256.empty() || !delta.target_weight_sha256.empty())
                throw std::invalid_argument("Native SSSP certification missing changed rank packet");
        } else if (delta.base_weight_sha256 != parent_hashes[rank] ||
                   delta.target_weight_sha256 != current_hashes[rank]) {
            throw std::invalid_argument("Native SSSP certification packet identity mismatch");
        }
        changed_positions += delta.positions.size();
    }
    if (changed_positions != producer_changed_positions)
        throw std::invalid_argument("Native SSSP certification packet coverage mismatch");
}

PartitionedSsspPlan rebuild_sssp_f64_from_parent(
    const PartitionedSsspPlan& parent,
    std::vector<std::vector<double> > current_weights) {
    return rebuild_sssp_f64_from_parent_impl(parent, std::move(current_weights), nullptr);
}

PartitionedSsspPlan rebuild_sssp_f64_from_native_delta(
    const PartitionedSsspPlan& parent,
    const std::vector<SsspNativeWeightDelta>& native_deltas) {
    if (native_deltas.size() != parent.partitions.size())
        throw std::invalid_argument("Native SSSP delta rank count mismatch");
    std::vector<std::vector<double> > weights;
    weights.reserve(parent.partitions.size());
    for (std::size_t rank = 0; rank < parent.partitions.size(); ++rank)
        weights.push_back(parent.partitions[rank].weights);
    // The private implementation validates each delta before applying it.
    return rebuild_sssp_f64_from_parent_impl(parent, std::move(weights), &native_deltas);
}

std::vector<double> partitioned_sssp_reference(
    const PartitionedSsspPlan& plan,
    std::uint64_t source) {
    const PartitionedSsspPlan checked = build_partitioned_sssp_plan(
        plan.partitions, plan.vertex_count);
    checked.owner_rank(source);
    const std::size_t count = checked_size(
        checked.vertex_count, "SSSP result exceeds host address space");
    typedef std::pair<double, std::uint64_t> QueueItem;
    std::vector<std::vector<std::pair<std::uint64_t, double> > > adjacency(
        count);
    for (std::size_t rank = 0; rank < checked.partitions.size(); ++rank) {
        const SsspHostPartition& partition = checked.partitions[rank];
        for (std::size_t row = 0; row < partition.owned_vertex_count; ++row) {
            const std::uint64_t global_source =
                partition.local_to_global[row];
            for (int edge = partition.offsets[row];
                 edge < partition.offsets[row + 1U]; ++edge) {
                const std::size_t index = static_cast<std::size_t>(edge);
                adjacency[static_cast<std::size_t>(global_source)].push_back(
                    std::make_pair(
                        partition.local_to_global[static_cast<std::size_t>(
                            partition.indices[index])],
                        partition.weights[index]));
            }
        }
    }
    std::vector<double> distance(
        count, std::numeric_limits<double>::infinity());
    std::priority_queue<QueueItem, std::vector<QueueItem>,
                        std::greater<QueueItem> > queue;
    distance[static_cast<std::size_t>(source)] = 0.0;
    queue.push(std::make_pair(0.0, source));
    while (!queue.empty()) {
        const QueueItem current = queue.top();
        queue.pop();
        if (current.first != distance[static_cast<std::size_t>(current.second)]) {
            continue;
        }
        const std::vector<std::pair<std::uint64_t, double> >& row =
            adjacency[static_cast<std::size_t>(current.second)];
        for (std::size_t edge = 0; edge < row.size(); ++edge) {
            const double candidate = current.first + row[edge].second;
            const std::size_t target = static_cast<std::size_t>(row[edge].first);
            if (candidate < distance[target]) {
                distance[target] = candidate;
                queue.push(std::make_pair(candidate, row[edge].first));
            }
        }
    }
    return distance;
}

std::vector<double> merge_partitioned_sssp_owner_distances(
    const PartitionedSsspPlan& plan,
    const std::vector<std::vector<double> >& owned_distances) {
    if (owned_distances.size() != plan.partitions.size()) {
        throw std::invalid_argument(
            "SSSP owner result rank count does not match the plan");
    }
    const std::size_t count = checked_size(
        plan.vertex_count, "SSSP result exceeds host address space");
    std::vector<double> result(
        count, std::numeric_limits<double>::infinity());
    for (std::size_t rank = 0; rank < plan.partitions.size(); ++rank) {
        const SsspHostPartition& partition = plan.partitions[rank];
        if (owned_distances[rank].size() != partition.owned_vertex_count) {
            throw std::invalid_argument(
                "SSSP owner result length does not match its partition");
        }
        for (std::size_t local = 0; local < partition.owned_vertex_count;
             ++local) {
            result[static_cast<std::size_t>(partition.local_to_global[local])] =
                owned_distances[rank][local];
        }
    }
    return result;
}

}  // namespace runtime
}  // namespace gpu_easygraph
