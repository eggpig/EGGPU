#include "partitioned_ego_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace gpu_easygraph {
namespace runtime {
namespace {

typedef std::vector<std::vector<std::uint64_t> > GlobalRows;

std::size_t checked_vertex_count(std::uint64_t count) {
    if (count == 0U || count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(
            "partitioned ego plan requires a nonempty host-addressable graph");
    }
    return static_cast<std::size_t>(count);
}

int checked_i32(std::size_t value, const char* context) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(context) + " exceeds int32 capacity");
    }
    return static_cast<int>(value);
}

void canonical_owner_row(
    const EgoInputPartitionView& source,
    std::size_t local,
    std::uint64_t vertex_count,
    std::vector<std::uint64_t>* row,
    std::size_t* non_self_entries) {
    if (!row || !non_self_entries || !source.offsets || !source.indices ||
        !source.local_to_global || local >= source.owned_vertex_count) {
        throw std::invalid_argument("invalid partition row scan request");
    }
    const std::vector<int>& offsets = *source.offsets;
    const std::vector<int>& indices = *source.indices;
    const std::vector<std::uint64_t>& local_to_global =
        *source.local_to_global;
    const int begin = offsets[local];
    const int end = offsets[local + 1U];
    if (begin < 0 || end < begin ||
        static_cast<std::size_t>(end) > indices.size()) {
        throw std::invalid_argument("invalid owner CSR offsets");
    }
    const std::uint64_t global = local_to_global[local];
    row->clear();
    row->reserve(static_cast<std::size_t>(end - begin));
    for (int edge = begin; edge < end; ++edge) {
        const int local_target = indices[static_cast<std::size_t>(edge)];
        if (local_target < 0 ||
            static_cast<std::size_t>(local_target) >= local_to_global.size()) {
            throw std::invalid_argument(
                "ego owner CSR target is outside its local map");
        }
        const std::uint64_t target =
            local_to_global[static_cast<std::size_t>(local_target)];
        if (target >= vertex_count) {
            throw std::invalid_argument(
                "ego owner CSR target is outside the global graph domain");
        }
        if (target != global) row->push_back(target);
    }
    *non_self_entries = row->size();
    std::sort(row->begin(), row->end());
    row->erase(std::unique(row->begin(), row->end()), row->end());
}

std::size_t incidence_position(
    const PartitionedEgoPlan& plan,
    const GlobalRows& logical,
    std::uint64_t source,
    std::uint64_t target) {
    const std::size_t source_index = static_cast<std::size_t>(source);
    const int partition_id = plan.owner_partition[source_index];
    const int owner_local = plan.owner_local_id[source_index];
    if (partition_id < 0 || owner_local < 0) {
        throw std::logic_error("ego owner catalog is incomplete");
    }
    const std::vector<std::uint64_t>& row = logical[source_index];
    const std::vector<std::uint64_t>::const_iterator found =
        std::lower_bound(row.begin(), row.end(), target);
    if (found == row.end() || *found != target) {
        throw std::logic_error("logical ego incidence is not symmetric");
    }
    const EgoHostPartition& partition =
        plan.partitions[static_cast<std::size_t>(partition_id)];
    return static_cast<std::size_t>(partition.offsets[
               static_cast<std::size_t>(owner_local)]) +
        static_cast<std::size_t>(found - row.begin());
}

std::uint8_t incidence_mutual(
    const PartitionedEgoPlan& plan,
    const GlobalRows& logical,
    std::uint64_t source,
    std::uint64_t target) {
    const std::size_t position = incidence_position(
        plan, logical, source, target);
    const int partition = plan.owner_partition[
        static_cast<std::size_t>(source)];
    return plan.partitions[static_cast<std::size_t>(partition)].mutual[
        position];
}

std::size_t vector_bytes(const std::vector<std::uint64_t>& values) {
    return values.size() * sizeof(std::uint64_t);
}

std::size_t vector_bytes(const std::vector<int>& values) {
    return values.size() * sizeof(int);
}

std::size_t vector_bytes(const std::vector<std::uint8_t>& values) {
    return values.size() * sizeof(std::uint8_t);
}

std::size_t vector_bytes(const std::vector<double>& values) {
    return values.size() * sizeof(double);
}

}  // namespace

EgoInputPartition::EgoInputPartition()
    : partition_id(-1), device_id(-1) {}

EgoInputPartitionView::EgoInputPartitionView()
    : partition_id(-1),
      device_id(-1),
      owned_vertex_count(0U),
      offsets(nullptr),
      indices(nullptr),
      local_to_global(nullptr) {}

EgoHostPartition::EgoHostPartition()
    : partition_id(-1),
      device_id(-1),
      remote_incidence_count(0U),
      unique_remote_rows(0U),
      reused_remote_rows(0U),
      remote_row_adjacency_entries(0U) {}

std::size_t EgoHostPartition::resident_bytes() const {
    return vector_bytes(offsets) + vector_bytes(mutual) +
        vector_bytes(neighbor_max_scale) + vector_bytes(common_product) +
        vector_bytes(indirect) + vector_bytes(degree) +
        vector_bytes(sum_scale);
}

PartitionedEgoPlan::PartitionedEgoPlan()
    : vertex_count(0U),
      directed_entry_count(0U),
      logical_edge_count(0U),
      duplicate_directed_entries_removed(0U) {}

std::size_t PartitionedEgoPlan::host_bytes() const {
    std::size_t bytes = owner_partition.size() * sizeof(int) +
        owner_local_id.size() * sizeof(int);
    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        bytes += partitions[rank].resident_bytes() +
            vector_bytes(partitions[rank].owned_global_ids);
    }
    return bytes;
}

std::shared_ptr<const PartitionedEgoPlan>
build_partitioned_ego_plan_from_views(
    const std::vector<EgoInputPartitionView>& input,
    std::uint64_t vertex_count,
    const std::string& semantic_key) {
    const std::size_t vertices = checked_vertex_count(vertex_count);
    if (input.empty() || semantic_key.empty()) {
        throw std::invalid_argument(
            "partitioned ego plan requires partitions and a semantic key");
    }
    if (input.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error("partition count exceeds int32 capacity");
    }

    std::shared_ptr<PartitionedEgoPlan> plan(new PartitionedEgoPlan());
    plan->semantic_key = semantic_key;
    plan->vertex_count = vertex_count;
    plan->owner_partition.assign(vertices, -1);
    plan->owner_local_id.assign(vertices, -1);
    plan->partitions.resize(input.size());

    std::unordered_set<int> devices;
    for (std::size_t rank = 0; rank < input.size(); ++rank) {
        const EgoInputPartitionView& source = input[rank];
        if (!source.offsets || !source.indices ||
            !source.local_to_global) {
            throw std::invalid_argument("ego partition view has null arrays");
        }
        const std::vector<int>& offsets = *source.offsets;
        const std::vector<int>& indices = *source.indices;
        const std::vector<std::uint64_t>& local_to_global =
            *source.local_to_global;
        if (source.partition_id != static_cast<int>(rank) ||
            source.device_id < 0 || !devices.insert(source.device_id).second) {
            throw std::invalid_argument(
                "ego partitions require contiguous IDs and distinct devices");
        }
        const std::size_t owned = source.owned_vertex_count;
        if (offsets.size() != owned + 1U ||
            local_to_global.size() < owned ||
            offsets.empty() || offsets.front() != 0 ||
            offsets.back() < 0 ||
            static_cast<std::size_t>(offsets.back()) != indices.size()) {
            throw std::invalid_argument("invalid owner CSR in ego partition");
        }
        EgoHostPartition& target = plan->partitions[rank];
        target.partition_id = source.partition_id;
        target.device_id = source.device_id;
        target.owned_global_ids.assign(
            local_to_global.begin(), local_to_global.begin() + owned);
        for (std::size_t local = 0; local < owned; ++local) {
            const std::uint64_t global = local_to_global[local];
            if (global >= vertex_count ||
                plan->owner_partition[static_cast<std::size_t>(global)] >= 0) {
                throw std::invalid_argument(
                    "ego owner rows do not form a unique global cover");
            }
            plan->owner_partition[static_cast<std::size_t>(global)] =
                static_cast<int>(rank);
            plan->owner_local_id[static_cast<std::size_t>(global)] =
                checked_i32(local, "owner-local vertex ID");
        }
    }
    for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
        if (plan->owner_partition[vertex] < 0) {
            throw std::invalid_argument(
                "ego owner rows do not cover every global vertex");
        }
    }

    // Stream the borrowed owner rows into the logical projection.  Only one
    // canonical row scratch is materialized at a time; the manifest CSR is
    // never copied into a second global directed-row catalog.
    GlobalRows logical(vertices);
    std::vector<std::uint64_t> row_scratch;
    for (std::size_t rank = 0; rank < input.size(); ++rank) {
        const EgoInputPartitionView& source_partition = input[rank];
        const std::vector<std::uint64_t>& local_to_global =
            *source_partition.local_to_global;
        for (std::size_t local = 0;
             local < source_partition.owned_vertex_count; ++local) {
            std::size_t raw_entries = 0U;
            canonical_owner_row(
                source_partition, local, vertex_count, &row_scratch,
                &raw_entries);
            plan->duplicate_directed_entries_removed +=
                static_cast<std::uint64_t>(raw_entries - row_scratch.size());
            plan->directed_entry_count +=
                static_cast<std::uint64_t>(row_scratch.size());
            const std::size_t source = static_cast<std::size_t>(
                local_to_global[local]);
            for (std::size_t edge = 0; edge < row_scratch.size(); ++edge) {
                const std::size_t target = static_cast<std::size_t>(
                    row_scratch[edge]);
                logical[source].push_back(static_cast<std::uint64_t>(target));
                logical[target].push_back(static_cast<std::uint64_t>(source));
            }
        }
    }
    std::vector<int> degree(vertices, 0);
    std::vector<int> sum_scale(vertices, 0);
    std::vector<int> max_scale(vertices, 0);
    for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
        std::vector<std::uint64_t>& row = logical[vertex];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        degree[vertex] = checked_i32(row.size(), "logical vertex degree");
    }

    // Allocate one incidence row for each owned vertex.  The aggregate number
    // of records is exactly 2|E_projection|, distributed by owner.
    for (std::size_t rank = 0; rank < plan->partitions.size(); ++rank) {
        EgoHostPartition& partition = plan->partitions[rank];
        const std::size_t owned = partition.owned_global_ids.size();
        partition.offsets.assign(owned + 1U, 0);
        partition.degree.resize(owned);
        partition.sum_scale.resize(owned);
        std::unordered_set<std::uint64_t> remote_rows;
        for (std::size_t local = 0; local < owned; ++local) {
            const std::uint64_t global = partition.owned_global_ids[local];
            const std::vector<std::uint64_t>& row =
                logical[static_cast<std::size_t>(global)];
            const std::size_t next =
                static_cast<std::size_t>(partition.offsets[local]) + row.size();
            partition.offsets[local + 1U] = checked_i32(
                next, "partitioned ego incidence count");
            partition.degree[local] = degree[static_cast<std::size_t>(global)];
            for (std::size_t edge = 0; edge < row.size(); ++edge) {
                const std::uint64_t neighbor = row[edge];
                if (plan->owner_partition[static_cast<std::size_t>(neighbor)] !=
                    static_cast<int>(rank)) {
                    ++partition.remote_incidence_count;
                    remote_rows.insert(neighbor);
                }
            }
        }
        const std::size_t incidences = static_cast<std::size_t>(
            partition.offsets.back());
        partition.mutual.assign(incidences, 0U);
        partition.neighbor_max_scale.assign(incidences, 0U);
        partition.unique_remote_rows = remote_rows.size();
        partition.reused_remote_rows = partition.remote_incidence_count -
            partition.unique_remote_rows;
        for (std::unordered_set<std::uint64_t>::const_iterator row =
                 remote_rows.begin(); row != remote_rows.end(); ++row) {
            partition.remote_row_adjacency_entries +=
                logical[static_cast<std::size_t>(*row)].size();
        }
    }

    // A second streaming scan marks directed presence in both owner records.
    // Duplicate input entries were removed inside each row, so an incidence
    // obtains the exact mutual value one or two without a global directed CSR.
    for (std::size_t rank = 0; rank < input.size(); ++rank) {
        const EgoInputPartitionView& source_partition = input[rank];
        const std::vector<std::uint64_t>& local_to_global =
            *source_partition.local_to_global;
        for (std::size_t local = 0;
             local < source_partition.owned_vertex_count; ++local) {
            std::size_t ignored_raw_entries = 0U;
            canonical_owner_row(
                source_partition, local, vertex_count, &row_scratch,
                &ignored_raw_entries);
            const std::uint64_t source = local_to_global[local];
            for (std::size_t edge = 0; edge < row_scratch.size(); ++edge) {
                const std::uint64_t target = row_scratch[edge];
                const std::size_t source_position = incidence_position(
                    *plan, logical, source, target);
                const std::size_t target_position = incidence_position(
                    *plan, logical, target, source);
                EgoHostPartition& source_owner = plan->partitions[
                    static_cast<std::size_t>(plan->owner_partition[
                        static_cast<std::size_t>(source)])];
                EgoHostPartition& target_owner = plan->partitions[
                    static_cast<std::size_t>(plan->owner_partition[
                        static_cast<std::size_t>(target)])];
                if (source_owner.mutual[source_position] >= 2U ||
                    target_owner.mutual[target_position] >= 2U) {
                    throw std::logic_error(
                        "directed ego incidence was counted more than once");
                }
                ++source_owner.mutual[source_position];
                ++target_owner.mutual[target_position];
            }
        }
    }

    for (std::size_t first = 0; first < vertices; ++first) {
        const std::vector<std::uint64_t>& row = logical[first];
        for (std::size_t edge = 0; edge < row.size(); ++edge) {
            const std::uint64_t second = row[edge];
            if (first >= second) continue;
            ++plan->logical_edge_count;
            const int mutual = static_cast<int>(incidence_mutual(
                *plan, logical, static_cast<std::uint64_t>(first), second));
            if (mutual < 1 || mutual > 2 ||
                sum_scale[first] > std::numeric_limits<int>::max() - mutual ||
                sum_scale[static_cast<std::size_t>(second)] >
                    std::numeric_limits<int>::max() - mutual) {
                throw std::overflow_error("invalid or overflowing ego mutual scale");
            }
            sum_scale[first] += mutual;
            sum_scale[static_cast<std::size_t>(second)] += mutual;
            max_scale[first] = std::max(max_scale[first], mutual);
            max_scale[static_cast<std::size_t>(second)] = std::max(
                max_scale[static_cast<std::size_t>(second)], mutual);
        }
    }

    // Complete the endpoint-local scale fields shared by all four metrics.
    for (std::size_t source = 0; source < vertices; ++source) {
        const int rank = plan->owner_partition[source];
        const int local = plan->owner_local_id[source];
        EgoHostPartition& partition =
            plan->partitions[static_cast<std::size_t>(rank)];
        partition.sum_scale[static_cast<std::size_t>(local)] =
            sum_scale[source];
        const std::vector<std::uint64_t>& row = logical[source];
        const std::size_t base = static_cast<std::size_t>(
            partition.offsets[static_cast<std::size_t>(local)]);
        for (std::size_t edge = 0; edge < row.size(); ++edge) {
            partition.neighbor_max_scale[base + edge] =
                static_cast<std::uint8_t>(max_scale[
                    static_cast<std::size_t>(row[edge])]);
        }
    }

    // Allocate the heavier two-hop arrays only after the two streaming scans.
    // At no point is the borrowed manifest copied into another global CSR.
    for (std::size_t rank = 0; rank < plan->partitions.size(); ++rank) {
        EgoHostPartition& partition = plan->partitions[rank];
        const std::size_t incidences = partition.mutual.size();
        partition.common_product.assign(incidences, 0U);
        partition.indirect.assign(incidences, 0.0);
    }

    // Each undirected edge is intersected once.  The resulting two-hop
    // statistics are copied into its two owner-side incidence records.
    for (std::size_t first = 0; first < vertices; ++first) {
        const std::vector<std::uint64_t>& first_row = logical[first];
        EgoHostPartition& first_partition = plan->partitions[
            static_cast<std::size_t>(plan->owner_partition[first])];
        const std::size_t first_base = static_cast<std::size_t>(
            first_partition.offsets[static_cast<std::size_t>(
                plan->owner_local_id[first])]);
        for (std::size_t edge = 0; edge < first_row.size(); ++edge) {
            const std::uint64_t second = first_row[edge];
            if (first >= second) continue;
            const std::vector<std::uint64_t>& second_row =
                logical[static_cast<std::size_t>(second)];
            EgoHostPartition& second_partition = plan->partitions[
                static_cast<std::size_t>(plan->owner_partition[
                    static_cast<std::size_t>(second)])];
            const std::size_t second_base = static_cast<std::size_t>(
                second_partition.offsets[static_cast<std::size_t>(
                    plan->owner_local_id[static_cast<std::size_t>(second)])]);
            std::size_t left = 0;
            std::size_t right = 0;
            std::uint64_t common_product = 0U;
            double indirect = 0.0;
            while (left < first_row.size() && right < second_row.size()) {
                if (first_row[left] < second_row[right]) {
                    ++left;
                } else if (second_row[right] < first_row[left]) {
                    ++right;
                } else {
                    const std::uint64_t common = first_row[left];
                    // The merge already locates both incidences. Reusing its
                    // positions avoids two binary searches per triangle.
                    const int first_mutual = static_cast<int>(
                        first_partition.mutual[first_base + left]);
                    const int second_mutual = static_cast<int>(
                        second_partition.mutual[second_base + right]);
                    const std::uint64_t product = static_cast<std::uint64_t>(
                        first_mutual * second_mutual);
                    if (common_product >
                        std::numeric_limits<std::uint64_t>::max() - product) {
                        throw std::overflow_error(
                            "ego common-neighbor product overflows uint64");
                    }
                    common_product += product;
                    const int scale = sum_scale[
                        static_cast<std::size_t>(common)];
                    if (scale <= 0) {
                        throw std::logic_error(
                            "common ego neighbor has no mutual scale");
                    }
                    indirect += static_cast<double>(product) /
                        static_cast<double>(scale);
                    ++left;
                    ++right;
                }
            }
            const std::size_t first_position = first_base + edge;
            const std::size_t second_position = incidence_position(
                *plan, logical, second, static_cast<std::uint64_t>(first));
            first_partition.common_product[first_position] = common_product;
            first_partition.indirect[first_position] = indirect;
            second_partition.common_product[second_position] = common_product;
            second_partition.indirect[second_position] = indirect;
        }
    }
    return plan;
}

std::shared_ptr<const PartitionedEgoPlan> build_partitioned_ego_plan(
    const std::vector<EgoInputPartition>& input,
    std::uint64_t vertex_count,
    const std::string& semantic_key) {
    std::vector<EgoInputPartitionView> views(input.size());
    for (std::size_t rank = 0; rank < input.size(); ++rank) {
        views[rank].partition_id = input[rank].partition_id;
        views[rank].device_id = input[rank].device_id;
        views[rank].owned_vertex_count =
            input[rank].owned_global_ids.size();
        views[rank].offsets = &input[rank].offsets;
        views[rank].indices = &input[rank].indices;
        views[rank].local_to_global = &input[rank].local_to_global;
    }
    return build_partitioned_ego_plan_from_views(
        views, vertex_count, semantic_key);
}

std::vector<double> evaluate_partitioned_ego_plan_cpu(
    const PartitionedEgoPlan& plan,
    PartitionedEgoMetric metric,
    const std::vector<std::uint64_t>& global_nodes) {
    std::vector<double> result;
    result.reserve(global_nodes.size());
    for (std::size_t index = 0; index < global_nodes.size(); ++index) {
        const std::uint64_t global = global_nodes[index];
        if (global >= plan.vertex_count) {
            throw std::out_of_range("ego result node is outside the graph");
        }
        const int rank = plan.owner_partition[static_cast<std::size_t>(global)];
        const int local = plan.owner_local_id[static_cast<std::size_t>(global)];
        const EgoHostPartition& partition =
            plan.partitions[static_cast<std::size_t>(rank)];
        const int degree = partition.degree[static_cast<std::size_t>(local)];
        const int scale = partition.sum_scale[static_cast<std::size_t>(local)];
        if (degree == 0) {
            result.push_back(metric == PartitionedEgoMetric::Hierarchy
                ? 0.0 : std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        const std::size_t begin = static_cast<std::size_t>(
            partition.offsets[static_cast<std::size_t>(local)]);
        const std::size_t end = static_cast<std::size_t>(
            partition.offsets[static_cast<std::size_t>(local) + 1U]);
        if (metric == PartitionedEgoMetric::EffectiveSize ||
            metric == PartitionedEgoMetric::Efficiency) {
            double value = 0.0;
            for (std::size_t edge = begin; edge < end; ++edge) {
                value += 1.0 - static_cast<double>(
                    partition.common_product[edge]) /
                    (static_cast<double>(scale) * static_cast<double>(
                        partition.neighbor_max_scale[edge]));
            }
            if (metric == PartitionedEgoMetric::Efficiency) {
                value /= static_cast<double>(scale);
            }
            result.push_back(value);
            continue;
        }
        std::vector<double> local_constraint;
        local_constraint.reserve(end - begin);
        double total = 0.0;
        for (std::size_t edge = begin; edge < end; ++edge) {
            const double normalized =
                (static_cast<double>(partition.mutual[edge]) +
                 partition.indirect[edge]) /
                static_cast<double>(scale);
            const double value = normalized * normalized;
            local_constraint.push_back(value);
            total += value;
        }
        if (metric == PartitionedEgoMetric::Constraint) {
            result.push_back(total);
            continue;
        }
        if (degree <= 1 || total <= 0.0) {
            result.push_back(0.0);
            continue;
        }
        double hierarchy = 0.0;
        for (std::size_t edge = 0; edge < local_constraint.size(); ++edge) {
            const double probability = local_constraint[edge] / total;
            hierarchy += probability *
                std::log(probability * static_cast<double>(degree)) /
                std::log(static_cast<double>(degree));
        }
        result.push_back(hierarchy);
    }
    return result;
}

}  // namespace runtime
}  // namespace gpu_easygraph
