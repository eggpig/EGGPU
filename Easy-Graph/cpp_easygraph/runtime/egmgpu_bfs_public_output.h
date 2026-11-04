#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace easygraph { namespace egmgpu {
struct BfsPublicSlice {
    std::uint64_t checked = 0, unreachable = 0, invalid = 0;
};

// Owner IDs and their disjoint complete cover are certified by the immutable
// BFS plan. Workers read their pinned D2H buffer and write final FP64 values
// once; neither an intermediate global int64 array nor a Python object is used.
inline BfsPublicSlice assemble_bfs_public_host(
    const std::uint64_t* global_ids, std::size_t count,
    const std::int64_t* observed, double* destination, std::uint64_t vertices) {
    BfsPublicSlice result;
    for (std::size_t local = 0; local < count; ++local) {
        ++result.checked;
        const auto global = global_ids[local];
        const auto value = observed[local];
        if (global >= vertices) { ++result.invalid; continue; }
        if (value == std::numeric_limits<std::int64_t>::max() || value < 0) {
            destination[global] = std::numeric_limits<double>::infinity();
            ++result.unreachable;
        } else if (static_cast<std::uint64_t>(value) >= vertices) {
            ++result.invalid;
        } else {
            destination[global] = static_cast<double>(value);
        }
    }
    return result;
}

struct BfsPublicObservation {
    std::atomic<std::uint64_t> checked{0}, unreachable{0}, invalid{0};
    void add(const BfsPublicSlice& slice) {
        checked.fetch_add(slice.checked, std::memory_order_relaxed);
        unreachable.fetch_add(slice.unreachable, std::memory_order_relaxed);
        invalid.fetch_add(slice.invalid, std::memory_order_relaxed);
    }
    void verify(std::uint64_t vertices) const {
        if (checked.load() != vertices || unreachable.load() > vertices || invalid.load())
            throw std::runtime_error("BFS worker FP64 output has an incomplete or invalid exact distance domain");
    }
};

// Caller/GIL-owned lifetime encloses execute_gang and its failure drain.
// Workers borrow only data(). Publish is allowed once, after their join.
class BfsPublicOutput {
    pybind11::array_t<double> values_;
    std::uint64_t count_;
    bool published_ = false;
public:
    explicit BfsPublicOutput(std::uint64_t count) : count_(count) {
        if (count > (std::uint64_t(1) << 53) ||
            count > static_cast<std::uint64_t>(std::numeric_limits<pybind11::ssize_t>::max()) / sizeof(double))
            throw std::overflow_error("BFS worker FP64 output requires an exact dense distance domain and addressable host storage");
        values_ = pybind11::array_t<double>(static_cast<pybind11::ssize_t>(count));
    }
    double* data() {
        if (published_) throw std::logic_error("BFS public output already published");
        return values_.mutable_data();
    }
    pybind11::array_t<double> publish(const BfsPublicObservation& observation) {
        if (published_) throw std::logic_error("BFS public output already published");
        observation.verify(count_);
        published_ = true;
        return std::move(values_);
    }
};

inline pybind11::dict bfs_public_output_proof(
    std::uint64_t count, const BfsPublicObservation& observation) {
    observation.verify(count);
    pybind11::dict proof;
    proof["version"] = "native_bfs_worker_public_fp64_v1";
    proof["requested"] = "1"; proof["eligible"] = true; proof["used"] = true;
    proof["reason"] = "complete_private_worker_scatter";
    proof["producer"] = "numpy_owned_fp64_worker_scatter_normalize";
    proof["complete_vertices"] = count;
    proof["checked_vertices"] = observation.checked.load();
    proof["unreachable_vertices"] = observation.unreachable.load();
    proof["invalid_vertices"] = observation.invalid.load();
    proof["values_bytes"] = count * sizeof(double);
    proof["intermediate_int64_bytes_avoided"] = count * sizeof(std::int64_t);
    proof["intermediate_payload_read_write_bytes_avoided"] = count * 2U * sizeof(std::int64_t);
    return proof;
}
}} // namespace easygraph::egmgpu
