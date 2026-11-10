#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace easygraph { namespace egmgpu {
inline bool sssp_numpy_output_requested() {
    const char* raw = std::getenv("EGGPU_NATIVE_SSSP_OWNED_OUTPUT");
    const std::string value = raw ? raw : "0";
    if (value != "0" && value != "1")
        throw std::invalid_argument("EGGPU_NATIVE_SSSP_OWNED_OUTPUT must be 0 or 1");
    return value == "1";
}

// Caller/GIL-owned object. Workers receive only data(), never a Python object.
// Its lifetime encloses execute_gang, including draining accepted tasks on error.
// Both routes allocate uninitialized request-private FP64 storage exactly once.
class SsspHostOutput {
    std::size_t count_;
    bool numpy_owned_, published_ = false;
    std::unique_ptr<double[]> legacy_;
    pybind11::array_t<double> numpy_;
public:
    SsspHostOutput(std::uint64_t count, bool numpy_owned)
        : count_(count), numpy_owned_(numpy_owned) {
        if (count > static_cast<std::uint64_t>(std::numeric_limits<pybind11::ssize_t>::max()) / sizeof(double))
            throw std::overflow_error("SSSP host output size overflow");
        if (numpy_owned_) numpy_ = pybind11::array_t<double>(static_cast<pybind11::ssize_t>(count));
        else legacy_.reset(new double[count_]);
    }
    double* data() {
        if (published_) throw std::logic_error("SSSP output already published");
        return numpy_owned_ ? numpy_.mutable_data() : legacy_.get();
    }
    pybind11::array_t<double> publish() {
        if (published_) throw std::logic_error("SSSP output already published");
        published_ = true;
        if (numpy_owned_) return std::move(numpy_);
        double* ptr = legacy_.get();
        pybind11::capsule owner(ptr, [](void* value) { delete[] static_cast<double*>(value); });
        legacy_.release();
        return pybind11::array_t<double>({static_cast<pybind11::ssize_t>(count_)}, {sizeof(double)}, ptr, owner);
    }
};

inline pybind11::dict sssp_host_output_proof(bool requested, bool dense, bool worker, std::uint64_t count) {
    const bool eligible = dense && worker, used = requested && eligible;
    pybind11::dict proof;
    proof["version"] = "native_sssp_host_ownership_v1";
    proof["requested"] = requested ? "1" : "0";
    proof["eligible"] = eligible; proof["used"] = used;
    proof["reason"] = used ? "private_numpy_owner" : requested ? "unsupported_output_route" : "disabled";
    proof["producer"] = !dense ? "eager_dict" : used ? "numpy_owned_fp64_worker_scatter" : worker ? "capsule_fp64_worker_scatter" : "caller_vector_capsule";
    proof["complete_vertices"] = count;
    proof["values_bytes"] = dense ? count * sizeof(double) : 0;
    return proof;
}
}} // namespace easygraph::egmgpu
