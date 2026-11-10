#include "egmgpu_triangle_count.h"
#ifdef EASYGRAPH_ENABLE_GPU
#include <runtime/triangle_count_plan.h>
#include <runtime/native_runtime.h>
#include <runtime/partitioned_device_state.h>
#include <runtime/device_phase_timer.h>
#include <functions/basic/triangle_count.cuh>
#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace {
namespace rt = gpu_easygraph::runtime;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void check(cudaError_t code, const char* op) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(code));
}
struct TcDeviceState {
    std::shared_ptr<rt::DeviceAllocation> row, col, source, count;
    int source_first = 0;
    std::uint64_t bytes = 0;
    double allocation_seconds = 0., h2d_seconds = 0.;
};
struct TcGroup {
    std::mutex lease;
    std::vector<std::shared_ptr<TcDeviceState> > states;
    bool ready = false;
};
struct TcRankMetrics {
    std::uint64_t count = 0;
    double reset_submit_seconds = 0., kernel_submit_seconds = 0.;
    double d2h_and_kernel_wait_seconds = 0., kernel_device_seconds = 0.;
};

class PreparedTriangleCount {
public:
    explicit PreparedTriangleCount(rt::TriangleHostPlan plan) : plan_(std::move(plan)) {}
    py::dict info() const {
        py::dict out;
        out["vertex_count"] = plan_.vertex_count;
        out["input_arcs"] = plan_.input_arcs;
        out["forward_edges"] = plan_.indices.size();
        out["parts"] = plan_.owners.size();
        out["owner_policy"] = plan_.owner_policy;
        out["orientation"] = "degree_then_original_id";
        out["kernel"] = "warp_owned_edge_binary_intersection_block_u64_reduce";
        out["triangle_upper_bound"] = plan_.triangle_upper_bound;
        out["validation_seconds"] = plan_.validation_seconds;
        out["orientation_seconds"] = plan_.orientation_seconds;
        out["owner_partition_seconds"] = plan_.owner_partition_seconds;
        out["host_plan_bytes"] = 4ULL * (plan_.offsets.size()+plan_.indices.size()+plan_.edge_sources.size());
        out["replicated_forward_csr_bytes_per_device"] = 4ULL * (plan_.offsets.size()+plan_.indices.size());
        out["owner_source_bytes_all_devices"] = 4ULL * plan_.edge_sources.size();
        out["output_scalar_bytes_per_rank"] = 8;
        py::list owners;
        for (const auto& owner : plan_.owners) {
            py::dict row;
            row["root_begin"] = owner.begin; row["root_end"] = owner.end;
            row["edge_begin"] = plan_.offsets[owner.begin];
            row["edge_end"] = plan_.offsets[owner.end];
            row["estimated_binary_search_work"] = owner.estimated_scan_work;
            owners.append(row);
        }
        out["owners"] = owners;
        return out;
    }

    py::dict upload(const std::vector<int>& devices, bool serial) {
        validate_devices(devices, serial);
        std::shared_ptr<TcGroup> group = acquire(devices, serial);
        bool hit = false;
        double wall = 0.;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lease(group->lease);
            hit = group->ready;
            const auto begin = Clock::now();
            if (!hit) {
                group->states.resize(devices.size());
                std::vector<rt::NativeRuntime::Task> tasks;
                for (std::size_t rank=0; rank<devices.size(); ++rank) {
                    tasks.push_back([this, group, rank, serial](int device) {
                        const int first = serial ? 0 : plan_.offsets[plan_.owners[rank].begin];
                        const int last = serial ? static_cast<int>(plan_.indices.size()) : plan_.offsets[plan_.owners[rank].end];
                        auto state = std::make_shared<TcDeviceState>();
                        state->source_first = first;
                        const auto allocation = Clock::now();
                        state->row = std::make_shared<rt::DeviceAllocation>(device, 4ULL*plan_.offsets.size());
                        if (!plan_.indices.empty()) state->col = std::make_shared<rt::DeviceAllocation>(device, 4ULL*plan_.indices.size());
                        if (last>first) state->source = std::make_shared<rt::DeviceAllocation>(device, 4ULL*(last-first));
                        state->count = std::make_shared<rt::DeviceAllocation>(device, 8);
                        state->allocation_seconds = seconds(allocation);
                        state->bytes = state->row->bytes()+(state->col ? state->col->bytes():0)+(state->source ? state->source->bytes():0)+8;
                        const auto copy = Clock::now();
                        check(cudaMemcpy(state->row->data(),plan_.offsets.data(),state->row->bytes(),cudaMemcpyHostToDevice),"TC forward row H2D");
                        if (state->col)
                            check(cudaMemcpy(state->col->data(),plan_.indices.data(),state->col->bytes(),cudaMemcpyHostToDevice),"TC forward column H2D");
                        if (state->source)
                            check(cudaMemcpy(state->source->data(),plan_.edge_sources.data()+first,state->source->bytes(),cudaMemcpyHostToDevice),"TC owned source H2D");
                        state->h2d_seconds = seconds(copy);
                        group->states[rank] = state;
                    });
                }
                rt::runtime_manager().execute_gang(devices,std::move(tasks));
                group->ready = true;
            }
            wall = seconds(begin);
        }
        py::dict out;
        out["cache_hit"] = hit; out["upload_gang_seconds"] = wall;
        py::list rows;
        for (std::size_t rank=0; rank<group->states.size(); ++rank) {
            const auto& state = *group->states[rank];
            py::dict row;
            row["logical_device"] = devices[rank]; row["resident_requested_bytes"] = state.bytes;
            row["allocation_seconds"] = hit ? 0. : state.allocation_seconds;
            row["h2d_wall_seconds"] = hit ? 0. : state.h2d_seconds;
            row["h2d_bytes"] = hit ? 0ULL : state.bytes-8;
            rows.append(row);
        }
        out["devices"] = rows;
        return out;
    }

    py::dict query(const std::vector<int>& devices, bool serial, bool profile) {
        validate_devices(devices, serial);
        auto group = acquire(devices, serial);
        std::vector<TcRankMetrics> metrics(plan_.owners.size());
        double gang_seconds = 0., sum_seconds = 0.;
        std::uint64_t total = 0;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lease(group->lease);
            if (!group->ready) throw std::runtime_error("TC prepare/upload is required before a resident query");
            auto run_rank = [&](std::size_t rank, const std::shared_ptr<TcDeviceState>& state) {
                TcRankMetrics& m = metrics[rank];
                const auto& owner = plan_.owners[rank];
                const int first = plan_.offsets[owner.begin], last = plan_.offsets[owner.end];
                auto* count = static_cast<unsigned long long*>(state->count->data());
                auto begin = Clock::now();
                check(cudaMemset(count,0,8),"TC scalar reset");
                m.reset_submit_seconds = seconds(begin);
                rt::DevicePhaseTimer kernel(profile);
                kernel.begin(nullptr);
                begin = Clock::now();
                const int* source = state->source ? static_cast<const int*>(state->source->data()) : nullptr;
                if (last > first) source += first-state->source_first;
                gpu_easygraph::launch_triangle_count(static_cast<const int*>(state->row->data()),
                    state->col ? static_cast<const int*>(state->col->data()) : nullptr,source,first,last-first,count,nullptr);
                check(cudaGetLastError(),"TC kernel launch");
                m.kernel_submit_seconds = seconds(begin);
                kernel.end(nullptr);
                begin = Clock::now();
                check(cudaMemcpy(&m.count,count,8,cudaMemcpyDeviceToHost),"TC scalar D2H");
                m.d2h_and_kernel_wait_seconds = seconds(begin);
                m.kernel_device_seconds = kernel.seconds();
            };
            std::vector<rt::NativeRuntime::Task> tasks;
            if (serial) {
                tasks.push_back([&](int) { for (std::size_t rank=0;rank<plan_.owners.size();++rank) run_rank(rank,group->states[0]); });
            } else {
                for (std::size_t rank=0;rank<plan_.owners.size();++rank)
                    tasks.push_back([&,rank](int) { run_rank(rank,group->states[rank]); });
            }
            auto begin = Clock::now();
            rt::runtime_manager().execute_gang(devices,std::move(tasks));
            gang_seconds = seconds(begin);
            begin = Clock::now();
            for (const auto& m : metrics) {
                if (m.count > UINT64_MAX-total) throw std::overflow_error("TC host scalar sum exceeds uint64");
                total += m.count;
            }
            sum_seconds = seconds(begin);
        }
        py::dict out;
        out["count"] = py::int_(total);
        out["gang_seconds"] = gang_seconds;
        out["host_scalar_sum_seconds"] = sum_seconds;
        out["profile_enabled"] = profile;
        out["execution"] = serial ? "same_owner_serial" : "parallel";
        out["per_rank"] = py::list();
        py::list rows;
        for (std::size_t rank=0;rank<metrics.size();++rank) {
            const auto& m = metrics[rank];
            py::dict row;
            row["rank"] = rank; row["count"] = py::int_(m.count);
            row["reset_submit_seconds"] = m.reset_submit_seconds;
            row["kernel_submit_seconds"] = m.kernel_submit_seconds;
            row["d2h_and_kernel_wait_seconds"] = m.d2h_and_kernel_wait_seconds;
            row["kernel_device_seconds"] = profile ? py::cast(m.kernel_device_seconds) : py::none();
            row["d2h_bytes"] = 8;
            rows.append(row);
        }
        out["per_rank"] = rows;
        return out;
    }
private:
    rt::TriangleHostPlan plan_;
    void validate_devices(const std::vector<int>& devices,bool serial) const {
        if (devices.size() != (serial ? 1 : plan_.owners.size()))
            throw std::invalid_argument("TC requires one fixed worker per owner, or one for same-owner serial");
        auto sorted=devices; std::sort(sorted.begin(),sorted.end());
        if (std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())
            throw std::invalid_argument("TC physical workers must be distinct");
        if (sorted!=devices) throw std::invalid_argument("TC logical device IDs must follow the native runtime's sorted worker order");
    }
    std::shared_ptr<TcGroup> acquire(const std::vector<int>& devices,bool serial) const {
        std::ostringstream key;
        key << "triangle_count_v1:" << plan_.identity << ':' << serial;
        for (int d:devices) key << ':' << d;
        return std::static_pointer_cast<TcGroup>(rt::runtime_manager().acquire_resident_resource(
            key.str(), [] { return std::static_pointer_cast<void>(std::make_shared<TcGroup>()); }));
    }
};
}
#endif

void bind_egmgpu_triangle_count(py::module_& module) {
#ifdef EASYGRAPH_ENABLE_GPU
    py::class_<PreparedTriangleCount,std::shared_ptr<PreparedTriangleCount> >(module,"TriangleCountPlan")
        .def("info",&PreparedTriangleCount::info)
        .def("upload",&PreparedTriangleCount::upload,py::arg("devices"),py::arg("serial")=false)
        .def("query",&PreparedTriangleCount::query,py::arg("devices"),py::arg("serial")=false,py::arg("profile")=false);
    module.def("cpp_egmgpu_prepare_triangle_count",
        [](py::array_t<int,py::array::c_style> row,py::array_t<int,py::array::c_style> col,int parts,const std::string& policy) {
            if (row.ndim()!=1 || col.ndim()!=1) throw std::invalid_argument("TC CSR arrays must be one-dimensional int32");
            py::gil_scoped_release release;
            return std::make_shared<PreparedTriangleCount>(rt::make_triangle_plan(
                row.data(),row.size(),col.data(),col.size(),parts,policy));
        },py::arg("offsets").noconvert(),py::arg("indices").noconvert(),py::arg("parts"),py::arg("owner_policy")="edge_work");
#endif
}
