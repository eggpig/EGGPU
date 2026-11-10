#pragma once
#include "egmgpu_owner_output_team.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace easygraph { namespace egmgpu {
struct WccPublicObservation {
    std::atomic<std::uint64_t> converted{0}, invalid{0}, certified{0}, noncanonical{0};
    void conversion(std::uint64_t count, std::uint64_t bad) {
        converted.fetch_add(count,std::memory_order_relaxed);
        invalid.fetch_add(bad,std::memory_order_relaxed);
    }
};
inline bool write_wcc_public_label(std::uint64_t global, std::uint64_t label,
                                   double* output, std::uint64_t count) {
    if(global>=count || label>=count)return false;
    output[global]=static_cast<double>(label);
    return true;
}
inline void assemble_wcc_public_owned(const std::uint64_t* ids,std::size_t size,
    const std::uint64_t* input,double* output,std::uint64_t count,WccPublicObservation& observation) {
    std::uint64_t bad=0;
    for(std::size_t local=0;local<size;++local)
        bad+=!write_wcc_public_label(ids[local],input[local],output,count);
    observation.conversion(size,bad);
}
inline void assemble_wcc_public_tiled(const std::uint64_t* source_for_global,
    const std::uint64_t* const* sources,double* output,std::size_t begin,std::size_t end,
    std::uint64_t count,WccPublicObservation& observation) {
    std::uint64_t bad=0;
    for(std::size_t global=begin;global<end;++global){
        const auto source=source_for_global[global];
        bad+=!write_wcc_public_label(global,sources[source>>32][static_cast<std::uint32_t>(source)],output,count);
    }
    observation.conversion(end-begin,bad);
}

// The calling/GIL thread owns this request-private array throughout execute_gang.
// Workers borrow only data(). Certification starts only after every producer
// has joined, so its indirect label reads never race an output write.
class WccPublicOutput {
    pybind11::array_t<double> values_;
    std::uint64_t count_;
    bool certified_=false,published_=false;
public:
    explicit WccPublicOutput(std::uint64_t count):count_(count){
        if(count>(std::uint64_t(1)<<53) || count>static_cast<std::uint64_t>(std::numeric_limits<pybind11::ssize_t>::max())/sizeof(double))
            throw std::overflow_error("WCC worker FP64 output requires an exact dense label domain and addressable host storage");
        values_=pybind11::array_t<double>(static_cast<pybind11::ssize_t>(count));
    }
    double* data(){
        if(certified_ || published_)throw std::logic_error("WCC public output is no longer writable");
        return values_.mutable_data();
    }
    // Caller obtains the raw pointer before releasing the GIL. No Python API
    // is called in the parallel certificate body or while the GIL is released.
    void certify(const double* values,WccPublicObservation& counts,int width,
                 owner_output::Observation& threads,const std::vector<int>* cpus=nullptr){
        if(certified_ || published_)throw std::logic_error("WCC public output already certified or published");
        if(counts.converted.load()!=count_ || counts.invalid.load())
            throw std::runtime_error("WCC worker output has an incomplete or invalid exact label domain");
        owner_output::run(count_,width,threads,[&](std::size_t begin,std::size_t end){
            std::uint64_t bad=0;
            for(std::size_t i=begin;i<end;++i){
                // Conversion checked every uint64 label before exact FP64
                // conversion; complete immutable-plan ownership was preserved.
                const auto label=static_cast<std::uint64_t>(values[i]);
                if(label>i || values[label]!=values[i])++bad;
            }
            counts.certified.fetch_add(end-begin,std::memory_order_relaxed);
            counts.noncanonical.fetch_add(bad,std::memory_order_relaxed);
        },cpus);
        if(counts.certified.load()!=count_ || counts.noncanonical.load())
            throw std::runtime_error("WCC worker output failed the complete canonical label certificate");
        certified_=true;
    }
    pybind11::array_t<double> publish(){
        if(!certified_ || published_)throw std::logic_error("WCC public output requires one completed certificate before publication");
        published_=true;return std::move(values_);
    }
};
inline pybind11::dict wcc_public_output_proof(std::uint64_t n,const WccPublicObservation& counts,int threads){
    if(counts.converted.load()!=n || counts.certified.load()!=n || counts.invalid.load() || counts.noncanonical.load())
        throw std::runtime_error("WCC public output proof has incomplete checks");
    pybind11::dict out;
    out["version"]="native_wcc_worker_public_fp64_v1";out["requested"]="1";
    out["eligible"]=true;out["used"]=true;out["reason"]="complete_private_worker_scatter_and_certificate";
    out["producer"]="numpy_owned_fp64_worker_scatter";out["complete_vertices"]=n;
    out["converted_vertices"]=counts.converted.load();out["certified_vertices"]=counts.certified.load();
    out["invalid_vertices"]=counts.invalid.load();out["noncanonical_vertices"]=counts.noncanonical.load();
    out["certificate_threads"]=threads;out["certificate_preserved"]=true;out["values_bytes"]=n*sizeof(double);
    out["intermediate_uint64_bytes_avoided"]=n*sizeof(std::uint64_t);
    out["intermediate_payload_write_bytes_avoided"]=n*sizeof(std::uint64_t);
    return out;
}
}} // namespace easygraph::egmgpu
