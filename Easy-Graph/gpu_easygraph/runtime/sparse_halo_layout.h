#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gpu_easygraph { namespace runtime {
struct SparseHaloRequestView {
    const std::vector<std::size_t>* offsets;
    const std::vector<int>* sources;
    const std::vector<int>* destinations;
};
struct SparseHaloRun {
    std::size_t peer;
    std::size_t send_offset;
    std::size_t destination;
    std::size_t count;
};
struct SparseHaloRank {
    std::size_t owned = 0, local = 0, receive_values = 0;
    std::vector<int> source_indices, packed_indices;
    std::vector<SparseHaloRun> sends, receives;
};
struct SparseHaloLayout {
    std::vector<SparseHaloRank> ranks;
    std::size_t halo_values = 0, transfer_runs = 0;
    std::size_t host_payload_bytes = 0, validation_peak_bytes = 0;
};
inline std::size_t sparse_checked_add(std::size_t a,std::size_t b) {
    if (b>std::numeric_limits<std::size_t>::max()-a) throw std::overflow_error("sparse halo count overflow");
    return a+b;
}
inline std::size_t sparse_checked_bytes(std::size_t count,std::size_t width) {
    if (width && count>std::numeric_limits<std::size_t>::max()/width) throw std::overflow_error("sparse halo byte overflow");
    return count*width;
}
// This immutable request transpose is built once PER CALL. The single host
// process already owns every rank's request lists; no network metadata exchange
// is needed. No persistent cache or GPU allocation is created by this function.
inline SparseHaloLayout sparse_halo_layout(
    const std::vector<std::size_t>& owned,const std::vector<std::size_t>& local,
    const std::vector<SparseHaloRequestView>& requests) {
    const std::size_t p=owned.size(),limit=static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (!p || local.size()!=p || requests.size()!=p) throw std::invalid_argument("sparse halo rank shape");
    SparseHaloLayout out;out.ranks.resize(p);std::vector<std::size_t> totals(p,0);
    for (std::size_t d=0;d<p;++d) {
        const auto& r=requests[d];
        if (owned[d]>local[d] || local[d]>limit || !r.offsets || !r.sources || !r.destinations)
            throw std::invalid_argument("sparse halo local shape or int32 index overflow");
        const auto& off=*r.offsets;
        if (off.size()!=p+1 || off.front()!=0 || off.back()!=r.sources->size() || r.sources->size()!=r.destinations->size() || r.sources->size()!=local[d]-owned[d])
            throw std::invalid_argument("sparse halo request shape or incomplete halo");
        out.ranks[d].owned=owned[d];out.ranks[d].local=local[d];out.ranks[d].receive_values=r.sources->size();
        out.halo_values=sparse_checked_add(out.halo_values,r.sources->size());
        for (std::size_t owner=0;owner<p;++owner) {
            if (off[owner]>off[owner+1] || off[owner+1]>off.back() || (owner==d && off[owner]!=off[owner+1]))
                throw std::invalid_argument("sparse halo owner interval or self halo");
            totals[owner]=sparse_checked_add(totals[owner],off[owner+1]-off[owner]);
            if (totals[owner]>limit) throw std::overflow_error("sparse halo packed int32 index overflow");
        }
    }
    sparse_checked_bytes(out.halo_values,16U);
    for (std::size_t owner=0;owner<p;++owner) {
        out.ranks[owner].source_indices.reserve(totals[owner]);
        out.ranks[owner].packed_indices.resize(totals[owner]);
        for (std::size_t i=0;i<totals[owner];++i) out.ranks[owner].packed_indices[i]=static_cast<int>(i);
    }
    for (std::size_t d=0;d<p;++d) {
        const auto& request=requests[d];const auto& off=*request.offsets;
        std::vector<unsigned char> seen(local[d]-owned[d],0);
        out.validation_peak_bytes=std::max(out.validation_peak_bytes,seen.capacity());
        for (std::size_t owner=0;owner<p;++owner) {
            const std::size_t first=off[owner],end=off[owner+1];
            auto& sender=out.ranks[owner];
            const std::size_t base=sender.source_indices.size();
            for (std::size_t i=first;i<end;++i) {
                const int source=(*request.sources)[i],destination=(*request.destinations)[i];
                if (source<0 || static_cast<std::size_t>(source)>=owned[owner] || destination<0 || static_cast<std::size_t>(destination)<owned[d] || static_cast<std::size_t>(destination)>=local[d])
                    throw std::invalid_argument("sparse halo source or destination index");
                auto& flag=seen[static_cast<std::size_t>(destination)-owned[d]];
                if (flag) throw std::invalid_argument("duplicate sparse halo destination");
                flag=1;sender.source_indices.push_back(source);
            }
            // A direct receive must be contiguous. Split arbitrary halo layouts
            // into runs; no renumbering, scatter, or silent fallback is required.
            for (std::size_t i=first;i<end;) {
                std::size_t stop=i+1;
                while (stop<end && (*request.destinations)[stop]==(*request.destinations)[stop-1]+1) ++stop;
                SparseHaloRun run{d,base+i-first,static_cast<std::size_t>((*request.destinations)[i]),stop-i};
                sender.sends.push_back(run);run.peer=owner;out.ranks[d].receives.push_back(run);
                out.transfer_runs=sparse_checked_add(out.transfer_runs,1);i=stop;
            }
        }
    }
    out.host_payload_bytes=sparse_checked_bytes(out.ranks.capacity(),sizeof(SparseHaloRank));
    for (const auto& r:out.ranks) {
        out.host_payload_bytes=sparse_checked_add(out.host_payload_bytes,sparse_checked_bytes(sparse_checked_add(r.source_indices.capacity(),r.packed_indices.capacity()),sizeof(int)));
        out.host_payload_bytes=sparse_checked_add(out.host_payload_bytes,sparse_checked_bytes(sparse_checked_add(r.sends.capacity(),r.receives.capacity()),sizeof(SparseHaloRun)));
    }
    return out;
}
}} // namespace
