#pragma once
#include "egmgpu_owner_output_team.h"
#include <algorithm>

namespace easygraph { namespace egmgpu { namespace host_halo {

// A cached immutable plan must fill each destination halo exactly once.
inline void validate(const std::vector<std::size_t>& offsets,
                     const std::vector<int>& source, const std::vector<int>& destination,
                     const std::vector<std::size_t>& source_counts,
                     std::size_t owned, std::size_t halo) {
    if (offsets.size() != source_counts.size()+1U || offsets.empty() || offsets.front() ||
        offsets.back() != source.size() || source.size() != destination.size() || source.size() != halo)
        throw std::invalid_argument("host halo plan shape mismatch");
    std::vector<unsigned char> seen(halo, 0U);
    for (std::size_t owner=0; owner<source_counts.size(); ++owner) {
        if (offsets[owner] > offsets[owner+1U] || offsets[owner+1U] > source.size())
            throw std::invalid_argument("host halo owner interval invalid");
        for (std::size_t i=offsets[owner]; i<offsets[owner+1U]; ++i) {
            if (source[i]<0 || static_cast<std::size_t>(source[i])>=source_counts[owner] ||
                destination[i]<0 || static_cast<std::size_t>(destination[i])<owned ||
                static_cast<std::size_t>(destination[i])-owned>=halo)
                throw std::invalid_argument("host halo source/destination outside actual values");
            if (seen[static_cast<std::size_t>(destination[i])-owned]++)
                throw std::invalid_argument("host halo destination written more than once");
        }
    }
}

// Call only with a validated plan and complete, immutable owned contributions.
// Owner-grouped int32 indices are the existing host upload plan, not a new copy.
inline void gather(const std::vector<std::size_t>& offsets,
                   const std::vector<int>& source, const std::vector<int>& destination,
                   const std::vector<const double*>& values, std::size_t owned,
                   double* halo, int width, owner_output::Observation& observation) {
    owner_output::run(source.size(), width, observation,
        [&](std::size_t begin, std::size_t end) noexcept {
            if (begin==end) return;
            std::size_t owner=static_cast<std::size_t>(
                std::upper_bound(offsets.begin(), offsets.end(), begin)-offsets.begin()-1);
            while (begin<end) {
                while (offsets[owner+1U]<=begin) ++owner;
                const auto stop=std::min(end, offsets[owner+1U]);
                const double* input=values[owner];
                for (; begin<stop; ++begin)
                    halo[static_cast<std::size_t>(destination[begin])-owned]=input[source[begin]];
            }
        });
}

} } }
