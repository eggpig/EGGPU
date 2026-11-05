#pragma once
#include <cstddef>
#include <cstdint>
// Called only after the existing setup-time full OID permutation validation.
// Conversion belongs to each query's dense_scatter_seconds interval.
inline void gum_fp32_dense_output(double* output, const float* values,
    const int64_t* ids, std::size_t n, bool identity) {
  if (identity) for (std::size_t i=0;i<n;++i) output[i]=static_cast<double>(values[i]);
  else for (std::size_t i=0;i<n;++i) output[ids[i]]=static_cast<double>(values[i]);
}
