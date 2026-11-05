#pragma once

#include "buffer_cache.h"

namespace gpu_easygraph {

// A CSR upload must own its host-registration lifetime. A range registered by
// another worker may be unregistered while our DMA is in flight. Copy such
// ranges into this worker's pinned staging buffer rather than borrowing them.
template <typename T>
inline int prepare_csr_h2d_source(
    const T* src, std::size_t count, PersistentPinnedBuffer& staging,
    const T** prepared, bool* owns_registration) {
    if (prepared == nullptr || owns_registration == nullptr) return EG_GPU_DEVICE_ERR;
    *prepared = nullptr;
    *owns_registration = false;
    if (count == 0) return EG_GPU_SUCC;
    if (src == nullptr || count > static_cast<std::size_t>(-1) / sizeof(T)) return EG_GPU_DEVICE_ERR;
    const std::size_t bytes = sizeof(T) * count;
    if (gpu_host_register_enabled() && bytes >= gpu_host_register_min_bytes()) {
        const cudaError_t status = cudaHostRegister(
            const_cast<T*>(src), bytes, cudaHostRegisterPortable);
        if (status == cudaSuccess) {
            *prepared = src;
            *owns_registration = true;
            return EG_GPU_SUCC;
        }
        // These are optional registration failures for which owned staging is
        // a valid fallback. Never clear a fatal execution/context error and
        // continue with a copy from the same damaged CUDA context.
        if (status != cudaErrorHostMemoryAlreadyRegistered &&
            status != cudaErrorMemoryAllocation &&
            status != cudaErrorInvalidValue &&
            status != cudaErrorNotSupported) return EG_GPU_DEVICE_ERR;
        const cudaError_t cleared = cudaGetLastError();
        if (cleared != cudaSuccess && cleared != status) return EG_GPU_DEVICE_ERR;
    }
    const int status = staging.ensure_bytes(bytes);
    if (status != EG_GPU_SUCC) return status;
    T* private_source = staging.as<T>();
    std::memcpy(private_source, src, bytes);
    *prepared = private_source;
    return EG_GPU_SUCC;
}

inline int release_csr_h2d_source(const void* ptr, bool owns_registration) {
    if (!owns_registration || ptr == nullptr) return EG_GPU_SUCC;
    const cudaError_t status = cudaHostUnregister(const_cast<void*>(ptr));
    return status == cudaSuccess ? EG_GPU_SUCC : EG_GPU_DEVICE_ERR;
}

} // namespace gpu_easygraph
