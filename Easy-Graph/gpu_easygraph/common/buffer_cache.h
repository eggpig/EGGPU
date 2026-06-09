#pragma once

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "err.h"

namespace gpu_easygraph {

class PersistentDeviceBuffer {
public:
    PersistentDeviceBuffer() = default;
    PersistentDeviceBuffer(const PersistentDeviceBuffer&) = delete;
    PersistentDeviceBuffer& operator=(const PersistentDeviceBuffer&) = delete;
    ~PersistentDeviceBuffer() { reset(); }

    int ensure_bytes(std::size_t bytes) {
        if (bytes <= capacity_bytes_) return EG_GPU_SUCC;
        void* new_ptr = nullptr;
        cudaError_t ret = cudaMalloc(&new_ptr, bytes);
        if (ret != cudaSuccess) {
            if (ret == cudaErrorMemoryAllocation) return EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM;
            return EG_GPU_DEVICE_ERR;
        }
        if (ptr_ != nullptr) cudaFree(ptr_);
        ptr_ = new_ptr;
        capacity_bytes_ = bytes;
        return EG_GPU_SUCC;
    }

    template <typename T>
    T* as() const {
        return reinterpret_cast<T*>(ptr_);
    }

    void* data() const { return ptr_; }

    std::size_t capacity_bytes() const { return capacity_bytes_; }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        capacity_bytes_ = 0;
    }

private:
    void* ptr_ = nullptr;
    std::size_t capacity_bytes_ = 0;
};

class PersistentPinnedBuffer {
public:
    PersistentPinnedBuffer() = default;
    PersistentPinnedBuffer(const PersistentPinnedBuffer&) = delete;
    PersistentPinnedBuffer& operator=(const PersistentPinnedBuffer&) = delete;
    ~PersistentPinnedBuffer() { reset(); }

    int ensure_bytes(std::size_t bytes) {
        if (bytes <= capacity_bytes_) return EG_GPU_SUCC;
        void* new_ptr = nullptr;
        cudaError_t ret = cudaMallocHost(&new_ptr, bytes);
        if (ret != cudaSuccess) {
            if (ret == cudaErrorMemoryAllocation) return EG_GPU_FAILED_TO_ALLOCATE_HOST_MEM;
            return EG_GPU_DEVICE_ERR;
        }
        if (ptr_ != nullptr) cudaFreeHost(ptr_);
        ptr_ = new_ptr;
        capacity_bytes_ = bytes;
        return EG_GPU_SUCC;
    }

    template <typename T>
    T* as() const {
        return reinterpret_cast<T*>(ptr_);
    }

    void* data() const { return ptr_; }

    std::size_t capacity_bytes() const { return capacity_bytes_; }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
        capacity_bytes_ = 0;
    }

private:
    void* ptr_ = nullptr;
    std::size_t capacity_bytes_ = 0;
};

template <typename T>
inline T* stage_to_pinned_or_null(
    const T* src,
    std::size_t count,
    PersistentPinnedBuffer& buf
) {
    if (src == nullptr || count == 0) return nullptr;
    int rc = buf.ensure_bytes(sizeof(T) * count);
    if (rc != EG_GPU_SUCC) return nullptr;
    T* dst = buf.as<T>();
    std::memcpy(dst, src, sizeof(T) * count);
    return dst;
}

inline bool gpu_host_register_enabled() {
    static bool initialized = false;
    static bool enabled = true;
    if (!initialized) {
        initialized = true;
        const char* v = std::getenv("EASYGRAPH_GPU_HOST_REGISTER");
        if (v != nullptr) {
            std::string s(v);
            for (char& c : s) c = (char)std::toupper((unsigned char)c);
            enabled = (s == "1" || s == "TRUE" || s == "ON" || s == "YES");
        }
    }
    return enabled;
}

inline std::size_t gpu_host_register_min_bytes() {
    static bool initialized = false;
    static std::size_t bytes = (std::size_t)1 << 20; // 1 MiB
    if (!initialized) {
        initialized = true;
        const char* v = std::getenv("EASYGRAPH_GPU_HOST_REGISTER_MIN_BYTES");
        if (v != nullptr) {
            unsigned long long parsed = std::strtoull(v, nullptr, 10);
            if (parsed > 0ULL) bytes = (std::size_t)parsed;
        }
    }
    return bytes;
}

template <typename T>
inline const T* prepare_h2d_source(
    const T* src,
    std::size_t count,
    PersistentPinnedBuffer& staging,
    bool* used_host_register
) {
    if (used_host_register != nullptr) *used_host_register = false;
    if (src == nullptr || count == 0) return nullptr;
    const std::size_t bytes = sizeof(T) * count;

    if (gpu_host_register_enabled() && bytes >= gpu_host_register_min_bytes()) {
        cudaError_t reg = cudaHostRegister((void*)src, bytes, cudaHostRegisterPortable);
        if (reg == cudaSuccess) {
            if (used_host_register != nullptr) *used_host_register = true;
            return src;
        }
        if (reg == cudaErrorHostMemoryAlreadyRegistered) {
            // We do not own this registration, so do not unregister later.
            (void)cudaGetLastError();
            return src;
        }
        (void)cudaGetLastError();
    }

    int rc = staging.ensure_bytes(bytes);
    if (rc == EG_GPU_SUCC) {
        T* dst = staging.as<T>();
        std::memcpy(dst, src, bytes);
        return dst;
    }
    return src;
}

template <typename T>
inline void release_h2d_source(const T* ptr, bool used_host_register) {
    if (!used_host_register || ptr == nullptr) return;
    (void)cudaHostUnregister((void*)ptr);
}

} // namespace gpu_easygraph
