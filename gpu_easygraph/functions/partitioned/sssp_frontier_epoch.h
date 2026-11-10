#pragma once

#include <climits>

namespace gpu_easygraph {
namespace partitioned {

// Query-local host state. The caller clears the existing membership array when
// advance() returns true, before any producer of that epoch can run. No epoch
// may overlap its successor; publication waits for every local/remote producer.
struct SsspFrontierEpoch {
    int value = 0;
    bool advance() {
        const bool clear = value == 0 || value == INT_MAX;
        value = clear ? 1 : value + 1;
        return clear;
    }
};

struct SsspNextFrontier {
    int* vertices;
    int* count;
    int epoch;
    SsspNextFrontier(int* vertices_ = nullptr, int* count_ = nullptr, int epoch_ = 0)
        : vertices(vertices_), count(count_), epoch(epoch_) {}
};

// The CPU invariant harness supplies atomicExch/atomicAdd with real host
// atomics. Production host translation units do not instantiate this helper.
#if defined(__CUDACC__) || defined(EGGPU_SSSP_EPOCH_HOST_TEST)
#if defined(__CUDACC__)
#define EGGPU_SSSP_QUEUE_INLINE __device__ __forceinline__
#else
#define EGGPU_SSSP_QUEUE_INLINE inline
#endif
template <bool DirectQueue>
EGGPU_SSSP_QUEUE_INLINE void mark_sssp_next_vertex(
    int vertex, int* flags, SsspNextFrontier next) {
    if (DirectQueue) {
        // Only the first successful relaxer per vertex/epoch owns a slot.
        // Capacity is owned_count: producers validate the owner-local ID and
        // the existing plan limits owned_count to INT_MAX. Later improvements
        // still update distances; the next round reads the completed values.
        if (atomicExch(flags + vertex, next.epoch) != next.epoch) {
            const int position = atomicAdd(next.count, 1);
            next.vertices[position] = vertex;
        }
    } else {
        atomicExch(flags + vertex, 1);
    }
}
#undef EGGPU_SSSP_QUEUE_INLINE
#endif

}  // namespace partitioned
}  // namespace gpu_easygraph
