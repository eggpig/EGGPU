#pragma once

#include <cstdint>
#include <cstddef>

namespace gpu_easygraph {
namespace runtime {

// Weight and wire encodings are independent immutable identities; state stays FP64.
enum class SsspWeightEncoding { F64, U8Exact };
inline const char* sssp_weight_encoding_name(SsspWeightEncoding encoding) {
    return encoding == SsspWeightEncoding::U8Exact ? "u8_exact" : "f64";
}

// Fixed wire format for one owner-directed distance candidate.
struct SsspRelaxation {
    std::int32_t owner_local_id;
    std::int32_t reserved;
    double distance;
};

static_assert(sizeof(SsspRelaxation) == 16,
              "SSSP relaxation wire format must remain 16 bytes");

enum class SsspWireEncoding { F64, U32Exact };
struct SsspRelaxationU32 {
    std::int32_t owner_local_id;
    std::uint32_t distance;
};
static_assert(sizeof(SsspRelaxationU32)==8,"compact SSSP wire must be 8 bytes");
inline const char* sssp_wire_encoding_name(SsspWireEncoding value) {
    return value==SsspWireEncoding::U32Exact ? "u32_exact" : "f64";
}
inline std::size_t sssp_wire_bytes(SsspWireEncoding value) {
    return value==SsspWireEncoding::U32Exact ? sizeof(SsspRelaxationU32) : sizeof(SsspRelaxation);
}

}  // namespace runtime
}  // namespace gpu_easygraph
