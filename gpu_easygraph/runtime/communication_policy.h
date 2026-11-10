#pragma once

#include "runtime_types.h"

#include <cstddef>
#include <cstdint>

namespace gpu_easygraph {
namespace runtime {

// Logical communication shapes exposed by function-family execution
// descriptors.  Kernels describe semantics; this policy chooses transport.
enum class MessagePattern {
    SparseOwnerDirected,
    DenseHalo,
    GlobalScalar,
    OutputGather,
};

enum class CombineOperation {
    None,
    Sum,
    Minimum,
    LabelMinimum,
};

enum class CommunicationPrimitive {
    LocalOnly,
    OwnerDirectedExchange,
    PeerBroadcast,
    CollectiveAllGather,
    CollectiveAllReduce,
    CoordinatorGather,
};

struct CommunicationCalibration {
    // These values are calibration outputs, not claims about any GPU model.
    // A production planner must replace defaults with measured values from
    // the current server before promoting the adaptive path to AVAILABLE.
    double sparse_density_cutover;
    std::uint64_t collective_byte_cutover;
    std::uint64_t direct_copy_byte_cutover;

    CommunicationCalibration();
};

struct CommunicationRequest {
    MessagePattern pattern;
    CombineOperation combine;
    std::uint64_t active_items;
    std::uint64_t total_items;
    std::uint64_t payload_bytes;
    std::size_t participant_count;
    bool nccl_available;
    bool fully_peer_connected;

    double density() const;
};

struct CommunicationDecision {
    CommunicationBackend backend;
    CommunicationPrimitive primitive;
    bool deduplicate;
    bool local_combine;
    bool overlap_eligible;
};

CommunicationDecision choose_communication(
    const CommunicationRequest& request,
    const CommunicationCalibration& calibration = CommunicationCalibration());

const char* communication_primitive_name(CommunicationPrimitive primitive);

}  // namespace runtime
}  // namespace gpu_easygraph
