#include "communication_policy.h"

#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {

CommunicationCalibration::CommunicationCalibration()
    : sparse_density_cutover(0.08),
      collective_byte_cutover(UINT64_C(65536)),
      direct_copy_byte_cutover(UINT64_C(4096)) {}

double CommunicationRequest::density() const {
    if (total_items == 0) {
        return 0.0;
    }
    return static_cast<double>(active_items) /
           static_cast<double>(total_items);
}

namespace {

void validate(
    const CommunicationRequest& request,
    const CommunicationCalibration& calibration) {
    if (request.participant_count == 0) {
        throw std::invalid_argument(
            "communication requires at least one participant");
    }
    if (request.active_items > request.total_items) {
        throw std::invalid_argument(
            "active communication items cannot exceed total items");
    }
    if (!(calibration.sparse_density_cutover > 0.0 &&
          calibration.sparse_density_cutover < 1.0)) {
        throw std::invalid_argument(
            "sparse density cutover must be strictly between zero and one");
    }
}

CommunicationBackend fallback_transport(const CommunicationRequest& request) {
    return request.fully_peer_connected
               ? CommunicationBackend::PeerToPeer
               : CommunicationBackend::HostStaging;
}

}  // namespace

CommunicationDecision choose_communication(
    const CommunicationRequest& request,
    const CommunicationCalibration& calibration) {
    validate(request, calibration);
    CommunicationDecision decision;
    decision.deduplicate = false;
    decision.local_combine = false;
    decision.overlap_eligible = false;

    if (request.participant_count == 1 || request.payload_bytes == 0) {
        decision.backend = CommunicationBackend::None;
        decision.primitive = CommunicationPrimitive::LocalOnly;
        return decision;
    }

    if (request.pattern == MessagePattern::SparseOwnerDirected &&
        request.density() < calibration.sparse_density_cutover) {
        decision.backend = fallback_transport(request);
        decision.primitive = CommunicationPrimitive::OwnerDirectedExchange;
        decision.deduplicate = true;
        decision.local_combine = request.combine != CombineOperation::None;
        decision.overlap_eligible = true;
        return decision;
    }

    if (request.pattern == MessagePattern::GlobalScalar) {
        decision.backend = request.nccl_available
                               ? CommunicationBackend::Nccl
                               : fallback_transport(request);
        decision.primitive = CommunicationPrimitive::CollectiveAllReduce;
        decision.local_combine = request.combine != CombineOperation::None;
        return decision;
    }

    if (request.pattern == MessagePattern::OutputGather) {
        // NCCL has point-to-point send/receive, but no native gather primitive.
        // The coordinator path therefore chooses direct peer copies when the
        // complete directed P2P matrix is valid and pinned host staging otherwise.
        decision.backend = fallback_transport(request);
        decision.primitive = CommunicationPrimitive::CoordinatorGather;
        decision.overlap_eligible = true;
        return decision;
    }

    if (request.pattern == MessagePattern::DenseHalo ||
        request.density() >= calibration.sparse_density_cutover) {
        if (request.nccl_available &&
            request.payload_bytes >= calibration.collective_byte_cutover) {
            decision.backend = CommunicationBackend::Nccl;
            decision.primitive = CommunicationPrimitive::CollectiveAllGather;
        } else {
            decision.backend = fallback_transport(request);
            decision.primitive = CommunicationPrimitive::PeerBroadcast;
        }
        decision.overlap_eligible = true;
        return decision;
    }

    throw std::logic_error("unhandled EGMGPU communication request");
}

const char* communication_primitive_name(CommunicationPrimitive primitive) {
    switch (primitive) {
        case CommunicationPrimitive::LocalOnly:
            return "local_only";
        case CommunicationPrimitive::OwnerDirectedExchange:
            return "owner_directed_exchange";
        case CommunicationPrimitive::PeerBroadcast:
            return "peer_broadcast";
        case CommunicationPrimitive::CollectiveAllGather:
            return "collective_all_gather";
        case CommunicationPrimitive::CollectiveAllReduce:
            return "collective_all_reduce";
        case CommunicationPrimitive::CoordinatorGather:
            return "coordinator_gather";
    }
    return "unknown";
}

}  // namespace runtime
}  // namespace gpu_easygraph
