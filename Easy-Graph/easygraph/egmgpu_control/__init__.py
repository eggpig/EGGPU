"""Reference control plane for EGMGPU.

This package deliberately contains no CUDA implementation.  It defines the
state, capability, and planning contracts that native workers can implement
without making unavailable multi-GPU paths appear executable.
"""

from .descriptors import CapabilityLevel
from .calibration import CALIBRATION_SCHEMA_VERSION
from .calibration import CalibrationCoefficients
from .calibration import CalibrationError
from .calibration import CalibrationKey
from .calibration import CalibrationObservation
from .calibration import CalibrationPolicy
from .calibration import CalibrationProfile
from .calibration import EvidenceFile
from .calibration import OracleDecision
from .calibration import SelectionMetrics
from .calibration import build_calibration_profile
from .calibration import evaluate_enumerated_oracle
from .descriptors import CapabilityRegistry
from .descriptors import DecompositionAxis
from .descriptors import ExecutionMode
from .descriptors import FunctionDescriptor
from .descriptors import FunctionFamily
from .descriptors import default_capability_registry
from .planner import AdaptivePlanner
from .planner import AnalyticalCostModel
from .planner import CapabilityUnavailableError
from .planner import CostBreakdown
from .planner import CostModelConfig
from .planner import DeviceProfile
from .planner import ExecutionPlan
from .planner import NoFeasiblePlanError
from .planner import PlanOverride
from .planner import TopologyProfile
from .planner import WorkloadEstimate
from .partition_manifest import ArrayFile
from .partition_manifest import MANIFEST_FORMAT
from .partition_manifest import MANIFEST_VERSION
from .partition_manifest import ManifestContentError
from .partition_manifest import ManifestError
from .partition_manifest import ManifestFileError
from .partition_manifest import ManifestPartitionMetadata
from .partition_manifest import ManifestPathError
from .partition_manifest import ManifestValidationReport
from .partition_manifest import PartitionedCSRManifest
from .partition_manifest import build_tiny_partition_fixture
from .partition_manifest import load_partition_manifest
from .partition_manifest import write_array_stream
from .state import CapacityError
from .state import GraphSnapshot
from .state import MultiDeviceGraphState
from .state import PartitionMetadata
from .state import ResidencyRecord
from .state import StateKey
from .state import StaleSnapshotError
from .state import ViewKey
from .state import validate_partition_set
from .streaming_partition_builder import PartitionBuildError
from .streaming_partition_builder import PartitionBuildReport
from .streaming_partition_builder import build_partitioned_csr_from_edge_list
from .streaming_partition_builder import build_partitioned_csr_from_host_csr
from .streaming_partition_builder import build_partitioned_csr_from_int32_csr

__all__ = [
    "AdaptivePlanner",
    "AnalyticalCostModel",
    "ArrayFile",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationCoefficients",
    "CalibrationError",
    "CalibrationKey",
    "CalibrationObservation",
    "CalibrationPolicy",
    "CalibrationProfile",
    "CapabilityLevel",
    "CapabilityRegistry",
    "CapabilityUnavailableError",
    "CapacityError",
    "CostBreakdown",
    "CostModelConfig",
    "DecompositionAxis",
    "DeviceProfile",
    "ExecutionMode",
    "ExecutionPlan",
    "EvidenceFile",
    "FunctionDescriptor",
    "FunctionFamily",
    "GraphSnapshot",
    "MANIFEST_FORMAT",
    "MANIFEST_VERSION",
    "ManifestContentError",
    "ManifestError",
    "ManifestFileError",
    "ManifestPartitionMetadata",
    "ManifestPathError",
    "ManifestValidationReport",
    "MultiDeviceGraphState",
    "NoFeasiblePlanError",
    "OracleDecision",
    "PartitionMetadata",
    "PartitionBuildError",
    "PartitionBuildReport",
    "PartitionedCSRManifest",
    "PlanOverride",
    "ResidencyRecord",
    "SelectionMetrics",
    "StateKey",
    "StaleSnapshotError",
    "TopologyProfile",
    "ViewKey",
    "WorkloadEstimate",
    "build_tiny_partition_fixture",
    "build_partitioned_csr_from_edge_list",
    "build_partitioned_csr_from_host_csr",
    "build_partitioned_csr_from_int32_csr",
    "build_calibration_profile",
    "default_capability_registry",
    "load_partition_manifest",
    "evaluate_enumerated_oracle",
    "write_array_stream",
    "validate_partition_set",
]
