"""Single-machine multi-GPU execution for EasyGraph.

EGMGPU adds explicit configuration, asynchronous submission, and workflow
execution through this namespace.  After explicit configuration and native
binding registration, the 16 existing ``easygraph`` analysis entry points
also redirect immutable native graphs and generation-tracked EasyGraph graphs
synchronously through the same runtime.  Mutable graphs are copied into a
private submission-time native snapshot.  Importing EasyGraph alone never
starts that runtime.
"""

from .api import call
from .api import configure
from .api import register_cooperative_executor
from .api import register_native_executor
from .api import run_workflow
from .api import shutdown
from .api import stats
from .api import submit
from .api import unregister_cooperative_executor
from .api import unregister_native_executor
from .cooperative import AdapterNotAvailableError
from .cooperative import AdapterReadiness
from .cooperative import AdapterStatus
from .cooperative import CooperativeAdapterError
from .cooperative import CooperativeAdapterRegistry
from .cooperative import ExecutionDescription
from .cooperative import FunctionExecutionDescriptor
from .cooperative import GPUProbeEvidence
from .cooperative import MergeStrategy
from .cooperative import ReplicatedSubsetAdapter
from .cooperative import ShardAssignment
from .cooperative import ShardAxis
from .cooperative import ShardResultError
from .cooperative import default_cooperative_adapter_registry
from .cooperative import default_function_descriptors
from .deployment_planner import CooperativeCapability
from .deployment_planner import DeploymentCapabilityError
from .deployment_planner import DeploymentDecision
from .deployment_planner import DeploymentPlanner
from .deployment_planner import DeploymentPlanningError
from .deployment_planner import FunctionCostHints
from .deployment_planner import MeasuredDeployment
from .deployment_planner import WorkloadEstimator
from .direct_dispatch import DIRECT_PUBLIC_FUNCTIONS
from .direct_dispatch import direct_original
from .direct_dispatch import is_direct_public_function
from .runtime import ConfigurationError
from .runtime import CPUTestRuntime
from .runtime import CooperativeExecutionError
from .runtime import EGMGPUError
from .runtime import EGMGPUFuture
from .runtime import EGMGPURuntime
from .runtime import RuntimeClosedError
from .runtime import UnsupportedExecutionModeError
from .runtime import WorkflowExecutionError
from .production_runtime import NativeOwnedRuntime
from .production_bootstrap import ProductionBootstrapReport
from .production_bootstrap import bootstrap_production_bindings
from .compatibility import CompatibilityResult
from .compatibility import ExecutorCompatibilityError
from .native_bindings import NATIVE_PROBE_SCHEMA
from .native_bindings import NativeBindingError
from .native_bindings import NativeBindingEvidenceError
from .native_bindings import NativeBindingProbeEvidence
from .native_bindings import NativeBindingReadiness
from .native_bindings import NativeBindingResultError
from .native_bindings import NativeBindingSpec
from .native_bindings import NativeRawResult
from .native_bindings import NativeRawResultAdapter
from .native_bindings import PublicResultKind
from .native_bindings import ReplicatedNativeBindingRegistry
from .native_bindings import ReplicatedNativeExecutor
from .native_bindings import default_replicated_native_registry
from .native_bindings import default_replicated_native_specs
from .native_bindings import load_native_probe_evidence
from .native_bindings import register_probed_replicated_bindings
from .partitioned_bindings import DEFAULT_PARTITIONED_EVIDENCE_INDEX
from .partitioned_bindings import PARTITIONED_EVIDENCE_SCHEMA
from .partitioned_bindings import ImmutableMultiPartitionGraph
from .partitioned_bindings import SingleOwnerSsspExecutor
from .partitioned_bindings import PartitionedBindingEvidence
from .partitioned_bindings import PartitionedBindingSpec
from .partitioned_bindings import PartitionedBindingStatus
from .partitioned_bindings import PartitionedEvidenceIndex
from .partitioned_bindings import PartitionedNativeBindingRegistry
from .partitioned_bindings import PartitionedRawResult
from .partitioned_bindings import default_partitioned_native_registry
from .partitioned_bindings import default_partitioned_native_specs
from .partitioned_bindings import load_partitioned_binding_evidence
from .partitioned_bindings import load_partitioned_graph
from .partitioned_bindings import register_partitioned_bindings
from .partitioned_bindings import register_probed_partitioned_bindings
from .partitioned_bindings import topology_fingerprint
from .sssp_policy import SsspExecutionPolicy
from .snapshot import GraphSnapshot
from .snapshot import GraphSnapshotError
from .snapshot import GraphSnapshotLease
from .snapshot import GraphSnapshotRegistry
from .snapshot import GraphGenerationChangedError
from .snapshot import NativePayloadUnavailableError
from .snapshot import NativeSnapshotPayload
from .snapshot import SnapshotCaptureError
from .snapshot import SnapshotConsistencyError
from .snapshot import SnapshotObservation
from .snapshot import SnapshotRecord
from .snapshot import StaleGraphGenerationError
from .single_bindings import SINGLE_NATIVE_BINDING
from .single_bindings import SINGLE_NATIVE_PROBE_SCHEMA
from .single_bindings import SingleBindingReadiness
from .single_bindings import SingleNativeBindingRegistry
from .single_bindings import SingleNativeBindingSpec
from .single_bindings import SingleNativeEvidence
from .single_bindings import SingleNativeExecutor
from .single_bindings import SingleResultKind
from .single_bindings import default_single_native_registry
from .single_bindings import default_single_native_specs
from .single_bindings import default_single_public_functions
from .single_bindings import load_single_native_evidence
from .single_bindings import register_probed_single_bindings
from .types import CallSpec
from .types import CooperativeContext
from .types import CooperativePlan
from .types import ExecutionMode
from .types import NativeCallContext
from .types import NativeExecutor
from .types import RequestSchedulingPolicy
from .types import RuntimeConfig


__all__ = [
    "SingleOwnerSsspExecutor",
    "SsspExecutionPolicy",
    "CallSpec",
    "AdapterNotAvailableError",
    "AdapterReadiness",
    "AdapterStatus",
    "ConfigurationError",
    "CompatibilityResult",
    "CooperativeContext",
    "CooperativeCapability",
    "CooperativeExecutionError",
    "CooperativeAdapterError",
    "CooperativeAdapterRegistry",
    "CooperativePlan",
    "CPUTestRuntime",
    "EGMGPUError",
    "EGMGPUFuture",
    "EGMGPURuntime",
    "ExecutionMode",
    "ExecutorCompatibilityError",
    "DeploymentCapabilityError",
    "DeploymentDecision",
    "DeploymentPlanner",
    "DeploymentPlanningError",
    "DIRECT_PUBLIC_FUNCTIONS",
    "ExecutionDescription",
    "FunctionExecutionDescriptor",
    "FunctionCostHints",
    "GPUProbeEvidence",
    "GraphSnapshot",
    "GraphSnapshotError",
    "GraphSnapshotLease",
    "GraphSnapshotRegistry",
    "GraphGenerationChangedError",
    "NativePayloadUnavailableError",
    "NATIVE_PROBE_SCHEMA",
    "NativeBindingError",
    "NativeBindingEvidenceError",
    "NativeBindingProbeEvidence",
    "NativeBindingReadiness",
    "NativeBindingResultError",
    "NativeBindingSpec",
    "NativeCallContext",
    "NativeExecutor",
    "NativeOwnedRuntime",
    "ProductionBootstrapReport",
    "NativeRawResult",
    "NativeRawResultAdapter",
    "NativeSnapshotPayload",
    "DEFAULT_PARTITIONED_EVIDENCE_INDEX",
    "PARTITIONED_EVIDENCE_SCHEMA",
    "ImmutableMultiPartitionGraph",
    "PartitionedBindingEvidence",
    "PartitionedBindingSpec",
    "PartitionedBindingStatus",
    "PartitionedEvidenceIndex",
    "PartitionedNativeBindingRegistry",
    "PartitionedRawResult",
    "MergeStrategy",
    "MeasuredDeployment",
    "ReplicatedSubsetAdapter",
    "ReplicatedNativeBindingRegistry",
    "ReplicatedNativeExecutor",
    "PublicResultKind",
    "RuntimeClosedError",
    "RuntimeConfig",
    "RequestSchedulingPolicy",
    "SnapshotCaptureError",
    "SnapshotConsistencyError",
    "SnapshotObservation",
    "SnapshotRecord",
    "ShardAssignment",
    "ShardAxis",
    "ShardResultError",
    "StaleGraphGenerationError",
    "SINGLE_NATIVE_BINDING",
    "SINGLE_NATIVE_PROBE_SCHEMA",
    "SingleBindingReadiness",
    "SingleNativeBindingRegistry",
    "SingleNativeBindingSpec",
    "SingleNativeEvidence",
    "SingleNativeExecutor",
    "SingleResultKind",
    "UnsupportedExecutionModeError",
    "WorkflowExecutionError",
    "call",
    "bootstrap_production_bindings",
    "configure",
    "default_cooperative_adapter_registry",
    "default_function_descriptors",
    "default_replicated_native_registry",
    "default_replicated_native_specs",
    "default_partitioned_native_registry",
    "default_partitioned_native_specs",
    "default_single_native_registry",
    "default_single_native_specs",
    "default_single_public_functions",
    "direct_original",
    "is_direct_public_function",
    "load_native_probe_evidence",
    "load_partitioned_binding_evidence",
    "load_partitioned_graph",
    "load_single_native_evidence",
    "register_cooperative_executor",
    "register_native_executor",
    "register_partitioned_bindings",
    "register_probed_partitioned_bindings",
    "register_probed_replicated_bindings",
    "register_probed_single_bindings",
    "run_workflow",
    "shutdown",
    "stats",
    "submit",
    "topology_fingerprint",
    "unregister_cooperative_executor",
    "unregister_native_executor",
    "WorkloadEstimator",
]

# Explicit isolated finite-window candidate, disabled unless configured.
from .api import configure_joint_window, submit_window
from .joint_window import WindowCall, WindowDeploymentPlanner

# Narrow explicit TC service extension; not part of the existing 16/6/4 bootstrap.
from .triangle_count_service import TriangleCountGraph, prepare_triangle_count
from .triangle_count_service import register_triangle_count_bindings, triangle_count
__all__ += ["TriangleCountGraph", "prepare_triangle_count", "register_triangle_count_bindings", "triangle_count"]


# Importing this explicit namespace installs idempotent wrappers; configure()
# remains explicit and workers start on the first qualified submission.
import easygraph as _easygraph
from .direct_dispatch import install_direct_public_api as _install_public_api
from .direct_dispatch import DIRECT_PUBLIC_FUNCTIONS as _direct_public_functions
# Explicit ``import easygraph.egmgpu`` may precede access to any root-level
# graph function. Finish the normal lazy root exports before installing its
# wrappers; importing ``easygraph.gpu`` alone never enters this namespace.
if not all(attribute in vars(_easygraph) for _, attribute in _direct_public_functions):
    _easygraph._load_legacy_exports()
_install_public_api(vars(_easygraph))
del _install_public_api, _easygraph, _direct_public_functions
