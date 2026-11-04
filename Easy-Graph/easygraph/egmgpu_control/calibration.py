"""Evidence-backed calibration for the EGMGPU adaptive planner.

Profiles in this module are learned only from retained, successful JSONL
records produced on one measured topology.  They do not promote a reference
implementation to an executable capability; they only allow AUTO to consider
an already available cooperative path once its evidence is sufficient.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .descriptors import ExecutionMode


CALIBRATION_SCHEMA_VERSION = 1
COOPERATIVE_MODES = frozenset(
    {
        ExecutionMode.REPLICATED_TASK_PARALLEL,
        ExecutionMode.PARTITIONED_ITERATIVE,
    }
)


class CalibrationError(ValueError):
    """Calibration evidence is invalid, incomplete, or incompatible."""


@dataclass(frozen=True)
class CalibrationPolicy:
    minimum_samples: int = 5
    maximum_age_seconds: float = 30.0 * 24.0 * 60.0 * 60.0
    minimum_multigpu_gain: float = 0.05
    ema_alpha: float = 0.20

    def __post_init__(self) -> None:
        if self.minimum_samples < 2:
            raise ValueError("minimum_samples must be at least two")
        if self.maximum_age_seconds <= 0:
            raise ValueError("maximum_age_seconds must be positive")
        if not 0 <= self.minimum_multigpu_gain < 1:
            raise ValueError("minimum_multigpu_gain must be in [0, 1)")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")


@dataclass(frozen=True, order=True)
class CalibrationKey:
    function: str
    mode: ExecutionMode
    device_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "function", self.function.strip().lower())
        object.__setattr__(self, "mode", ExecutionMode.parse(self.mode))
        if not self.function:
            raise ValueError("function cannot be empty")
        if self.device_count <= 0:
            raise ValueError("device_count must be positive")
        if self.mode is ExecutionMode.SINGLE_DEVICE and self.device_count != 1:
            raise ValueError("single mode requires device_count=1")
        if self.mode in COOPERATIVE_MODES and self.device_count < 2:
            raise ValueError("cooperative modes require at least two devices")

    def encode(self) -> str:
        return "{}|{}|{}".format(self.function, self.mode.value, self.device_count)

    @classmethod
    def decode(cls, value: str) -> "CalibrationKey":
        function, mode, device_count = value.split("|", 2)
        return cls(function, ExecutionMode.parse(mode), int(device_count))


@dataclass(frozen=True)
class CalibrationObservation:
    observation_id: str
    workload_key: str
    topology_fingerprint: str
    function: str
    mode: ExecutionMode
    device_count: int
    captured_at_utc: str
    compute_seconds: float
    state_seconds: float
    communication_seconds: float
    merge_seconds: float
    materialize_seconds: float
    total_seconds: float
    result_digest: str
    feature_scale: float = 1.0
    raw_path: str = ""
    raw_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "function", self.function.strip().lower())
        object.__setattr__(self, "mode", ExecutionMode.parse(self.mode))
        if (
            not self.observation_id
            or not self.workload_key
            or not self.topology_fingerprint
            or not self.result_digest
        ):
            raise ValueError(
                "observation identity, workload key, topology, and digest are required"
            )
        CalibrationKey(self.function, self.mode, self.device_count)
        _parse_utc(self.captured_at_utc)
        for name in (
            "compute_seconds",
            "state_seconds",
            "communication_seconds",
            "merge_seconds",
            "materialize_seconds",
            "total_seconds",
            "feature_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and non-negative".format(name))
        if self.feature_scale <= 0 or self.total_seconds <= 0:
            raise ValueError("feature_scale and total_seconds must be positive")

    @property
    def key(self) -> CalibrationKey:
        return CalibrationKey(self.function, self.mode, self.device_count)


@dataclass(frozen=True)
class CalibrationCoefficients:
    sample_count: int
    unique_workloads: int
    compute: float
    state: float
    communication: float
    merge: float
    materialize: float
    total: float
    ema_total: float
    newest_at_utc: str
    break_even_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sample_count <= 0 or self.unique_workloads <= 0:
            raise ValueError("coefficient sample counts must be positive")
        _parse_utc(self.newest_at_utc)
        for name in (
            "compute",
            "state",
            "communication",
            "merge",
            "materialize",
            "total",
            "ema_total",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} coefficient must be finite and non-negative".format(name))
        if self.break_even_scale is not None and self.break_even_scale <= 0:
            raise ValueError("break_even_scale must be positive")


@dataclass(frozen=True)
class EvidenceFile:
    path: str
    sha256: str
    record_count: int


@dataclass(frozen=True)
class CalibrationProfile:
    topology_fingerprint: str
    created_at_utc: str
    policy: CalibrationPolicy
    coefficients: Mapping[CalibrationKey, CalibrationCoefficients]
    observations: Tuple[CalibrationObservation, ...]
    evidence_files: Tuple[EvidenceFile, ...]
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.topology_fingerprint:
            raise ValueError("topology_fingerprint cannot be empty")
        _parse_utc(self.created_at_utc)
        if any(
            observation.topology_fingerprint != self.topology_fingerprint
            for observation in self.observations
        ):
            raise ValueError("profile cannot mix topology fingerprints")

    def coefficient(
        self, function: str, mode: ExecutionMode, device_count: int
    ) -> Optional[CalibrationCoefficients]:
        return self.coefficients.get(
            CalibrationKey(function, ExecutionMode.parse(mode), device_count)
        )

    def sufficient(
        self,
        function: str,
        mode: ExecutionMode,
        device_count: int,
        *,
        topology_fingerprint: Optional[str] = None,
        now_utc: Optional[str] = None,
    ) -> bool:
        mode = ExecutionMode.parse(mode)
        if topology_fingerprint != self.topology_fingerprint:
            return False
        coefficient = self.coefficient(function, mode, device_count)
        if coefficient is None or coefficient.sample_count < self.policy.minimum_samples:
            return False
        now = _parse_utc(now_utc) if now_utc else datetime.now(timezone.utc)
        age = (now - _parse_utc(coefficient.newest_at_utc)).total_seconds()
        return 0 <= age <= self.policy.maximum_age_seconds

    def with_online_observation(
        self, observation: CalibrationObservation
    ) -> "CalibrationProfile":
        """Append raw evidence and update EMA without discarding history."""

        if observation.topology_fingerprint != self.topology_fingerprint:
            raise CalibrationError(
                "online observation has a different topology fingerprint"
            )
        if any(
            item.observation_id == observation.observation_id
            for item in self.observations
        ):
            raise CalibrationError("duplicate observation_id")
        comparable = tuple(
            item
            for item in self.observations
            if item.workload_key == observation.workload_key
            and item.function == observation.function
        )
        if any(item.result_digest != observation.result_digest for item in comparable):
            raise CalibrationError(
                "online observation result digest disagrees with retained evidence"
            )
        if any(item.feature_scale != observation.feature_scale for item in comparable):
            raise CalibrationError(
                "online observation work_units disagree with retained evidence"
            )
        now = datetime.now(timezone.utc)
        age = (now - _parse_utc(observation.captured_at_utc)).total_seconds()
        if age < 0 or age > self.policy.maximum_age_seconds:
            raise CalibrationError("online observation is stale or from the future")
        coefficients = dict(self.coefficients)
        old = coefficients.get(observation.key)
        all_observations = self.observations + (observation,)
        if old is None:
            coefficients[observation.key] = _fit_group(
                (observation,), self.policy, all_observations
            )
        else:
            samples = tuple(
                item for item in self.observations if item.key == observation.key
            )
            updated = _fit_group(
                samples + (observation,), self.policy, all_observations
            )
            coefficients[observation.key] = replace(
                updated,
                ema_total=(
                    self.policy.ema_alpha
                    * (observation.total_seconds / observation.feature_scale)
                    + (1.0 - self.policy.ema_alpha) * old.ema_total
                ),
            )
        return CalibrationProfile(
            topology_fingerprint=self.topology_fingerprint,
            created_at_utc=_utc_now(),
            policy=self.policy,
            coefficients=coefficients,
            observations=all_observations,
            evidence_files=self.evidence_files,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "topology_fingerprint": self.topology_fingerprint,
            "created_at_utc": self.created_at_utc,
            "policy": asdict(self.policy),
            "coefficients": {
                key.encode(): asdict(value)
                for key, value in sorted(
                    self.coefficients.items(), key=lambda item: item[0].encode()
                )
            },
            "observations": [
                {
                    **asdict(item),
                    "mode": item.mode.value,
                }
                for item in self.observations
            ],
            "evidence_files": [asdict(item) for item in self.evidence_files],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationProfile":
        if int(value.get("schema_version", -1)) != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationError("unsupported calibration schema version")
        return cls(
            topology_fingerprint=str(value["topology_fingerprint"]),
            created_at_utc=str(value["created_at_utc"]),
            policy=CalibrationPolicy(**value["policy"]),
            coefficients={
                CalibrationKey.decode(key): CalibrationCoefficients(**item)
                for key, item in value["coefficients"].items()
            },
            observations=tuple(
                CalibrationObservation(
                    **{**item, "mode": ExecutionMode.parse(item["mode"])}
                )
                for item in value["observations"]
            ),
            evidence_files=tuple(
                EvidenceFile(**item) for item in value["evidence_files"]
            ),
        )

    @classmethod
    def load(cls, path: Path, *, verify_evidence: bool = True) -> "CalibrationProfile":
        profile = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        if verify_evidence:
            for evidence in profile.evidence_files:
                source = Path(evidence.path)
                if not source.is_file() or _sha256(source) != evidence.sha256:
                    raise CalibrationError(
                        "calibration evidence checksum mismatch: {}".format(source)
                    )
        return profile


@dataclass(frozen=True)
class OracleDecision:
    workload_key: str
    chosen_mode: ExecutionMode
    chosen_device_count: int
    chosen_seconds: float
    oracle_mode: ExecutionMode
    oracle_device_count: int
    oracle_seconds: float
    regret: float
    exact_selection: bool
    regression: bool


@dataclass(frozen=True)
class SelectionMetrics:
    workload_count: int
    selection_accuracy: float
    geometric_mean_regret: float
    maximum_regret: float
    regression_rate: float
    decisions: Tuple[OracleDecision, ...]


def evaluate_enumerated_oracle(
    candidates: Mapping[str, Sequence[Tuple[ExecutionMode, int, float]]],
    choices: Mapping[str, Tuple[ExecutionMode, int]],
) -> SelectionMetrics:
    """Compare selected plans with an actually enumerated per-workload oracle."""

    if not candidates:
        raise CalibrationError("oracle evaluation requires at least one workload")
    decisions = []
    for workload_key, rows in sorted(candidates.items()):
        if workload_key not in choices:
            raise CalibrationError("missing chosen plan for {}".format(workload_key))
        normalized = []
        for mode, count, seconds in rows:
            mode = ExecutionMode.parse(mode)
            if not math.isfinite(seconds) or seconds <= 0:
                raise CalibrationError("oracle latencies must be finite and positive")
            normalized.append((mode, int(count), float(seconds)))
        by_plan = {(mode, count): seconds for mode, count, seconds in normalized}
        if len(by_plan) != len(normalized):
            raise CalibrationError("duplicate enumerated plan for {}".format(workload_key))
        chosen_key = (
            ExecutionMode.parse(choices[workload_key][0]),
            int(choices[workload_key][1]),
        )
        if chosen_key not in by_plan:
            raise CalibrationError("chosen plan was not enumerated for {}".format(workload_key))
        oracle_key, oracle_seconds = min(
            by_plan.items(),
            key=lambda item: (item[1], item[0][1], item[0][0].value),
        )
        chosen_seconds = by_plan[chosen_key]
        single_values = [
            seconds for (mode, count), seconds in by_plan.items()
            if mode is ExecutionMode.SINGLE_DEVICE and count == 1
        ]
        regression = bool(single_values and chosen_seconds > min(single_values))
        decisions.append(
            OracleDecision(
                workload_key=workload_key,
                chosen_mode=chosen_key[0],
                chosen_device_count=chosen_key[1],
                chosen_seconds=chosen_seconds,
                oracle_mode=oracle_key[0],
                oracle_device_count=oracle_key[1],
                oracle_seconds=oracle_seconds,
                regret=chosen_seconds / oracle_seconds,
                exact_selection=chosen_key == oracle_key,
                regression=regression,
            )
        )
    regrets = [item.regret for item in decisions]
    return SelectionMetrics(
        workload_count=len(decisions),
        selection_accuracy=(
            sum(item.exact_selection for item in decisions) / len(decisions)
        ),
        geometric_mean_regret=math.exp(
            statistics.fmean(math.log(value) for value in regrets)
        ),
        maximum_regret=max(regrets),
        regression_rate=sum(item.regression for item in decisions) / len(decisions),
        decisions=tuple(decisions),
    )


def build_calibration_profile(
    raw_paths: Sequence[Path],
    *,
    topology_fingerprint: str,
    policy: CalibrationPolicy = CalibrationPolicy(),
    now_utc: Optional[str] = None,
) -> CalibrationProfile:
    """Fit coefficients from checksum-bound, complete JSONL observations."""

    if not raw_paths:
        raise CalibrationError("at least one raw JSONL path is required")
    now = _parse_utc(now_utc) if now_utc else datetime.now(timezone.utc)
    observations = []
    evidence_files = []
    seen_ids = set()
    for path_value in raw_paths:
        path = Path(path_value).resolve()
        if not path.is_file():
            raise CalibrationError("raw result path does not exist: {}".format(path))
        checksum = _sha256(path)
        count = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            count += 1
            try:
                record = json.loads(line)
                observation = _observation_from_record(record, str(path), checksum)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CalibrationError("{}:{}: {}".format(path, line_number, exc)) from exc
            if observation.topology_fingerprint != topology_fingerprint:
                raise CalibrationError(
                    "raw results mix or mismatch hardware topology fingerprints"
                )
            age = (now - _parse_utc(observation.captured_at_utc)).total_seconds()
            if age < 0 or age > policy.maximum_age_seconds:
                raise CalibrationError(
                    "stale or future calibration observation {}".format(
                        observation.observation_id
                    )
                )
            if observation.observation_id in seen_ids:
                raise CalibrationError("duplicate calibration observation_id")
            seen_ids.add(observation.observation_id)
            observations.append(observation)
        evidence_files.append(EvidenceFile(str(path), checksum, count))
    grouped: dict[CalibrationKey, list[CalibrationObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.key, []).append(observation)
    correctness_groups: dict[tuple[str, str], set[str]] = {}
    scales: dict[str, set[float]] = {}
    for observation in observations:
        correctness_groups.setdefault(
            (
                observation.workload_key,
                observation.function,
            ),
            set(),
        ).add(observation.result_digest)
        scales.setdefault(observation.workload_key, set()).add(
            observation.feature_scale
        )
    inconsistent = [
        key for key, digests in correctness_groups.items() if len(digests) != 1
    ]
    if inconsistent:
        raise CalibrationError(
            "result digest mismatch in calibration workload {}".format(
                inconsistent[0]
            )
        )
    mixed_scales = [key for key, values in scales.items() if len(values) != 1]
    if mixed_scales:
        raise CalibrationError(
            "workload_key has inconsistent work_units: {}".format(mixed_scales[0])
        )
    coefficients = {
        key: _fit_group(tuple(rows), policy, tuple(observations))
        for key, rows in grouped.items()
    }
    return CalibrationProfile(
        topology_fingerprint=topology_fingerprint,
        created_at_utc=now.isoformat(),
        policy=policy,
        coefficients=coefficients,
        observations=tuple(observations),
        evidence_files=tuple(evidence_files),
    )


def _observation_from_record(
    record: Mapping[str, Any], raw_path: str, raw_sha256: str
) -> CalibrationObservation:
    if record.get("status") != "ok":
        raise CalibrationError("failed records cannot calibrate AUTO")
    if int(record.get("schema_version", -1)) < 1:
        raise CalibrationError("unsupported raw timing schema")
    topology = record.get("topology_fingerprint")
    workload_key = record.get("workload_key")
    timing = record.get("timing")
    if not topology or not workload_key or not isinstance(timing, Mapping):
        raise CalibrationError("record lacks topology, workload key, or timing")
    required = {
        "first_state_build",
        "resident_public_call",
        "device_processing",
        "communication",
        "native_merge",
        "output_gather",
        "python_materialization",
    }
    missing = sorted(name for name in required if timing.get(name) is None)
    if missing:
        raise CalibrationError("record lacks required calibrated timers: {}".format(missing))
    output_gather = timing["output_gather"]
    merge = float(timing["native_merge"]) + float(output_gather)
    feature_scale = float(record.get("calibration_features", {}).get("work_units", 1.0))
    return CalibrationObservation(
        observation_id=str(record["run_id"]),
        workload_key=str(workload_key),
        topology_fingerprint=str(topology),
        function=str(record["function"]),
        mode=ExecutionMode.parse(record["mode"]),
        device_count=len(record["devices"]),
        captured_at_utc=str(record["finished_at_utc"]),
        compute_seconds=float(timing["device_processing"]),
        state_seconds=float(timing["first_state_build"]),
        communication_seconds=float(timing["communication"]),
        merge_seconds=merge,
        materialize_seconds=float(timing["python_materialization"]),
        total_seconds=float(timing["resident_public_call"]),
        result_digest=str(record["result_digest"]),
        feature_scale=feature_scale,
        raw_path=raw_path,
        raw_sha256=raw_sha256,
    )


def _fit_group(
    rows: Sequence[CalibrationObservation],
    policy: CalibrationPolicy,
    all_rows: Sequence[CalibrationObservation],
) -> CalibrationCoefficients:
    ordered = tuple(sorted(rows, key=lambda item: _parse_utc(item.captured_at_utc)))
    normalized = lambda name: [float(getattr(item, name)) / item.feature_scale for item in ordered]
    total_values = normalized("total_seconds")
    ema = total_values[0]
    for value in total_values[1:]:
        ema = policy.ema_alpha * value + (1.0 - policy.ema_alpha) * ema
    break_even = _break_even(ordered, all_rows, policy.minimum_multigpu_gain)
    return CalibrationCoefficients(
        sample_count=len(ordered),
        unique_workloads=len({item.workload_key for item in ordered}),
        compute=statistics.fmean(normalized("compute_seconds")),
        state=statistics.fmean(normalized("state_seconds")),
        communication=statistics.fmean(normalized("communication_seconds")),
        merge=statistics.fmean(normalized("merge_seconds")),
        materialize=statistics.fmean(normalized("materialize_seconds")),
        total=statistics.fmean(total_values),
        ema_total=ema,
        newest_at_utc=ordered[-1].captured_at_utc,
        break_even_scale=break_even,
    )


def _break_even(
    rows: Sequence[CalibrationObservation],
    all_rows: Sequence[CalibrationObservation],
    minimum_gain: float,
) -> Optional[float]:
    if not rows or rows[0].mode not in COOPERATIVE_MODES:
        return None
    single: dict[str, list[float]] = {}
    cooperative: dict[str, list[CalibrationObservation]] = {}
    for item in all_rows:
        if item.function != rows[0].function:
            continue
        if item.mode is ExecutionMode.SINGLE_DEVICE and item.device_count == 1:
            single.setdefault(item.workload_key, []).append(item.total_seconds)
    for item in rows:
        cooperative.setdefault(item.workload_key, []).append(item)
    qualifying = []
    for workload_key, items in cooperative.items():
        if workload_key not in single:
            continue
        if statistics.fmean(item.total_seconds for item in items) <= (
            statistics.fmean(single[workload_key]) * (1.0 - minimum_gain)
        ):
            qualifying.append(items[0].feature_scale)
    return min(qualifying) if qualifying else None


def _parse_utc(value: Optional[str]) -> datetime:
    if not value:
        raise ValueError("UTC timestamp is required")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
