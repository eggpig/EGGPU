import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


BENCHMARKING = Path(__file__).resolve().parents[1]
if str(BENCHMARKING) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING))

import run_eggpu_scaling as scaling
from stable_timing_protocol import (
    CONTROLLED_THREAD_ENVIRONMENT,
    DEFAULT_MEDIAN_OVER_MIN_LIMIT,
    FORMAL_BATCH_SELECTION_POLICY,
    GATE_CALIBRATION,
    PAPER_ESTIMATOR,
    VARIANCE_POLICY,
    StableTimingProtocolError,
    apply_controlled_thread_environment,
    format_linux_list,
    gate_calibration_payload,
    parse_linux_list,
    summarize_five_samples,
    validate_controlled_execution,
    validate_gate_calibration,
)


def placement(cpu_ids=(0, 1), numa_ids=(0,)):
    return {
        "cpu_affinity_ids": list(cpu_ids),
        "cpu_affinity_list": format_linux_list(cpu_ids),
        "controlled_thread_environment": dict(
            CONTROLLED_THREAD_ENVIRONMENT
        ),
        "numactl": {
            "available": True,
            "returncode": 0,
            "parsed": {"membind_ids": list(numa_ids)},
        },
    }


class StableTimingProtocolTests(unittest.TestCase):
    def test_gate_calibration_is_hash_bound_and_returns_isolated_copy(self):
        validated = validate_gate_calibration(
            gate_calibration_payload(),
            repo_root=BENCHMARKING.parent,
        )

        self.assertEqual(validated, GATE_CALIBRATION)
        validated["calibration_evidence"][0]["e2e_raw5_seconds"][0] = -1
        self.assertEqual(
            GATE_CALIBRATION["calibration_evidence"][0][
                "e2e_raw5_seconds"
            ][0],
            0.141710006,
        )

    def test_gate_calibration_rejects_any_full_dictionary_drift(self):
        drifted = gate_calibration_payload()
        drifted["calibration_evidence"][0][
            "results_samples_sha256"
        ] = "0" * 64

        with self.assertRaisesRegex(
            StableTimingProtocolError,
            "shared canonical definition",
        ):
            validate_gate_calibration(
                drifted,
                repo_root=BENCHMARKING.parent,
            )

    def test_linux_list_parser_and_formatter_are_canonical(self):
        self.assertEqual(
            parse_linux_list("32-39,96-103"),
            list(range(32, 40)) + list(range(96, 104)),
        )
        self.assertEqual(
            format_linux_list([3, 2, 1, 7, 10, 9, 8]),
            "1-3,7-10",
        )

    def test_controlled_thread_environment_overrides_host_defaults(self):
        environment = {
            "OMP_NUM_THREADS": "64",
            "OPENBLAS_NUM_THREADS": "32",
            "UNRELATED": "preserved",
        }

        observed = apply_controlled_thread_environment(environment)

        self.assertEqual(observed, CONTROLLED_THREAD_ENVIRONMENT)
        self.assertEqual(environment["UNRELATED"], "preserved")

    def test_ordinary_variance_passes_and_sd_cv_are_reported(self):
        # CV is deliberately above 10%; it is diagnostic, not the hard gate.
        result = summarize_five_samples((1.0, 1.2, 1.4, 1.6, 1.8))

        self.assertEqual(result["stability_status"], "pass")
        self.assertEqual(
            result["batch_acceptance_status"], "accepted_complete_batch"
        )
        self.assertEqual(result["submission_seconds"], 1.0)
        self.assertEqual(result["paper_estimator"], PAPER_ESTIMATOR)
        self.assertEqual(result["variance_policy"], VARIANCE_POLICY)
        self.assertGreater(result["coefficient_of_variation"], 0.10)
        self.assertGreater(result["sample_std_seconds"], 0.0)

    def test_single_catastrophic_high_sample_rejects_entire_batch(self):
        result = summarize_five_samples((1.0, 1.1, 1.2, 1.3, 100.0))

        self.assertEqual(result["stability_status"], "fail")
        self.assertEqual(
            result["batch_acceptance_status"], "rejected_entire_batch"
        )
        self.assertIsNone(result["submission_seconds"])
        self.assertIn(
            "max_over_median_exceeds_limit", result["failure_reasons"]
        )

    def test_unique_catastrophic_low_sample_rejects_entire_batch(self):
        result = summarize_five_samples((0.1, 1.0, 1.1, 1.2, 1.3))

        self.assertEqual(result["stability_status"], "fail")
        self.assertIsNone(result["submission_seconds"])
        self.assertIn(
            "median_over_min_exceeds_limit", result["failure_reasons"]
        )

    def test_calibrated_repeatable_bimodality_passes_default_gate(self):
        result = summarize_five_samples(
            (0.127117831, 0.141710006, 0.286958758, 0.338744991, 0.5971683)
        )

        self.assertEqual(DEFAULT_MEDIAN_OVER_MIN_LIMIT, 3.0)
        self.assertEqual(result["stability_status"], "pass")
        self.assertAlmostEqual(
            result["median_over_min"],
            GATE_CALIBRATION["independent_batch_median_over_min"][0],
        )
        self.assertEqual(result["submission_seconds"], 0.127117831)

    def test_zero_minimum_in_nonzero_batch_fails_but_all_zero_passes(self):
        rejected = summarize_five_samples((0.0, 1.0, 1.0, 1.0, 1.0))
        accepted = summarize_five_samples((0.0, 0.0, 0.0, 0.0, 0.0))

        self.assertEqual(rejected["stability_status"], "fail")
        self.assertIn(
            "zero_minimum_in_nonzero_batch", rejected["failure_reasons"]
        )
        self.assertEqual(accepted["stability_status"], "pass")

    def test_exact_five_samples_are_required(self):
        with self.assertRaises(StableTimingProtocolError):
            summarize_five_samples((1.0, 1.0, 1.0, 1.0))

    def test_external_taskset_and_numactl_declarations_are_verified(self):
        observed = placement(cpu_ids=(2, 3), numa_ids=(1,))

        validate_controlled_execution(
            observed,
            expected_cpu_affinity="2-3",
            expected_numa_nodes="1",
        )
        with self.assertRaisesRegex(
            StableTimingProtocolError, "CPU affinity"
        ):
            validate_controlled_execution(
                observed,
                expected_cpu_affinity="0-1",
                expected_numa_nodes="1",
            )
        with self.assertRaisesRegex(
            StableTimingProtocolError, "NUMA memory binding"
        ):
            validate_controlled_execution(
                observed,
                expected_cpu_affinity="2-3",
                expected_numa_nodes="0",
            )

    def test_numa_validation_falls_back_to_libnuma_when_cli_is_absent(self):
        observed = placement(cpu_ids=(2, 3), numa_ids=(1,))
        observed["numactl"] = {
            "available": False,
            "returncode": None,
            "parsed": {},
        }
        observed["libnuma"] = {
            "available": True,
            "membind_ids": [1],
            "membind_list": "1",
        }

        validate_controlled_execution(
            observed,
            expected_cpu_affinity="2-3",
            expected_numa_nodes="1",
        )

    def test_scaling_control_rejects_resume_and_mixed_measurement_pass(self):
        base = {
            "eggpu_stable_timing_protocol": True,
            "repeat": 5,
            "timing_processes": 0,
            "memory_repeat": 0,
            "resume": True,
            "stability_max_over_median_limit": 5.0,
            "stability_median_over_min_limit": 3.0,
            "expected_cpu_affinity": "",
            "expected_numa_nodes": "",
        }
        with self.assertRaisesRegex(SystemExit, "--no-resume"):
            scaling.configure_stable_timing_protocol(
                SimpleNamespace(**base)
            )

        base["resume"] = False
        base["memory_repeat"] = 3
        with self.assertRaisesRegex(SystemExit, "--memory-repeat 0"):
            scaling.configure_stable_timing_protocol(
                SimpleNamespace(**base)
            )

    def test_scaling_control_records_placement_before_measurement(self):
        args = SimpleNamespace(
            eggpu_stable_timing_protocol=True,
            repeat=5,
            timing_processes=0,
            memory_repeat=0,
            resume=False,
            stability_max_over_median_limit=5.0,
            stability_median_over_min_limit=3.0,
            expected_cpu_affinity="2-3",
            expected_numa_nodes="1",
        )
        observed = placement(cpu_ids=(2, 3), numa_ids=(1,))
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(
                scaling,
                "collect_execution_placement",
                return_value=observed,
            ):
                scaling.configure_stable_timing_protocol(args)

            self.assertEqual(
                args.stable_timing_protocol_metadata["variance_policy"],
                VARIANCE_POLICY,
            )
            self.assertEqual(
                args.stable_timing_protocol_metadata[
                    "expected_cpu_affinity"
                ],
                "2-3",
            )
            self.assertEqual(
                args.stable_timing_protocol_metadata[
                    "gate_calibration"
                ]["median_over_min_limit"],
                3.0,
            )
            self.assertEqual(
                args.stable_timing_protocol_metadata[
                    "formal_batch_selection_policy"
                ],
                FORMAL_BATCH_SELECTION_POLICY,
            )
            self.assertEqual(
                os.environ["EGGPU_STABLE_TIMING_PROTOCOL"], "TRUE"
            )


if __name__ == "__main__":
    unittest.main()
