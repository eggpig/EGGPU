import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent


ACTIVE_RUNTIME_FILES = (
    REPO / "Easy-Graph/easygraph/utils/gpu_runtime.py",
    REPO / "Easy-Graph/easygraph/utils/gpu_eggpu_backend.py",
    ROOT / "benchmarking/library_baselines.py",
    ROOT / "benchmarking/run_full_baselines.py",
    ROOT / "benchmarking/run_eggpu_correctness_gate.py",
    ROOT / "benchmarking/run_eggpu_ablations.py",
    ROOT / "benchmarking/run_eggpu_workflow_reuse.py",
    ROOT / "benchmarking/run_eggpu_first_use.py",
    ROOT / "benchmarking/run_eggpu_natural_workflow.py",
    ROOT / "benchmarking/eggpu_reuse_ablation.py",
    ROOT / "benchmarking/run_closeness_large_supplement.py",
    ROOT / "benchmarking/run_build_time_benchmark.py",
    ROOT / "benchmarking/preflight_full_eval_ready.py",
    ROOT / "benchmarking/preflight_structural_scanv.py",
    ROOT / "benchmarking/audit_backend_separation.py",
    ROOT / "benchmarking/audit_full_result.py",
    ROOT / "benchmarking/summarize_final_result.py",
    ROOT / "run_main_and_ablation.sh",
    ROOT / "run_complete_paper_experiments.sh",
    REPO / "scripts/run_smoke.sh",
)


class RuntimeBackendContractTests(unittest.TestCase):
    def test_active_runtime_has_no_legacy_backend_selector(self):
        forbidden = (
            "EASYGRAPH_GPU_BACKEND",
            "--easygraph-gpu-backend",
            "gpu_mine_backend",
        )
        failures = []
        for path in ACTIVE_RUNTIME_FILES:
            self.assertTrue(path.exists(), path)
            text = path.read_text()
            for token in forbidden:
                if token in text:
                    failures.append(f"{path.relative_to(REPO)} contains {token}")
        self.assertEqual([], failures, "\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
