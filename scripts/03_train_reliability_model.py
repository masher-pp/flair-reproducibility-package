from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from _runtime import pytorch_python


ROOT = Path(__file__).resolve().parent.parent
RELIABILITY_DIR = ROOT / "src" / "reliability_model"
TRAIN_SCRIPT = RELIABILITY_DIR / "06_hpo_log_mlp_random.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3: reproduce the v3 Log-MLP-HPO100 Reliability model."
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--hpo-only", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate v3 inputs and environment without training.",
    )
    return parser.parse_args()


def required_files() -> list[Path]:
    return [
        TRAIN_SCRIPT,
        RELIABILITY_DIR / "reliability_features.py",
        RELIABILITY_DIR / "log_mlp_reliability.py",
        ROOT / "results/intermediate/calibration_features_ae_pre.csv",
        ROOT / "results/intermediate/test_features_ae_pre.csv",
        ROOT / "results/intermediate/test_labels_ae_pre.csv",
        ROOT / "data/splits/deployment/deployment.csv",
        ROOT / "results/hpo/mlp_log_hpo100_20260824/baseline_log_mlp_test15_metrics.csv",
    ]


def main() -> None:
    args = parse_args()
    runtime_python = pytorch_python()
    missing = [path for path in required_files() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing v3 Reliability inputs:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    cmd = [
        str(runtime_python),
        str(TRAIN_SCRIPT),
        "--n-trials",
        str(args.n_trials),
        "--workers",
        str(args.workers),
    ]
    if args.reset:
        cmd.append("--reset")
    if args.smoke_test:
        cmd.append("--smoke-test")
    if args.hpo_only:
        cmd.append("--hpo-only")
    if args.check:
        cmd.append("--check")

    print("[Step 3] Python runtime:", runtime_python)
    print("[Step 3] cwd:", RELIABILITY_DIR)
    print("[Step 3] command:", " ".join(cmd))
    subprocess.run(cmd, cwd=RELIABILITY_DIR, check=True)


if __name__ == "__main__":
    main()
