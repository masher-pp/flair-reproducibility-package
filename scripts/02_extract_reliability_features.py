from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from _runtime import pytorch_python


ROOT = Path(__file__).resolve().parent.parent
AE_DIR = ROOT / "src" / "reliability_model"
EXTRACT_SCRIPT = AE_DIR / "02_generate_ae_pre_merged_features.py"
MAIN_MODEL_ROOT = ROOT / "src" / "main_model"
BEST_DIR = ROOT / "models" / "main_model"
SPLIT_DIR = ROOT / "data" / "splits" / "deployment"
CURRENT_TEST_CSV = SPLIT_DIR / "deployment_test.csv"
CALIBRATION_FEATURES_CSV = ROOT / "results" / "intermediate" / "calibration_features_ae_pre.csv"
TEST_FEATURES_CSV = ROOT / "results" / "intermediate" / "test_features_ae_pre.csv"
TEST_LABELS_CSV = ROOT / "results" / "intermediate" / "test_labels_ae_pre.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2: create a sealed 85/15 split and extract AE features.")
    parser.add_argument(
        "--ae_pre_test_csv",
        default=None,
        help="Defaults to data/source/ae_pre_holdout_candidate_pool.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--feature_n_jobs", type=int, default=8)
    parser.add_argument("--feature_backend", choices=["threading", "process"], default="threading")
    parser.add_argument("--feature_chunk_size", type=int, default=512)
    parser.add_argument("--rebuild_merged_split", action="store_true", help="Force a fresh canonical-pair grouped 85/15 split.")
    parser.add_argument(
        "--reuse_intermediate_cache",
        action="store_true",
        help="Reuse Step 2 prediction/feature caches. Disabled by default for reproducibility.",
    )
    parser.add_argument("--check", action="store_true", help="Only check files and print the command; do not extract.")
    return parser.parse_args()


def required_files() -> list[Path]:
    files = [
        EXTRACT_SCRIPT,
        MAIN_MODEL_ROOT / "data_loading.py",
        MAIN_MODEL_ROOT / "model_factory.py",
        MAIN_MODEL_ROOT / "trainer.py",
        SPLIT_DIR / "deployment.csv",
        CURRENT_TEST_CSV,
    ]
    for fold in range(1, 6):
        files.append(BEST_DIR / f"fold_{fold:02d}" / "best_model.pt")
    return files


def main() -> None:
    args = parse_args()
    runtime_python = pytorch_python()
    missing = [path for path in required_files() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files. Step 1 must finish before Step 2.\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    cmd = [
        str(runtime_python),
        str(EXTRACT_SCRIPT),
        "--split",
        "ae",
        "--best_dir",
        str(BEST_DIR),
        "--split_dir",
        str(SPLIT_DIR),
        "--current_test_csv",
        str(CURRENT_TEST_CSV),
        "--calibration_output_csv",
        str(CALIBRATION_FEATURES_CSV),
        "--test_features_output_csv",
        str(TEST_FEATURES_CSV),
        "--test_labels_output_csv",
        str(TEST_LABELS_CSV),
        "--summary_json",
        str(ROOT / "results" / "intermediate" / "offline_feature_generation_summary_ae_pre.json"),
        "--seed",
        str(args.seed),
        "--test_fraction",
        str(args.test_fraction),
        "--feature_n_jobs",
        str(args.feature_n_jobs),
        "--feature_backend",
        args.feature_backend,
        "--feature_chunk_size",
        str(args.feature_chunk_size),
    ]
    if args.ae_pre_test_csv:
        cmd.extend(["--ae_pre_test_csv", args.ae_pre_test_csv])
    if args.rebuild_merged_split:
        cmd.append("--rebuild_merged_split")
    if args.reuse_intermediate_cache:
        cmd.append("--reuse_intermediate_cache")
    print("[Step 2] cwd:", AE_DIR)
    print("[Step 2] main model code:", MAIN_MODEL_ROOT)
    print("[Step 2] command:", " ".join(cmd))
    if args.check:
        print("[Step 2] check passed; extraction not started.")
        return
    env = dict(os.environ)
    env["MAIN_MODEL_ROOT"] = str(MAIN_MODEL_ROOT)
    env["MODEL_CODE_ROOT"] = str(MAIN_MODEL_ROOT)
    subprocess.run(cmd, cwd=AE_DIR, env=env, check=True)


if __name__ == "__main__":
    main()
