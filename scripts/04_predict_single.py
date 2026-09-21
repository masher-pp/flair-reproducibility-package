from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from _runtime import pytorch_python


ROOT = Path(__file__).resolve().parent.parent
AE_DIR = ROOT / "src" / "reliability_model"
PREDICT_SCRIPT = AE_DIR / "predict_single_with_ae.py"
MAIN_MODEL_ROOT = ROOT / "src" / "main_model"
BEST_DIR = ROOT / "models" / "main_model"
AE_MODEL = ROOT / "models" / "reliability_model" / "mlp_log_hpo100_reliability_model_bundle.pt"
DEPLOY_CACHE = ROOT / "models" / "reliability_model" / "prediction_deploy_cache_solvent32.pkl"
OFFLINE_FEATURE_CSV = ROOT / "results" / "intermediate" / "calibration_features_ae_pre.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict Abs/Emi/Plqy/Em from the mean of 5 main models and report only the "
            "numeric pre_AE error estimate for each property."
        )
    )
    parser.add_argument("--smiles", default=None, help="Solute SMILES. If omitted, the inner script asks interactively.")
    parser.add_argument("--solvent", default=None, help="Solvent SMILES. If omitted, the inner script asks interactively.")
    parser.add_argument("--build_cache_only", action="store_true", help="Build deployment cache and exit.")
    parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild deployment cache even if it already exists.")
    parser.add_argument("--json", action="store_true", help="Print compact JSON only.")
    parser.add_argument("--check", action="store_true", help="Only check required files and print the command.")
    return parser.parse_args()


def required_files(build_cache_only: bool) -> list[Path]:
    files = [
        PREDICT_SCRIPT,
        MAIN_MODEL_ROOT / "data_loading.py",
        MAIN_MODEL_ROOT / "model_factory.py",
        MAIN_MODEL_ROOT / "trainer.py",
        ROOT / "data" / "splits" / "deployment" / "deployment.csv",
        AE_MODEL,
        OFFLINE_FEATURE_CSV,
    ]
    for fold in range(1, 6):
        files.append(BEST_DIR / f"fold_{fold:02d}" / "best_model.pt")
    return files


def main() -> None:
    args = parse_args()
    runtime_python = pytorch_python()
    missing = [path for path in required_files(args.build_cache_only) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files. Run Steps 1-3 before single-molecule prediction.\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    cmd = [
        str(runtime_python),
        str(PREDICT_SCRIPT),
        "--split",
        "ae",
        "--best_dir",
        str(BEST_DIR),
        "--ae_model",
        str(AE_MODEL),
        "--deploy_cache",
        str(DEPLOY_CACHE),
        "--offline_csv",
        str(OFFLINE_FEATURE_CSV),
    ]
    if args.smiles:
        cmd.extend(["--smiles", args.smiles])
    if args.solvent:
        cmd.extend(["--solvent", args.solvent])
    if args.build_cache_only:
        cmd.append("--build_cache_only")
    if args.rebuild_cache:
        cmd.append("--rebuild_cache")
    if args.json:
        cmd.append("--json")

    if args.check:
        print("[Predict] cwd:", AE_DIR)
        print("[Predict] Python runtime:", runtime_python)
        print("[Predict] main model code:", MAIN_MODEL_ROOT)
        print("[Predict] command:", " ".join(cmd))
        print("[Predict] check passed; prediction not started.")
        return
    env = dict(os.environ)
    env["MAIN_MODEL_ROOT"] = str(MAIN_MODEL_ROOT)
    env["MODEL_CODE_ROOT"] = str(MAIN_MODEL_ROOT)
    subprocess.run(cmd, cwd=AE_DIR, env=env, check=True)


if __name__ == "__main__":
    main()
