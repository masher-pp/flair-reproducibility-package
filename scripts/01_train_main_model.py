from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from _runtime import pytorch_python


ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "src" / "main_model"
TRAIN_SCRIPT = CODE_DIR / "run_online_model_fivefold_training.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: train the 5 DL main-model folds.")
    parser.add_argument("--single_run", type=int, default=None, help="Run only one fold, e.g. --single_run 1.")
    parser.add_argument("--check", action="store_true", help="Only check files and print the command; do not train.")
    return parser.parse_args()


def required_files() -> list[Path]:
    files = [TRAIN_SCRIPT, ROOT / "models" / "pretrained" / "MORE.pth"]
    split_dir = ROOT / "data" / "splits" / "deployment"
    files.extend([split_dir / "deployment.csv", split_dir / "deployment_test.csv"])
    return files


def main() -> None:
    args = parse_args()
    runtime_python = pytorch_python()
    missing = [path for path in required_files() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(f"  - {p}" for p in missing))

    cmd = [str(runtime_python), str(TRAIN_SCRIPT)]
    if args.single_run is not None:
        cmd.extend(["--single_run", str(args.single_run)])

    print("[Step 1] cwd:", CODE_DIR)
    print("[Step 1] command:", " ".join(cmd))
    if args.check:
        print("[Step 1] check passed; training not started.")
        return
    subprocess.run(cmd, cwd=CODE_DIR, check=True)


if __name__ == "__main__":
    main()
