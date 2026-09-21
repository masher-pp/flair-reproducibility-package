from __future__ import annotations

import subprocess
from pathlib import Path


PYTORCH_PYTHON = Path("/opt/anaconda3/envs/Pytorch/bin/python3.10")
REQUIRED_IMPORTS = (
    "torch, torch_geometric, torch_scatter, rdkit, sklearn, pandas, numpy"
)


def pytorch_python() -> Path:
    """Return this workstation's validated Pytorch interpreter."""
    if not PYTORCH_PYTHON.is_file():
        raise FileNotFoundError(
            "The required Pytorch interpreter is missing: "
            f"{PYTORCH_PYTHON}"
        )
    probe = subprocess.run(
        [str(PYTORCH_PYTHON), "-c", f"import {REQUIRED_IMPORTS}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        raise RuntimeError(
            "The required Pytorch environment is not usable. "
            f"Interpreter: {PYTORCH_PYTHON}\n{detail}"
        )
    return PYTORCH_PYTHON.resolve()
