from __future__ import annotations

import argparse
import hashlib
import pickle
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NAME = "FLAIR-reproducibility-package-v3-log-mlp-hpo100"
OFFICIAL_MODEL = "mlp_log_hpo100_reliability_model_bundle.pt"
OFFICIAL_CACHE = "prediction_deploy_cache_solvent32.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the v3 Log-MLP HPO100 release archive.")
    parser.add_argument("--destination_parent", default=str(ROOT.parent))
    parser.add_argument("--name", default=DEFAULT_NAME)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_ignore(path: str, names: list[str]) -> set[str]:
    directory = Path(path)
    relative = directory.relative_to(ROOT)
    ignored = {
        name
        for name in names
        if name == ".DS_Store" or name == "__pycache__" or name.endswith(".pyc")
    }
    if relative == Path("."):
        ignored.add("MANIFEST.sha256")
    if relative == Path("models/reliability_model"):
        allowed = {OFFICIAL_MODEL, OFFICIAL_CACHE}
        ignored.update(name for name in names if name not in allowed)
    if relative == Path("results"):
        ignored.add("experiments")
    if relative == Path("results/final"):
        allowed = {"mlp_log_hpo100_20260824"}
        ignored.update(name for name in names if name not in allowed)
    if relative == Path("results/hpo"):
        allowed = {"mlp_log_hpo100_20260824"}
        ignored.update(name for name in names if name not in allowed)
    if relative == Path("results/hpo/mlp_log_hpo100_20260824"):
        ignored.add("cv_cache")
    return ignored


def normalize_release_metadata(package_root: Path) -> None:
    """Remove workstation-specific absolute paths from distributed metadata."""
    source_prefix = str(ROOT)
    text_suffixes = {".json", ".csv", ".md", ".txt", ".log", ".cff"}
    for path in package_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        if source_prefix in text:
            path.write_text(text.replace(source_prefix, "."), encoding="utf-8")

    cache_path = package_root / "models/reliability_model" / OFFICIAL_CACHE
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    cache.update(
        {
            "offline_csv": "../../results/intermediate/calibration_features_ae_pre.csv",
            "train_and_val_csv": "../../data/splits/deployment/deployment.csv",
        }
    )
    tmp_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_cache.open("wb") as handle:
        pickle.dump(cache, handle)
    tmp_cache.replace(cache_path)


def write_manifest(package_root: Path) -> Path:
    manifest = package_root / "MANIFEST.sha256"
    lines = []
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        if path == manifest:
            continue
        relative = path.relative_to(package_root).as_posix()
        lines.append(f"{sha256(path)}  ./{relative}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def verify_required_files(package_root: Path) -> None:
    required = [
        package_root / "README.md",
        package_root / "RELEASE_NOTES.md",
        package_root / "environment/pytorch_environment.txt",
        package_root / "data/splits/deployment/deployment.csv",
        package_root / "data/splits/deployment/deployment_test.csv",
        package_root / "MANIFEST.sha256",
        package_root / "scripts/03_train_reliability_model.py",
        package_root / "scripts/04_predict_single.py",
        package_root / "scripts/_runtime.py",
        package_root / "src/reliability_model/06_hpo_log_mlp_random.py",
        package_root / "src/reliability_model/log_mlp_reliability.py",
        package_root / "src/reliability_model/reliability_features.py",
        package_root / "src/reliability_model/main_model_inference.py",
        package_root / "models/reliability_model" / OFFICIAL_MODEL,
        package_root / "models/reliability_model" / OFFICIAL_CACHE,
        package_root / "results/final/mlp_log_hpo100_20260824/validation.json",
        package_root / "results/final/mlp_log_hpo100_20260824/test15_by_property.csv",
        package_root / "results/hpo/mlp_log_hpo100_20260824/trials.csv",
        package_root / "results/hpo/mlp_log_hpo100_20260824/best_params.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Release package is missing required files:\n" + "\n".join(missing))


def verify_v3_only(package_root: Path) -> None:
    forbidden_paths = [
        package_root / "backups",
        package_root / ("batch_predict_" + "35.py"),
        package_root / ("batch_predict_" + "35_full4.py"),
        package_root / ("batch_predict_" + "bins.py"),
    ]
    present = [str(path) for path in forbidden_paths if path.exists()]
    if present:
        raise RuntimeError("Superseded files entered the v3 release:\n" + "\n".join(present))

    forbidden_text = (
        "FLAIR-reproducibility-package-" + "v1",
        "FLAIR-reproducibility-package-" + "v2",
        "reliability_" + "rf_",
        "rf_" + "property_specific",
        "legacy " + "RF",
        "/Users/" + "yczhao",
    )
    text_suffixes = {".py", ".json", ".csv", ".md", ".txt", ".cff"}
    violations = []
    for path in package_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        matches = [token for token in forbidden_text if token in text]
        if matches:
            violations.append(f"{path.relative_to(package_root)}: {matches}")
    if violations:
        raise RuntimeError(
            "Non-v3 text remains in the release:\n" + "\n".join(violations)
        )


def main() -> None:
    args = parse_args()
    destination_parent = Path(args.destination_parent).expanduser().resolve()
    package_root = destination_parent / args.name
    archive = destination_parent / f"{args.name}.zip"
    checksum = destination_parent / f"{args.name}.zip.sha256"
    existing = [path for path in [package_root, archive, checksum] if path.exists()]
    if existing:
        raise FileExistsError("Release targets already exist:\n" + "\n".join(str(path) for path in existing))

    shutil.copytree(ROOT, package_root, ignore=release_ignore, copy_function=shutil.copy2)
    normalize_release_metadata(package_root)
    write_manifest(package_root)
    verify_required_files(package_root)
    verify_v3_only(package_root)
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=destination_parent, base_dir=args.name)
    with zipfile.ZipFile(archive) as handle:
        bad_file = handle.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP CRC verification failed at {bad_file}")
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    file_count = sum(1 for path in package_root.rglob("*") if path.is_file())
    print(f"package_root={package_root}")
    print(f"archive={archive}")
    print(f"archive_sha256={sha256(archive)}")
    print(f"file_count={file_count}")
    print(f"archive_size_bytes={archive.stat().st_size}")


if __name__ == "__main__":
    main()
