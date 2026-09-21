from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator


CGSD_NAMES = ["ET30", "SA", "SB", "SdP", "SP"]





def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    """ SMILES ； None。"""
    if smiles is None:
        return None

    s = str(smiles).strip()
    if s == "" or s.lower() == "nan":
        return None

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def _sanitize_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_name_key(value: Optional[str]) -> Optional[str]:
    text = _sanitize_text(value)
    if text is None:
        return None
    return " ".join(text.lower().split())


def _mol_from_smiles(smiles: Optional[str]) -> Optional[Chem.Mol]:
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    return Chem.MolFromSmiles(canonical)


def _to_float32_1d(x: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)





def morgan_fingerprint(
    smiles: Optional[str],
    radius: int = 4,
    n_bits: int = 256,
    use_chirality: bool = False,
) -> np.ndarray:
    """ SMILES  Morgan fingerprint，/ 0。"""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return np.zeros((int(n_bits),), dtype=np.float32)

    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
        includeChirality=bool(use_chirality),
    )
    fp = fpgen.GetFingerprint(mol)
    out = np.zeros((int(n_bits),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, out)
    return out.astype(np.float32, copy=False)





DEFAULT_RDKIT_DESCRIPTOR_NAMES = [
    name
    for name, fn in Descriptors._descList
    if callable(fn)
]


def available_rdkit_descriptor_names() -> list[str]:
    return list(DEFAULT_RDKIT_DESCRIPTOR_NAMES)


def rdkit_descriptors(
    smiles: Optional[str],
    descriptor_names: Optional[Sequence[str]] = None,
    nan_value: float = 0.0,
    inf_value: float = 0.0,
) -> np.ndarray:
    """ SMILES  RDKit ，/ 0。 NaN/Inf 。"""
    names = list(descriptor_names) if descriptor_names is not None else list(DEFAULT_RDKIT_DESCRIPTOR_NAMES)
    funcs = []
    for name in names:
        fn = getattr(Descriptors, str(name), None)
        if fn is None or not callable(fn):
            raise ValueError(f" RDKit descriptor: {name}")
        funcs.append(fn)

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return np.zeros((len(funcs),), dtype=np.float32)

    values: list[float] = []
    for fn in funcs:
        try:
            v = float(fn(mol))
        except Exception:
            v = float(nan_value)

        if np.isnan(v):
            v = float(nan_value)
        elif np.isposinf(v) or np.isneginf(v):
            v = float(inf_value)

        if not np.isfinite(v):
            v = float(nan_value)

        values.append(v)

    return _to_float32_1d(values)





@dataclass
class CGSDTable:
    by_smiles: Dict[str, np.ndarray]
    by_name: Dict[str, np.ndarray]

    def __post_init__(self) -> None:
        norm_smiles: Dict[str, np.ndarray] = {}
        for key, value in self.by_smiles.items():
            canon = canonicalize_smiles(key)
            if canon is None:
                continue
            arr = _to_float32_1d(value)
            if arr.shape[0] != len(CGSD_NAMES):
                raise ValueError(
                    f"CGSD  {len(CGSD_NAMES)}， {arr.shape[0]}，smiles={key!r}"
                )
            norm_smiles[canon] = arr

        norm_name: Dict[str, np.ndarray] = {}
        for key, value in self.by_name.items():
            nk = _normalize_name_key(key)
            if nk is None:
                continue
            arr = _to_float32_1d(value)
            if arr.shape[0] != len(CGSD_NAMES):
                raise ValueError(
                    f"CGSD  {len(CGSD_NAMES)}， {arr.shape[0]}，name={key!r}"
                )
            norm_name[nk] = arr

        self.by_smiles = norm_smiles
        self.by_name = norm_name

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Sequence[float] | np.ndarray]) -> "CGSDTable":
        by_smiles: Dict[str, np.ndarray] = {}
        by_name: Dict[str, np.ndarray] = {}
        for key, value in mapping.items():
            arr = _to_float32_1d(value)
            if canonicalize_smiles(key) is not None:
                by_smiles[key] = arr
            by_name[key] = arr
        return cls(by_smiles=by_smiles, by_name=by_name)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        solvent_col: str = "solvent",
        value_cols: Sequence[str] = CGSD_NAMES,
    ) -> "CGSDTable":
        df = pd.read_csv(csv_path)
        if solvent_col not in df.columns:
            raise ValueError(f"CGSD CSV : {solvent_col!r}")
        miss = [c for c in value_cols if c not in df.columns]
        if miss:
            raise ValueError(f"CGSD CSV : {miss}")

        by_smiles: Dict[str, np.ndarray] = {}
        by_name: Dict[str, np.ndarray] = {}
        for _, row in df.iterrows():
            solvent = row[solvent_col]
            vals = np.asarray([row[c] for c in value_cols], dtype=np.float32)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            name_key = _normalize_name_key(solvent)
            if name_key is not None:
                by_name[name_key] = vals

            canon = canonicalize_smiles(solvent)
            if canon is not None:
                by_smiles[canon] = vals

        return cls(by_smiles=by_smiles, by_name=by_name)

    def get(
        self,
        smiles: Optional[str] = None,
        *,
        solvent_name: Optional[str] = None,
        on_missing: str = "raise",
    ) -> np.ndarray:
        name_key = _normalize_name_key(solvent_name)
        if name_key is not None and name_key in self.by_name:
            return self.by_name[name_key].copy()

        canon = canonicalize_smiles(smiles)
        if canon is not None and canon in self.by_smiles:
            return self.by_smiles[canon].copy()

        if on_missing == "zeros":
            return np.zeros((len(CGSD_NAMES),), dtype=np.float32)

        raise KeyError(
            f"CGSD : solvent_name={solvent_name!r}, smiles={smiles!r}"
        )





def _normalize_include(include: str | Sequence[str] | None, solvent_mode: Optional[str] = None) -> list[str]:
    source = include if include is not None else solvent_mode
    if source is None:
        raise ValueError(" include  solvent_mode")

    if isinstance(source, str):
        text = source.strip().lower()
        if not text:
            raise ValueError("solvent_mode/include ")
        parts = [p.strip() for p in text.split("+") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in source if str(p).strip()]

    alias_map = {
        "morganfp": "morgan",
        "morgan_fingerprint": "morgan",
        "morganfingerprint": "morgan",
    }
    parts = [alias_map.get(p, p) for p in parts]

    valid = {"morgan", "rdkit", "cgsd"}
    bad = [p for p in parts if p not in valid]
    if bad:
        raise ValueError(f" solvent block: {bad}； morgan / rdkit / cgsd")

    out: list[str] = []
    seen = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    if not out:
        raise ValueError("include ")
    return out


def encode_solvent_feature(
    smiles: Optional[str],
    mode: str,
    *,
    solvent_name: Optional[str] = None,
    cgsd_table: Optional[CGSDTable] = None,
    cgsd_on_missing: str = "raise",
    descriptor_names: Optional[Sequence[str]] = None,
    rdkit_nan_value: float = 0.0,
    rdkit_inf_value: float = 0.0,
    morgan_radius: int = 4,
    morgan_n_bits: int = 256,
    morgan_use_chirality: bool = False,
) -> np.ndarray:
    text = str(mode).strip().lower()
    if text in {"morganfp", "morgan_fingerprint", "morganfingerprint"}:
        text = "morgan"
    if text == "morgan":
        return morgan_fingerprint(
            smiles,
            radius=morgan_radius,
            n_bits=morgan_n_bits,
            use_chirality=morgan_use_chirality,
        )
    if text == "rdkit":
        return rdkit_descriptors(
            smiles,
            descriptor_names=descriptor_names,
            nan_value=rdkit_nan_value,
            inf_value=rdkit_inf_value,
        )
    if text == "cgsd":
        if cgsd_table is None:
            raise ValueError("mode='cgsd'  cgsd_table")
        return cgsd_table.get(smiles=smiles, solvent_name=solvent_name, on_missing=cgsd_on_missing)
    raise ValueError(f" mode: {mode!r}")


def encode_and_concat_solvent_features(
    smiles: Optional[str] = None,
    solvent_mode: Optional[str] = None,
    *,
    solvent_name: Optional[str] = None,
    include: str | Sequence[str] | None = None,
    cgsd_table: Optional[CGSDTable] = None,
    morgan_kwargs: Optional[Mapping[str, object]] = None,
    rdkit_kwargs: Optional[Mapping[str, object]] = None,
    cgsd_kwargs: Optional[Mapping[str, object]] = None,
    cgsd_on_missing: str = "raise",
    descriptor_names: Optional[Sequence[str]] = None,
    rdkit_nan_value: float = 0.0,
    rdkit_inf_value: float = 0.0,
    morgan_radius: int = 4,
    morgan_n_bits: int = 256,
    morgan_use_chirality: bool = False,
) -> torch.Tensor:
    """
    ：
    1) encode_and_concat_solvent_features(smiles, solvent_mode="morgan+cgsd", ...)
    2) encode_and_concat_solvent_features(smiles=..., solvent_name=..., include=[...], morgan_kwargs=..., ...)
     torch.float32 ， data_loading.py 。
    """
    parts = _normalize_include(include=include, solvent_mode=solvent_mode)

    mk = dict(morgan_kwargs or {})
    rk = dict(rdkit_kwargs or {})
    ck = dict(cgsd_kwargs or {})

    
    if "radius" not in mk:
        mk["radius"] = morgan_radius
    if "n_bits" not in mk:
        mk["n_bits"] = morgan_n_bits
    if "use_chirality" not in mk:
        mk["use_chirality"] = morgan_use_chirality
    if "descriptor_names" not in rk:
        rk["descriptor_names"] = descriptor_names
    if "on_missing" not in ck:
        ck["on_missing"] = cgsd_on_missing

    feats: list[np.ndarray] = []
    for part in parts:
        if part == "morgan":
            feats.append(
                morgan_fingerprint(
                    smiles,
                    radius=int(mk.get("radius", 4)),
                    n_bits=int(mk.get("n_bits", 256)),
                    use_chirality=bool(mk.get("use_chirality", False)),
                )
            )
        elif part == "rdkit":
            feats.append(
                rdkit_descriptors(
                    smiles,
                    descriptor_names=rk.get("descriptor_names", None),
                    nan_value=float(rk.get("nan_value", 0.0)),
                    inf_value=float(rk.get("inf_value", 0.0)),
                )
            )
        elif part == "cgsd":
            if cgsd_table is None:
                raise ValueError("include/solvent_mode  'cgsd' ， cgsd_table")
            feats.append(
                cgsd_table.get(
                    smiles=smiles,
                    solvent_name=solvent_name,
                    on_missing=str(ck.get("on_missing", "raise")),
                )
            )
        else:
            raise ValueError(f" solvent block: {part}")

    if len(feats) == 1:
        arr = feats[0].astype(np.float32, copy=False)
    else:
        arr = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(arr)


def get_solvent_feature_dim(
    solvent_mode: str | Sequence[str],
    *,
    descriptor_names: Optional[Sequence[str]] = None,
    morgan_n_bits: int = 256,
) -> int:
    parts = _normalize_include(include=solvent_mode)
    dim = 0
    for part in parts:
        if part == "morgan":
            dim += int(morgan_n_bits)
        elif part == "rdkit":
            dim += len(list(descriptor_names) if descriptor_names is not None else DEFAULT_RDKIT_DESCRIPTOR_NAMES)
        elif part == "cgsd":
            dim += len(CGSD_NAMES)
        else:
            raise ValueError(f" mode: {part}")
    return int(dim)


__all__ = [
    "CGSD_NAMES",
    "CGSDTable",
    "DEFAULT_RDKIT_DESCRIPTOR_NAMES",
    "available_rdkit_descriptor_names",
    "canonicalize_smiles",
    "encode_and_concat_solvent_features",
    "encode_solvent_feature",
    "get_solvent_feature_dim",
    "morgan_fingerprint",
    "rdkit_descriptors",
]
