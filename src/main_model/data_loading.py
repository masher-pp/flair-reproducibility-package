from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from solvent_encoders import CGSDTable, canonicalize_smiles, encode_and_concat_solvent_features


TARGET_COLS = ["abs", "emi", "plqy", "em"]
SMILES_COL = "smiles"
SOLVENT_COL = "solvent"              
SOLVENT_RAW_COL = "solvent_raw"      
SAMPLE_ID_COL = "sample_id"
RAW_ROW_ID_COL = "raw_row_id"
MANIFEST_REQUIRED_COLS = [
    SAMPLE_ID_COL,
    RAW_ROW_ID_COL,
    SMILES_COL,
    SOLVENT_COL,
    SOLVENT_RAW_COL,
] + TARGET_COLS





ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATION_TYPES = [
    int(Chem.rdchem.HybridizationType.SP),
    int(Chem.rdchem.HybridizationType.SP2),
    int(Chem.rdchem.HybridizationType.SP3),
]

ATOM_FDIM = (
    (len(ATOM_TYPES) + 1)
    + (len(ATOM_DEGREES) + 1)
    + (len(FORMAL_CHARGES) + 1)
    + (len(HYBRIDIZATION_TYPES) + 1)
    + 2
)
BOND_FDIM = 6


class MolSolventData(Data):
    """
     PyG Data。

    ：
    1) ：x / edge_index / edge_attr
    2)  MORE ：more_x / more_edge_index / more_edge_attr
    3) ：solvent_feat / solv_cond
    4) ：solvent_x / solvent_edge_index / solvent_edge_attr
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "solvent_edge_index":
            if hasattr(self, "solvent_x") and isinstance(self.solvent_x, torch.Tensor):
                return int(self.solvent_x.size(0))
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "solvent_edge_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def one_hot_with_unk(value, choices: list[int]) -> list[float]:
    out = [0.0] * (len(choices) + 1)
    if value in choices:
        out[choices.index(value)] = 1.0
    else:
        out[-1] = 1.0
    return out


def atom_features(atom: Chem.Atom) -> list[float]:
    feats: list[float] = []
    feats += one_hot_with_unk(atom.GetAtomicNum(), ATOM_TYPES)
    feats += one_hot_with_unk(atom.GetDegree(), ATOM_DEGREES)
    feats += one_hot_with_unk(atom.GetFormalCharge(), FORMAL_CHARGES)
    feats += one_hot_with_unk(int(atom.GetHybridization()), HYBRIDIZATION_TYPES)
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))
    return feats


def bond_features(bond: Chem.Bond) -> list[float]:
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def make_dummy_graph() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.zeros((1, ATOM_FDIM), dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float32)
    return x, edge_index, edge_attr


def smiles_to_graph(smiles: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    canonical = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical) if canonical is not None else None
    if mol is None:
        return make_dummy_graph()

    x_list = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(x_list, dtype=torch.float32)

    edge_index_list: list[list[int]] = []
    edge_attr_list: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_attr_list.append(bf)
        edge_attr_list.append(bf)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float32)

    return x, edge_index, edge_attr








MORE_ALLOWABLE_FEATURES = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
        Chem.rdchem.ChiralType.CHI_ALLENE,
        Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,
        Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
    ],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        Chem.rdchem.BondType.UNSPECIFIED,
        Chem.rdchem.BondType.QUADRUPLE,
        Chem.rdchem.BondType.QUINTUPLE,
        Chem.rdchem.BondType.HEXTUPLE,
        Chem.rdchem.BondType.ONEANDAHALF,
        Chem.rdchem.BondType.TWOANDAHALF,
        Chem.rdchem.BondType.THREEANDAHALF,
        Chem.rdchem.BondType.FOURANDAHALF,
        Chem.rdchem.BondType.FIVEANDAHALF,
        Chem.rdchem.BondType.IONIC,
        Chem.rdchem.BondType.HYDROGEN,
        Chem.rdchem.BondType.THREECENTER,
        Chem.rdchem.BondType.DATIVEONE,
        Chem.rdchem.BondType.DATIVE,
        Chem.rdchem.BondType.DATIVEL,
        Chem.rdchem.BondType.DATIVER,
        Chem.rdchem.BondType.OTHER,
        Chem.rdchem.BondType.ZERO,
    ],
    "possible_bond_dirs": [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.BEGINDASH,
        Chem.rdchem.BondDir.BEGINWEDGE,
        Chem.rdchem.BondDir.EITHERDOUBLE,
        Chem.rdchem.BondDir.UNKNOWN,
    ],
}


def _safe_index(values: list, item, default: int = 0) -> int:
    try:
        return values.index(item)
    except ValueError:
        return int(default)


def make_more_dummy_graph() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.zeros((1, 2), dtype=torch.long)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 2), dtype=torch.long)
    return x, edge_index, edge_attr


def smiles_to_more_graph(smiles: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    canonical = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical) if canonical is not None else None
    if mol is None:
        return make_more_dummy_graph()

    atom_features_list = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atomic_num_idx = _safe_index(
            MORE_ALLOWABLE_FEATURES["possible_atomic_num_list"],
            atomic_num,
            default=0,
        )
        chirality_idx = _safe_index(
            MORE_ALLOWABLE_FEATURES["possible_chirality_list"],
            atom.GetChiralTag(),
            default=0,
        )
        atom_features_list.append([atomic_num_idx, chirality_idx])

    if len(atom_features_list) == 0:
        return make_more_dummy_graph()

    x = torch.tensor(np.asarray(atom_features_list), dtype=torch.long)

    edge_index_list: list[tuple[int, int]] = []
    edge_attr_list: list[list[int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type_idx = _safe_index(
            MORE_ALLOWABLE_FEATURES["possible_bonds"],
            bond.GetBondType(),
            default=4,
        )
        bond_dir_idx = _safe_index(
            MORE_ALLOWABLE_FEATURES["possible_bond_dirs"],
            bond.GetBondDir(),
            default=0,
        )
        edge_feature = [bond_type_idx, bond_dir_idx]

        edge_index_list.append((i, j))
        edge_index_list.append((j, i))
        edge_attr_list.append(edge_feature)
        edge_attr_list.append(edge_feature)

    if edge_index_list:
        edge_index = torch.tensor(np.asarray(edge_index_list).T, dtype=torch.long).contiguous()
        edge_attr = torch.tensor(np.asarray(edge_attr_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return x, edge_index, edge_attr


def _build_more_graph_cache(smiles_series: pd.Series) -> Dict[Optional[str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    cache: Dict[Optional[str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    uniq_smiles: list[Optional[str]] = []
    seen = set()
    for value in smiles_series.tolist():
        smi = _sanitize_smiles_value(value)
        if smi not in seen:
            seen.add(smi)
            uniq_smiles.append(smi)

    for smi in uniq_smiles:
        cache[smi] = smiles_to_more_graph(smi)
    return cache





def _is_missing_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _sanitize_text_value(value) -> Optional[str]:
    if _is_missing_value(value):
        return None
    text = str(value).strip()
    return text if text else None


def _safe_canonicalize_smiles(value) -> Optional[str]:
    text = _sanitize_text_value(value)
    if text is None:
        return None
    try:
        return canonicalize_smiles(text)
    except Exception:
        return None


def _sanitize_smiles_value(value) -> Optional[str]:
    return _safe_canonicalize_smiles(value)


def _canonicalize_smiles_series_with_stats(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    out: list[Optional[str]] = []
    missing_count = 0
    invalid_count = 0
    valid_count = 0

    for value in series.tolist():
        text = _sanitize_text_value(value)
        if text is None:
            out.append(None)
            missing_count += 1
            continue
        try:
            canonical = canonicalize_smiles(text)
        except Exception:
            canonical = None
        if canonical is None:
            out.append(None)
            invalid_count += 1
        else:
            out.append(canonical)
            valid_count += 1

    return pd.Series(out, index=series.index, dtype=object), {
        "valid_count": int(valid_count),
        "invalid_count": int(invalid_count),
        "missing_count": int(missing_count),
        "skipped_count": int(invalid_count + missing_count),
    }


def _clone_graph_tuple(
    graph_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return graph_tuple


def _build_graph_cache(smiles_series: pd.Series) -> Dict[Optional[str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    cache: Dict[Optional[str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    uniq_smiles: list[Optional[str]] = []
    seen = set()
    for value in smiles_series.tolist():
        smi = _sanitize_smiles_value(value)
        if smi not in seen:
            seen.add(smi)
            uniq_smiles.append(smi)

    for smi in uniq_smiles:
        cache[smi] = smiles_to_graph(smi)
    return cache


def _normalize_solvent_blocks(solvent_mode: str | Sequence[str]) -> list[str]:
    if isinstance(solvent_mode, str):
        raw = str(solvent_mode).replace(",", "+").split("+")
        blocks = [x.strip().lower() for x in raw if x.strip()]
    else:
        blocks = [str(x).strip().lower() for x in solvent_mode if str(x).strip()]

    if len(blocks) == 0:
        raise ValueError("solvent_mode ")

    alias_map = {
        "morganfp": "morgan",
        "morgan_fingerprint": "morgan",
        "morganfingerprint": "morgan",
    }
    blocks = [alias_map.get(x, x) for x in blocks]

    allowed = {"morgan", "rdkit", "cgsd"}
    unknown = [x for x in blocks if x not in allowed]
    if unknown:
        raise ValueError(f" solvent_mode: {unknown}； {sorted(allowed)} ")

    out: list[str] = []
    seen = set()
    for x in blocks:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _solvent_mode_requires_smiles(solvent_blocks: Sequence[str]) -> bool:
    return any(block in {"morgan", "rdkit"} for block in solvent_blocks)


def _strict_smiles_solvent_required(
    solvent_blocks: Sequence[str],
    drop_invalid_solvent_in_graph: bool,
) -> bool:
    return bool(_solvent_mode_requires_smiles(solvent_blocks) and drop_invalid_solvent_in_graph)


def _build_solvent_feature_cache(
    solvent_smiles_series: pd.Series,
    solvent_raw_series: pd.Series,
    *,
    solvent_blocks: Sequence[str],
    cgsd_table: Optional[CGSDTable] = None,
    morgan_kwargs: Optional[Mapping[str, Any]] = None,
    rdkit_kwargs: Optional[Mapping[str, Any]] = None,
    cgsd_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[tuple[Optional[str], Optional[str]], torch.Tensor]:
    cache: Dict[tuple[Optional[str], Optional[str]], torch.Tensor] = {}
    uniq_keys: list[tuple[Optional[str], Optional[str]]] = []
    seen = set()

    for s_smiles, s_raw in zip(solvent_smiles_series.tolist(), solvent_raw_series.tolist()):
        key = (_sanitize_smiles_value(s_smiles), _sanitize_text_value(s_raw))
        if key not in seen:
            seen.add(key)
            uniq_keys.append(key)

    for sol_smiles, sol_raw in uniq_keys:
        cache[(sol_smiles, sol_raw)] = encode_and_concat_solvent_features(
            smiles=sol_smiles,
            solvent_name=sol_raw,
            include=solvent_blocks,
            cgsd_table=cgsd_table,
            morgan_kwargs=dict(morgan_kwargs or {}),
            rdkit_kwargs=dict(rdkit_kwargs or {}),
            cgsd_kwargs=dict(cgsd_kwargs or {}),
        )

    return cache


def _pin_memory_enabled(pin_memory: Optional[bool]) -> bool:
    if pin_memory is None:
        return torch.cuda.is_available()
    return bool(pin_memory)


def _persistent_workers_enabled(num_workers: int, persistent_workers: Optional[bool]) -> bool:
    if int(num_workers) <= 0:
        return False
    if persistent_workers is None:
        return True
    return bool(persistent_workers)


def _resolve_cgsd_table(
    cgsd_table: Optional[CGSDTable] = None,
    cgsd_csv_path: Optional[str] = None,
) -> Optional[CGSDTable]:
    if cgsd_table is not None:
        return cgsd_table
    if cgsd_csv_path is not None:
        return CGSDTable.from_csv(cgsd_csv_path)
    return None





def compute_target_stats(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    values = df[TARGET_COLS].to_numpy(dtype=np.float32)
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)

    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)
    return mean, std


def standardize_targets(y_np: np.ndarray, mask_np: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = y_np.astype(np.float32, copy=True)
    valid = mask_np.astype(bool)
    if valid.any():
        out[valid] = (out[valid] - mean[valid]) / std[valid]
    return out





def build_manifest_df(
    csv_path: str,
    solvent_mode: str | Sequence[str] = "morgan",
    drop_invalid_solvent_in_graph: bool = False,
    cv_fold: Optional[int] = None,
    fold_role: Optional[str] = None,
) -> pd.DataFrame:
    solvent_blocks = _normalize_solvent_blocks(solvent_mode)

    df = pd.read_csv(csv_path)
    if cv_fold is not None:
        fold = int(cv_fold)
        role = str(fold_role or "").strip().lower()
        if "cv_fold" not in df.columns:
            raise ValueError(f"CSV {csv_path} is missing the cv_fold column required for fold selection.")
        if role not in {"train", "valid", "validation"}:
            raise ValueError(f"fold_role must be 'train' or 'valid', got {fold_role!r}.")
        fold_values = pd.to_numeric(df["cv_fold"], errors="raise").astype(int)
        mask = fold_values.ne(fold) if role == "train" else fold_values.eq(fold)
        df = df.loc[mask].copy().reset_index(drop=True)
        if "split" in df.columns:
            df["split"] = "train" if role == "train" else "valid"
    required = TARGET_COLS + [SMILES_COL, SOLVENT_COL]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"CSV: {miss}；={list(df.columns)}")

    df = df.copy()
    total_rows_before = int(len(df))
    df[RAW_ROW_ID_COL] = np.arange(len(df), dtype=np.int64)
    df[SMILES_COL], solute_stats = _canonicalize_smiles_series_with_stats(df[SMILES_COL])

    raw_solvent = df[SOLVENT_COL].copy()
    df[SOLVENT_RAW_COL] = raw_solvent.map(_sanitize_text_value)
    df[SOLVENT_COL], solvent_stats = _canonicalize_smiles_series_with_stats(raw_solvent)

    before_drop_solute = len(df)
    df = df.dropna(subset=[SMILES_COL]).reset_index(drop=True)
    dropped_solute_rows = int(before_drop_solute - len(df))

    dropped_solvent_rows = 0
    solvent_smiles_required = _strict_smiles_solvent_required(
        solvent_blocks=solvent_blocks,
        drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
    )

    if (not solvent_smiles_required) and _solvent_mode_requires_smiles(solvent_blocks):
        unresolved = int(solvent_stats["invalid_count"] + solvent_stats["missing_count"])
        if unresolved > 0:
            raise ValueError(
                " solvent_mode  solvent SMILES（morgan/rdkit），"
                f" {csv_path}  {unresolved} / solvent。"
                " 0 ， solvent，"
                " drop_invalid_solvent_in_graph  True。"
            )

    if solvent_smiles_required:
        before_drop_solvent = len(df)
        df = df.dropna(subset=[SOLVENT_COL]).reset_index(drop=True)
        dropped_solvent_rows = int(before_drop_solvent - len(df))

    if len(df) == 0:
        raise ValueError(" SMILES ，，")

    df[SAMPLE_ID_COL] = np.arange(len(df), dtype=np.int64)

    ordered_cols = [
        SAMPLE_ID_COL,
        RAW_ROW_ID_COL,
        SMILES_COL,
        SOLVENT_COL,
        SOLVENT_RAW_COL,
    ] + TARGET_COLS
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra_cols].reset_index(drop=True)

    invalid_smiles_stats = {
        "csv_path": str(csv_path),
        "solvent_mode": "+".join(solvent_blocks),
        "total_rows_before": int(total_rows_before),
        "kept_rows": int(len(df)),
        "dropped_total_rows": int(total_rows_before - len(df)),
        "solute_skipped_rows": int(dropped_solute_rows),
        "solute_invalid_count": int(solute_stats["invalid_count"]),
        "solute_missing_count": int(solute_stats["missing_count"]),
        "solvent_smiles_required": int(solvent_smiles_required),
        "solvent_skipped_rows": int(dropped_solvent_rows),
        "solvent_invalid_count": int(solvent_stats["invalid_count"]),
        "solvent_missing_count": int(solvent_stats["missing_count"]),
    }
    df.attrs["invalid_smiles_stats"] = invalid_smiles_stats

    print(
        f"[SMILES Check] {csv_path} |  {invalid_smiles_stats['solute_skipped_rows']}  "
        f"( {invalid_smiles_stats['solute_invalid_count']} /  {invalid_smiles_stats['solute_missing_count']})"
        + (
            f" |  {invalid_smiles_stats['solvent_skipped_rows']}  "
            f"( {invalid_smiles_stats['solvent_invalid_count']} /  {invalid_smiles_stats['solvent_missing_count']})"
            if solvent_smiles_required else " |  SMILES "
        )
        + f" |  {invalid_smiles_stats['kept_rows']}/{invalid_smiles_stats['total_rows_before']} "
    )
    return df


def save_manifest(df: pd.DataFrame, manifest_path: str) -> None:
    missing = [c for c in MANIFEST_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest : {missing}")
    df.to_csv(manifest_path, index=False)


def load_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if SOLVENT_RAW_COL not in df.columns:
        df[SOLVENT_RAW_COL] = df[SOLVENT_COL].map(_sanitize_text_value)

    missing = [c for c in MANIFEST_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest : {missing}")
    return df.reset_index(drop=True)


def build_split_from_manifest(
    manifest_df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= float(val_ratio) < 1.0):
        raise ValueError(f"val_ratio  0 <= val_ratio < 1， {val_ratio}")

    n = len(manifest_df)
    if n < 2:
        raise ValueError(f"（n={n}）， 2  train/val")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    n_val = int(round(n * val_ratio))
    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if n_val >= n:
        n_val = n - 1

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"：train={len(train_idx)}, val={len(val_idx)}；"
            " val_ratio "
        )

    sample_ids = manifest_df[SAMPLE_ID_COL].to_numpy(dtype=np.int64)
    train_sample_ids = sample_ids[train_idx]
    val_sample_ids = sample_ids[val_idx]
    return train_sample_ids, val_sample_ids


def save_split(train_sample_ids: np.ndarray, val_sample_ids: np.ndarray, split_path: str) -> None:
    np.savez(
        split_path,
        train_sample_ids=np.asarray(train_sample_ids, dtype=np.int64),
        val_sample_ids=np.asarray(val_sample_ids, dtype=np.int64),
    )


def load_split(split_path: str) -> Tuple[np.ndarray, np.ndarray]:
    obj = np.load(split_path)
    if "train_sample_ids" not in obj or "val_sample_ids" not in obj:
        raise ValueError("split  train_sample_ids  val_sample_ids")
    train_sample_ids = np.asarray(obj["train_sample_ids"], dtype=np.int64)
    val_sample_ids = np.asarray(obj["val_sample_ids"], dtype=np.int64)
    return train_sample_ids, val_sample_ids


def prepare_reproducible_protocol(
    csv_path: str,
    manifest_path: str,
    split_path: str,
    solvent_mode: str | Sequence[str],
    val_ratio: float = 0.15,
    seed: int = 42,
    drop_invalid_solvent_in_graph: bool = False,
    overwrite_manifest: bool = False,
    overwrite_split: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    manifest_exists = pd.io.common.file_exists(manifest_path)
    split_exists = pd.io.common.file_exists(split_path)

    if (not manifest_exists) or overwrite_manifest:
        manifest_df = build_manifest_df(
            csv_path=csv_path,
            solvent_mode=solvent_mode,
            drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
        )
        save_manifest(manifest_df, manifest_path)
    else:
        manifest_df = load_manifest(manifest_path)

    if (not split_exists) or overwrite_split:
        train_sample_ids, val_sample_ids = build_split_from_manifest(
            manifest_df=manifest_df,
            val_ratio=val_ratio,
            seed=seed,
        )
        save_split(train_sample_ids, val_sample_ids, split_path)
    else:
        train_sample_ids, val_sample_ids = load_split(split_path)

    return manifest_df, train_sample_ids, val_sample_ids


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FluorSolventDataset(Dataset):
    """
     PyG Data：
      - : x / edge_index / edge_attr
      - : y (4), y_mask (4，1=, 0=)
      - :
          solvent_feat: [1, D]， MLP 
          solv_cond:     solvent_feat ，
          solvent_x / solvent_edge_index / solvent_edge_attr：， GAT / Transformer 
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        solvent_mode: str | Sequence[str] = "morgan",
        solvent_vocab: Optional[Dict[str, int]] = None,
        unk_token: str = "<UNK>",
        drop_invalid_solvent_in_graph: bool = False,
        df: Optional[pd.DataFrame] = None,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
        cache_graphs: bool = True,
        *,
        cgsd_table: Optional[CGSDTable] = None,
        cgsd_csv_path: Optional[str] = None,
        morgan_kwargs: Optional[Mapping[str, Any]] = None,
        rdkit_kwargs: Optional[Mapping[str, Any]] = None,
        cgsd_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        del solvent_vocab, unk_token

        self.solvent_blocks = _normalize_solvent_blocks(solvent_mode)
        self.drop_invalid_solvent_in_graph = bool(drop_invalid_solvent_in_graph)
        self.cache_graphs = bool(cache_graphs)

        if df is None:
            if csv_path is None:
                raise ValueError("csv_path  df  None")
            df = build_manifest_df(
                csv_path=csv_path,
                solvent_mode=self.solvent_blocks,
                drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
            )
        else:
            df = df.copy().reset_index(drop=True)
            if SOLVENT_RAW_COL not in df.columns:
                df[SOLVENT_RAW_COL] = df[SOLVENT_COL].map(_sanitize_text_value)
            missing = [c for c in MANIFEST_REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f" manifest DataFrame : {missing}")

        if len(df) == 0:
            raise ValueError("，")

        self.df = df.reset_index(drop=True)
        self.solvent_mode = "+".join(self.solvent_blocks)
        self.solvent_feature_dim: Optional[int] = None

        if target_mean is None or target_std is None:
            target_mean, target_std = compute_target_stats(self.df)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        if self.target_mean.shape != (len(TARGET_COLS),) or self.target_std.shape != (len(TARGET_COLS),):
            raise ValueError("target_mean / target_std  (4,)")

        self.cgsd_table = _resolve_cgsd_table(cgsd_table=cgsd_table, cgsd_csv_path=cgsd_csv_path)
        self.morgan_kwargs = dict(morgan_kwargs or {})
        self.rdkit_kwargs = dict(rdkit_kwargs or {})
        self.cgsd_kwargs = dict(cgsd_kwargs or {})

        if "cgsd" in self.solvent_blocks and self.cgsd_table is None:
            raise ValueError("solvent_mode  'cgsd' ， cgsd_table  cgsd_csv_path")

        self.mol_graph_cache: Optional[Dict[Optional[str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
        self.more_graph_cache: Optional[Dict[Optional[str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
        self.solvent_graph_cache: Optional[Dict[Optional[str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
        self.solvent_feature_cache: Optional[Dict[tuple[Optional[str], Optional[str]], torch.Tensor]] = None

        if self.cache_graphs:
            self.mol_graph_cache = _build_graph_cache(self.df[SMILES_COL])
            self.more_graph_cache = _build_more_graph_cache(self.df[SMILES_COL])
            self.solvent_graph_cache = _build_graph_cache(self.df[SOLVENT_COL])
            self.solvent_feature_cache = _build_solvent_feature_cache(
                self.df[SOLVENT_COL],
                self.df[SOLVENT_RAW_COL],
                solvent_blocks=self.solvent_blocks,
                cgsd_table=self.cgsd_table,
                morgan_kwargs=self.morgan_kwargs,
                rdkit_kwargs=self.rdkit_kwargs,
                cgsd_kwargs=self.cgsd_kwargs,
            )
            if len(self.solvent_feature_cache) > 0:
                self.solvent_feature_dim = int(next(iter(self.solvent_feature_cache.values())).numel())

    def __len__(self) -> int:
        return len(self.df)

    def _get_mol_graph(self, mol_smiles: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mol_smiles = _sanitize_smiles_value(mol_smiles)
        if self.mol_graph_cache is None:
            return smiles_to_graph(mol_smiles)
        graph = self.mol_graph_cache.get(mol_smiles)
        if graph is None:
            graph = smiles_to_graph(mol_smiles)
            self.mol_graph_cache[mol_smiles] = graph
        return _clone_graph_tuple(graph)

    def _get_more_graph(self, mol_smiles: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mol_smiles = _sanitize_smiles_value(mol_smiles)
        if self.more_graph_cache is None:
            return smiles_to_more_graph(mol_smiles)
        graph = self.more_graph_cache.get(mol_smiles)
        if graph is None:
            graph = smiles_to_more_graph(mol_smiles)
            self.more_graph_cache[mol_smiles] = graph
        return _clone_graph_tuple(graph)

    def _get_solvent_features(self, sol_smiles: Optional[str], sol_raw: Optional[str]) -> torch.Tensor:
        key = (_sanitize_smiles_value(sol_smiles), _sanitize_text_value(sol_raw))
        if self.solvent_feature_cache is None:
            feat = encode_and_concat_solvent_features(
                smiles=key[0],
                solvent_name=key[1],
                include=self.solvent_blocks,
                cgsd_table=self.cgsd_table,
                morgan_kwargs=self.morgan_kwargs,
                rdkit_kwargs=self.rdkit_kwargs,
                cgsd_kwargs=self.cgsd_kwargs,
            )
        else:
            feat = self.solvent_feature_cache.get(key)
            if feat is None:
                feat = encode_and_concat_solvent_features(
                    smiles=key[0],
                    solvent_name=key[1],
                    include=self.solvent_blocks,
                    cgsd_table=self.cgsd_table,
                    morgan_kwargs=self.morgan_kwargs,
                    rdkit_kwargs=self.rdkit_kwargs,
                    cgsd_kwargs=self.cgsd_kwargs,
                )
                self.solvent_feature_cache[key] = feat

        if self.solvent_feature_dim is None:
            self.solvent_feature_dim = int(feat.numel())
        return feat

    def _get_solvent_graph(
        self,
        sol_smiles: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sol_smiles = _sanitize_smiles_value(sol_smiles)
        if self.solvent_graph_cache is None:
            return smiles_to_graph(sol_smiles)
        graph = self.solvent_graph_cache.get(sol_smiles)
        if graph is None:
            graph = smiles_to_graph(sol_smiles)
            self.solvent_graph_cache[sol_smiles] = graph
        return _clone_graph_tuple(graph)

    def __getitem__(self, idx: int) -> Data:
        row = self.df.iloc[idx]

        mol_smiles = _sanitize_smiles_value(row[SMILES_COL])
        sol_smiles = _sanitize_smiles_value(row[SOLVENT_COL])
        sol_raw = _sanitize_text_value(row[SOLVENT_RAW_COL])

        y_np = row[TARGET_COLS].to_numpy(dtype=np.float32)
        mask_np = ~np.isnan(y_np)
        y_np = np.nan_to_num(y_np, nan=0.0)
        y_np = standardize_targets(y_np, mask_np, self.target_mean, self.target_std)

        y = torch.tensor(y_np, dtype=torch.float32).view(1, -1)
        y_mask = torch.tensor(mask_np.astype(np.float32), dtype=torch.float32).view(1, -1)

        x, edge_index, edge_attr = self._get_mol_graph(mol_smiles)
        data = MolSolventData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            y_mask=y_mask,
            smiles=mol_smiles or "",
            solvent_smiles=sol_smiles or "",
            solvent_raw=sol_raw or "",
            sample_id=int(row[SAMPLE_ID_COL]),
            raw_row_id=int(row[RAW_ROW_ID_COL]),
        )

        more_x, more_edge_index, more_edge_attr = self._get_more_graph(mol_smiles)
        data.more_x = more_x
        data.more_edge_index = more_edge_index
        data.more_edge_attr = more_edge_attr

        solvent_feat = self._get_solvent_features(sol_smiles, sol_raw).view(1, -1)
        data.solvent_feat = solvent_feat
        data.solv_cond = solvent_feat

        solvent_x, solvent_edge_index, solvent_edge_attr = self._get_solvent_graph(sol_smiles)
        data.solvent_x = solvent_x
        data.solvent_edge_index = solvent_edge_index
        data.solvent_edge_attr = solvent_edge_attr

        return data


def _subset_indices_from_sample_ids(dataset_df: pd.DataFrame, sample_ids: np.ndarray) -> list[int]:
    id_to_idx = {int(sid): i for i, sid in enumerate(dataset_df[SAMPLE_ID_COL].tolist())}
    subset_idx = []
    missing_ids = []
    for sid in np.asarray(sample_ids, dtype=np.int64).tolist():
        if int(sid) not in id_to_idx:
            missing_ids.append(int(sid))
        else:
            subset_idx.append(id_to_idx[int(sid)])

    if missing_ids:
        preview = missing_ids[:10]
        raise ValueError(
            f"split  {len(missing_ids)}  sample_id  dataset ，"
            f": {preview}。 manifest  split 。"
        )
    return subset_idx


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    follow_batch,
    seed: int,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 2,
):
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        follow_batch=follow_batch,
        worker_init_fn=seed_worker,
        pin_memory=_pin_memory_enabled(pin_memory),
    )

    if shuffle:
        loader_kwargs["generator"] = torch.Generator().manual_seed(seed)

    persistent = _persistent_workers_enabled(num_workers, persistent_workers)
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = persistent
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return PyGDataLoader(**loader_kwargs)


def make_loaders(
    csv_path: Optional[str] = None,
    solvent_mode: str | Sequence[str] = "morgan",
    batch_size: int = 64,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    drop_invalid_solvent_in_graph: bool = False,
    manifest_path: Optional[str] = None,
    split_path: Optional[str] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 2,
    cache_graphs: bool = True,
    *,
    cgsd_table: Optional[CGSDTable] = None,
    cgsd_csv_path: Optional[str] = None,
    morgan_kwargs: Optional[Mapping[str, Any]] = None,
    rdkit_kwargs: Optional[Mapping[str, Any]] = None,
    cgsd_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[PyGDataLoader, PyGDataLoader, None]:
    if not (0.0 <= float(val_ratio) < 1.0):
        raise ValueError(f"val_ratio  0 <= val_ratio < 1， {val_ratio}")

    solvent_blocks = _normalize_solvent_blocks(solvent_mode)
    resolved_cgsd_table = _resolve_cgsd_table(cgsd_table=cgsd_table, cgsd_csv_path=cgsd_csv_path)

    if manifest_path is not None:
        manifest_df = load_manifest(manifest_path)
    else:
        if csv_path is None:
            raise ValueError("csv_path  manifest_path  None")
        manifest_df = build_manifest_df(
            csv_path=csv_path,
            solvent_mode=solvent_blocks,
            drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
        )

    n = len(manifest_df)
    if n < 2:
        raise ValueError(f"（n={n}）， 2  train/val")

    if split_path is not None:
        train_sample_ids, val_sample_ids = load_split(split_path)
        train_idx = _subset_indices_from_sample_ids(manifest_df, train_sample_ids)
        val_idx = _subset_indices_from_sample_ids(manifest_df, val_sample_ids)
    else:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()

        n_val = int(round(n * val_ratio))
        if val_ratio > 0 and n_val == 0:
            n_val = 1
        if n_val >= n:
            n_val = n - 1

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"：train={len(train_idx)}, val={len(val_idx)}；"
            " val_ratio "
        )

    train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
    val_df = manifest_df.iloc[val_idx].reset_index(drop=True)
    target_mean, target_std = compute_target_stats(train_df)

    train_ds = FluorSolventDataset(
        df=train_df,
        solvent_mode=solvent_blocks,
        drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
        target_mean=target_mean,
        target_std=target_std,
        cache_graphs=cache_graphs,
        cgsd_table=resolved_cgsd_table,
        morgan_kwargs=morgan_kwargs,
        rdkit_kwargs=rdkit_kwargs,
        cgsd_kwargs=cgsd_kwargs,
    )
    val_ds = FluorSolventDataset(
        df=val_df,
        solvent_mode=solvent_blocks,
        drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
        target_mean=target_mean,
        target_std=target_std,
        cache_graphs=cache_graphs,
        cgsd_table=resolved_cgsd_table,
        morgan_kwargs=morgan_kwargs,
        rdkit_kwargs=rdkit_kwargs,
        cgsd_kwargs=cgsd_kwargs,
    )

    train_loader = _make_loader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        follow_batch=["solvent_x"],
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _make_loader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        follow_batch=["solvent_x"],
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, None


def make_train_val_test_loaders(
    train_csv_path: str,
    val_csv_path: str,
    test_csv_path: str,
    solvent_mode: str | Sequence[str] = "morgan",
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    drop_invalid_solvent_in_graph: bool = False,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 2,
    cache_graphs: bool = True,
    *,
    cgsd_table: Optional[CGSDTable] = None,
    cgsd_csv_path: Optional[str] = None,
    morgan_kwargs: Optional[Mapping[str, Any]] = None,
    rdkit_kwargs: Optional[Mapping[str, Any]] = None,
    cgsd_kwargs: Optional[Mapping[str, Any]] = None,
    cv_fold: Optional[int] = None,
) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader, None]:
    solvent_blocks = _normalize_solvent_blocks(solvent_mode)
    resolved_cgsd_table = _resolve_cgsd_table(cgsd_table=cgsd_table, cgsd_csv_path=cgsd_csv_path)

    train_df = build_manifest_df(
        csv_path=train_csv_path,
        solvent_mode=solvent_blocks,
        drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
        cv_fold=cv_fold,
        fold_role="train" if cv_fold is not None else None,
    )
    val_df = build_manifest_df(
        csv_path=val_csv_path,
        solvent_mode=solvent_blocks,
        drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
        cv_fold=cv_fold,
        fold_role="valid" if cv_fold is not None else None,
    )
    test_df = build_manifest_df(
        csv_path=test_csv_path,
        solvent_mode=solvent_blocks,
        drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
    )

    target_mean, target_std = compute_target_stats(train_df)

    train_ds = FluorSolventDataset(
        df=train_df,
        solvent_mode=solvent_blocks,
        drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
        target_mean=target_mean,
        target_std=target_std,
        cache_graphs=cache_graphs,
        cgsd_table=resolved_cgsd_table,
        morgan_kwargs=morgan_kwargs,
        rdkit_kwargs=rdkit_kwargs,
        cgsd_kwargs=cgsd_kwargs,
    )
    val_ds = FluorSolventDataset(
        df=val_df,
        solvent_mode=solvent_blocks,
        drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
        target_mean=target_mean,
        target_std=target_std,
        cache_graphs=cache_graphs,
        cgsd_table=resolved_cgsd_table,
        morgan_kwargs=morgan_kwargs,
        rdkit_kwargs=rdkit_kwargs,
        cgsd_kwargs=cgsd_kwargs,
    )
    test_ds = FluorSolventDataset(
        df=test_df,
        solvent_mode=solvent_blocks,
        drop_invalid_solvent_in_graph=drop_invalid_solvent_in_graph,
        target_mean=target_mean,
        target_std=target_std,
        cache_graphs=cache_graphs,
        cgsd_table=resolved_cgsd_table,
        morgan_kwargs=morgan_kwargs,
        rdkit_kwargs=rdkit_kwargs,
        cgsd_kwargs=cgsd_kwargs,
    )
    train_ds.invalid_smiles_stats = dict(train_df.attrs.get("invalid_smiles_stats", {}))
    val_ds.invalid_smiles_stats = dict(val_df.attrs.get("invalid_smiles_stats", {}))
    test_ds.invalid_smiles_stats = dict(test_df.attrs.get("invalid_smiles_stats", {}))

    train_loader = _make_loader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        follow_batch=["solvent_x"],
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _make_loader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        follow_batch=["solvent_x"],
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _make_loader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        follow_batch=["solvent_x"],
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, test_loader, None


__all__ = [
    "ATOM_FDIM",
    "BOND_FDIM",
    "FluorSolventDataset",
    "MANIFEST_REQUIRED_COLS",
    "RAW_ROW_ID_COL",
    "SAMPLE_ID_COL",
    "SMILES_COL",
    "SOLVENT_COL",
    "SOLVENT_RAW_COL",
    "TARGET_COLS",
    "build_manifest_df",
    "build_split_from_manifest",
    "compute_target_stats",
    "load_manifest",
    "load_split",
    "make_loaders",
    "make_train_val_test_loaders",
    "prepare_reproducible_protocol",
    "save_manifest",
    "save_split",
    "seed_worker",
    "smiles_to_graph",
    "standardize_targets",
]
