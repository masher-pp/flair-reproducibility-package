
"""
AE 、scaffold split、、 PCA 。

：
1.  AE ，reference set  checkpoint 。
2. ，reference set 。
3.  solvent ， solute  solvent 。
"""

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import (
    DESCRIPTOR_PCA_COMPONENTS,
    AE_FEATURE_N_JOBS,
    FINGERPRINT_NBITS,
    FINGERPRINT_RADIUS,
    LOCAL_DENSITY_RADIUS,
    PROPERTIES,
    RANDOM_SEED,
    SMILES_COLUMN,
    SOLUTE_ECFP_NBITS,
    SOLUTE_ECFP_RADIUS,
    SOLVENT_ECFP_NBITS,
    SOLVENT_ECFP_RADIUS,
    SOLVENT_COLUMN,
)


FEATURE_NAMES_SOLUTE = [
    "solute_max_tanimoto",
    "solute_min_descriptor_distance",
    "solute_scaffold_novel",
    "solute_scaffold_similarity",
    "solute_local_density",
]

FEATURE_NAMES_SOLVENT = [f"solvent_ecfp_{i:03d}" for i in range(SOLVENT_ECFP_NBITS)]
FEATURE_NAMES_SOLUTE_ECFP = [f"solute_ecfp_{i:03d}" for i in range(SOLUTE_ECFP_NBITS)]
TANIMOTO_NEIGHBOR_FEATURE_NAMES = [
    "solute_top3_mean_tanimoto",
    "solute_top5_mean_tanimoto",
    "solute_neighbor_count_ge_0_5",
    "solute_neighbor_count_ge_0_7",
    "solute_neighbor_count_ge_0_8",
    "solute_top1_top2_tanimoto_gap",
    "solute_same_scaffold_neighbor_count_ge_0_5",
    "solute_other_scaffold_neighbor_count_ge_0_5",
]
SOLVENT_REFERENCE_FEATURE_NAMES = [
    "solvent_train_and_val_count",
    "solvent_max_tanimoto",
    "solvent_top3_mean_tanimoto",
    "solvent_top5_mean_tanimoto",
    "solvent_neighbor_count_ge_0_5",
    "solvent_neighbor_count_ge_0_7",
    "solvent_neighbor_count_ge_0_8",
    "solvent_top1_top2_tanimoto_gap",
]
SOLVENT_SCAFFOLD_FEATURE_NAMES = [
    "solvent_has_murcko_scaffold",
    "solvent_scaffold_train_and_val_count",
    "solvent_scaffold_novel",
    "solvent_scaffold_similarity",
    "solvent_same_scaffold_neighbor_count_ge_0_5",
    "solvent_other_scaffold_neighbor_count_ge_0_5",
]

FOLD_VARIANCE_FEATURE = "fold_variance"


def _safe_float_array(values: Sequence) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ smiles / solvent / 。"""
    rename = {}
    lower_to_col = {str(c).lower(): c for c in df.columns}

    smiles_candidates = [SMILES_COLUMN, "SMILES", "smile", "Smiles", "solute", "solute_smiles"]
    for c in smiles_candidates:
        if c in df.columns:
            rename[c] = SMILES_COLUMN
            break
        if c.lower() in lower_to_col:
            rename[lower_to_col[c.lower()]] = SMILES_COLUMN
            break

    solvent_candidates = [SOLVENT_COLUMN, "SOLVENT", "solvent_smiles", "Solvent", "solvent_smi"]
    for c in solvent_candidates:
        if c in df.columns:
            rename[c] = SOLVENT_COLUMN
            break
        if c.lower() in lower_to_col:
            rename[lower_to_col[c.lower()]] = SOLVENT_COLUMN
            break

    for prop in PROPERTIES:
        if prop in df.columns:
            continue
        if prop.lower() in lower_to_col:
            rename[lower_to_col[prop.lower()]] = prop

    if rename:
        df = df.rename(columns=rename)
    return df


def load_training_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_columns(df)
    if SMILES_COLUMN not in df.columns:
        raise ValueError(f" {SMILES_COLUMN} 。")
    missing_props = [p for p in PROPERTIES if p not in df.columns]
    if missing_props:
        raise ValueError(f": {missing_props}")
    return df


def load_training_data(path: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """： smiles_list  properties 。"""
    df = load_training_dataframe(path)
    smiles = df[SMILES_COLUMN].astype(str).tolist()
    props = {p: pd.to_numeric(df[p], errors="coerce").to_numpy(dtype=float) for p in PROPERTIES}
    return smiles, props


def mol_from_smiles(smiles: object) -> Optional[Chem.Mol]:
    if pd.isna(smiles):
        return None
    smi = str(smiles).strip()
    if not smi:
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def get_morgan_fingerprint(
    mol: Optional[Chem.Mol],
    radius: int = FINGERPRINT_RADIUS,
    n_bits: int = FINGERPRINT_NBITS,
):
    if mol is None:
        return None
    with rdBase.BlockLogs():
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def get_scaffold_smiles(mol: Optional[Chem.Mol]) -> str:
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return ""


def build_unique_tanimoto_reference(smiles_values: Sequence[object]) -> Tuple[List, List[str]]:
    """Build one fingerprint per unique canonical solute for neighborhood counts."""
    fingerprints = []
    scaffolds = []
    seen = set()
    for smiles in smiles_values:
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen:
            continue
        fingerprint = get_morgan_fingerprint(mol)
        if fingerprint is None:
            continue
        seen.add(canonical)
        fingerprints.append(fingerprint)
        scaffolds.append(get_scaffold_smiles(mol))
    return fingerprints, scaffolds


def canonical_smiles(smiles: object) -> str:
    """Return a canonical SMILES key, or an empty string for invalid input."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def build_solvent_tanimoto_reference(solvent_values: Sequence[object]) -> Dict:
    """Build exact occurrence counts and one Morgan fingerprint per unique solvent."""
    counts: Dict[str, int] = {}
    fingerprints = []
    fingerprint_scaffolds = []
    scaffold_counts: Dict[str, int] = {}
    scaffold_fingerprints = []
    seen_scaffolds = set()
    seen = set()
    for solvent in solvent_values:
        canonical = canonical_smiles(solvent)
        if not canonical:
            continue
        counts[canonical] = counts.get(canonical, 0) + 1
        scaffold = get_scaffold_smiles(mol_from_smiles(canonical))
        if scaffold:
            scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
            if scaffold not in seen_scaffolds:
                scaffold_fingerprint = get_morgan_fingerprint(mol_from_smiles(scaffold))
                if scaffold_fingerprint is not None:
                    seen_scaffolds.add(scaffold)
                    scaffold_fingerprints.append(scaffold_fingerprint)
        if canonical in seen:
            continue
        mol = mol_from_smiles(canonical)
        fingerprint = get_morgan_fingerprint(mol)
        if fingerprint is None:
            continue
        seen.add(canonical)
        fingerprints.append(fingerprint)
        fingerprint_scaffolds.append(scaffold)
    if not fingerprints:
        raise ValueError("No valid solvent SMILES were available for the Tanimoto reference.")
    return {
        "counts": counts,
        "fingerprints": fingerprints,
        "fingerprint_scaffolds": fingerprint_scaffolds,
        "scaffold_counts": scaffold_counts,
        "scaffold_set": set(scaffold_counts),
        "scaffold_fingerprints": scaffold_fingerprints,
        "n_rows": int(sum(counts.values())),
        "n_unique": int(len(fingerprints)),
        "radius": int(FINGERPRINT_RADIUS),
        "n_bits": int(FINGERPRINT_NBITS),
    }


def solvent_reference_features(solvent: object, reference: Dict) -> Dict[str, float]:
    """Return exact occurrence count and Tanimoto-neighborhood features for a solvent."""
    values = {
        name: 0.0
        for name in [*SOLVENT_REFERENCE_FEATURE_NAMES, *SOLVENT_SCAFFOLD_FEATURE_NAMES]
    }
    canonical = canonical_smiles(solvent)
    mol = mol_from_smiles(canonical) if canonical else None
    scaffold = get_scaffold_smiles(mol)
    count = float(reference.get("counts", {}).get(canonical, 0)) if canonical else 0.0
    values["solvent_train_and_val_count"] = count
    values["solvent_has_murcko_scaffold"] = 1.0 if scaffold else 0.0
    values["solvent_scaffold_train_and_val_count"] = (
        float(reference.get("scaffold_counts", {}).get(scaffold, 0)) if scaffold else 0.0
    )
    values["solvent_scaffold_novel"] = (
        0.0 if scaffold and scaffold in reference.get("scaffold_set", set()) else 1.0
    )
    if scaffold:
        scaffold_fingerprint = get_morgan_fingerprint(mol_from_smiles(scaffold))
        reference_scaffold_fingerprints = list(reference.get("scaffold_fingerprints", []))
        if scaffold_fingerprint is not None and reference_scaffold_fingerprints:
            values["solvent_scaffold_similarity"] = float(
                max(DataStructs.BulkTanimotoSimilarity(scaffold_fingerprint, reference_scaffold_fingerprints))
            )

    fingerprint = get_morgan_fingerprint(mol)
    reference_fingerprints = list(reference.get("fingerprints", []))
    if fingerprint is None or not reference_fingerprints:
        return values

    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fingerprints),
        dtype=float,
    )
    top_count = min(5, len(similarities))
    top = np.partition(similarities, len(similarities) - top_count)[-top_count:]
    top = np.sort(top)[::-1]
    near_mask = similarities >= 0.5
    reference_scaffolds = np.asarray(reference.get("fingerprint_scaffolds", []), dtype=object)
    same_scaffold_mask = (
        near_mask & (reference_scaffolds == scaffold)
        if scaffold and len(reference_scaffolds) == len(similarities)
        else np.zeros_like(near_mask)
    )
    values.update(
        {
            "solvent_max_tanimoto": float(top[0]),
            "solvent_top3_mean_tanimoto": float(top[: min(3, len(top))].mean()),
            "solvent_top5_mean_tanimoto": float(top.mean()),
            "solvent_neighbor_count_ge_0_5": float(np.count_nonzero(similarities >= 0.5)),
            "solvent_neighbor_count_ge_0_7": float(np.count_nonzero(similarities >= 0.7)),
            "solvent_neighbor_count_ge_0_8": float(np.count_nonzero(similarities >= 0.8)),
            "solvent_top1_top2_tanimoto_gap": (
                float(top[0] - top[1]) if len(top) > 1 else float(top[0])
            ),
            "solvent_same_scaffold_neighbor_count_ge_0_5": float(np.count_nonzero(same_scaffold_mask)),
            "solvent_other_scaffold_neighbor_count_ge_0_5": float(
                np.count_nonzero(near_mask & ~same_scaffold_mask)
            ),
        }
    )
    return values


def tanimoto_neighborhood_features(
    mol: Optional[Chem.Mol],
    reference_fingerprints: Sequence,
    reference_scaffolds: Sequence[str],
) -> Dict[str, float]:
    values = {name: 0.0 for name in TANIMOTO_NEIGHBOR_FEATURE_NAMES}
    fingerprint = get_morgan_fingerprint(mol)
    if fingerprint is None or not reference_fingerprints:
        return values

    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(fingerprint, list(reference_fingerprints)),
        dtype=float,
    )
    top_count = min(5, len(similarities))
    top = np.partition(similarities, len(similarities) - top_count)[-top_count:]
    top = np.sort(top)[::-1]
    near_mask = similarities >= 0.5
    scaffold = get_scaffold_smiles(mol)
    scaffold_array = np.asarray(reference_scaffolds, dtype=object)
    same_scaffold_mask = near_mask & (scaffold_array == scaffold) if scaffold else np.zeros_like(near_mask)

    values.update(
        {
            "solute_max_tanimoto": float(top[0]),
            "solute_top3_mean_tanimoto": float(top[: min(3, len(top))].mean()),
            "solute_top5_mean_tanimoto": float(top.mean()),
            "solute_neighbor_count_ge_0_5": float(np.count_nonzero(near_mask)),
            "solute_neighbor_count_ge_0_7": float(np.count_nonzero(similarities >= 0.7)),
            "solute_neighbor_count_ge_0_8": float(np.count_nonzero(similarities >= 0.8)),
            "solute_top1_top2_tanimoto_gap": float(top[0] - top[1]) if len(top) > 1 else float(top[0]),
            "solute_same_scaffold_neighbor_count_ge_0_5": float(np.count_nonzero(same_scaffold_mask)),
            "solute_other_scaffold_neighbor_count_ge_0_5": float(np.count_nonzero(near_mask & ~same_scaffold_mask)),
        }
    )
    return values


def compute_descriptors(mol: Optional[Chem.Mol]) -> Optional[np.ndarray]:
    if mol is None:
        return None
    values = []
    for _, func in Descriptors._descList:
        try:
            v = float(func(mol))
        except Exception:
            v = np.nan
        if not math.isfinite(v):
            v = np.nan
        values.append(v)
    return np.asarray(values, dtype=float)


def resolve_n_jobs(n_jobs: Optional[int] = None) -> int:
    value = AE_FEATURE_N_JOBS if n_jobs is None else int(n_jobs)
    if value < 0:
        return max(1, (os.cpu_count() or 1) + 1 + value)
    return max(1, value)


def threaded_map(func, items: Sequence, n_jobs: Optional[int] = None) -> List:
    jobs = resolve_n_jobs(n_jobs)
    if jobs <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(func, items))


def compute_descriptors_many(
    mols: Sequence[Optional[Chem.Mol]],
    n_jobs: Optional[int] = None,
) -> List[Optional[np.ndarray]]:
    return threaded_map(compute_descriptors, list(mols), n_jobs=n_jobs)


def fit_descriptor_pipeline(
    mols: Sequence[Optional[Chem.Mol]],
    n_components: int = DESCRIPTOR_PCA_COMPONENTS,
    n_jobs: Optional[int] = None,
) -> Dict:
    desc_list = []
    for desc in compute_descriptors_many(mols, n_jobs=n_jobs):
        if desc is not None:
            desc_list.append(desc)
    if len(desc_list) == 0:
        raise ValueError(" PCA 。")

    X = np.vstack(desc_list)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    max_components = max(1, min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    pca = PCA(n_components=max_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    return {"imputer": imputer, "scaler": scaler, "pca": pca, "n_components": max_components}


def transform_mol_to_pca(mol: Optional[Chem.Mol], pipeline: Dict) -> Optional[np.ndarray]:
    desc = compute_descriptors(mol)
    if desc is None:
        return None
    X_imp = pipeline["imputer"].transform([desc])
    X_scaled = pipeline["scaler"].transform(X_imp)
    return pipeline["pca"].transform(X_scaled)[0]


def transform_mols_to_pca(
    mols: Sequence[Optional[Chem.Mol]],
    pipeline: Dict,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    dim = int(pipeline["n_components"])
    rows = np.full((len(mols), dim), np.nan, dtype=float)
    descs = compute_descriptors_many(mols, n_jobs=n_jobs)
    valid_indices = [idx for idx, desc in enumerate(descs) if desc is not None]
    if not valid_indices:
        return rows

    X = np.vstack([descs[idx] for idx in valid_indices])
    X_imp = pipeline["imputer"].transform(X)
    X_scaled = pipeline["scaler"].transform(X_imp)
    rows[np.asarray(valid_indices, dtype=int)] = pipeline["pca"].transform(X_scaled)
    return rows


def _scaffold_and_fp(mol: Optional[Chem.Mol]):
    scaf = get_scaffold_smiles(mol)
    if not scaf:
        return "", None
    return scaf, get_morgan_fingerprint(Chem.MolFromSmiles(scaf))


def build_reference_cache(
    mols: Sequence[Optional[Chem.Mol]],
    n_components: int = DESCRIPTOR_PCA_COMPONENTS,
) -> Dict:
    valid_mols = [m for m in mols if m is not None]
    if len(valid_mols) == 0:
        raise ValueError("reference set 。")

    
    
    
    
    unique_mols: Dict[str, Chem.Mol] = {}
    mol_keys = []
    for mol in valid_mols:
        try:
            key = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            key = f"__mol_{len(mol_keys)}"
        mol_keys.append(key)
        if key not in unique_mols:
            unique_mols[key] = mol

    descriptor_cache = {}
    fp_cache = {}
    scaffold_cache = {}
    for key, mol in unique_mols.items():
        descriptor_cache[key] = compute_descriptors(mol)
        fp_cache[key] = get_morgan_fingerprint(mol)
        scaffold_cache[key] = _scaffold_and_fp(mol)

    desc_list = [descriptor_cache[key] for key in mol_keys if descriptor_cache.get(key) is not None]
    if len(desc_list) == 0:
        raise ValueError("reference set  PCA 。")

    X = np.vstack(desc_list)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    max_components = max(1, min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    pca = PCA(n_components=max_components, random_state=RANDOM_SEED)
    pca_embeddings = pca.fit_transform(X_scaled)
    pipeline = {"imputer": imputer, "scaler": scaler, "pca": pca, "n_components": max_components}

    fps = [fp_cache.get(key) for key in mol_keys]

    scaffold_smiles = []
    scaffold_fps = []
    for key in mol_keys:
        scaf, fp = scaffold_cache.get(key, ("", None))
        if scaf:
            scaffold_smiles.append(scaf)
            scaffold_fps.append(fp)

    return {
        "mols": valid_mols,
        "fps": fps,
        "descriptor_pipeline": pipeline,
        "pca_embeddings": pca_embeddings,
        "scaffold_set": set(scaffold_smiles),
        "scaffold_fps": scaffold_fps,
    }


def max_tanimoto_to_train_set(mol: Optional[Chem.Mol], train_mols: Sequence[Chem.Mol], train_fps=None) -> float:
    if mol is None or not train_mols:
        return 0.0
    query_fp = get_morgan_fingerprint(mol)
    fps = train_fps if train_fps is not None else [get_morgan_fingerprint(m) for m in train_mols]
    fps = [fp for fp in fps if fp is not None]
    if query_fp is None or not fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, fps)
    return float(np.max(sims)) if sims else 0.0


def min_euclidean_distance_to_train(
    mol: Optional[Chem.Mol],
    descriptor_pipeline: Dict,
    train_pca_embeddings: np.ndarray,
) -> float:
    if mol is None or train_pca_embeddings is None or len(train_pca_embeddings) == 0:
        return 999.0
    emb = transform_mol_to_pca(mol, descriptor_pipeline)
    if emb is None:
        return 999.0
    valid = np.asarray(train_pca_embeddings, dtype=float)
    valid = valid[np.all(np.isfinite(valid), axis=1)]
    if len(valid) == 0:
        return 999.0
    distances = np.linalg.norm(valid - emb.reshape(1, -1), axis=1)
    return float(np.min(distances))


def scaffold_novelty(mol: Optional[Chem.Mol], train_scaffolds_set: set) -> float:
    scaf = get_scaffold_smiles(mol)
    if not scaf:
        return 1.0
    return 0.0 if scaf in train_scaffolds_set else 1.0


def scaffold_similarity(mol: Optional[Chem.Mol], train_scaffold_fps: Sequence) -> float:
    scaf = get_scaffold_smiles(mol)
    if not scaf or not train_scaffold_fps:
        return 0.0
    scaf_mol = Chem.MolFromSmiles(scaf)
    qfp = get_morgan_fingerprint(scaf_mol)
    fps = [fp for fp in train_scaffold_fps if fp is not None]
    if qfp is None or not fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(qfp, fps)
    return float(np.max(sims)) if sims else 0.0


def local_density(
    mol: Optional[Chem.Mol],
    descriptor_pipeline: Dict,
    train_pca_embeddings: np.ndarray,
    radius: float = LOCAL_DENSITY_RADIUS,
) -> float:
    if mol is None or train_pca_embeddings is None or len(train_pca_embeddings) == 0:
        return 0.0
    emb = transform_mol_to_pca(mol, descriptor_pipeline)
    if emb is None:
        return 0.0
    valid = np.asarray(train_pca_embeddings, dtype=float)
    valid = valid[np.all(np.isfinite(valid), axis=1)]
    if len(valid) == 0:
        return 0.0
    distances = np.linalg.norm(valid - emb.reshape(1, -1), axis=1)
    return float(np.sum(distances <= radius))


def extract_single_molecule_features(mol: Optional[Chem.Mol], cache: Dict, prefix: str) -> Tuple[List[float], List[str]]:
    
    
    
    names = list(FEATURE_NAMES_SOLUTE if prefix == "solute" else FEATURE_NAMES_SOLVENT)
    if mol is None:
        return [0.0, 999.0, 1.0, 0.0, 0.0], names

    features = [
        max_tanimoto_to_train_set(mol, cache["mols"], cache.get("fps")),
        min_euclidean_distance_to_train(mol, cache["descriptor_pipeline"], cache["pca_embeddings"]),
        scaffold_novelty(mol, cache["scaffold_set"]),
        scaffold_similarity(mol, cache["scaffold_fps"]),
        local_density(mol, cache["descriptor_pipeline"], cache["pca_embeddings"]),
    ]
    return features, names


def extract_solvent_ecfp_features(mol: Optional[Chem.Mol]) -> Tuple[List[float], List[str]]:
    """Solvent features are raw ECFP/Morgan bits; solute features stay unchanged."""
    names = list(FEATURE_NAMES_SOLVENT)
    zeros = [0.0] * int(SOLVENT_ECFP_NBITS)
    if mol is None:
        return zeros, names

    fp = get_morgan_fingerprint(mol, radius=SOLVENT_ECFP_RADIUS, n_bits=SOLVENT_ECFP_NBITS)
    if fp is None:
        return zeros, names
    arr = np.zeros((int(SOLVENT_ECFP_NBITS),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float).tolist(), names


def extract_solute_ecfp_features(mol: Optional[Chem.Mol]) -> Tuple[List[float], List[str]]:
    """Optional solute ECFP bits for rank/Spearman experiments."""
    names = list(FEATURE_NAMES_SOLUTE_ECFP)
    zeros = [0.0] * int(SOLUTE_ECFP_NBITS)
    if mol is None:
        return zeros, names

    fp = get_morgan_fingerprint(mol, radius=SOLUTE_ECFP_RADIUS, n_bits=SOLUTE_ECFP_NBITS)
    if fp is None:
        return zeros, names
    arr = np.zeros((int(SOLUTE_ECFP_NBITS),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float).tolist(), names


def extract_features_for_error_model(
    solute_mol: Optional[Chem.Mol],
    solute_cache: Dict,
    solvent_mol: Optional[Chem.Mol] = None,
    solvent_cache: Optional[Dict] = None,
    fold_variance: float = 0.0,
    base_prediction: Optional[float] = None,
) -> Tuple[np.ndarray, List[str]]:
    """/。

    base_prediction 。
    “/”， base_prediction 
     OOD ；，。
    """
    features, names = extract_single_molecule_features(solute_mol, solute_cache, "solute")

    if solvent_mol is not None or solvent_cache is not None:
        solvent_features, solvent_names = extract_solvent_ecfp_features(solvent_mol)
        features.extend(solvent_features)
        names.extend(solvent_names)

    features.append(0.0 if not np.isfinite(fold_variance) else float(fold_variance))
    names.append(FOLD_VARIANCE_FEATURE)

    if base_prediction is not None:
        try:
            pred_value = float(base_prediction)
        except Exception:
            pred_value = np.nan
        features.append(0.0 if not np.isfinite(pred_value) else pred_value)
        names.append("base_prediction")

    return np.asarray(features, dtype=float), names


def scaffold_split(smiles_list: Sequence[str], n_folds: int = 5, seed: int = RANDOM_SEED) -> List[List[int]]:
    """ Bemis-Murcko scaffold  k-fold。"""
    scaffold_to_indices: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles_list):
        mol = mol_from_smiles(smi)
        scaf = get_scaffold_smiles(mol) or f"NO_SCAFFOLD_{idx}"
        scaffold_to_indices.setdefault(scaf, []).append(idx)

    groups = list(scaffold_to_indices.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    folds: List[List[int]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for group in groups:
        target = int(np.argmin(fold_sizes))
        folds[target].extend(group)
        fold_sizes[target] += len(group)

    return [sorted(fold) for fold in folds]


def records_from_dataframe(df: pd.DataFrame) -> List[Dict]:
    has_solvent = SOLVENT_COLUMN in df.columns
    records = []
    for _, row in df.iterrows():
        rec = {
            "smiles": str(row[SMILES_COLUMN]).strip(),
            "solute_mol": mol_from_smiles(row[SMILES_COLUMN]),
        }
        if has_solvent:
            rec["solvent"] = str(row[SOLVENT_COLUMN]).strip() if not pd.isna(row[SOLVENT_COLUMN]) else ""
            rec["solvent_mol"] = mol_from_smiles(row[SOLVENT_COLUMN])
        else:
            rec["solvent"] = None
            rec["solvent_mol"] = None
        records.append(rec)
    return records
