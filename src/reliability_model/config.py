from __future__ import annotations

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "models", "reliability_model")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "results", "intermediate")

SMILES_COLUMN = "smiles"
SOLVENT_COLUMN = "solvent"
PROPERTIES = ["plqy", "emi", "em", "abs"]
MAIN_MODEL_TARGET_ORDER = ["abs", "emi", "plqy", "em"]
RANDOM_SEED = 42


CV_MORE_WEIGHT_PATH = os.path.join(BASE_DIR, "..", "..", "models", "pretrained", "MORE.pth")


FINGERPRINT_RADIUS = 2
FINGERPRINT_NBITS = 2048
DESCRIPTOR_PCA_COMPONENTS = 50
LOCAL_DENSITY_RADIUS = 0.4
SOLVENT_ECFP_RADIUS = 2
SOLVENT_ECFP_NBITS = 256
SOLUTE_ECFP_RADIUS = 2
SOLUTE_ECFP_NBITS = 512
USE_SOLVENT_FEATURES = True


USE_ENSEMBLE_VARIANCE_FEATURE = False


ERROR_EPSILON_BY_PROPERTY = {
    "abs": 1.0,
    "emi": 1.0,
    "em": 0.01,
    "plqy": 0.005,
}


ERROR_LOSS_FLOOR_BY_PROPERTY = {
    "abs": 1.0,
    "emi": 1.0,
    "em": 0.01,
    "plqy": 0.005,
}
ERROR_LOSS_SMOOTH_WIDTH_BY_PROPERTY = {
    "abs": 4.0,
    "emi": 4.0,
    "em": 0.04,
    "plqy": 0.015,
}

ERROR_MODEL_MIN_SAMPLES = 20





AE_FEATURE_N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))
AE_FEATURE_BACKEND = "threading"
AE_FEATURE_CHUNK_SIZE = 512
