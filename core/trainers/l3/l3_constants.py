"""L3 values that are definitions or numeric floors, not training hyperparameters (C-class)."""

from __future__ import annotations

from core.trainers.constants import PA_STATE_FEATURES

# Binary exit head labels (LightGBM / policy dataset)
EXIT_LABEL_HOLD = 0  # OK-HARDCODED: binary definition
EXIT_LABEL_EXIT = 1  # OK-HARDCODED: binary definition

PA_STATE_FEATURE_COUNT = len(PA_STATE_FEATURES)  # OK-HARDCODED: tied to PA_STATE_FEATURES

# Platt / isotonic
PLATT_N_PARAMS = 2  # OK-HARDCODED: logistic on scalar logit
ISOTONIC_MIN_UNIQUE = 5  # OK-HARDCODED: isotonic needs variety in x

# Cox partial hazard / baseline
COX_LOG_PARTIAL_HAZARD_FLOOR = 1e-12  # OK-HARDCODED: numerical stability
