"""L3 values that are definitions or numeric floors, not training hyperparameters (C-class)."""

from __future__ import annotations

# Platt / isotonic
ISOTONIC_MIN_UNIQUE = 5  # OK-HARDCODED: isotonic needs variety in x

# Cox partial hazard / baseline
COX_LOG_PARTIAL_HAZARD_FLOOR = 1e-12  # OK-HARDCODED: numerical stability
