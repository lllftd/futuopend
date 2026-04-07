from __future__ import annotations

import gc
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from core.pa_rules import add_pa_features
from core.tcn_pa_state import FocalLoss, PAStateTCN

from core.trainers.tcn_constants import *

def _tqdm_disabled() -> bool:
    """Align with tqdm policy: DISABLE_TQDM=1 off; non-TTY off unless FORCE_TQDM=1."""
    d = os.environ.get("DISABLE_TQDM", "").strip().lower()
    if d in {"1", "true", "yes"}:
        return True
    if not sys.stderr.isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return True
    return False


def _tq(it, **kwargs):
    """Iterator progress bar (epochs, etc.)."""
    return tqdm(it, disable=_tqdm_disabled(), **kwargs)


def _pbar(**kwargs):
    """Manual tqdm (e.g. subprocess folds completed). Caller should use ``with`` / ``.update()``."""
    return tqdm(disable=_tqdm_disabled(), **kwargs)


def _pick_tcn_train_device() -> torch.device:
    """CUDA > MPS > CPU; set TORCH_DEVICE=cuda:0 / mps / cpu to force."""
    forced = os.environ.get("TORCH_DEVICE", "").strip()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


