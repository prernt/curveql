# app/config.py
"""
Global constants and settings dataclasses.
No Streamlit or heavy scientific dependencies — safe to import first.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root & sys.path bootstrap
# ---------------------------------------------------------------------------
# config.py lives in <repo>/app/, so parents[1] is the repo root. Computed
# locally (not imported from growthqa.config) because this bootstrap has to
# run before growthqa is guaranteed to be importable at all.
ROOT = Path(__file__).resolve().parents[1]

for _cand in {ROOT, ROOT / "src", Path.cwd(), Path.cwd() / "src"}:
    if _cand.exists():
        _sp = str(_cand)
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

# ---------------------------------------------------------------------------
# Well-known paths and preprocessing defaults, shared with the training
# pipeline. Imported from growthqa.config -- the single source of truth --
# rather than redefined here, now that sys.path includes src/.
# ---------------------------------------------------------------------------
from growthqa.config import (
    MODEL_DIR,
    TRAIN_META_CSV as TRAIN_META,
    STEP_HOURS as _STEP_HOURS,
    MIN_POINTS as _MIN_POINTS,
    TMAX_HOURS as _TMAX_HOURS,
    SMOOTH_METHOD as _SMOOTH_METHOD,
    SMOOTH_WINDOW as _SMOOTH_WINDOW,
    NORMALIZE as _NORMALIZE,
)


# ---------------------------------------------------------------------------
# Settings dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InferenceSettings:
    """Pre-processing options, partially exposed in the UI."""
    # Blank handling
    input_is_raw: bool         = False
    global_blank: float | None = None

    # Fixed pipeline defaults (not shown in UI) -- pinned to the same values
    # training uses (growthqa.config), so an uploaded curve is preprocessed
    # identically to the training set.
    step: float                = _STEP_HOURS
    min_points: int            = _MIN_POINTS
    auto_tmax: bool            = False
    auto_tmax_coverage: float  = 0.8
    tmax_hours: float | None   = _TMAX_HOURS

    # Locked values
    clip_negatives: bool = False
    smooth_method: str   = _SMOOTH_METHOD
    smooth_window: int   = _SMOOTH_WINDOW
    normalize: str       = _NORMALIZE


@dataclass
class GrofitOptions:
    """Options forwarded to the Grofit fitting pipeline."""
    response_var: str          = "mu"
    have_atleast: int          = 6
    fit_opt: str               = "b"
    gc_boot_B: int             = 200
    dr_boot_B: int             = 300
    spline_auto_cv: bool       = True
    spline_s: float | None     = None   # legacy manual spline smoothing
    dr_s: float | None         = None   # legacy manual DR smoothing
    smooth_gc: float | None    = None   # Grofit-R smooth.gc spar ∈ (0,1]
    smooth_dr: float | None    = None   # Grofit-R smooth.dr spar ∈ (0,1]
    dr_x_transform: str | None = None
    dr_y_transform: str | None = None
    bootstrap_method: str      = None