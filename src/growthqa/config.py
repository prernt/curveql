from __future__ import annotations

"""
Central configuration for the GrowthQA pipeline.

Every hardcoded, static, or global value that affects preprocessing,
training, or inference lives here, and only here. Before this file existed,
the same handful of values (repo paths, the canonical grid step, sparsity
thresholds, the training random seed, the Stage 1 feature list) were each
independently redefined in three or four different files -- e.g. ROOT was
computed separately in train_from_meta.py, auto_train_classifier.py, and
app/config.py, and min_points/low_res_threshold were hardcoded again at the
one inference call site instead of reusing the values training already
pins. That meant a change made in one place silently had no effect anywhere
else. Every other module now imports these values from here instead of
redefining them.

If a new hardcoded value is about to be added anywhere in the codebase, it
belongs here instead.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
# This file lives at <repo>/src/growthqa/config.py, so parents[2] is <repo>.
ROOT = Path(__file__).resolve().parents[2]

TRAIN_META_CSV = ROOT / "data" / "train_data" / "training_meta.csv"
MODEL_DIR = ROOT / "classifier_output" / "saved_models_selected"
LOCKFILE_OUT = ROOT / "classifier_output" / "requirements_lock.txt"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Canonical grid / sparsity thresholds
# ---------------------------------------------------------------------------
# Used identically for training-set construction (build_meta_dataset) and for
# every uploaded curve at inference (infer_labels.
# run_label_inference_from_uploaded_wide). An uploaded curve MUST be
# preprocessed with exactly these same values as the training set, or Stage
# 1's predictions are not meaningful -- this is the one set of values where
# "defined in one place" isn't just tidiness, it's a correctness requirement.
STEP_HOURS = 0.5
TMAX_HOURS = 16.0
MIN_POINTS = 3
LOW_RES_THRESHOLD = 7
SMOOTH_METHOD = "SGF"
SMOOTH_WINDOW = 5
NORMALIZE = "MINMAX"

# ---------------------------------------------------------------------------
# Truncation augmentation (training-set construction only; see
# preprocess.truncation_augment.augment_raw_wide)
# ---------------------------------------------------------------------------
TRUNC_HORIZONS = [8.0, 10.0, 12.0, 14.75, 16.0]
TRUNC_PER_CURVE = 3
TRUNC_SEED = 123

# ---------------------------------------------------------------------------
# Gap augmentation (training-set construction only; see
# preprocess.gap_augment.augment_raw_wide_with_gaps)
# ---------------------------------------------------------------------------
# Injects real internal gaps -- not just tail-truncation -- into a subset of
# training curves, at the raw-data level, before interpolation. Exists
# because the pre-existing training set (synthetic + lab, with or without
# truncation augmentation) had ZERO curves with genuine internal gaps: every
# curve was either fully dense or cleanly truncated at the tail. A model
# trained on that data has no real evidence for what a multi-hour gap means,
# and tree-based models (RF/HGB) extrapolate poorly past the edge of their
# training range -- confirmed directly: real lab curves with gaps of
# 2-6 hours and 40-80% missingness were being scored using training
# statistics that topped out at 1.5 hours and 17.6% missing.
GAP_AUG_FRACTION = 0.30       # fraction of curves (per source) to draw gap-augmented variants from
GAP_AUG_PER_CURVE = 2         # gap-augmented variants generated per chosen curve
GAP_AUG_SEED = 456
GAP_MIN_HOURS = 2.0           # "block" pattern: one contiguous missing stretch, duration in [GAP_MIN_HOURS, GAP_MAX_HOURS]
GAP_MAX_HOURS = 6.0
GAP_MIN_MISSING_FRAC = 0.40   # "scattered" pattern: individually-missing points, until this fraction of real points is gone
GAP_MAX_MISSING_FRAC = 0.80

# ---------------------------------------------------------------------------
# Out-of-distribution gap/missingness override (inference-time safety net)
# ---------------------------------------------------------------------------
# Independent of the ML model and of too_sparse (which only looks at point
# COUNT, not gap size or overall missingness). A curve whose max_gap_hours or
# missing_frac_on_grid falls beyond what the training data actually covers is
# not something Stage 1's opinion should be trusted on, regardless of its
# confidence -- so this is enforced as a plain threshold, not learned.
#
# Calibrated with headroom above the training ceiling GAP_MAX_HOURS /
# GAP_MAX_MISSING_FRAC establish above, not above the model's ORIGINAL
# (pre-gap-augmentation) training range. Recalibrate these two values
# whenever training_meta.csv is rebuilt with a different gap-augmentation
# range: they should always sit just past whatever the training data's own
# max_gap_hours / missing_frac_on_grid columns actually reach, not be picked
# independently of it. Verified directly against a rebuilt training_meta.csv:
# gap-augmented rows realize max_gap_hours up to ~8.5h (GAP_MAX_HOURS=6.0 is
# the sampled block *duration*, not a hard ceiling on the realized raw gap)
# and missing_frac_on_grid up to ~0.78 -- these two constants sit above both.
MAX_GAP_HOURS_OVERRIDE = 10.0
MISSING_FRAC_OVERRIDE = 0.85

# ---------------------------------------------------------------------------
# Stage 1 candidate feature pool
# ---------------------------------------------------------------------------
# The deliberate, documented search space for feature selection -- NOT the
# same thing as "every numeric column build_model_matrix happens to leave
# after dropping identifiers". too_sparse / low_resolution /
# grid_resolution_mismatch are excluded here even though build_model_matrix
# would otherwise include them, because they are constant by design on any
# training corpus (they only vary on genuinely sparse real-world uploads at
# inference, which is their actual job -- see infer_labels.py's hard
# override and app/results.py's audit display). Including them in a
# selection search wastes analysis effort on columns that can never be
# selected.
#
# Organized into four groups, each answering one distinct question about a
# curve. Every feature earns its place by covering a failure mode none of
# the others do; features that were merely "not correlated with anything"
# but didn't answer a clearly distinct question (time_of_max_slope,
# time_of_max_OD, plateau_OD, symmetry_factor, num_slope_sign_changes) were
# deliberately left out of this set, not because they are wrong, but because
# their inclusion couldn't be justified past "it's available" -- a smaller,
# fully-justified pool was chosen over a maximal one.
#
# See Stage1_Feature_Analysis_And_Selection.ipynb for the CV-based ranking,
# statistical tests, and final selection built on top of this pool.
STAGE1_FEATURE_GROUPS = {
    # "Can we even trust the shape we're about to measure?"
    "observation_quality": [
        "observed_tmax",            # how much of the curve was actually observed
        "n_points_observed",        # raw point count: data density (raw-data-based, not grid-based)
        "max_gap_hours",            # largest real gap between measurements (raw-data-based)
        "missing_frac_on_grid",     # measurement density relative to the canonical grid (raw-data-based)
    ],
    # "Where did it start and end?"
    "level": [
        "initial_OD",               # starting level
        "final_OD",                 # ending level
    ],
    # "How did it get from start to end?"
    "growth_dynamics": [
        "net_change_per_hour",      # average rate over the whole window
        "max_slope",                # peak instantaneous rate
        "auc_per_hour",             # average level over time (distinct from rate)
        "lag_time_est",             # onset of active growth
        "growth_phase_duration",    # duration of the active growth phase
    ],
    # "Does the trajectory look like real growth, or an artifact?"
    "shape_integrity": [
        "monotonicity_fraction",    # overall directional consistency
        "largest_drop_frac",        # worst single decline
        "multi_phase_flag",         # diauxic / double-peak detector
        "roughness",                # raw jaggedness (includes trend)
        "noise_residual_std",       # noise after removing trend (isolates noise alone)
    ],
}
STAGE1_CANDIDATE_POOL = [f for group in STAGE1_FEATURE_GROUPS.values() for f in group]

# Production feature set. Evaluated once, on a held-out test set untouched by
# any selection step, against Top-10 and Top-8 (by cross-validated
# permutation importance) and a greedy, CV-stability-voted subset -- see
# Stage1_Feature_Analysis_And_Selection.ipynb, Section 7. The full 16-feature
# pool won outright (best or tied-best balanced accuracy, F1, and ROC-AUC
# across LR/RF/HGB), so nothing is dropped from it for production.
STAGE1_SELECTED_FEATURES = list(STAGE1_CANDIDATE_POOL)

# ---------------------------------------------------------------------------
# Column roles for the Stage 1 feature matrix (build_model_matrix)
# ---------------------------------------------------------------------------
IDENTIFIER_COLS = {
    "FileName",
    "Test Id",
    "Model Name",
    "Concentration",
    "base_curve_id",
    "aug_id",
    # Provenance/bookkeeping columns, not curve-shape features.
    # tmax_original/is_synthetic mostly just encode which source file a
    # curve came from (near-perfectly correlated, r=0.99) rather than
    # anything about the curve's shape -- keeping them as features would
    # risk the model learning "which dataset" instead of "is this valid".
    # tmax_original is also NaN for every real inference curve (only ever
    # set by the training-time augmentation step), so it would contribute
    # nothing at deployment regardless of training-time appearance.
    "tmax_original",
    "train_horizon",
    "is_synthetic",
    # is_censored IS meaningfully computed at inference (unlike the above),
    # but it is a binary threshold of observed_tmax at exactly 16h, which
    # is already a continuous candidate feature above -- it can't add
    # information observed_tmax doesn't already carry more precisely.
    "is_censored",
    # Bookkeeping from gap_augment.augment_raw_wide_with_gaps: which
    # training rows are gap-augmented, and by which pattern. Useful for
    # analysis (e.g. checking model performance specifically on
    # gap-augmented rows) but not a property of the curve itself, and
    # gap_pattern is a non-numeric string with no meaning for non-augmented
    # rows anyway.
    "gap_augmented",
    "gap_pattern",
}
LEAKAGE_COLS = {"best_model_name"}