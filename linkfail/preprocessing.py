"""Data wrangling and feature engineering (Section III of the paper).

Steps
1. Fill loss readings that are missing because of a fibre cut (LOS alarm)
   with the expected raw value of a cut fibre.
2. Linearly interpolate every other missing cell between neighbouring time
   bins, separately per link (i.e. per traffic direction) and per span.
3. Per-span diff features  (Eq. 1)
       x1_n = amp_gain_n - target_gain_n
       x2_n = target_span_loss_n - span_loss_n
4. E2E aggregation over the 19 spans of a link in each 2-minute bin (Eq. 2)
       X1 = sum_n x1_n ,  X2 = sum_n x2_n
5. Join the OSNR-based labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def fill_missing(tele: pd.DataFrame) -> pd.DataFrame:
    tele = tele.sort_values(["link_id", "span", "timestamp"]).copy()
    cut = tele["los_alarm"].eq(1) & tele["span_loss"].isna()
    tele.loc[cut, "span_loss"] = tele.loc[cut, "target_span_loss"] + C.FIBER_CUT_LOSS_DB
    for col in ("span_loss", "amp_gain"):
        tele[col] = (
            tele.groupby(["link_id", "span"], sort=False)[col]
            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        )
    return tele


def span_diffs(tele: pd.DataFrame) -> pd.DataFrame:
    tele = tele.copy()
    tele["x1"] = tele["amp_gain"] - tele["target_gain"]
    tele["x2"] = tele["target_span_loss"] - tele["span_loss"]
    return tele


def aggregate(tele: pd.DataFrame) -> pd.DataFrame:
    return (
        tele.groupby(["link_id", "timestamp"], as_index=False)
        .agg(X1=("x1", "sum"), X2=("x2", "sum"))
    )


def residual_matrix(tele: pd.DataFrame) -> pd.DataFrame:
    """Wide per-span residuals used for localisation.

    loss_res_n > 0 : span n loses more than planned (fibre problem)
    gain_res_n > 0 : amplifier n gives less gain than planned (amp problem)
    """
    t = tele.assign(loss_res=-tele["x2"], gain_res=-tele["x1"])
    wide = t.pivot_table(index=["link_id", "timestamp"], columns="span",
                         values=["loss_res", "gain_res"])
    wide.columns = [f"{m}_{s:02d}" for m, s in wide.columns]
    return wide.reset_index()


def build_features(tele: pd.DataFrame, health: pd.DataFrame):
    """Full preprocessing. Returns (features_df, residuals_df)."""
    tele = span_diffs(fill_missing(tele))
    feats = aggregate(tele).merge(health, on=["link_id", "timestamp"], how="inner")
    feats = feats.sort_values(["link_id", "timestamp"]).reset_index(drop=True)
    resid = residual_matrix(tele).merge(
        health[["link_id", "timestamp", "y", "fault_type", "fault_span"]],
        on=["link_id", "timestamp"])
    resid = resid.sort_values(["link_id", "timestamp"]).reset_index(drop=True)
    return feats, resid


def features_from_snapshot(snap: pd.DataFrame) -> np.ndarray:
    """X1, X2 for a single snapshot table (one row per span)."""
    x1 = (snap["amp_gain"] - snap["target_gain"]).sum()
    x2 = (snap["target_span_loss"] - snap["span_loss"]).sum()
    return np.array([[x1, x2]])
