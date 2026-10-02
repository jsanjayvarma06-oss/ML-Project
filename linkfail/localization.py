"""Failure localisation and flapping detection.

Once the E2E classifier flags a link as not healthy, the per-span residuals
tell us *where* the problem is:

    loss_res_n = span_loss_n - target_span_loss_n   (> 0: fibre lossier than plan)
    gain_res_n = target_gain_n - amp_gain_n          (> 0: amplifier under-performing)

Rule-based localiser: the faulty element is the span/amp with the largest
positive residual. A loss residual above ~25 dB is reported as a fibre cut.

Flapping detector: a link whose predicted health toggles many times within a
short sliding window is reported as flapping (intermittent connector, etc.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CUT_THRESHOLD_DB = 25.0


def _residual_arrays(resid: pd.DataFrame):
    spans = range(1, C.N_SPANS + 1)
    loss = resid[[f"loss_res_{s:02d}" for s in spans]].to_numpy()
    gain = resid[[f"gain_res_{s:02d}" for s in spans]].to_numpy()
    return loss, gain


def localize(resid: pd.DataFrame) -> pd.DataFrame:
    """Return predicted faulty span (1-based), element type and residual size."""
    loss, gain = _residual_arrays(resid)
    score = np.maximum(loss, gain)
    span_idx = score.argmax(axis=1)
    rows = np.arange(len(span_idx))
    l, g = loss[rows, span_idx], gain[rows, span_idx]
    element = np.where(l >= g, "fiber", "amplifier")
    element = np.where(l >= CUT_THRESHOLD_DB, "fiber_cut", element)
    return pd.DataFrame({
        "pred_span": span_idx + 1,
        "pred_element": element,
        "residual_db": score[rows, span_idx].round(2),
    }, index=resid.index)


def localize_snapshot(snap: pd.DataFrame) -> dict:
    """Localise on a single snapshot table (one row per span)."""
    loss = (snap["span_loss"] - snap["target_span_loss"]).to_numpy()
    gain = (snap["target_gain"] - snap["amp_gain"]).to_numpy()
    score = np.maximum(loss, gain)
    k = int(score.argmax())
    element = "fiber" if loss[k] >= gain[k] else "amplifier"
    if loss[k] >= CUT_THRESHOLD_DB:
        element = "fiber_cut"
    return {"span": k + 1, "element": element, "residual_db": float(score[k]),
            "loss_res": loss, "gain_res": gain}


TRUE_ELEMENT = {
    "fiber_degradation": "fiber",
    "flapping": "fiber",
    "amplifier_degradation": "amplifier",
    "fiber_cut": "fiber_cut",
}


def localization_report(resid: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Score localisation on samples that are truly faulty and predicted faulty."""
    mask = (resid["y"].to_numpy() == 1) & (np.asarray(y_pred) == 1) & (resid["fault_span"].to_numpy() > 0)
    sub = resid[mask]
    loc = localize(sub)
    true_el = sub["fault_type"].map(TRUE_ELEMENT)
    span_ok = loc["pred_span"].to_numpy() == sub["fault_span"].to_numpy()
    el_ok = loc["pred_element"].to_numpy() == true_el.to_numpy()
    per_type = (pd.DataFrame({"type": sub["fault_type"], "ok": span_ok})
                .groupby("type")["ok"].mean().round(4).to_dict())
    all_faults = resid[resid["fault_span"].to_numpy() > 0]
    all_ok = localize(all_faults)["pred_span"].to_numpy() == all_faults["fault_span"].to_numpy()
    return {
        "n_evaluated": int(mask.sum()),
        "span_accuracy_all_fault_samples": float(all_ok.mean()),
        "n_all_fault_samples": int(len(all_faults)),
        "span_accuracy": float(span_ok.mean()),
        "element_accuracy": float(el_ok.mean()),
        "span_and_element_accuracy": float((span_ok & el_ok).mean()),
        "span_accuracy_by_fault": per_type,
    }


def flapping_scores(pred: pd.Series, window: int = 15) -> pd.Series:
    """Number of health-state toggles in the trailing `window` bins."""
    toggles = pred.diff().abs().fillna(0)
    return toggles.rolling(window, min_periods=1).sum()


def detect_flapping(feats: pd.DataFrame, pred_col: str = "pred",
                    window: int = 15, min_toggles: int = 4) -> pd.DataFrame:
    """Flag (link, time) points where the predicted health is flapping."""
    out = feats[["link_id", "timestamp", pred_col]].copy()
    out["toggles"] = (out.groupby("link_id", group_keys=False)[pred_col]
                      .apply(lambda s: flapping_scores(s, window)))
    out["flapping_alarm"] = (out["toggles"] >= min_toggles).astype(int)
    return out


def flapping_report(feats: pd.DataFrame, alarms: pd.DataFrame) -> dict:
    """Event-level recall over flapping events and false-alarm rate elsewhere."""
    df = feats[["fault_type", "event_id", "link_id"]].join(alarms["flapping_alarm"])
    flap = df[df["fault_type"] == "flapping"]
    events = flap.groupby(["link_id", "event_id"])["flapping_alarm"].max()
    other = df[df["fault_type"] != "flapping"]
    return {
        "flapping_events": int(len(events)),
        "events_detected": int(events.sum()),
        "event_recall": float(events.mean()) if len(events) else float("nan"),
        "false_alarm_rate_non_flapping_bins": float(other["flapping_alarm"].mean()),
    }
