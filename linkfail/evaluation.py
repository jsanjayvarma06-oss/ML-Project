"""Metrics used to compare the classifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def scores(model, X):
    """Continuous score for ROC: probability if available, else margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def classification_metrics(model, X, y) -> dict:
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores(model, X))),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def early_warning(feats: pd.DataFrame, pred_col: str = "pred", bin_minutes: int = 2) -> dict:
    """How long before a hard outage did the classifier first flag the link?

    For every fault event that eventually causes an outage, lead time is the
    gap between the first bin predicted 'not healthy' and the first outage bin.
    Fibre cuts are excluded: they are instantaneous, nothing can precede them.
    """
    leads = []
    ev = feats[(feats["event_id"] >= 0) & (feats["fault_type"] != "fiber_cut")]
    for _, g in ev.groupby(["link_id", "event_id"]):
        g = g.reset_index(drop=True)
        if not g["outage"].any():
            continue
        first_outage = int(g["outage"].to_numpy().argmax())
        flagged = g.index[g[pred_col] == 1]
        first_flag = int(flagged.min()) if len(flagged) else None
        if first_flag is None or first_flag > first_outage:
            leads.append(-1)            # missed / late
        else:
            leads.append((first_outage - first_flag) * bin_minutes)
    leads = np.array(leads)
    hit = leads[leads >= 0]
    return {
        "events_with_outage": int(len(leads)),
        "flagged_before_or_at_outage": int((leads >= 0).sum()),
        "flagged_strictly_before_outage": int((leads > 0).sum()),
        "median_lead_minutes": float(np.median(hit)) if len(hit) else 0.0,
        "mean_lead_minutes": float(np.mean(hit)) if len(hit) else 0.0,
    }
