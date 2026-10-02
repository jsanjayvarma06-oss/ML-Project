"""End-to-end experiment: data -> features -> models -> evaluation -> artefacts."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from . import plots
from .evaluation import classification_metrics, early_warning
from .localization import (detect_flapping, flapping_report, localization_report,
                           localize)
from .models import make_models
from .preprocessing import build_features
from .simulator import generate_dataset

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "GDA": "gda.joblib",
    "Linear SVM": "linear_svm.joblib",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step_generate(seed=C.SEED):
    log("Generating synthetic telemetry (20 links x 19 spans x 300 bins)...")
    tele, health = generate_dataset(seed=seed)
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    tele.to_csv(C.TELEMETRY_FILE, index=False, compression="gzip")
    health.to_csv(C.HEALTH_FILE, index=False)
    log(f"  telemetry rows={len(tele):,}  missing cells="
        f"{int(tele[['span_loss', 'amp_gain']].isna().sum().sum()):,}  -> {C.TELEMETRY_FILE.name}")
    return tele, health


def step_preprocess():
    log("Preprocessing: fibre-cut fill, interpolation, Eq.1 diffs, Eq.2 aggregation...")
    tele = pd.read_csv(C.TELEMETRY_FILE, parse_dates=["timestamp"])
    health = pd.read_csv(C.HEALTH_FILE, parse_dates=["timestamp"])
    feats, resid = build_features(tele, health)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_csv(C.FEATURES_FILE, index=False)
    resid.to_csv(C.RESIDUALS_FILE, index=False)
    log(f"  {len(feats)} samples, {feats[C.TARGET].mean():.1%} not healthy -> {C.FEATURES_FILE.name}")
    return feats, resid


def _route(link_ids: pd.Series) -> np.ndarray:
    return link_ids.str.slice(0, 3).to_numpy()


def step_train_evaluate(seed=C.SEED):
    feats = pd.read_csv(C.FEATURES_FILE, parse_dates=["timestamp"])
    resid = pd.read_csv(C.RESIDUALS_FILE, parse_dates=["timestamp"])
    X = feats[C.FEATURES].to_numpy()
    y = feats[C.TARGET].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=C.TEST_SIZE,
                                      random_state=seed, stratify=y)
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
    log(f"Split: {len(idx_tr)} train / {len(idx_te)} test (75/25, stratified)")

    # ---- the paper's three classifiers (from scratch) --------------------
    models = make_models(seed)
    metrics, results = {}, {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        metrics[name] = classification_metrics(m, Xte, yte)
        metrics[name]["train_accuracy"] = float((m.predict(Xtr) == ytr).mean())
        metrics[name]["fit_seconds"] = round(time.time() - t0, 3)
        metrics[name]["theta"] = [float(v) for v in m.theta_]
        log(f"  {name:<20} test acc={metrics[name]['accuracy']:.4f}  "
            f"F1={metrics[name]['f1']:.4f}  AUC={metrics[name]['roc_auc']:.4f}")
    lr = models["Logistic Regression"]
    Xtr_s = lr._scale(Xtr)
    metrics["Logistic Regression"]["final_cost"] = float(
        lr.cost(np.hstack([np.ones((len(Xtr_s), 1)), Xtr_s]), ytr, lr.theta_scaled_))
    metrics["Logistic Regression"]["newton_iterations"] = lr.n_iter_
    results["paper_models"] = metrics

    # ---- validate our implementations against scikit-learn ---------------
    refs = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=1e4)),
        "GDA": LinearDiscriminantAnalysis(),
        "Linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
    }
    agreement = {}
    for name, ref in refs.items():
        ref.fit(Xtr, ytr)
        agreement[name] = {
            "sklearn_test_accuracy": float((ref.predict(Xte) == yte).mean()),
            "prediction_agreement": float((ref.predict(Xte) == models[name].predict(Xte)).mean()),
        }
        log(f"  sklearn {name:<12} acc={agreement[name]['sklearn_test_accuracy']:.4f}  "
            f"agreement with ours={agreement[name]['prediction_agreement']:.4f}")
    results["sklearn_validation"] = agreement

    # ---- robustness: k-fold CV and leave-routes-out ----------------------
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    gkf = GroupKFold(5)
    groups = _route(feats["link_id"])
    cv = {}
    for name, m in models.items():
        s1 = cross_val_score(clone(m), X, y, cv=skf)
        s2 = cross_val_score(clone(m), X, y, cv=gkf, groups=groups)
        cv[name] = {"stratified_5fold_mean": float(s1.mean()), "stratified_5fold_std": float(s1.std()),
                    "route_grouped_5fold_mean": float(s2.mean()), "route_grouped_5fold_std": float(s2.std())}
        log(f"  CV {name:<20} 5-fold={s1.mean():.4f}±{s1.std():.4f}  "
            f"unseen-routes={s2.mean():.4f}±{s2.std():.4f}")
    results["cross_validation"] = cv

    # ---- extensions ------------------------------------------------------
    ext = {}
    rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")).fit(Xtr, ytr)
    ext["SVM (RBF kernel)"] = classification_metrics(rbf, Xte, yte)
    res_cols = [c for c in resid.columns if c.startswith(("loss_res_", "gain_res_"))]
    Xr = np.hstack([X, resid[res_cols].to_numpy()])
    span_lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    span_lr.fit(Xr[idx_tr], ytr)
    ext["LR + 38 per-span residuals"] = classification_metrics(span_lr, Xr[idx_te], yte)
    for k, v in ext.items():
        log(f"  extension {k:<28} acc={v['accuracy']:.4f}")
    results["extensions"] = ext

    # ---- localisation ----------------------------------------------------
    pred_te = lr.predict(Xte)
    rep = localization_report(resid.iloc[idx_te], pred_te)
    faulty_tr = resid.iloc[idx_tr].query("y == 1 and fault_span > 0")
    faulty_te_mask = (resid.iloc[idx_te]["y"].to_numpy() == 1) & (pred_te == 1) & \
                     (resid.iloc[idx_te]["fault_span"].to_numpy() > 0)
    faulty_te = resid.iloc[idx_te][faulty_te_mask]
    ml_loc = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=5000))
    ml_loc.fit(faulty_tr[res_cols], faulty_tr["fault_span"])
    rep["ml_localizer_span_accuracy"] = float((ml_loc.predict(faulty_te[res_cols]) ==
                                               faulty_te["fault_span"]).mean())
    log(f"  localisation: rule span acc={rep['span_accuracy']:.4f}  element acc="
        f"{rep['element_accuracy']:.4f}  ML(multinomial LR) span acc={rep['ml_localizer_span_accuracy']:.4f}")
    results["localization"] = rep

    # ---- timeline analysis with out-of-fold predictions (unseen routes) --
    feats["pred"] = cross_val_predict(clone(lr), X, y, cv=gkf, groups=groups)
    alarms = detect_flapping(feats)
    results["flapping"] = flapping_report(feats, alarms)
    results["early_warning"] = early_warning(feats)
    log(f"  flapping: {results['flapping']}")
    log(f"  early warning: {results['early_warning']}")

    results["dataset"] = {
        "samples": int(len(y)), "train": int(len(idx_tr)), "test": int(len(idx_te)),
        "not_healthy_fraction": float(y.mean()),
        "fault_type_counts": feats["fault_type"].value_counts().to_dict(),
        "links": int(feats["link_id"].nunique()), "spans_per_link": C.N_SPANS,
    }

    # ---- artefacts -------------------------------------------------------
    C.MODELS.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        joblib.dump(m, C.MODELS / MODEL_FILES[name])
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    feats.to_csv(C.RESULTS / "oof_predictions.csv", index=False,
                 columns=["link_id", "timestamp", "X1", "X2", "osnr_db", "y", "pred",
                          "fault_type", "fault_span"])

    log("Rendering figures...")
    plots.dataset_scatter(X, y, C.FIGURES / "dataset_scatter.png")
    plots.decision_boundaries(models, Xte, yte, C.FIGURES / "decision_boundaries.png")
    plots.confusion_matrices(metrics, C.FIGURES / "confusion_matrices.png")
    plots.roc_curves(models, Xte, yte, C.FIGURES / "roc_curves.png")
    plots.training_curves(lr, models["Linear SVM"], C.FIGURES / "training_curves.png")
    busiest = feats.groupby("link_id")["y"].sum().idxmax()
    plots.link_timeline(feats[feats["link_id"] == busiest].reset_index(drop=True),
                        C.FIGURES / "link_timeline.png")
    rows = [(n, metrics[n]["accuracy"], "paper") for n in models] + \
           [(n, v["accuracy"], "ext") for n, v in ext.items()]
    plots.model_comparison(rows, C.FIGURES / "model_comparison.png")
    ex = faulty_te[faulty_te["fault_type"] == "fiber_degradation"].iloc[0]
    loc = localize(faulty_te.loc[[ex.name]]).iloc[0]
    plots.localization_example(
        np.array([ex[f"loss_res_{s:02d}"] for s in range(1, C.N_SPANS + 1)]),
        np.array([ex[f"gain_res_{s:02d}"] for s in range(1, C.N_SPANS + 1)]),
        int(ex["fault_span"]),
        f"Localisation example ({ex['link_id']}): predicted span {loc.pred_span} "
        f"({loc.pred_element}), true span {int(ex['fault_span'])}",
        C.FIGURES / "localization_example.png")
    write_summary(results)
    log(f"Done. Metrics -> {C.RESULTS / 'metrics.json'}, figures -> {C.FIGURES}")
    return results


def write_summary(r: dict):
    pm = r["paper_models"]
    lines = ["# Results summary", "",
             "Generated by `python run_pipeline.py`. All numbers are on the 25 % held-out test set "
             "unless stated otherwise.", "",
             "## Paper classifiers (implemented from scratch)", "",
             "| Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC | θ0 | θ1 | θ2 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, m in pm.items():
        t = m["theta"]
        lines.append(f"| {n} | {m['accuracy']:.2%} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['f1']:.3f} | {m['roc_auc']:.3f} | {t[0]:.3f} | {t[1]:.3f} | {t[2]:.3f} |")
    lines += ["", f"Logistic-regression final cost J(θ) = {pm['Logistic Regression']['final_cost']:.5f} "
              f"after {pm['Logistic Regression']['newton_iterations']} Newton iterations.", "",
              "## Agreement with scikit-learn", "", "| Model | sklearn accuracy | Prediction agreement |",
              "|---|---|---|"]
    for n, a in r["sklearn_validation"].items():
        lines.append(f"| {n} | {a['sklearn_test_accuracy']:.2%} | {a['prediction_agreement']:.2%} |")
    lines += ["", "## Cross-validation", "", "| Model | Stratified 5-fold | Unseen routes (GroupKFold) |",
              "|---|---|---|"]
    for n, c in r["cross_validation"].items():
        lines.append(f"| {n} | {c['stratified_5fold_mean']:.2%} ± {c['stratified_5fold_std']:.2%} | "
                     f"{c['route_grouped_5fold_mean']:.2%} ± {c['route_grouped_5fold_std']:.2%} |")
    lines += ["", "## Extensions", "", "| Model | Accuracy | F1 |", "|---|---|---|"]
    for n, m in r["extensions"].items():
        lines.append(f"| {n} | {m['accuracy']:.2%} | {m['f1']:.3f} |")
    loc, fl, ew = r["localization"], r["flapping"], r["early_warning"]
    lines += ["", "## Localisation, flapping and early warning", "",
              f"- Rule-based span localisation accuracy: **{loc['span_accuracy']:.2%}** "
              f"(element type {loc['element_accuracy']:.2%}, n={loc['n_evaluated']})",
              f"- ML (multinomial LR) span localisation accuracy: **{loc['ml_localizer_span_accuracy']:.2%}**",
              f"- Rule-based span accuracy on *every* fault-present sample, incl. minor faults the "
              f"classifier calls healthy: {loc['span_accuracy_all_fault_samples']:.2%} (n={loc['n_all_fault_samples']})",
              f"- Flapping events detected: {fl['events_detected']}/{fl['flapping_events']} "
              f"(false-alarm rate on other bins {fl['false_alarm_rate_non_flapping_bins']:.2%})",
              f"- Degradations that later caused an outage: {ew['events_with_outage']}; flagged before "
              f"the outage: {ew['flagged_strictly_before_outage']}; median lead time "
              f"{ew['median_lead_minutes']:.0f} min", ""]
    (C.RESULTS / "summary.md").write_text("\n".join(lines))


def run_all(seed=C.SEED):
    step_generate(seed)
    step_preprocess()
    return step_train_evaluate(seed)
