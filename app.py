"""Interactive demo for the review:  streamlit run app.py"""
import io
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from linkfail import config as C
from linkfail import plots
from linkfail.localization import detect_flapping, localize_snapshot
from linkfail.pipeline import MODEL_FILES
from linkfail.preprocessing import features_from_snapshot
from linkfail.simulator import Fault, LinkDesign, snapshot

st.set_page_config(page_title="Link Failure Prediction", layout="wide")


@st.cache_resource
def load_models():
    missing = [f for f in MODEL_FILES.values() if not (C.MODELS / f).exists()]
    if missing:
        st.error("Trained models not found. Run `python run_pipeline.py` first.")
        st.stop()
    return {n: joblib.load(C.MODELS / f) for n, f in MODEL_FILES.items()}


@st.cache_data
def load_data():
    feats = pd.read_csv(C.RESULTS / "oof_predictions.csv", parse_dates=["timestamp"])
    metrics = json.loads((C.RESULTS / "metrics.json").read_text())
    return feats, metrics


models = load_models()
feats, metrics = load_data()

st.title("Link Failure Prediction & Localization in Cloud-Scale Optical Networks")
st.caption("Supervised learning on E2E span-loss / amplifier-gain telemetry: "
           "Logistic Regression (Newton), GDA and Linear SVM, all implemented from scratch.")

tab_inject, tab_replay, tab_perf = st.tabs(["🔧 Fault injection", "📈 Network replay", "📊 Model performance"])

# ---------------------------------------------------------------------------
with tab_inject:
    st.subheader("Inject a fault into a 19-span link and watch the models react")
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Fault type", ["none", "fiber_degradation", "amplifier_degradation", "fiber_cut"])
    span = c2.slider("Faulty span", 1, C.N_SPANS, 8)
    sev = c3.slider("Severity [dB]", 0.0, 16.0, 6.0, 0.5, disabled=kind in ("none", "fiber_cut"))
    comp = c4.slider("Amplifier compensation [dB]", 0.0, 3.0, 0.0, 0.5,
                     disabled=kind != "fiber_degradation")
    noise = st.checkbox("Add telemetry noise", value=True)

    faults = [] if kind == "none" else [Fault(kind, span - 1, C.FIBER_CUT_LOSS_DB if kind == "fiber_cut" else sev, comp)]
    snap, gsnr, rx, y_true = snapshot(LinkDesign.nominal(), faults, noise=noise)
    x = features_from_snapshot(snap)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("X1  Σ(gain − target)", f"{x[0, 0]:.2f} dB")
    m2.metric("X2  Σ(target − loss)", f"{x[0, 1]:.2f} dB")
    m3.metric("Physical GSNR", f"{gsnr:.1f} dB", help="Ground truth from the optical link budget")
    m4.metric("Ground truth", "NOT HEALTHY" if y_true else "Healthy")

    cols = st.columns(len(models))
    for col, (name, m) in zip(cols, models.items()):
        pred = int(m.predict(x)[0])
        if hasattr(m, "predict_proba"):
            detail = f"P(failure) = {m.predict_proba(x)[0, 1]:.3f}"
        else:
            detail = f"margin = {m.decision_function(x)[0]:+.2f}"
        verdict = "🔴 NOT HEALTHY" if pred else "🟢 Healthy"
        ok = "✅ correct" if pred == y_true else "❌ wrong"
        col.markdown(f"**{name}**  \n{verdict}  \n{detail}  \n{ok}")

    left, right = st.columns([1, 1])
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        sample = feats.sample(1500, random_state=0)
        plots._scatter(ax, sample[["X1", "X2"]].to_numpy(), sample["y"].to_numpy(), s=6)
        styles = {"Logistic Regression": "-", "GDA": "--", "Linear SVM": ":"}
        for name, m in models.items():
            plots._boundary(ax, m.theta_, (-16, 6), color=plots.MODEL_COLORS[name], ls=styles[name], label=name)
        ax.scatter(*x[0], s=220, marker="*", c=plots.INK, edgecolors="white", zorder=5, label="This snapshot")
        ax.set_xlim(-16, 6)
        ax.set_ylim(-20, 8)
        if x[0, 1] < -20:
            ax.annotate("fibre cut: X2 ≈ %.0f dB (off-chart)" % x[0, 1], (0.02, 0.03),
                        xycoords="axes fraction", fontsize=8)
        ax.legend(fontsize=7, loc="lower left")
        st.pyplot(fig)
    with right:
        loc = localize_snapshot(snap)
        fig, ax = plt.subplots(figsize=(5, 4))
        s = np.arange(1, C.N_SPANS + 1)
        ax.bar(s - 0.2, np.clip(loc["loss_res"], None, 20), 0.4, color=plots.ORANGE, label="Span loss − target")
        ax.bar(s + 0.2, loc["gain_res"], 0.4, color=plots.BLUE, label="Target gain − amp gain")
        ax.set_xticks(s)
        ax.set_xlabel("Span / amplifier")
        ax.set_ylabel("Residual [dB] (clipped at 20)")
        ax.legend(fontsize=7)
        ax.set_title("Per-span residuals used for localisation")
        st.pyplot(fig)
        if any(int(m.predict(x)[0]) for m in models.values()):
            st.error(f"**Localised:** span {loc['span']} — {loc['element'].replace('_', ' ')} "
                     f"(residual {loc['residual_db']:.1f} dB)")
        else:
            st.success("Link healthy — no localisation needed.")

# ---------------------------------------------------------------------------
with tab_replay:
    st.subheader("Replay 10 hours of telemetry for one link (out-of-fold predictions)")
    link = st.selectbox("Link", sorted(feats["link_id"].unique()),
                        index=sorted(feats["link_id"].unique()).index("R07-ZA"))
    d = feats[feats["link_id"] == link].reset_index(drop=True)
    alarms = detect_flapping(d)
    d["flapping_alarm"] = alarms["flapping_alarm"]
    buf = io.BytesIO()
    plots.link_timeline(d, buf)
    st.image(buf.getvalue())
    acc = (d["pred"] == d["y"]).mean()
    a, b, c = st.columns(3)
    a.metric("Accuracy on this link", f"{acc:.1%}")
    b.metric("Bins flagged not healthy", int(d["pred"].sum()))
    c.metric("Flapping alarms (bins)", int(d["flapping_alarm"].sum()))
    st.dataframe(
        d[d["pred"] == 1][["timestamp", "X1", "X2", "osnr_db", "fault_type", "fault_span", "flapping_alarm"]]
        .rename(columns={"fault_type": "true fault", "fault_span": "true span"}),
        width="stretch", height=260)

# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader("Held-out test set (1500 samples)")
    pm = metrics["paper_models"]
    table = pd.DataFrame({n: {"Accuracy": v["accuracy"], "Precision": v["precision"], "Recall": v["recall"],
                              "F1": v["f1"], "ROC-AUC": v["roc_auc"]} for n, v in pm.items()}).T
    for n, v in metrics["extensions"].items():
        table.loc[n + " (extension)"] = [v["accuracy"], v["precision"], v["recall"], v["f1"], v["roc_auc"]]
    st.dataframe(table.style.format("{:.4f}"), width="stretch")
    for f in ["decision_boundaries", "confusion_matrices", "roc_curves", "model_comparison", "training_curves"]:
        st.image(str(C.FIGURES / f"{f}.png"))
