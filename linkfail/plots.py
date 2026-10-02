"""Figures for the report, slides and README."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .evaluation import scores

# Validated categorical slots (blue, orange, aqua) + recessive ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {"Logistic Regression": BLUE, "GDA": ORANGE, "Linear SVM": AQUA}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def _scatter(ax, X, y, s=10):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=s, c=BLUE, marker="o", alpha=0.45,
               linewidths=0, label="Healthy (y=0)")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=s + 6, c=ORANGE, marker="x", alpha=0.7,
               linewidths=0.9, label="Not healthy (y=1)")
    ax.set_xlabel("X1 = Σ (amp gain − target gain)  [dB]")
    ax.set_ylabel("X2 = Σ (target span loss − span loss)  [dB]")


def _boundary(ax, theta, xlim, **kw):
    """Plot theta0 + theta1*x1 + theta2*x2 = 0."""
    x1 = np.linspace(*xlim, 200)
    x2 = -(theta[0] + theta[1] * x1) / theta[2]
    ax.plot(x1, x2, lw=2, **kw)


def dataset_scatter(X, y, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _scatter(axes[0], X, y)
    axes[0].set_title("All 6000 samples (fibre cuts at X2 ≈ −40 dB)")
    _scatter(axes[1], X, y)
    axes[1].set_xlim(-16, 6)
    axes[1].set_ylim(-20, 8)
    axes[1].set_title("Zoom on the decision region")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def decision_boundaries(models, X, y, path):
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), sharey=True)
    xlim, ylim = (-16, 6), (-20, 8)
    for ax, (name, m) in zip(axes, models.items()):
        _scatter(ax, X, y, s=8)
        _boundary(ax, m.theta_, xlim, color=INK, label="Decision boundary")
        acc = (m.predict(X) == y).mean()
        ax.set_title(f"{name} — test accuracy {acc:.1%}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if ax is not axes[0]:
            ax.set_ylabel("")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion_matrices(metrics, path):
    names = list(metrics)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3))
    for ax, name in zip(axes, names):
        c = metrics[name]["confusion"]
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.sum() / 2 else INK)
        ax.set_xticks([0, 1], ["Healthy", "Not healthy"])
        ax.set_yticks([0, 1], ["Healthy", "Not healthy"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def roc_curves(models, X, y, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(y, scores(m, X))
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS.get(name, MUTED), label=name)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def training_curves(lr, svm, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(range(len(lr.cost_history_)), lr.cost_history_, "-o", ms=4, lw=2, color=BLUE)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Newton iteration")
    axes[0].set_ylabel("J(θ)  (log scale)")
    axes[0].set_title(f"Logistic regression: converged in {lr.n_iter_} Newton steps")
    axes[1].plot(svm.objective_history_, lw=2, color=AQUA)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Primal SVM objective")
    axes[1].set_title("Linear SVM: Pegasos sub-gradient descent")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def link_timeline(link_df, path):
    """Time series of one link: features, OSNR and predicted vs true health."""
    t = (link_df["timestamp"] - link_df["timestamp"].iloc[0]).dt.total_seconds() / 3600
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    axes[0].plot(t, link_df["X1"], lw=1.5, color=BLUE, label="X1 (gain diff)")
    axes[0].plot(t, link_df["X2"], lw=1.5, color=ORANGE, label="X2 (loss diff)")
    axes[0].set_ylabel("dB")
    axes[0].legend(loc="lower left", ncol=2)
    axes[0].set_title(f"Link {link_df['link_id'].iloc[0]}: telemetry features, OSNR and health")
    axes[1].plot(t, link_df["osnr_db"].clip(lower=5), lw=1.5, color=INK, label="Generalised OSNR")
    axes[1].axhline(17, color=ORANGE, ls="--", lw=1, label="Unhealthy threshold 17 dB")
    axes[1].axhline(13, color=MUTED, ls=":", lw=1, label="Outage threshold 13 dB")
    axes[1].set_ylabel("dB")
    axes[1].legend(loc="lower left", ncol=3, fontsize=7)
    axes[2].step(t, link_df["y"] + 0.04, where="post", lw=1.5, color=MUTED, label="True label")
    axes[2].step(t, link_df["pred"] - 0.04, where="post", lw=1.5, color=AQUA, label="LR prediction")
    axes[2].set_yticks([0, 1], ["Healthy", "Not healthy"])
    axes[2].set_xlabel("Time [hours]")
    axes[2].legend(loc="upper right", ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def model_comparison(rows, path):
    """rows: list of (label, accuracy, group)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    labels = [r[0] for r in rows]
    accs = [r[1] * 100 for r in rows]
    colors = [BLUE if r[2] == "paper" else AQUA for r in rows]
    bars = ax.barh(labels, accs, color=colors, height=0.6)
    for b, a in zip(bars, accs):
        ax.text(a + 0.15, b.get_y() + b.get_height() / 2, f"{a:.1f}%", va="center", fontsize=8, color=INK)
    lo = max(0, min(accs) - 5)
    ax.set_xlim(lo, 100.8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Test accuracy [%]  (axis starts at {lo:.0f}%)")
    ax.set_title("Paper models (blue) vs. extensions (aqua)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def localization_example(loss_res, gain_res, true_span, title, path):
    spans = np.arange(1, len(loss_res) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    w = 0.4
    ax.bar(spans - w / 2, loss_res, width=w, color=ORANGE, label="Span loss − target")
    ax.bar(spans + w / 2, gain_res, width=w, color=BLUE, label="Target gain − amp gain")
    ax.axvline(true_span, color=MUTED, ls="--", lw=1, label=f"True faulty span {true_span}")
    ax.set_xticks(spans)
    ax.set_xlabel("Span / amplifier")
    ax.set_ylabel("Residual [dB]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
