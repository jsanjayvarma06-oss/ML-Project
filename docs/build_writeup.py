"""Build the two-page project write-up PDF from results/metrics.json.

    python docs/build_writeup.py
"""
import json
import sys
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from linkfail import config as C  # noqa: E402

OUT = ROOT / "docs" / "Writeup_Link_Failure_Prediction.pdf"

# DejaVu ships with matplotlib and has Greek/maths glyphs (Σ, θ, ≈, ≥).
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", FONT_DIR / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FONT_DIR / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FONT_DIR / "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

ACCENT = colors.HexColor("#256abf")
INK = colors.HexColor("#1a1a19")
body = ParagraphStyle("body", fontName="DV", fontSize=8.4, leading=10.6, textColor=INK,
                      alignment=TA_JUSTIFY, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="DV-B", fontSize=14, leading=17, alignment=TA_CENTER,
                    textColor=INK, spaceAfter=2)
sub = ParagraphStyle("sub", fontName="DV", fontSize=8.5, leading=11, alignment=TA_CENTER,
                     textColor=colors.HexColor("#52514e"), spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="DV-B", fontSize=9.6, leading=12, textColor=ACCENT,
                    spaceBefore=5, spaceAfter=2)
cap = ParagraphStyle("cap", parent=body, fontSize=7.4, leading=9, alignment=TA_CENTER,
                     textColor=colors.HexColor("#52514e"))
cell = ParagraphStyle("cell", fontName="DV", fontSize=7.6, leading=9.2, textColor=INK)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")


def P(text, style=body):
    return Paragraph(text, style)


def table(rows, widths):
    data = [[P(c, cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eefa")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#b0afa8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return t


def fig(name, width_mm):
    path = C.FIGURES / f"{name}.png"
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_mm * mm, height=width_mm * mm * h / w)


def build():
    r = json.loads((C.RESULTS / "metrics.json").read_text())
    pm, cv, ext = r["paper_models"], r["cross_validation"], r["extensions"]
    loc, fl, ew, ds = r["localization"], r["flapping"], r["early_warning"], r["dataset"]
    lr, gda, svm = pm["Logistic Regression"], pm["GDA"], pm["Linear SVM"]
    pct = lambda v: f"{100 * v:.1f}%"

    s = []
    s.append(P("Link Failure Prediction and Localization in Cloud-Scale Networks "
               "using Supervised Learning", h1))
    s.append(P("UE24CS352A Machine Learning — Mini-Project Write-up &nbsp;|&nbsp; "
               "Team: J Sanjay Varma (PES1UG24CS194) &amp; Kanak Pandey (PES1UG24CS212)", sub))

    s.append(P("1. Problem statement", h2))
    s.append(P(
        "Cloud providers carry traffic over optical links, and each end-to-end (E2E) link crosses many fibre "
        "spans and inline amplifiers. Fibres that slowly get lossier, amplifiers that lose gain, loose "
        "connectors and fibre cuts all reduce the optical signal-to-noise ratio (OSNR) until traffic fails. "
        "Following Bakhtiari [1], we predict from routine telemetry whether a link is <b>healthy</b> or "
        "<b>not healthy</b>, so operators can act before customers see an outage. We then <b>localise</b> "
        "the faulty span and element so a repair can be scheduled."))

    s.append(P("2. Dataset", h2))
    s.append(P(
        f"The paper's live network data is not public, so we built a physics-based telemetry simulator "
        f"that matches its format: <b>{ds['links']} directional E2E links</b> (10 routes × 2 traffic "
        f"directions), <b>{C.N_SPANS} spans</b> each, and 4 raw metrics per span every <b>2 minutes</b> "
        f"(span loss, target span loss, amplifier gain, target gain). This gives <b>{ds['samples']} "
        f"samples</b>, split 75/25 into {ds['train']} for training and {ds['test']} for testing, as in the "
        f"paper. We injected four fault types with known locations: fibre degradation, amplifier "
        f"degradation, flapping connector and fibre cut. Slow faults ramp up gradually; others are abrupt "
        f"steps. As in the paper, labels come from OSNR. We compute a generalised OSNR from an ASE + "
        f"GN-model nonlinear-noise link budget, and a sample is <i>not healthy</i> "
        f"({pct(ds['not_healthy_fraction'])} of samples) if any of the following holds: OSNR &lt; 17 dB, "
        f"received power more than 7 dB below nominal, or loss of signal. The telemetry includes "
        f"measurement noise, about 2% missing cells, whole missing bins, and no loss readings during fibre "
        f"cuts."))

    s.append(P("3. Approach", h2))
    s.append(P(
        "<b>Preprocessing (paper §III).</b> Loss readings missing because of a fibre cut are filled with the "
        "expected cut value (target + 40 dB). All other gaps are filled by linear interpolation in time, "
        "separately per link (traffic direction) and span. Per-span differences "
        "x<sub>1</sub> = gain − target gain and x<sub>2</sub> = target loss − span loss (Eq. 1) are summed "
        "over the 19 spans to give the two E2E features X1 = Σx<sub>1</sub> and X2 = Σx<sub>2</sub> (Eq. 2)."))
    s.append(P(
        f"<b>Classifiers (paper §IV), written from scratch in NumPy.</b> (i) <i>Logistic regression</i>: "
        f"minimises the cross-entropy J(θ) using Newton's method, converging in "
        f"{lr['newton_iterations']} iterations to J = {lr['final_cost']:.4f}. (ii) <i>GDA</i>: closed-form "
        f"maximum-likelihood estimates of φ, μ<sub>0</sub>, μ<sub>1</sub> and a shared Σ, which give a "
        f"linear boundary at p(y|x) = 0.5. (iii) <i>Linear SVM</i>: minimises the soft-margin hinge loss in "
        f"the primal with the mini-batch Pegasos sub-gradient solver. Each model was checked against "
        f"scikit-learn (99.8–100% agreement on test predictions). <b>Localisation:</b> the faulty element is "
        f"the span with the largest positive loss or gain residual. A link is reported as <b>flapping</b> "
        f"when its predicted health toggles at least 4 times in 30 minutes."))

    s.append(P("4. Implementation overview", h2))
    s.append(P(
        "The code is a Python package, <font face='DV-B'>linkfail/</font>, with modules for configuration, "
        "the simulator, preprocessing, models, localisation, evaluation, plots and the pipeline. "
        "<font face='DV-B'>run_pipeline.py</font> regenerates all data, models, metrics and figures in about "
        "20 seconds. <font face='DV-B'>app.py</font> is a Streamlit demo: it injects a fault into any span "
        "and shows each model's decision and the localised span live. "
        "<font face='DV-B'>predict.py</font> provides command-line inference, and a 12-test pytest suite "
        "checks the models against scikit-learn, the physics and the preprocessing. Robustness was measured "
        "with stratified 5-fold CV and with GroupKFold over routes, so the test links are never seen in "
        "training."))

    s.append(P("5. Results", h2))
    rows = [["Classifier", "Acc.", "Prec.", "Recall", "F1", "AUC", "5-fold CV", "Unseen routes", "Paper"]]
    for name, paper in (("Logistic Regression", "99%"), ("GDA", "97%"), ("Linear SVM", "99%")):
        m, c = pm[name], cv[name]
        rows.append([name, pct(m["accuracy"]), f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                     f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}",
                     f"{pct(c['stratified_5fold_mean'])} ± {100 * c['stratified_5fold_std']:.1f}",
                     f"{pct(c['route_grouped_5fold_mean'])} ± {100 * c['route_grouped_5fold_std']:.1f}", paper])
    for name, m in ext.items():
        rows.append([name + " *", pct(m["accuracy"]), f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                     f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}", "–", "–", "–"])
    s.append(table(rows, [44 * mm, 14 * mm, 14 * mm, 14 * mm, 13 * mm, 13 * mm, 22 * mm, 22 * mm, 13 * mm]))
    s.append(P("Table 1. Held-out test set (1500 samples). * = extension beyond the paper.", cap))
    s.append(fig("decision_boundaries", 172))
    s.append(P("Figure 1. Test data and the learnt linear decision boundaries (compare with paper Fig. 4–5).", cap))
    s.append(P(
        f"Learnt coefficients (θ0, θ1, θ2) are LR ({lr['theta'][0]:.2f}, {lr['theta'][1]:.2f}, "
        f"{lr['theta'][2]:.2f}), GDA ({gda['theta'][0]:.2f}, {gda['theta'][1]:.2f}, {gda['theta'][2]:.2f}) "
        f"and SVM ({svm['theta'][0]:.2f}, {svm['theta'][1]:.2f}, {svm['theta'][2]:.2f}). For LR and SVM, "
        f"θ1 ≈ θ2, so the rule they learn is \"alarm when the net power deficit X1 + X2 falls below about "
        f"−7 dB\", which is the physically correct rule. "
        f"<b>Localisation</b> identified the right span and element type for {pct(loc['span_accuracy'])} of "
        f"detected failures (n = {loc['n_evaluated']}). Over every fault-present sample, including minor "
        f"faults, span accuracy was {pct(loc['span_accuracy_all_fault_samples'])}, and a learnt "
        f"multinomial-LR localiser reached {pct(loc['ml_localizer_span_accuracy'])}. "
        f"<b>Flapping</b> detection found {fl['events_detected']} of {fl['flapping_events']} events, with a "
        f"{pct(fl['false_alarm_rate_non_flapping_bins'])} false-alarm rate. "
        f"<b>Early warning:</b> {ew['flagged_before_or_at_outage']} of {ew['events_with_outage']} "
        f"degradations that ended in an outage were flagged no later than the outage, and "
        f"{ew['flagged_strictly_before_outage']} were flagged before it (mean lead time "
        f"{ew['mean_lead_minutes']:.0f} min)."))

    s.append(Spacer(1, 3))
    s.append(fig("link_timeline", 150))
    s.append(P("Figure 2. Ten hours of one link: features, physical OSNR, and the true label against the "
               "logistic-regression prediction (out-of-fold, link unseen in training).", cap))
    s.append(fig("localization_example", 140))
    s.append(P("Figure 3. Per-span residuals of a detected failure; the largest residual pinpoints the faulty "
               "span and whether the fibre or the amplifier is at fault.", cap))

    s.append(P("6. Discussion and conclusions", h2))
    s.append(P(
        f"We reproduce the paper's ranking: <b>logistic regression ≈ linear SVM "
        f"({pct(lr['accuracy'])} / {pct(svm['accuracy'])}) &gt; GDA ({pct(gda['accuracy'])})</b>. Accuracy "
        f"is the same on unseen routes, so the models generalise across links. GDA trails because its "
        f"assumption of one Gaussian per class with a shared covariance does not hold. The not-healthy class "
        f"has several clusters: gain loss, fibre loss, and fibre cuts at X2 ≈ −40 dB. These stretch Σ and "
        f"tilt the boundary, so GDA's recall drops to {gda['recall']:.2f}. Our absolute accuracy is lower than "
        f"the paper's 99% because our labels follow OSNR physics. Summing over 19 spans discards where the "
        f"fault is: a deficit in span 1 harms all 19 downstream amplifiers, but the same deficit in span 19 "
        f"harms only one. Two results support this. An RBF-kernel SVM gains only "
        f"{100 * (ext['SVM (RBF kernel)']['accuracy'] - svm['accuracy']):.1f} points, so the boundary really "
        f"is close to linear, as the paper argued. Adding the 38 per-span residuals helps more "
        f"({pct(ext['LR + 38 per-span residuals']['accuracy'])}). <b>Conclusion:</b> two aggregated "
        f"telemetry features and a linear classifier are enough to detect failing links reliably and cheaply "
        f"at cloud scale, and per-span residuals then localise the fault. Future work: forecast OSNR to "
        f"increase lead time, add span-position-aware features, and validate on real operator data."))

    s.append(P("References", h2))
    s.append(P("[1] Z. Bakhtiari, \"Link Failure Prediction and Localization in Cloud Scale Networks using "
               "Supervised Learning,\" Stanford University. &nbsp;[2] S. Shalev-Shwartz et al., \"Pegasos: "
               "Primal Estimated sub-GrAdient SOlver for SVM,\" ICML 2007. &nbsp;[3] P. Poggiolini, \"The GN "
               "model of non-linear propagation in uncompensated coherent optical systems,\" JLT 2012.",
                 ParagraphStyle("ref", parent=body, fontSize=7.4, leading=9)))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=13 * mm, bottomMargin=12 * mm,
                            title="Link Failure Prediction and Localization - Write-up",
                            author="UE24CS352A Mini-Project Team")
    doc.build(s)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
