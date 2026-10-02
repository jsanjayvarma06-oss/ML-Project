"""Build the review slide deck (16:9 .pptx) from results/metrics.json and figures.

    python docs/build_slides.py
"""
import json
import sys
from pathlib import Path

from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from linkfail import config as C  # noqa: E402

OUT = ROOT / "docs" / "Slides_Link_Failure_Prediction.pptx"

NAVY = RGBColor(0x0B, 0x1F, 0x3A)
NAVY_2 = RGBColor(0x16, 0x33, 0x5C)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA = RGBColor(0x1B, 0xAF, 0x7A)
INK = RGBColor(0x1A, 0x1A, 0x19)
MUTED = RGBColor(0x52, 0x51, 0x4E)
TINT = RGBColor(0xEE, 0xF3, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ICE = RGBColor(0xCA, 0xDC, 0xFC)
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------------------
def bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=INK, font=BODY, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space_after=6):
    """runs: str | list of paragraphs; a paragraph is str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        parts = [(para, {})] if isinstance(para, str) else para
        for t, o in parts:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = o.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        parts = [(item, {})] if isinstance(item, str) else item
        r0 = p.add_run()
        r0.text = "▸  "
        r0.font.size, r0.font.color.rgb, r0.font.name = Pt(size), ORANGE, BODY
        for t, o in parts:
            r = p.add_run()
            r.text = t
            r.font.name = BODY
            r.font.size = Pt(size)
            r.font.bold = o.get("bold", False)
            r.font.color.rgb = o.get("color", color)
    return tb


def box(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, radius=0.08):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def circle_num(slide, x, y, n, d=0.55, fill=ORANGE):
    c = box(slide, x, y, d, d, fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(16), True, WHITE, BODY
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return c


def image(slide, name, x, y, w=None, h=None):
    path = C.FIGURES / f"{name}.png"
    iw, ih = PILImage.open(path).size
    if w and h:                               # fit inside box, keep aspect
        if iw / ih > w / h:
            h2 = w * ih / iw
            y, h = y + (h - h2) / 2, h2
        else:
            w2 = h * iw / ih
            x, w = x + (w - w2) / 2, w2
    elif w:
        h = w * ih / iw
    else:
        w = h * iw / ih
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def title(slide, t, sub=None, dark=False):
    text(slide, 0.6, 0.42, 12.1, 0.8, t, size=34, font=HEAD, bold=True, color=WHITE if dark else NAVY)
    if sub:
        text(slide, 0.6, 1.18, 12.1, 0.45, sub, size=16, color=ICE if dark else MUTED)


def stat(slide, x, y, w, value, label, color=BLUE, fill=TINT, vsize=34):
    box(slide, x, y, w, 1.35, fill)
    text(slide, x + 0.2, y + 0.12, w - 0.4, 0.7, value, size=vsize, bold=True, color=color, font=HEAD)
    text(slide, x + 0.2, y + 0.84, w - 0.4, 0.45, label, size=12, color=MUTED)


def footer(slide, n, dark=False):
    text(slide, 11.9, 7.0, 0.9, 0.3, str(n), size=11, color=ICE if dark else MUTED, align=PP_ALIGN.RIGHT)


def new_slide(dark=False):
    s = prs.slides.add_slide(BLANK)
    bg(s, NAVY if dark else WHITE)
    return s


# ---------------------------------------------------------------------------
def build():
    r = json.loads((C.RESULTS / "metrics.json").read_text())
    pm, cv, ext = r["paper_models"], r["cross_validation"], r["extensions"]
    loc, fl, ew, ds = r["localization"], r["flapping"], r["early_warning"], r["dataset"]
    lr, gda, svm = pm["Logistic Regression"], pm["GDA"], pm["Linear SVM"]
    pct = lambda v: f"{100 * v:.1f}%"
    n = 0

    # 1 — Title --------------------------------------------------------------
    s = new_slide(dark=True)
    n += 1
    text(s, 0.8, 0.8, 8, 0.4, "UE24CS352A  ·  MACHINE LEARNING MINI-PROJECT", size=14, color=ICE, bold=True)
    text(s, 0.8, 1.6, 11.5, 2.2, "Link Failure Prediction & Localization in Cloud-Scale Networks",
         size=44, font=HEAD, bold=True, color=WHITE)
    text(s, 0.8, 3.75, 11, 0.6, "Supervised learning on optical-line telemetry: "
         "Logistic Regression · GDA · Linear SVM", size=20, color=ICE)
    # motif: a stylised optical line of spans
    y0 = 5.25
    for i in range(10):
        x0 = 0.8 + i * 1.18
        c = box(s, x0, y0, 0.32, 0.32, ORANGE if i == 6 else AQUA, shape=MSO_SHAPE.OVAL)
        if i < 9:
            ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0 + 0.32), Inches(y0 + 0.16),
                                        Inches(x0 + 1.18), Inches(y0 + 0.16))
            ln.line.color.rgb = ICE
            ln.line.width = Pt(2)
    text(s, 0.8, 5.75, 11, 0.4, "fibre span  ●  amplifier  ●  the orange one is failing — can we spot it from telemetry?",
         size=12, color=ICE)
    text(s, 0.8, 6.5, 11, 0.4, "Team: J Sanjay Varma (PES1UG24CS194)  ·  Kanak Pandey (PES1UG24CS212)", size=16, color=WHITE)
    s.notes_slide.notes_text_frame.text = (
        "Introduce the team and the problem: we reproduce and extend a Stanford paper that predicts optical "
        "link failures from span-loss and amplifier-gain telemetry, and then localises the failing span.")

    # 2 — Problem ------------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "The problem: failing links hurt customers", "Every E2E link crosses 19 fibre spans and amplifiers — any one can degrade")
    bullets(s, 0.6, 1.95, 6.0, 4.6, [
        [("Cloud services ride on ", {}), ("IP-over-optical", {"bold": True}), (" links spanning hundreds of km", {})],
        [("Fibres age, bend or get cut; amplifiers lose gain; connectors ", {}), ("flap", {"bold": True})],
        [("Each fault lowers the ", {}), ("OSNR", {"bold": True}), (" until traffic fails", {})],
        [("Reactive repair = outage + long maintenance window + hard to find ", {}), ("where", {"bold": True})],
        [("Goal: ", {"bold": True, "color": ORANGE}),
         ("flag marginal links ", {"bold": True}), ("before", {"bold": True, "color": ORANGE}),
         (" the outage and ", {}), ("localise", {"bold": True}), (" the faulty span", {})],
    ], size=17)
    box(s, 7.0, 1.95, 5.75, 4.6, TINT)
    text(s, 7.3, 2.15, 5.2, 0.4, "What the network gives us every 2 minutes", size=15, bold=True, color=NAVY)
    rows = [("Span loss", "measured fibre attenuation"), ("Target span loss", "planned value"),
            ("Amplifier gain", "measured"), ("Target gain", "planned value")]
    for i, (a, b) in enumerate(rows):
        yy = 2.8 + i * 0.78
        circle_num(s, 7.3, yy, i + 1, d=0.5, fill=BLUE)
        text(s, 8.0, yy + 0.02, 4.5, 0.3, a, size=16, bold=True, color=INK)
        text(s, 8.0, yy + 0.32, 4.5, 0.3, b, size=12, color=MUTED)
    text(s, 7.3, 5.95, 5.2, 0.5, "× 19 spans × 20 links  →  can ML read link health from this?",
         size=13, color=NAVY, bold=True)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Explain why proactive detection matters for cloud providers. The four raw metrics are exactly the ones "
        "the paper uses.")

    # 3 — Dataset -------------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "Dataset: same shape as the paper", "Live operator data is private, so we built a physics-based telemetry simulator")
    stats = [(str(ds["links"]), "directional E2E links"), (str(C.N_SPANS), "spans per link"),
             (f"{ds['samples']:,}", "samples (2-min bins)"), (pct(ds["not_healthy_fraction"]), "labelled not healthy")]
    for i, (v, l) in enumerate(stats):
        stat(s, 0.6 + i * 3.08, 1.9, 2.85, v, l, color=ORANGE if i == 3 else BLUE)
    text(s, 0.6, 3.55, 6, 0.4, "Injected faults (with ground-truth span)", size=16, bold=True, color=NAVY)
    ft = ds["fault_type_counts"]
    bullets(s, 0.6, 4.05, 6.0, 2.8, [
        [("Fibre degradation", {"bold": True}), (f"  1–14 dB, ramps or steps ({ft['fiber_degradation']} bins)", {})],
        [("Amplifier degradation", {"bold": True}), (f"  gain ↓, noise figure ↑ ({ft['amplifier_degradation']})", {})],
        [("Flapping connector", {"bold": True}), (f"  intermittent loss ({ft['flapping']})", {})],
        [("Fibre cut", {"bold": True}), (f"  +40 dB, no loss reading / LOS ({ft['fiber_cut']})", {})],
    ], size=15)
    text(s, 6.9, 3.55, 6, 0.4, "Labels from OSNR (as in the paper)", size=16, bold=True, color=NAVY)
    bullets(s, 6.9, 4.05, 5.9, 2.8, [
        "Generalised OSNR = ASE noise + GN-model nonlinear noise over all 19 amplifiers",
        [("Not healthy", {"bold": True, "color": ORANGE}), (" if OSNR < 17 dB, Rx power > 7 dB low, or loss of signal", {})],
        "Realistic mess: 0.12 dB noise, ~2 % missing cells, missing bins, no data during cuts",
        "75 / 25 split → 4500 train, 1500 test",
    ], size=15)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Q&A prep: Why simulate? The paper's data is from a private time-series DB and its appendix link is dead. "
        "The simulator reproduces the paper's format exactly and gives ground-truth fault locations, which the "
        "paper never had, so localisation can be scored. Physics: constant-gain amplifiers, so a power deficit "
        "propagates downstream; ASE OSNR per amp = 58 + P_in - NF; NLI via GN model.")

    # 4 — Preprocessing -------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "Preprocessing & feature engineering", "Paper §III: from 4 × 19 raw values to 2 features per link per bin")
    steps = [("Fibre-cut fill", "LOS-flagged gaps ← target + 40 dB"),
             ("Interpolate", "linear in time, per link direction & span"),
             ("Span diffs (Eq. 1)", "x1 = gain − target gain\nx2 = target loss − span loss"),
             ("Aggregate (Eq. 2)", "X1 = Σ x1 ,  X2 = Σ x2\nover 19 spans"),
             ("Join labels", "OSNR-based y ∈ {0, 1}")]
    for i, (a, b) in enumerate(steps):
        x0 = 0.6 + i * 2.5
        box(s, x0, 2.1, 2.2, 2.3, TINT)
        circle_num(s, x0 + 0.2, 2.3, i + 1)
        text(s, x0 + 0.2, 3.0, 1.85, 0.45, a, size=15, bold=True, color=NAVY)
        text(s, x0 + 0.2, 3.45, 1.85, 0.9, b, size=12, color=INK)
        if i < 4:
            ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x0 + 2.22), Inches(3.1), Inches(0.26), Inches(0.3))
            ar.fill.solid()
            ar.fill.fore_color.rgb = ORANGE
            ar.line.fill.background()
    box(s, 0.6, 4.85, 12.1, 1.75, NAVY, radius=0.06)
    text(s, 0.95, 5.05, 11.4, 0.4, "Why these two features?", size=16, bold=True, color=WHITE)
    text(s, 0.95, 5.5, 11.4, 1.0,
         "X1 + X2 is the net optical power the link has lost versus its design. A healthy link sits near "
         "(0, 0); fibre problems push X2 negative, amplifier problems push X1 negative. The features are "
         "independent of each link's absolute span lengths, so one model works for every link.",
         size=14, color=ICE)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Walk through linkfail/preprocessing.py. Mention the fibre-cut special case is straight from the paper: "
        "missing because of a root-cause failure is filled with the expected cut value, not interpolated.")

    # 5 — Models --------------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "Three linear classifiers, written from scratch",
          "NumPy implementations in linkfail/models.py, checked against scikit-learn")
    cards = [
        ("Logistic Regression", BLUE, "Discriminative",
         ["J(θ) = −1/n Σ [y log h + (1−y) log(1−h)]", "Newton's method: θ ← θ − H⁻¹∇J",
          f"Converged in {lr['newton_iterations']} iterations, J = {lr['final_cost']:.4f}"]),
        ("GDA", ORANGE, "Generative",
         ["y ~ Bernoulli(φ),  x | y ~ N(μ_y, Σ)", "Closed-form MLE for φ, μ₀, μ₁, shared Σ",
          "Shared Σ ⇒ linear boundary at p(y|x) = 0.5"]),
        ("Linear SVM", AQUA, "Max-margin",
         ["min λ/2‖w‖² + 1/n Σ max(0, 1 − y(wᵀx + b))", "Mini-batch Pegasos sub-gradient solver",
          "Iterate averaging, 200 epochs"]),
    ]
    agree = r["sklearn_validation"]
    for i, (name, col, kind, lines) in enumerate(cards):
        x0 = 0.6 + i * 4.1
        box(s, x0, 1.95, 3.85, 3.9, TINT)
        b = box(s, x0 + 0.3, 2.2, 0.4, 0.4, col, shape=MSO_SHAPE.OVAL)
        text(s, x0 + 0.85, 2.18, 2.9, 0.45, name, size=19, bold=True, color=NAVY, font=HEAD)
        text(s, x0 + 0.3, 2.75, 3.3, 0.3, kind.upper(), size=11, bold=True, color=MUTED)
        bullets(s, x0 + 0.3, 3.2, 3.35, 2.3, lines, size=13)
        text(s, x0 + 0.3, 5.3, 3.3, 0.4, f"sklearn agreement: {pct(agree[name]['prediction_agreement'])}",
             size=12, bold=True, color=col)
    text(s, 0.6, 6.2, 12.1, 0.5, "All three take raw (X1, X2); LR and SVM standardise internally and report θ in "
         "raw dB units.", size=13, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Q&A prep: Why Newton? Two features means a 3×3 Hessian, so each step is cheap and convergence is "
        "quadratic. GDA vs LR: GDA makes stronger assumptions (Gaussian classes), which help when true, hurt when "
        "not. Pegasos: projected stochastic sub-gradient on the primal; step 1/(λt).")

    # 6 — Data view -----------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "What the data looks like", "Each dot is one link in one 2-minute bin")
    image(s, "dataset_scatter", 0.4, 1.9, w=8.6)
    bullets(s, 9.2, 1.95, 3.6, 4.8, [
        [("Healthy", {"bold": True, "color": BLUE}), (" links cluster at (0, 0)", {})],
        [("Amplifier faults", {"bold": True}), (" spread left (X1 ≪ 0)", {})],
        [("Fibre faults", {"bold": True}), (" spread down (X2 ≪ 0)", {})],
        [("Fibre cuts", {"bold": True}), (" form a separate cluster at X2 ≈ −40 dB", {})],
        [("The boundary is ", {}), ("nearly linear", {"bold": True}),
         (", but points near it overlap", {})],
    ], size=15)
    footer(s, n)

    # 7 — Results -------------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "Results: decision boundaries on the test set", "1500 held-out samples; compare with Fig. 4–5 of the paper")
    image(s, "decision_boundaries", 0.45, 1.75, w=12.4)
    stat(s, 0.6, 5.45, 3.9, pct(lr["accuracy"]), "Logistic Regression  (paper: 99 %)", vsize=28)
    stat(s, 4.72, 5.45, 3.9, pct(gda["accuracy"]), "GDA  (paper: 97 %)", color=ORANGE, vsize=28)
    stat(s, 8.84, 5.45, 3.9, pct(svm["accuracy"]), "Linear SVM  (paper: 99 %)", color=AQUA, vsize=28)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        f"LR coefficients θ = ({lr['theta'][0]:.2f}, {lr['theta'][1]:.2f}, {lr['theta'][2]:.2f}). "
        "θ1 ≈ θ2, so the learnt rule is 'alarm when X1 + X2 is below about −7 dB', i.e. net power deficit, which "
        "is the physically right rule.")

    # 8 — Metrics -------------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "Beyond accuracy: robust on unseen links",
          "Precision/recall on the test set, and cross-validation over the whole dataset")
    hdr = ["Classifier", "Acc.", "Precision", "Recall", "F1", "AUC", "5-fold CV", "Unseen routes"]
    data = [hdr]
    for name in ("Logistic Regression", "GDA", "Linear SVM"):
        m, c = pm[name], cv[name]
        data.append([name, pct(m["accuracy"]), f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}",
                     f"{m['roc_auc']:.3f}", f"{pct(c['stratified_5fold_mean'])} ± {100 * c['stratified_5fold_std']:.1f}",
                     f"{pct(c['route_grouped_5fold_mean'])} ± {100 * c['route_grouped_5fold_std']:.1f}"])
    widths = [2.6, 1.1, 1.25, 1.1, 1.0, 1.0, 1.9, 2.1]
    tbl = s.shapes.add_table(len(data), len(hdr), Inches(0.6), Inches(1.9), Inches(sum(widths)),
                             Inches(0.45 * len(data))).table
    for j, wdt in enumerate(widths):
        tbl.columns[j].width = Inches(wdt)
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if i == 0 else (TINT if i % 2 else WHITE)
            p = cell.text_frame.paragraphs[0]
            p.text = ""
            run = p.add_run()
            run.text = val
            run.font.size = Pt(14)
            run.font.name = BODY
            run.font.bold = i == 0 or j == 0
            run.font.color.rgb = WHITE if i == 0 else INK
    image(s, "confusion_matrices", 0.6, 4.0, h=2.75)
    image(s, "roc_curves", 9.75, 3.95, h=2.85)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Unseen routes = GroupKFold where entire fibre routes (both directions) are held out, so this is the "
        "accuracy on a link the model has never seen. GDA's precision is fine but recall is only 0.59: it misses "
        "many failures.")

    # 9 — Discussion -----------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "Why GDA trails — and why we're below 99 %", "The ranking matches the paper; the gap has a physical cause")
    image(s, "model_comparison", 0.4, 2.2, w=6.85)
    box(s, 7.45, 1.85, 5.3, 2.25, TINT)
    text(s, 7.7, 2.0, 4.9, 0.4, "GDA's assumption breaks", size=17, bold=True, color=ORANGE)
    text(s, 7.7, 2.45, 4.9, 1.6, "'Not healthy' is not one Gaussian: amplifier, fibre and cut clusters stretch "
         f"the shared Σ and tilt the boundary, so recall falls to {gda['recall']:.2f}.", size=16)
    box(s, 7.45, 4.3, 5.3, 2.45, TINT)
    text(s, 7.7, 4.45, 4.9, 0.4, "Σ over spans hides where the fault is", size=17, bold=True, color=BLUE)
    text(s, 7.7, 4.9, 4.9, 1.8, "A 7 dB loss in span 1 degrades all 19 downstream amplifiers; the same loss in "
         "span 19 affects only one. The RBF SVM adds just "
         f"{100 * (ext['SVM (RBF kernel)']['accuracy'] - svm['accuracy']):.1f} pts, while per-span features reach "
         f"{pct(ext['LR + 38 per-span residuals']['accuracy'])}.", size=16)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Key Q&A point: our labels come from a real OSNR link budget, in which the position of the fault matters. "
        "The aggregated features cannot see position, so a fraction of the boundary points are ambiguous. That is "
        "information loss, not a failure of linear models: the nonlinear RBF SVM barely helps.")

    # 10 — Localisation ---------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "Localisation, flapping & early warning", "Turning a red flag into a work order")
    image(s, "localization_example", 0.5, 1.75, w=7.6)
    text(s, 0.6, 4.75, 7.4, 1.6,
         [[("Rule: ", {"bold": True}), ("the faulty element is the span with the largest positive residual — "
                                        "loss residual means fibre, gain residual means amplifier, ≥ 25 dB means cut.", {})],
          [("Flapping: ", {"bold": True}), ("≥ 4 health toggles within 30 minutes.", {})]], size=14)
    stat(s, 8.5, 1.8, 4.25, pct(loc["span_accuracy"]), f"span + element correct (n = {loc['n_evaluated']} detected failures)",
         color=AQUA)
    stat(s, 8.5, 3.35, 4.25, f"{fl['events_detected']}/{fl['flapping_events']}",
         f"flapping events caught ({pct(fl['false_alarm_rate_non_flapping_bins'])} false-alarm rate)")
    stat(s, 8.5, 4.9, 4.25, f"{ew['flagged_before_or_at_outage']}/{ew['events_with_outage']}",
         f"outages flagged in time ({ew['flagged_strictly_before_outage']} early, mean {ew['mean_lead_minutes']:.0f} min ahead)",
         color=ORANGE)
    text(s, 0.6, 6.45, 12, 0.4, f"Harder test, every fault incl. minor ones: {pct(loc['span_accuracy_all_fault_samples'])} "
         f"span accuracy · learnt multinomial-LR localiser: {pct(loc['ml_localizer_span_accuracy'])}",
         size=12, color=MUTED)
    footer(s, n)

    # 11 — Live demo -------------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "Live demo", "streamlit run app.py")
    image(s, "link_timeline", 6.6, 1.7, w=6.2)
    demo = [("Fault injection", "Pick a fault type, span and severity → see each model's verdict, the point on "
                                "the decision boundaries, and the localised span"),
            ("Network replay", "10 h of any link with out-of-fold predictions and flapping alarms"),
            ("Model performance", "Metrics table and every figure"),
            ("CLI fallback", "python predict.py --snapshot examples/degraded_span7.csv")]
    for i, (a, b) in enumerate(demo):
        yy = 1.85 + i * 1.2
        circle_num(s, 0.6, yy, i + 1)
        text(s, 1.35, yy - 0.02, 4.9, 0.4, a, size=17, bold=True, color=NAVY)
        text(s, 1.35, yy + 0.38, 4.9, 0.75, b, size=13, color=INK)
    text(s, 6.6, 5.7, 6.2, 0.6, "Replay view: the model tracks a slow fibre degradation, a hard outage and a flapping "
         "amplifier.", size=12, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Demo script: (1) none → all healthy. (2) fiber_degradation span 8, 4 dB → healthy; raise to 9 dB → LR & SVM "
        "flag it, localised to span 8. (3) Same severity at span 19 → OSNR barely moves (position effect). "
        "(4) amplifier_degradation 6 dB → borderline, GDA disagrees. (5) fiber_cut → everyone flags, 'fiber_cut' "
        "localised. Then switch to Network replay R07-ZA.")

    # 12 — Code ------------------------------------------------------------------
    s = new_slide()
    n += 1
    title(s, "Implementation & repository", "One command reproduces every number in this deck")
    box(s, 0.6, 1.85, 6.1, 4.55, NAVY, radius=0.04)
    tree = ("run_pipeline.py      one-command experiment\n"
            "app.py               Streamlit live demo\n"
            "predict.py           CLI inference\n"
            "linkfail/\n"
            "  simulator.py       telemetry + link budget\n"
            "  preprocessing.py   Eq. 1 / Eq. 2 features\n"
            "  models.py          LR · GDA · SVM\n"
            "  localization.py    span + flapping\n"
            "  evaluation.py      metrics, lead time\n"
            "  pipeline.py        CV, extensions\n"
            "tests/               12 pytest tests\n"
            "docs/                write-up + slides")
    text(s, 0.9, 2.15, 5.6, 4.1, tree.split("\n"), size=14, font="Courier New", color=WHITE, space_after=3)
    bullets(s, 7.1, 1.95, 5.6, 4.8, [
        [("Reproducible: ", {"bold": True}), ("fixed seed, whole pipeline in ~20 s", {})],
        [("Validated: ", {"bold": True}), ("our models agree with scikit-learn on 99.8–100 % of test predictions", {})],
        [("Tested: ", {"bold": True}), ("physics, interpolation, Eq. 2 identity, localisation", {})],
        [("Honest evaluation: ", {"bold": True}), ("held-out test set, 5-fold CV, unseen-route CV, out-of-fold timeline", {})],
        [("Contributions: ", {"bold": True}), ("J Sanjay Varma — simulator, preprocessing, localisation, demo · "
                                               "Kanak Pandey — models, evaluation, extensions, report", {})],
    ], size=15)
    footer(s, n)

    # 13 — Conclusion --------------------------------------------------------------
    s = new_slide(dark=True)
    n += 1
    title(s, "Conclusions", dark=True)
    pts = [("Two telemetry features + a linear model", f"detect failing links at {pct(svm['accuracy'])} accuracy, and "
                                                      "are just as accurate on links never seen in training"),
           ("Paper's ranking reproduced", "LR ≈ Linear SVM > GDA; GDA's Gaussian assumption fails on multi-modal faults"),
           ("Localisation is nearly free", "per-span residuals pinpoint the span and the element (fibre, amplifier or cut)"),
           ("Future work", "OSNR forecasting for longer lead time · position-aware features · real operator data")]
    for i, (a, b) in enumerate(pts):
        yy = 1.6 + i * 1.3
        circle_num(s, 0.7, yy, i + 1, fill=AQUA if i < 3 else ORANGE)
        text(s, 1.5, yy - 0.04, 11, 0.45, a, size=20, bold=True, color=WHITE, font=HEAD)
        text(s, 1.5, yy + 0.44, 11, 0.5, b, size=15, color=ICE)
    text(s, 0.7, 6.75, 11.5, 0.4, "Thank you — questions?", size=18, bold=True, color=WHITE)
    footer(s, n, dark=True)

    prs.save(OUT)
    print(f"Wrote {OUT} ({n} slides)")


if __name__ == "__main__":
    build()
