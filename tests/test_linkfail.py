import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from linkfail import config as C
from linkfail.localization import localize_snapshot
from linkfail.models import GDA, LinearSVM, LogisticRegressionNewton, sigmoid
from linkfail.preprocessing import build_features, features_from_snapshot, fill_missing
from linkfail.simulator import Fault, LinkDesign, generate_dataset, link_budget, snapshot


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 1.0, (400, 2))
    X1 = rng.normal([3, 3], 1.0, (400, 2))
    X = np.vstack([X0, X1]) * [5, 1] + [-20, 10]      # unequal scales on purpose
    y = np.r_[np.zeros(400, int), np.ones(400, int)]
    return X, y


@pytest.fixture(scope="module")
def small_dataset():
    tele, health = generate_dataset(seed=1, n_routes=2, n_bins=60)
    return tele, health


# ---- models --------------------------------------------------------------
def test_sigmoid_is_stable():
    np.testing.assert_allclose(sigmoid(np.array([-1e6, 0, 1e6])), [0.0, 0.5, 1.0], atol=1e-12)


def test_logistic_newton_matches_sklearn(blobs):
    X, y = blobs
    ours = LogisticRegressionNewton(l2=0.0).fit(X, y)
    ref = LogisticRegression(C=1e8, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(ours.theta_[1:], ref.coef_[0], rtol=1e-3)
    np.testing.assert_allclose(ours.theta_[0], ref.intercept_[0], rtol=1e-3)
    assert ours.cost_history_[-1] < ours.cost_history_[0]
    assert ours.n_iter_ < 20                      # Newton converges quickly


def test_gda_parameters_and_lda_agreement(blobs):
    X, y = blobs
    g = GDA().fit(X, y)
    assert g.phi_ == pytest.approx(0.5)
    np.testing.assert_allclose(g.mu1_, X[y == 1].mean(0))
    assert (g.predict(X) == LinearDiscriminantAnalysis().fit(X, y).predict(X)).mean() == 1.0
    p = g.predict_proba(X)
    np.testing.assert_allclose(p.sum(1), 1.0)


def test_linear_svm_separates_blobs(blobs):
    X, y = blobs
    svm = LinearSVM(epochs=50).fit(X, y)
    assert (svm.predict(X) == y).mean() > 0.95


# ---- simulator -----------------------------------------------------------
def test_generator_is_deterministic_and_shaped(small_dataset):
    tele, health = small_dataset
    tele2, _ = generate_dataset(seed=1, n_routes=2, n_bins=60)
    pd.testing.assert_frame_equal(tele, tele2)
    assert len(health) == 2 * 2 * 60
    assert len(tele) == len(health) * C.N_SPANS
    assert set(health["y"].unique()) <= {0, 1}


def test_physics_more_loss_means_lower_osnr():
    d = LinkDesign.nominal()
    _, g0, _, y0 = snapshot(d, [], noise=False)
    _, g1, _, y1 = snapshot(d, [Fault("fiber_degradation", 0, 10.0)], noise=False)
    _, g2, _, y2 = snapshot(d, [Fault("fiber_cut", 5, 40.0)], noise=False)
    assert g0 > g1 > g2
    assert (y0, y1, y2) == (0, 1, 1)


def test_link_budget_healthy_power_is_preserved():
    d = LinkDesign.nominal()
    _, rx = link_budget(d.target_span_loss[None], d.target_span_loss[None], d.noise_figure[None])
    assert rx[0] == pytest.approx(C.LAUNCH_POWER_DBM)


# ---- preprocessing -------------------------------------------------------
def test_fill_missing_interpolates_and_fills_cuts():
    ts = pd.date_range("2026-01-01", periods=3, freq="2min")
    tele = pd.DataFrame({
        "timestamp": list(ts) * 2, "link_id": "L", "span": [1, 1, 1, 2, 2, 2],
        "span_loss": [10.0, np.nan, 12.0, 15.0, np.nan, 15.0],
        "target_span_loss": [10.0] * 3 + [15.0] * 3,
        "amp_gain": [10.0, np.nan, 10.0, 15.0, 15.0, 15.0],
        "target_gain": [10.0] * 3 + [15.0] * 3,
        "los_alarm": [0, 0, 0, 0, 1, 0],
    })
    out = fill_missing(tele).set_index(["span", "timestamp"])
    assert out.loc[(1, ts[1]), "span_loss"] == pytest.approx(11.0)       # interpolated
    assert out.loc[(1, ts[1]), "amp_gain"] == pytest.approx(10.0)
    assert out.loc[(2, ts[1]), "span_loss"] == pytest.approx(15.0 + C.FIBER_CUT_LOSS_DB)  # cut fill


def test_build_features_no_missing_and_eq2(small_dataset):
    tele, health = small_dataset
    feats, resid = build_features(tele, health)
    assert len(feats) == len(health)
    assert feats[["X1", "X2"]].notna().all().all()
    loss_cols = [c for c in resid if c.startswith("loss_res_")]
    gain_cols = [c for c in resid if c.startswith("gain_res_")]
    np.testing.assert_allclose(-resid[loss_cols].sum(axis=1), feats["X2"], atol=1e-9)
    np.testing.assert_allclose(-resid[gain_cols].sum(axis=1), feats["X1"], atol=1e-9)


# ---- localisation --------------------------------------------------------
@pytest.mark.parametrize("kind,span,element", [
    ("fiber_degradation", 3, "fiber"),
    ("amplifier_degradation", 11, "amplifier"),
    ("fiber_cut", 17, "fiber_cut"),
])
def test_localize_snapshot(kind, span, element):
    snap, *_ = snapshot(LinkDesign.nominal(), [Fault(kind, span, 9.0 if kind != "fiber_cut" else 40.0)])
    loc = localize_snapshot(snap)
    assert loc["span"] == span + 1
    assert loc["element"] == element
    assert features_from_snapshot(snap).shape == (1, 2)
