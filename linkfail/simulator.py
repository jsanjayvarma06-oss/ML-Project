"""Synthetic optical-line telemetry generator.

The paper trains on live telemetry from a production time-series database that
is not publicly available. This module reproduces the same data shape from
first principles so the full pipeline can be run end to end:

* 10 fibre routes x 2 traffic directions = 20 end-to-end (E2E) links
* every link has 19 spans; each span = fibre + inline amplifier
* 4 raw metrics per span every 2 minutes: span loss, target span loss,
  amplifier gain, target gain
* faults (fibre degradation, amplifier degradation, flapping connector,
  fibre cut) are injected with a known span so localisation can be scored
* the link's generalised OSNR is computed with an ASE + GN-model NLI budget
  and used for labelling, exactly like the paper uses OSNR for labels
* telemetry gaps are injected (random cells, whole bins, loss-of-signal
  during fibre cuts) so the preprocessing has to do real wrangling
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

FAULT_TYPES = ("fiber_degradation", "amplifier_degradation", "flapping", "fiber_cut")
FAULT_PROBS = (0.38, 0.32, 0.18, 0.12)


@dataclass
class LinkDesign:
    """Provisioned (target) values of one directional E2E link."""

    link_id: str
    target_span_loss: np.ndarray
    target_gain: np.ndarray
    noise_figure: np.ndarray
    loss_bias: np.ndarray = field(default=None)   # real fibre vs. plan
    gain_bias: np.ndarray = field(default=None)   # amplifier calibration offset

    @classmethod
    def random(cls, link_id: str, rng: np.random.Generator) -> "LinkDesign":
        length_km = rng.uniform(55, 95, C.N_SPANS)
        target_loss = np.round(0.21 * length_km + rng.uniform(1.0, 2.5, C.N_SPANS), 2)
        target_gain = np.round(target_loss + rng.uniform(-0.3, 0.3, C.N_SPANS), 2)
        return cls(
            link_id=link_id,
            target_span_loss=target_loss,
            target_gain=target_gain,
            noise_figure=rng.uniform(4.5, 5.5, C.N_SPANS),
            loss_bias=rng.normal(0.0, 0.25, C.N_SPANS),
            gain_bias=rng.normal(0.0, 0.2, C.N_SPANS),
        )

    @classmethod
    def nominal(cls, link_id: str = "demo") -> "LinkDesign":
        """Deterministic textbook link used by the demo app."""
        rng = np.random.default_rng(7)
        d = cls.random(link_id, rng)
        d.loss_bias = np.zeros(C.N_SPANS)
        d.gain_bias = np.zeros(C.N_SPANS)
        return d


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def link_budget(span_loss: np.ndarray, gain: np.ndarray, nf: np.ndarray):
    """Propagate the signal through the line and return (GSNR dB, Rx power dBm).

    All inputs have shape (T, N_SPANS) and hold *true* (not measured) values.
    Amplifiers run in constant-gain mode, so a power deficit created in one
    span persists downstream and degrades every following amplifier's OSNR.
    """
    span_loss = np.atleast_2d(span_loss)
    gain = np.atleast_2d(gain)
    nf = np.atleast_2d(nf)
    p = np.full(span_loss.shape[0], C.LAUNCH_POWER_DBM)
    inv_snr = np.zeros_like(p)
    for i in range(span_loss.shape[1]):
        snr_nli = -C.NLI_ETA_DB - 2.0 * p             # GN model: P_NLI ~ eta * P^3
        inv_snr += 10 ** (-snr_nli / 10)
        p_rx = p - span_loss[:, i]
        osnr_ase = C.OSNR_CONST_DB + p_rx - nf[:, i]
        inv_snr += 10 ** (-osnr_ase / 10)
        p = np.minimum(p_rx + gain[:, i], C.AMP_SAT_POWER_DBM)
    gsnr = -10 * np.log10(inv_snr)
    return gsnr, p


def label_health(gsnr: np.ndarray, rx_power: np.ndarray, los: np.ndarray):
    """Return (y, outage) where y=1 means the link is NOT healthy.

    Not healthy = OSNR below the margin threshold, or received power below the
    receiver's operating range, or a hard outage (LOS / OSNR below FEC limit).
    """
    lost = los | (rx_power < C.LAUNCH_POWER_DBM - C.LOS_MARGIN_DB)
    outage = lost | (gsnr < C.OSNR_OUTAGE_DB)
    rx_low = rx_power < C.LAUNCH_POWER_DBM - C.RX_LOW_MARGIN_DB
    y = outage | (gsnr < C.OSNR_UNHEALTHY_DB) | rx_low
    return y.astype(int), outage.astype(int)


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------
@dataclass
class Fault:
    kind: str
    span: int                 # 0-based span index
    severity: float           # dB
    compensate: float = 0.0   # dB of gain the amplifier adds back (fibre faults)


def apply_fault(loss, gain, nf, fault: Fault, intensity: np.ndarray | float, rng=None):
    """Modify true per-span values in place. `intensity` in [0, 1] per bin."""
    k = fault.span
    intensity = np.asarray(intensity, dtype=float)
    los = np.zeros(loss.shape[0], dtype=bool)
    if fault.kind == "fiber_degradation":
        extra = fault.severity * intensity
        loss[:, k] += extra
        gain[:, k] += np.minimum(extra, fault.compensate)
    elif fault.kind == "amplifier_degradation":
        drop = fault.severity * intensity
        gain[:, k] -= drop
        nf[:, k] += 0.4 * drop
    elif fault.kind == "flapping":
        loss[:, k] += fault.severity * intensity
    elif fault.kind == "fiber_cut":
        on = intensity > 0
        loss[on, k] += C.FIBER_CUT_LOSS_DB
        los = on
    return los


def _event_profile(kind: str, duration: int, rng: np.random.Generator) -> np.ndarray:
    """Time profile of a fault's intensity over its duration."""
    if kind == "flapping":
        state, out = rng.random() < 0.5, []
        for _ in range(duration):
            if rng.random() < 0.35:
                state = not state
            out.append(float(state))
        out[0] = 1.0
        return np.array(out)
    if kind == "fiber_cut" or rng.random() < 0.3:
        return np.ones(duration)                        # abrupt step
    ramp_len = max(2, int(0.6 * duration))              # slow degradation
    return np.concatenate([np.linspace(0.05, 1.0, ramp_len), np.ones(duration - ramp_len)])


def _sample_fault(rng: np.random.Generator) -> Fault:
    kind = rng.choice(FAULT_TYPES, p=FAULT_PROBS)
    span = int(rng.integers(0, C.N_SPANS))
    if kind == "fiber_degradation":
        comp = rng.uniform(0.5, 3.0) if rng.random() < 0.35 else 0.0
        return Fault(kind, span, rng.uniform(1.0, 14.0), comp)
    if kind == "amplifier_degradation":
        return Fault(kind, span, rng.uniform(1.0, 12.0))
    if kind == "flapping":
        return Fault(kind, span, rng.uniform(6.0, 20.0))
    return Fault(kind, span, C.FIBER_CUT_LOSS_DB)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def simulate_link(design: LinkDesign, n_bins: int, rng: np.random.Generator,
                  start: pd.Timestamp):
    """Simulate one directional link. Returns (telemetry_long_df, health_df)."""
    T, S = n_bins, C.N_SPANS
    t = np.arange(T)
    # slow temperature-driven drift + per-bin measurement-level jitter
    drift = 0.08 * np.sin(2 * np.pi * t / rng.uniform(150, 400) + rng.uniform(0, 6.28))
    loss = design.target_span_loss + design.loss_bias + drift[:, None] * rng.uniform(0.3, 1.0, S)
    loss = loss + rng.normal(0, 0.15, (T, S))
    gain = design.target_gain + design.gain_bias + rng.normal(0, 0.1, (T, S))
    nf = np.tile(design.noise_figure, (T, 1))
    los = np.zeros(T, dtype=bool)

    fault_type = np.array(["none"] * T, dtype=object)
    fault_span = np.full(T, -1)
    event_id = np.full(T, -1)

    t0, ev = int(rng.integers(5, 30)), 0
    while t0 < T:
        fault = _sample_fault(rng)
        duration = int(rng.integers(10, 61)) if fault.kind != "fiber_cut" else int(rng.integers(5, 21))
        duration = min(duration, T - t0)
        prof = np.zeros(T)
        prof[t0:t0 + duration] = _event_profile(fault.kind, duration, rng)
        los |= apply_fault(loss, gain, nf, fault, prof)
        sl = slice(t0, t0 + duration)
        fault_type[sl], fault_span[sl], event_id[sl] = fault.kind, fault.span + 1, ev
        ev += 1
        t0 += duration + int(rng.exponential(25)) + 3

    gsnr, rx = link_budget(loss, gain, nf)
    y, outage = label_health(gsnr, rx, los)

    # ---- measured telemetry (noise + gaps) ---------------------------------
    m_loss = loss + rng.normal(0, C.LOSS_MEAS_NOISE_DB, (T, S))
    m_gain = gain + rng.normal(0, C.GAIN_MEAS_NOISE_DB, (T, S))
    m_loss[rng.random((T, S)) < 0.015] = np.nan
    m_gain[rng.random((T, S)) < 0.015] = np.nan
    gap_bins = rng.random(T) < 0.005
    m_loss[gap_bins], m_gain[gap_bins] = np.nan, np.nan
    los_alarm = np.zeros((T, S), dtype=int)
    for ti in np.where(los)[0]:                     # no light -> no loss reading
        k = fault_span[ti] - 1
        m_loss[ti, k] = np.nan
        los_alarm[ti, k] = 1

    ts = start + pd.to_timedelta(t * C.BIN_MINUTES, unit="min")
    tele = pd.DataFrame({
        "timestamp": np.repeat(ts, S),
        "link_id": design.link_id,
        "span": np.tile(np.arange(1, S + 1), T),
        "span_loss": m_loss.ravel().round(3),
        "target_span_loss": np.tile(design.target_span_loss, T),
        "amp_gain": m_gain.ravel().round(3),
        "target_gain": np.tile(design.target_gain, T),
        "los_alarm": los_alarm.ravel(),
    })
    health = pd.DataFrame({
        "timestamp": ts,
        "link_id": design.link_id,
        "osnr_db": gsnr.round(3),
        "rx_power_dbm": rx.round(3),
        "y": y,
        "outage": outage,
        "fault_type": fault_type,
        "fault_span": fault_span,
        "event_id": event_id,
    })
    return tele, health


def generate_dataset(seed: int = C.SEED, n_routes: int = C.N_ROUTES, n_bins: int = C.N_BINS):
    """Generate telemetry and ground-truth health for every directional link."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-09-01 00:00:00")
    teles, healths = [], []
    for r in range(1, n_routes + 1):
        for direction in ("AZ", "ZA"):
            design = LinkDesign.random(f"R{r:02d}-{direction}", rng)
            tele, health = simulate_link(design, n_bins, rng, start)
            teles.append(tele)
            healths.append(health)
    return pd.concat(teles, ignore_index=True), pd.concat(healths, ignore_index=True)


def snapshot(design: LinkDesign, faults: list[Fault], noise: bool = True, seed: int = 0):
    """Single 2-minute telemetry snapshot of a link with the given faults.

    Used by the interactive demo. Returns (measured_df, gsnr, rx_power, y).
    """
    rng = np.random.default_rng(seed)
    S = C.N_SPANS
    loss = (design.target_span_loss + design.loss_bias)[None, :].astype(float).copy()
    gain = (design.target_gain + design.gain_bias)[None, :].astype(float).copy()
    nf = design.noise_figure[None, :].astype(float).copy()
    los = np.zeros(1, dtype=bool)
    for f in faults:
        los |= apply_fault(loss, gain, nf, f, np.ones(1))
    gsnr, rx = link_budget(loss, gain, nf)
    y, _ = label_health(gsnr, rx, los)
    if noise:
        loss = loss + rng.normal(0, C.LOSS_MEAS_NOISE_DB, (1, S))
        gain = gain + rng.normal(0, C.GAIN_MEAS_NOISE_DB, (1, S))
    df = pd.DataFrame({
        "span": np.arange(1, S + 1),
        "span_loss": loss[0],
        "target_span_loss": design.target_span_loss,
        "amp_gain": gain[0],
        "target_gain": design.target_gain,
    })
    return df, float(gsnr[0]), float(rx[0]), int(y[0])
