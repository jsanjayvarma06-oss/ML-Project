"""Classify a link from the command line.

    python predict.py --x1 -0.5 --x2 -9.0
    python predict.py --snapshot examples/degraded_span7.csv
"""
import argparse

import joblib
import pandas as pd

from linkfail import config as C
from linkfail.localization import localize_snapshot
from linkfail.pipeline import MODEL_FILES
from linkfail.preprocessing import features_from_snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x1", type=float, help="Σ(amp gain − target gain) [dB]")
    ap.add_argument("--x2", type=float, help="Σ(target span loss − span loss) [dB]")
    ap.add_argument("--snapshot", help="CSV with columns span,span_loss,target_span_loss,amp_gain,target_gain")
    args = ap.parse_args()

    snap = None
    if args.snapshot:
        snap = pd.read_csv(args.snapshot)
        x = features_from_snapshot(snap)
    elif args.x1 is not None and args.x2 is not None:
        x = [[args.x1, args.x2]]
    else:
        ap.error("give --snapshot or both --x1 and --x2")

    print(f"Features: X1 = {x[0][0]:.2f} dB, X2 = {x[0][1]:.2f} dB\n")
    any_fail = False
    for name, fname in MODEL_FILES.items():
        m = joblib.load(C.MODELS / fname)
        pred = int(m.predict(x)[0])
        any_fail |= bool(pred)
        extra = (f"P(failure)={m.predict_proba(x)[0, 1]:.3f}" if hasattr(m, "predict_proba")
                 else f"margin={m.decision_function(x)[0]:+.2f}")
        print(f"  {name:<20} {'NOT HEALTHY' if pred else 'healthy':<12} {extra}")
    if snap is not None and any_fail:
        loc = localize_snapshot(snap)
        print(f"\nLocalised fault: span {loc['span']} ({loc['element']}), residual {loc['residual_db']:.1f} dB")


if __name__ == "__main__":
    main()
