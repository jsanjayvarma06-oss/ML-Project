"""Command-line entry point.

    python run_pipeline.py              # generate data, preprocess, train, evaluate
    python run_pipeline.py --step train # re-run only training/evaluation
"""
import argparse

from linkfail import config as C
from linkfail import pipeline


def main():
    ap = argparse.ArgumentParser(description="Link failure prediction & localisation pipeline")
    ap.add_argument("--step", choices=["all", "generate", "preprocess", "train"], default="all")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    if args.step in ("all", "generate"):
        pipeline.step_generate(args.seed)
    if args.step in ("all", "preprocess"):
        pipeline.step_preprocess()
    if args.step in ("all", "train"):
        pipeline.step_train_evaluate(args.seed)


if __name__ == "__main__":
    main()
