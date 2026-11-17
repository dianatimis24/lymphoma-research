"""Run the case-study experiments described in a configuration file.

Usage:
    python scripts/run_experiments.py configs/case_study_experiments.yaml e1 e2 e3
    python scripts/run_experiments.py configs/case_study_experiments.yaml all
"""

import argparse
import logging
import sys

import pandas as pd

from lymphdecoder.experiments import EXPERIMENTS, run


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="YAML file with the experiment settings")
    parser.add_argument("experiments", nargs="+", help=f"any of {', '.join(EXPERIMENTS)} or all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    try:
        results = run(args.config, [e.lower() for e in args.experiments])
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"Error: {err}")
    pd.set_option("display.width", 160)
    for name, result in results.items():
        print(f"\n=== {name.upper()} ===")
        print(result if not isinstance(result, pd.DataFrame) else result.to_string())


if __name__ == "__main__":
    main()
