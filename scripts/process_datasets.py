"""Convert, filter and describe all datasets listed in a configuration file.

Usage:
    python scripts/process_datasets.py configs/case_study_datasets.yaml [--overwrite]
"""

import argparse
import logging
import sys

from lymphdecoder.processing import ProcessingConfig, process


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="YAML file describing the datasets")
    parser.add_argument("--overwrite", action="store_true", help="process samples again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    try:
        meta = process(ProcessingConfig.load(args.config), overwrite=args.overwrite)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"Error: {err}")
    print(meta.groupby(["class", "platform"]).size().to_string())
    print(f"{len(meta)} samples")


if __name__ == "__main__":
    main()
