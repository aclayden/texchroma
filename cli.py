"""texchroma CLI

Colour correction flags
-----------------------
--correct <IMAGE>   Capture ColorChecker swatches, save YAML, then process.
--swatches <YAML>   Load a saved CCM and apply during processing.
(neither)           Interactive prompt: load YAML, skip, or quit.
--no-correct        Skip correction without prompting.
"""

import os
import sys
import yaml
import argparse

from texchroma import image_processing as img
from texchroma import colour_correction as cc


def _resolve_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.n_clusters is not None:
        config["n_clusters"] = args.n_clusters
    if args.mask_threshold is not None:
        config["mask_threshold"] = args.mask_threshold
    return config


def _load_ccm(path):
    if not os.path.isfile(path):
        sys.exit(f"Swatch file not found: {path}")
    _, ccm, meta = cc.load_swatches(path)
    print(f"CCM loaded: {path} (captured {meta['timestamp'][:10]})")
    return ccm


def _prompt_correction_choice(config_dir="config"):
    existing = sorted(
        os.path.join(config_dir, f) for f in os.listdir(config_dir)
        if f.startswith("swatches_") and f.endswith(".yaml")
    ) if os.path.isdir(config_dir) else []

    print("\nNo colour correction flag supplied.")
    if existing:
        print("Available swatch files:")
        for p in existing:
            print(f"  {p}")
    print("  [c] capture new swatches  [s] load swatch YAML  [n] skip  [q] quit\n")

    while True:
        choice = input("Choice: ").strip().lower()
        if choice == "q":
            sys.exit(0)
        if choice == "n":
            return None
        if choice == "c":
            image = input("Path to ColorChecker image: ").strip()
            if not os.path.isfile(image):
                print(f"  Not found: {image}")
                continue
            ccm, path = cc.run_correction_session(image, config_dir)
            return ccm
        if choice == "s":
            path = input("Path to swatch YAML: ").strip()
            if os.path.isfile(path):
                return _load_ccm(path)
            print(f"  Not found: {path}")


def main():
    parser = argparse.ArgumentParser(prog="texchroma",
        description="Textile colour measurement pipeline.")

    parser.add_argument("--image-dir",      default="images")
    parser.add_argument("--output-dir",     default="outputs")
    parser.add_argument("--config",         default="config/config.yaml")
    parser.add_argument("--n-clusters",     type=int)
    parser.add_argument("--mask-threshold", type=float)
    parser.add_argument("--single",         metavar="FILE")
    parser.add_argument("--purge",          action="store_true")
    parser.add_argument("--purge-all",      action="store_true")
    parser.add_argument("--no-correct",     action="store_true")

    corr = parser.add_mutually_exclusive_group()
    corr.add_argument("--correct",  metavar="CHECKER_IMAGE")
    corr.add_argument("--swatches", metavar="SWATCHES_YAML")

    args = parser.parse_args()
    config_dir = os.path.dirname(args.config) or "config"

    if args.correct:
        ccm, _ = cc.run_correction_session(args.correct, config_dir)
    elif args.swatches:
        ccm = _load_ccm(args.swatches)
    elif args.no_correct:
        ccm = None
    else:
        ccm = _prompt_correction_choice(config_dir)

    if args.purge_all:
        img.purge_images(args.image_dir, args.output_dir, purge_all=True)
    elif args.purge:
        img.purge_images(args.image_dir, args.output_dir)
    elif args.single:
        img.process_single_image(args.single, _resolve_config(args), args.output_dir, ccm=ccm)
    else:
        img.process_all_images(args.image_dir, args.output_dir, _resolve_config(args), ccm=ccm)


if __name__ == "__main__":
    main()