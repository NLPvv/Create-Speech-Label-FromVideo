"""Crop frame folders without removing source images.

Run from the repository root with ``python -m utils.batch_main_cut --help``.
"""

import argparse
from pathlib import Path

def run_crop_for_all_folders(
    root_dir,
    output_root=None,
    *,
    ymin,
    ymax,
    xmin,
    xmax,
    num_workers=16,
):
    """Return crop results per folder, leaving all source folders intact.

    Existing ``*_crop`` folders and the selected output root are never treated
    as inputs. By default each input folder gets a sibling ``<name>_crop``.
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    destination = Path(output_root).resolve() if output_root is not None else root
    sources = sorted(
        entry for entry in root.iterdir()
        if entry.is_dir() and not entry.name.endswith("_crop") and entry.resolve() != destination
    )
    if not sources:
        raise ValueError(f"No input frame folders found in: {root}")

    tasks = [(source, destination / f"{source.name}_crop") for source in sources]
    # Validate every task before writing anything, including symlink aliases.
    resolved_sources = [source.resolve() for source in sources]
    for _, target in tasks:
        target = target.resolve()
        for source in resolved_sources:
            if source == target or source in target.parents or target in source.parents:
                raise ValueError(f"Input and output directories overlap: {source}, {target}")

    # Keep --help and input validation usable without installing image libraries.
    from core.cropper import Cropper

    results = {}
    for source, target in tasks:
        print(f"Cropping {source} -> {target}")
        results[source.name] = Cropper(
            input_dir=str(source),
            crop_dir=str(target),
            ymin=ymin,
            ymax=ymax,
            xmin=xmin,
            xmax=xmax,
            save_crop=True,
            num_workers=num_workers,
        ).run()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Crop frame folders; source files are preserved.")
    parser.add_argument("root_dir", help="Directory containing one frame folder per video")
    parser.add_argument("--output-root", help="Output parent directory; defaults to root_dir")
    parser.add_argument("--roi", nargs=4, type=int, required=True,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="Subtitle crop bounds")
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        xmin, ymin, xmax, ymax = args.roi
        run_crop_for_all_folders(
            args.root_dir,
            output_root=args.output_root,
            ymin=ymin,
            ymax=ymax,
            xmin=xmin,
            xmax=xmax,
            num_workers=args.num_workers,
        )
    except ImportError as exc:
        parser.exit(1, f"Missing dependency: {exc}. Install requirements.txt first.\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

