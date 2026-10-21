"""Recognize cropped frame folders, reusing one OCR model per worker."""

import argparse
import logging
import multiprocessing as mp
import os
from pathlib import Path

from core.ocr_module import OCRProcessor
from core.subtitle_generator import SubtitleGenerator

LOGGER = logging.getLogger(__name__)


def gpu_worker(task_queue, gpu_id, args):
    # CUDA visibility must be set before Paddle is imported by the lazy loader.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "gpu:0" if gpu_id is not None else args.device
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(processName)s %(levelname)s: %(message)s")
    ocrer = OCRProcessor(
        input_dir="", device=device, enable_hpi=args.hpi,
        batch_size=args.batch_size, score=args.score, save_json=args.save_json,
        verbose=args.verbose, model_name=args.model_name,
    )
    failed = False
    while True:
        # A blocking get plus sentinels avoids Queue feeder/get_nowait races.
        folder = task_queue.get()
        if folder is None:
            break
        folder = Path(folder)
        name = folder.name[:-len(args.suffix)] if args.suffix else folder.name
        name = name or folder.name
        try:
            results = ocrer.run(str(folder), str(args.output_dir / "json" / name))
            if not results:
                LOGGER.warning("No text passed the confidence threshold: %s", folder)
            SubtitleGenerator(
                results, fps=args.fps, sim_threshold=args.sim_threshold,
                max_gap_frames=args.max_gap_frames,
            ).generate_srt(str(args.output_dir / f"{name}.srt"))
        except Exception:
            failed = True
            LOGGER.exception("Failed to process %s on %s", folder, device)
    if failed:
        raise SystemExit(1)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--suffix", default="_crop", help="Input folder suffix; empty string selects all folders")
    parser.add_argument("--gpus", type=int, nargs="+", help="GPU IDs; one process/model per GPU")
    parser.add_argument("--device", default="cpu", help="Device for a single worker when --gpus is omitted")
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--score", type=float, default=0.9)
    parser.add_argument("--sim-threshold", type=float, default=80)
    parser.add_argument("--max-gap-frames", type=int, default=1)
    parser.add_argument("--model-name", default="PP-OCRv5_server_rec")
    parser.add_argument("--hpi", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.base_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.base_dir}")
    if args.gpus and (any(gpu < 0 for gpu in args.gpus) or len(set(args.gpus)) != len(args.gpus)):
        parser.error("GPU IDs must be nonnegative and unique")
    try:
        OCRProcessor("", batch_size=args.batch_size, score=args.score,
                     device=args.device, model_name=args.model_name)
        SubtitleGenerator([], fps=args.fps, sim_threshold=args.sim_threshold,
                          max_gap_frames=args.max_gap_frames)
    except ValueError as exc:
        parser.error(str(exc))
    folders = sorted(path for path in args.base_dir.iterdir()
                     if path.is_dir() and path.name.endswith(args.suffix)
                     and path.resolve() != args.output_dir.resolve())
    if not folders:
        parser.error(f"No input folders ending with {args.suffix!r} in {args.base_dir}")
    workers = (args.gpus or [None])[:len(folders)]
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    for folder in folders:
        task_queue.put(str(folder))
    for _ in workers:
        task_queue.put(None)
    processes = [context.Process(target=gpu_worker, args=(task_queue, gpu, args)) for gpu in workers]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    return int(any(process.exitcode != 0 for process in processes))


if __name__ == "__main__":
    raise SystemExit(main())



