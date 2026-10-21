"""Extract hardcoded, single-line subtitles from a video."""

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Optional, Sequence

from core.frame_skip import BatchFFmpeg
from core.ocr_module import OCRProcessor
from core.subtitle_generator import SubtitleGenerator

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--roi", type=int, nargs=4, required=True,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="Subtitle rectangle in displayed video pixels")
    parser.add_argument("--output", type=Path, help="SRT path (default: video.with_suffix('.srt'))")
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--fps", type=float, default=25, help="Sampling FPS, also used for SRT timing")
    parser.add_argument("--quality", type=int, default=2, help="FFmpeg JPEG quality: 2 is high quality")
    parser.add_argument("--device", default="cpu", help="Paddle device, e.g. cpu or gpu:0")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--score", type=float, default=0.9, help="Minimum OCR confidence [0, 1]")
    parser.add_argument("--sim-threshold", type=float, default=80, help="Text similarity [0, 100]")
    parser.add_argument("--max-gap-frames", type=int, default=1,
                        help="Maximum adjacent frame-number difference allowed in one subtitle")
    parser.add_argument("--model-name", default="PP-OCRv5_server_rec")
    parser.add_argument("--hpi", action="store_true", help="Enable optional Paddle high-performance inference")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print each OCR result")
    parser.add_argument("--keep-full-frames", action="store_true",
                        help="Save full frames and crop with OpenCV instead of cropping in FFmpeg")
    parser.add_argument("--workers", type=int, default=2, help="OpenCV crop worker threads")
    return parser


def run_pipeline(args: argparse.Namespace) -> Path:
    if not args.video.is_file():
        raise FileNotFoundError(f"Video does not exist: {args.video}")
    output = args.output if args.output is not None else args.video.with_suffix(".srt")
    if output.suffix.lower() != ".srt":
        raise ValueError("Output path must use the .srt extension")
    if (output.resolve() == args.video.resolve()
            or (output.exists() and output.samefile(args.video))):
        raise ValueError("Subtitle output must not overwrite the input video")
    xmin, ymin, xmax, ymax = args.roi
    if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
        raise ValueError("ROI must satisfy 0 <= xmin < xmax and 0 <= ymin < ymax")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    # Validate all tunable stages before starting expensive extraction.
    subtitle_options = dict(fps=args.fps, sim_threshold=args.sim_threshold,
                            max_gap_frames=args.max_gap_frames)
    SubtitleGenerator([], **subtitle_options)
    ocrer = OCRProcessor(
        input_dir="", device=args.device, enable_hpi=args.hpi,
        batch_size=args.batch_size, score=args.score, save_json=args.save_json,
        verbose=args.verbose, model_name=args.model_name,
    )
    extractor = BatchFFmpeg(
        output_root=args.work_dir / ("frames" if args.keep_full_frames else "crop"),
        fps=args.fps, quality=args.quality,
        roi=None if args.keep_full_frames else tuple(args.roi),
    )
    for dependency in ("paddleocr", "paddle"):
        if importlib.util.find_spec(dependency) is None:
            raise RuntimeError(f"Missing {dependency}; follow the installation steps in README.md")
    if args.keep_full_frames:
        extractor.validate_video_roi(str(args.video), tuple(args.roi))

    LOGGER.info("Extracting video at %s FPS", args.fps)
    image_dir = extractor.process_video(str(args.video))
    json_namespace = Path(image_dir.name)
    if args.keep_full_frames:
        from core.cropper import Cropper
        crop_dir = args.work_dir / "crop" / image_dir.name / f"roi_{xmin}_{ymin}_{xmax}_{ymax}"
        Cropper(str(image_dir), str(crop_dir), ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
                num_workers=args.workers).run()
        image_dir = crop_dir
        json_namespace /= crop_dir.name

    LOGGER.info("Recognizing subtitle frames in %s", image_dir)
    ocr_results = ocrer.run(str(image_dir), str(args.work_dir / "json" / json_namespace))
    if not ocr_results:
        LOGGER.warning("No text passed the confidence threshold; writing an empty SRT")
    SubtitleGenerator(ocr_results, **subtitle_options).generate_srt(str(output))
    LOGGER.info("Subtitle saved: %s", output)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    try:
        run_pipeline(args)
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

