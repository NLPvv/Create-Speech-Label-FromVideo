"""Extract video frames without reusing incomplete or unrelated output."""

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
from uuid import uuid4


class BatchFFmpeg:
    """Run FFmpeg and return the directory containing a complete frame sequence.

    Existing files are preserved. A fresh sibling directory is allocated whenever
    an existing directory cannot be safely reused. ``roi`` uses
    ``(xmin, ymin, xmax, ymax)`` coordinates after FFmpeg's automatic rotation.
    """

    MANIFEST_NAME = ".frames-complete.json"
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self, output_root, fps=25, quality=2, skip_if_exists=True, nj=16,
        roi=None, ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe",
    ):
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise ValueError("fps must be a positive finite number")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number")
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 31:
            raise ValueError("quality must be an integer between 1 and 31")
        if isinstance(nj, bool) or not isinstance(nj, int) or nj < 1:
            raise ValueError("nj must be a positive integer")
        if not isinstance(skip_if_exists, bool):
            raise ValueError("skip_if_exists must be a boolean")
        self.output_root = Path(output_root).expanduser()
        self.fps = float(fps)
        self.quality = quality
        self.skip_if_exists = skip_if_exists
        self.nj = nj
        self.roi = self._normalize_roi(roi)
        self.ffmpeg_bin = os.fspath(ffmpeg_bin)
        self.ffprobe_bin = os.fspath(ffprobe_bin)

    @staticmethod
    def _normalize_roi(roi):
        if roi is None:
            return None
        if not isinstance(roi, (tuple, list)) or len(roi) != 4:
            raise ValueError("roi must contain xmin, ymin, xmax, ymax")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in roi):
            raise ValueError("roi coordinates must be integers")
        xmin, ymin, xmax, ymax = roi
        if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
            raise ValueError("roi must define a nonempty rectangle with nonnegative coordinates")
        return tuple(roi)

    @staticmethod
    def _source_signature(source):
        stat = source.stat()
        return {"path": str(source), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}

    def _settings(self):
        return {"fps": self.fps, "quality": self.quality,
                "roi": list(self.roi) if self.roi is not None else None,
                "roi_coordinate_space": "autorotated"}

    def _inventory(self, directory, stem):
        """Reject missing, empty, renamed, or additional image frames."""
        pattern = re.compile(re.escape(stem) + r"_([0-9]{4,})\.jpg\Z")
        frames = []
        for path in directory.iterdir():
            if path.suffix.lower() not in self.IMAGE_SUFFIXES:
                continue
            match = pattern.fullmatch(path.name)
            if not match or path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected image in frame directory: {path}")
            stat = path.stat()
            if stat.st_size <= 0:
                raise ValueError(f"Empty frame: {path}")
            frames.append((int(match.group(1)), path.name, stat.st_size))
        frames.sort()
        if not frames:
            raise ValueError(f"FFmpeg produced no frames in {directory}")
        for expected, (index, filename, _) in enumerate(frames, 1):
            if index != expected or filename != f"{stem}_{expected:04d}.jpg":
                raise ValueError(f"Incomplete frame sequence in {directory}")
        return {"count": len(frames), "bytes": sum(frame[2] for frame in frames)}

    def _completed(self, directory, source, signature):
        try:
            with (directory / self.MANIFEST_NAME).open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            if not isinstance(manifest, dict):
                return False
            if (manifest.get("version") != 1 or manifest.get("source") != signature
                    or manifest.get("settings") != self._settings()):
                return False
            return manifest.get("frames") == self._inventory(directory, source.stem)
        except (OSError, ValueError, TypeError):
            return False

    def _existing_output(self, source, signature):
        primary = self.output_root / source.stem
        if self._completed(primary, source, signature):
            return primary
        prefix = f"{source.stem}.run-"
        for candidate in sorted(self.output_root.iterdir()):
            if candidate.name.startswith(prefix) and self._completed(candidate, source, signature):
                return candidate
        return None

    def _new_output(self, stem):
        primary = self.output_root / stem
        try:
            primary.mkdir()
            return primary
        except FileExistsError:
            # Even an empty existing directory may belong to another active run.
            directory = self.output_root / f"{stem}.run-{uuid4().hex}"
            directory.mkdir()
            return directory

    def validate_video_roi(self, video_path, roi=None):
        """Validate ROI against displayed pixels, including rotation metadata.

        An explicit ``roi`` overrides this extractor's crop configuration without
        changing it, so full-frame extraction can validate a later image crop.
        Return ``(display_width, display_height)``, or ``None`` without an ROI.
        """
        roi = self._normalize_roi(self.roi if roi is None else roi)
        if roi is None:
            return None
        if not os.fspath(video_path).strip():
            raise ValueError("video_path must not be empty")
        source = Path(video_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Video path must be a file: {source}")
        cmd = [self.ffprobe_bin, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
               "-of", "json", str(source)]
        try:
            result = subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stream = json.loads(result.stdout)["streams"][0]
            width, height = int(stream["width"]), int(stream["height"])
            if width < 1 or height < 1:
                raise ValueError("Video dimensions must be positive")
            rotation = (stream.get("tags") or {}).get("rotate", 0)
            for side_data in stream.get("side_data_list", []):
                if "rotation" in side_data:
                    rotation = side_data["rotation"]
                    break
            rotation = float(rotation)
            if not math.isfinite(rotation):
                raise ValueError("Video rotation must be finite")
            rotation %= 360
            # Match FFmpeg's transpose threshold; other angles retain the
            # original canvas size in its default rotate filter.
            if abs(rotation - 90) < 1 or abs(rotation - 270) < 1:
                width, height = height, width
        except FileNotFoundError as exc:
            raise RuntimeError(f"Cannot find ffprobe executable: {self.ffprobe_bin}") from exc
        except (subprocess.CalledProcessError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Cannot read video dimensions: {source}") from exc
        if roi[2] > width or roi[3] > height:
            raise ValueError(f"roi {roi} exceeds video dimensions {width}x{height} after automatic rotation")
        return width, height

    def process_video(self, video_path):
        """Return a complete output ``Path``; extraction failures raise an error."""
        if not os.fspath(video_path).strip():
            raise ValueError("video_path must not be empty")
        source = Path(video_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Video path must be a file: {source}")
        signature = self._source_signature(source)
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.skip_if_exists:
            existing = self._existing_output(source, signature)
            if existing is not None:
                return existing
        self.validate_video_roi(source)
        directory = self._new_output(source.stem)
        filters = [f"fps={self.fps:.15g}"]
        if self.roi is not None:
            xmin, ymin, xmax, ymax = self.roi
            filters.append(f"crop={xmax - xmin}:{ymax - ymin}:{xmin}:{ymin}:exact=1")
        # image2 treats percent signs as sequence placeholders, including those
        # originating in the video filename.
        output_pattern = str(directory / f"{source.stem}_").replace("%", "%%") + "%04d.jpg"
        cmd = [self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostats", "-xerror",
               "-nostdin", "-n", "-i", str(source), "-map", "0:v:0",
               "-vf", ",".join(filters), "-qscale:v", str(self.quality),
               "-start_number", "1", output_pattern]
        try:
            subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Cannot find ffmpeg executable: {self.ffmpeg_bin}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()[-4000:]
            raise RuntimeError(
                f"FFmpeg extraction failed for {source}; incomplete output retained at {directory}"
                + (f": {detail}" if detail else "")
            ) from exc
        inventory = self._inventory(directory, source.stem)
        if self._source_signature(source) != signature:
            raise RuntimeError(f"Source video changed during extraction: {source}")
        manifest = {"version": 1, "source": signature,
                    "settings": self._settings(), "frames": inventory}
        # Write only after FFmpeg succeeded and all output frames were checked.
        # An interrupted JSON write is treated as incomplete on the next run.
        with (directory / self.MANIFEST_NAME).open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False)
            handle.write("\n")
        return directory

    def run_batch(self, list_file):
        """Extract paths listed one per line, preserving their order in results."""
        with Path(list_file).open(encoding="utf-8") as handle:
            video_list = [line.strip() for line in handle if line.strip()]
        if self.nj == 1:
            return [self.process_video(video) for video in video_list]
        with ThreadPoolExecutor(max_workers=self.nj) as executor:
            return list(executor.map(self.process_video, video_list))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract video frames safely with FFmpeg")
    parser.add_argument("list_file", help="UTF-8 text file containing one video path per line")
    parser.add_argument("--output-root", default="./frame_output")
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--quality", type=int, default=2)
    parser.add_argument("--nj", type=int, default=4, help="Number of parallel FFmpeg processes")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--no-skip", action="store_true", help="Always extract to a new directory")
    args = parser.parse_args(argv)
    try:
        extractor = BatchFFmpeg(args.output_root, fps=args.fps, quality=args.quality,
                                skip_if_exists=not args.no_skip, nj=args.nj, roi=args.roi)
        for directory in extractor.run_batch(args.list_file):
            print(directory)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
