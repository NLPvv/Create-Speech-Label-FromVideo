import math
import os
import re
from collections import Counter
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import Iterable, Tuple, Union

try:
    from thefuzz import fuzz
except ImportError:
    fuzz = None

OCRItem = Union[
    Tuple[str, str, float],
    Tuple[Tuple[str], Tuple[str], float],
]


def _similarity_ratio(first: str, second: str) -> int:
    if fuzz is not None:
        return fuzz.ratio(first, second)
    if first == second:
        return 100
    if not first or not second:
        return 0

    # Same normalized insertion/deletion score as fuzz.ratio, without requiring
    # the optional package for subtitle-only usage. Keep the shorter DP row.
    if len(first) < len(second):
        first, second = second, first
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second, 1):
            current.append(
                previous[index - 1] + 1
                if left == right else max(previous[index], current[-1])
            )
        previous = current
    return round(200 * previous[-1] / (len(first) + len(second)))


class SubtitleGenerator:

    def __init__(self, ocr_results: Iterable[OCRItem], fps: float = 25,
                 sim_threshold: int = 90, use_vsf=False, *,
                 max_gap_frames: int = 1, frame_index_base: int = 1):
        """Merge OCR frames into subtitles.

        Frame indices start at 1 by default, matching ffmpeg's image output.
        ``max_gap_frames`` is the largest difference between consecutive frame
        indices that may belong to the same subtitle (1 means adjacent frames).
        """
        if (isinstance(fps, bool) or not isinstance(fps, Real)
                or not math.isfinite(fps) or fps <= 0):
            raise ValueError("fps must be a positive finite number")
        if (isinstance(sim_threshold, bool) or not isinstance(sim_threshold, Real)
                or not math.isfinite(sim_threshold) or not 0 <= sim_threshold <= 100):
            raise ValueError("sim_threshold must be between 0 and 100")
        if (isinstance(max_gap_frames, bool) or not isinstance(max_gap_frames, int)
                or max_gap_frames < 1):
            raise ValueError("max_gap_frames must be a positive integer")
        if (isinstance(frame_index_base, bool)
                or not isinstance(frame_index_base, int)
                or frame_index_base not in (0, 1)):
            raise ValueError("frame_index_base must be 0 or 1")
        if isinstance(ocr_results, (str, bytes)):
            raise ValueError("ocr_results must be an iterable of OCR records")
        try:
            self.ocr_results = list(ocr_results)
        except TypeError as exc:
            raise ValueError("ocr_results must be an iterable of OCR records") from exc
        self.fps = fps
        self._fps_fraction = Fraction(str(fps))
        self.sim_threshold = sim_threshold
        self.use_vsf = use_vsf
        self.max_gap_frames = max_gap_frames
        self.frame_index_base = frame_index_base
        self.raw_list = []

    # ========= Basic tools ========= #

    def _extract_frame(self, path: str) -> int:
        try:
            path = os.fspath(path)
        except TypeError as exc:
            raise ValueError("OCR image path must be a string or PathLike") from exc
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise ValueError("OCR image path must be a non-empty valid path")
        # Read only the basename: digits in a parent directory are not frames.
        filename = re.split(r"[/\\]", path)[-1]
        match = re.search(r"(\d+)\.(?:jpe?g|png|bmp|webp|tiff?)$", filename, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Image filename must end with a frame number: {path!r}")
        frame = int(match.group(1))
        if frame < self.frame_index_base:
            raise ValueError(
                f"Frame {frame} is below frame_index_base={self.frame_index_base}: {path!r}"
            )
        return frame

    def _frame_to_tc(self, frame: int) -> str:
        if (isinstance(frame, bool) or not isinstance(frame, int)
                or frame < self.frame_index_base):
            raise ValueError("frame must be an integer at or above frame_index_base")
        elapsed_frames = frame - self.frame_index_base
        # Round the exact rational time to milliseconds once. Floating-point
        # subtraction previously turned e.g. 1.16 seconds into 1,159 ms.
        numerator = elapsed_frames * 1000 * self._fps_fraction.denominator
        denominator = self._fps_fraction.numerator
        total_ms = (2 * numerator + denominator) // (2 * denominator)
        seconds, milliseconds = divmod(total_ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    @staticmethod
    def _unwrap_field(value):
        if isinstance(value, (tuple, list)) and len(value) == 1:
            return value[0]
        return value

    def _convert_to_raw_list(self):
        raw = []
        for index, item in enumerate(self.ocr_results):
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError(f"OCR record {index} must contain (path, text, score)")
            frame = self._extract_frame(self._unwrap_field(item[0]))
            text = self._unwrap_field(item[1])
            if not isinstance(text, str):
                raise ValueError(f"OCR record {index} text must be a string")
            text = " ".join(text.split())
            if not text:
                continue
            raw.append([frame, "0,0,0,0", text])

        raw.sort(key=lambda x: x[0])
        self.raw_list = raw

    def _remove_duplicate_subtitle(self):
        """Return (first frame, last frame, most frequent text) per segment."""
        if not self.raw_list:
            return []
        result = []
        start_frame = previous_frame = self.raw_list[0][0]
        base_text = self.raw_list[0][2].replace(" ", "")
        counter = Counter([self.raw_list[0][2]])
        for frame, _, text in self.raw_list[1:]:
            clean_text = text.replace(" ", "")
            if (frame - previous_frame > self.max_gap_frames
                    or _similarity_ratio(base_text, clean_text) < self.sim_threshold):
                result.append((start_frame, previous_frame, counter.most_common(1)[0][0]))
                start_frame = frame
                base_text = clean_text
                counter = Counter()
            counter[text] += 1
            previous_frame = frame
        result.append((start_frame, previous_frame, counter.most_common(1)[0][0]))
        return result

    # ========= export SRT ========= #

    def generate_srt(self, save_path="subtitle.srt"):
        try:
            output = os.fspath(save_path)
        except TypeError as exc:
            raise ValueError("save_path must be a string or PathLike") from exc
        if not isinstance(output, str) or not output.strip() or "\x00" in output:
            raise ValueError("save_path must be a non-empty valid path")
        self._convert_to_raw_list()
        merged = self._remove_duplicate_subtitle()

        lines = []
        for idx, (start_f, end_f, text) in enumerate(merged, 1):
            start_tc = self._frame_to_tc(start_f)
            end_tc = self._frame_to_tc(end_f + 1)

            lines.append(str(idx))
            lines.append(f"{start_tc} --> {end_tc}")
            lines.append(text)
            lines.append("")

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Subtitle saved: {output_path}")
        return str(output_path)

