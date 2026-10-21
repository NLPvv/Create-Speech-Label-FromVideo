import os
import cv2
import logging
import threading
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, List, Tuple
from tqdm import tqdm

try:
    from turbojpeg import TurboJPEG
except (ImportError, OSError):
    TurboJPEG = None


logger = logging.getLogger(__name__)


class CropError(RuntimeError):
    """An image could not be read, cropped, or saved."""


class Cropper:
    def __init__(
        self,
        input_dir: str,
        crop_dir: str,
        ymin: int = 0,
        ymax: int = 1000,
        xmin: int = 0,
        xmax: int = 1000,
        save_crop: bool = True,
        num_workers: int = 2
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if ymax <= ymin or xmax <= xmin or ymax <= 0 or xmax <= 0:
            raise ValueError("ROI must have a positive width and height")

        self.input_dir: str = os.fspath(input_dir)
        self.crop_dir: str = os.fspath(crop_dir)

        self.ymin: int = ymin
        self.ymax: int = ymax
        self.xmin: int = xmin
        self.xmax: int = xmax

        self.save_crop: bool = save_crop
        self.num_workers: int = num_workers

        self.supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        # Native TurboJPEG libraries are optional. Each worker owns its decoder;
        # a decoder must not be shared by concurrent threads.
        self._decoder_local = threading.local()

    def _jpeg_decoder(self):
        if not hasattr(self._decoder_local, "decoder"):
            decoder = None
            if TurboJPEG is not None:
                try:
                    decoder = TurboJPEG()
                except Exception:
                    logger.debug("TurboJPEG unavailable; using OpenCV", exc_info=True)
            self._decoder_local.decoder = decoder
        return self._decoder_local.decoder

    def fast_read(self, path: str):
        """Decode JPEGs with TurboJPEG when available, otherwise use OpenCV."""
        if Path(path).suffix.lower() in (".jpg", ".jpeg"):
            decoder = self._jpeg_decoder()
            if decoder is not None:
                try:
                    with open(path, "rb") as image_file:
                        image = decoder.decode(image_file.read())
                    if image is not None:
                        return image
                except Exception:
                    logger.debug("TurboJPEG failed for %s; using OpenCV", path, exc_info=True)
        return cv2.imread(path)

    def crop_all_images(self, filename: str) -> Tuple[str, Any]:
        """Return ``(filename, saved_path)`` or ``(filename, cropped_array)``."""
        img_path = os.path.join(self.input_dir, filename)
        try:
            img = self.fast_read(img_path)
            if img is None:
                raise CropError(f"Cannot decode image: {img_path}")

            h, w = img.shape[:2]
            y1, y2 = max(0, self.ymin), min(h, self.ymax)
            x1, x2 = max(0, self.xmin), min(w, self.xmax)
            if y1 >= y2 or x1 >= x2:
                raise CropError(
                    f"ROI does not intersect image {img_path} ({w}x{h}): "
                    f"x=[{self.xmin}, {self.xmax}), y=[{self.ymin}, {self.ymax})"
                )
            crop = img[y1:y2, x1:x2]

            if self.save_crop:
                save_path = os.path.join(self.crop_dir, filename)
                params = (
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                    if filename.lower().endswith((".jpg", ".jpeg")) else []
                )
                if not cv2.imwrite(save_path, crop, params):
                    raise CropError(f"Cannot write cropped image: {save_path}")
                return filename, save_path
            # NumPy slices retain the complete source image. A copy keeps only
            # the requested ROI alive when callers collect in-memory results.
            return filename, crop.copy()
        except CropError:
            raise
        except Exception as exc:
            raise CropError(f"Failed to crop image {img_path}: {exc}") from exc

    def run(self) -> List[Tuple[str, Any]]:
        """Crop all images in filename order, raising on any failed image.

        Saving returns paths rather than retaining all decoded images in memory.
        With ``save_crop=False``, results contain the cropped image arrays.
        Existing successful output is preserved if a later image fails.
        """
        source = Path(self.input_dir).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Input image directory does not exist: {self.input_dir}")
        if self.save_crop:
            destination = Path(self.crop_dir).resolve()
            if source == destination or source in destination.parents or destination in source.parents:
                raise ValueError("Input and crop directories must not overlap")

        files = sorted(
            entry.name for entry in source.iterdir()
            if entry.is_file() and entry.suffix.lower() in self.supported_ext
        )
        if not files:
            raise ValueError(f"No supported images found in: {self.input_dir}")
        if self.save_crop:
            if destination.exists():
                if not destination.is_dir():
                    raise ValueError(f"Crop output path is not a directory: {self.crop_dir}")
                input_names = set(files)
                stale_images = sorted(
                    entry.name for entry in destination.iterdir()
                    if entry.is_file() and entry.suffix.lower() in self.supported_ext
                    and entry.name not in input_names
                )
                if stale_images:
                    raise ValueError(
                        f"Crop directory contains images absent from the current input: {self.crop_dir} "
                        f"({', '.join(stale_images[:5])}). "
                        "Use a new output directory to preserve existing files."
                    )
            os.makedirs(self.crop_dir, exist_ok=True)

        workers = min(self.num_workers, cpu_count(), len(files))
        if workers <= 1:
            return [self.crop_all_images(filename) for filename in tqdm(files, desc="Processing Cropper")]

        # Small batches distribute work evenly even when there are fewer than
        # 50 images, and ordered iteration gives callers deterministic results.
        with ThreadPool(workers) as pool:
            return list(tqdm(
                pool.imap(self.crop_all_images, files, chunksize=1),
                total=len(files),
                desc="Processing Cropper",
            ))
