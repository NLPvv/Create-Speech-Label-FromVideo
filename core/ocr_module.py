"""Batched text recognition with one lazily loaded model per processor."""

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

LOGGER = logging.getLogger(__name__)
OCRItem = Tuple[str, str, float]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class OCRProcessor:
    def __init__(
        self,
        input_dir: str,
        output_dir: str = "json_rec/output",
        device: str = "gpu:0",
        batch_size: int = 1,
        score: float = 0.8,
        save_json: bool = False,
        enable_hpi: bool = False,
        verbose: bool = False,
        model_name: str = "PP-OCRv5_server_rec",
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("score must be between 0 and 1")
        if not device.strip() or not model_name.strip():
            raise ValueError("device and model_name must not be empty")
        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.device: str = device
        self.batch_size: int = batch_size
        self.score: float = score
        self.save_json: bool = save_json
        self.hpi: bool = enable_hpi
        self.verbose: bool = verbose
        self.model_name = model_name
        self.ocr = None

    def _get_model(self):
        if self.ocr is None:
            try:
                from paddleocr import TextRecognition
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR is unavailable. Install requirements.txt and the "
                    "PaddlePaddle runtime for your CPU/CUDA environment; see README.md."
                ) from exc
            self.ocr = TextRecognition(
                model_name=self.model_name,
                enable_hpi=self.hpi,
                device=self.device,
            )
        return self.ocr

    def run(self, input_dir: Optional[str] = None,
            output_dir: Optional[str] = None) -> List[OCRItem]:
        """Recognize a directory; subsequent calls reuse the loaded model."""
        source = Path(input_dir if input_dir is not None else self.input_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {source}")
        images = sorted(str(path) for path in source.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No supported images found in: {source}")
        destination = Path(output_dir if output_dir is not None else self.output_dir)
        if self.save_json:
            destination.mkdir(parents=True, exist_ok=True)

        results_list: List[OCRItem] = []
        processed = 0
        for res in self._get_model().predict_iter(input=images, batch_size=self.batch_size):
            processed += 1
            if self.verbose:
                res.print()
            if self.save_json:
                json_path = destination / (Path(res["input_path"]).name + ".json")
                res.save_to_json(save_path=str(json_path))

            ocr_path = str(res["input_path"])
            ocr_text = str(res["rec_text"]).strip()
            ocr_score = float(res["rec_score"])
            if math.isfinite(ocr_score) and ocr_score >= self.score and ocr_text:
                results_list.append((ocr_path, ocr_text, ocr_score))
        if processed != len(images):
            raise RuntimeError(
                f"OCR returned {processed} results for {len(images)} images in {source}"
            )
        LOGGER.info("OCR: retained %d/%d frames from %s", len(results_list), processed, source)
        return results_list
