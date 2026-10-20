import os
from paddleocr import TextRecognition
from paddleocr import PaddleOCR
from typing import List, Tuple, Any


class OCRProcessor:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        device: str = "gpu:0",
        batch_size: int = 1,
        score: float = 0.8,
        save_json: bool = False,
        enable_hpi: bool = True,
        verbose: bool = True
    ):
        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.device: str = device
        self.batch_size: int = batch_size
        self.score: float = score
        self.save_json: bool = save_json
        self.hpi: bool = enable_hpi
        self.verbose: bool = verbose

        if self.save_json:
            os.makedirs(self.output_dir, exist_ok=True)

        # INIT OCR model
        self.ocr = TextRecognition(
                model_name = "PP-OCRv5_server_rec",
                enable_hpi = self.hpi,
                device = self.device
                )

    def run(self) -> List[Tuple[str, Any]]:
        results_list: List[Tuple[str, Any]] = []

        results = self.ocr.predict_iter(
            input=self.input_dir,
            batch_size = self.batch_size
        )
        
        for idx, res in enumerate(results):
            
            if self.verbose:
                res.print()

            if self.save_json:
                json_path = os.path.join(self.output_dir, f"res_{idx + 1}.json")
                res.save_to_json(json_path)
                print(f"[save] {json_path}")

            ocr_path = res['input_path'],
            ocr_text = res['rec_text'],
            ocr_score = res['rec_score']

            # Text confidence
            if ocr_score >= self.score:
                results_list.append((ocr_path, ocr_text, ocr_score))

        return results_list
