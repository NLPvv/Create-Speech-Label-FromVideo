import os
import cv2
from turbojpeg import TurboJPEG
import logging
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from typing import List, Tuple
from tqdm import tqdm


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
        self.input_dir: str = input_dir
        self.crop_dir: str = crop_dir

        self.ymin: int = ymin
        self.ymax: int = ymax
        self.xmin: int = xmin
        self.xmax: int = xmax

        self.save_crop: bool = save_crop
        self.num_workers: int = num_workers

        self.supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        self.jpeg = TurboJPEG()
        cv2.setNumThreads(4)

    def fast_read(self, path: str):
        """
        TurboJPEG replace cv2.imread，jpg/jpeg 

        """
        ext = path.lower().split(".")[-1]
        if ext in ("jpg", "jpeg"):
            with open(path, "rb") as f:
                return self.jpeg.decode(f.read())

        return cv2.imread(path)

    def crop_all_images(self,   filename: str)-> Tuple[str, any]:

        img_path = os.path.join(self.input_dir, filename)

        try:
            img = self.fast_read(img_path)
            if img is None:
                return (filename, None)
            
            h, w = img.shape[:2]
            y1, y2 = max(0, self.ymin), min(h, self.ymax)
            x1, x2 = max(0, self.xmin), min(w, self.xmax)

            crop = img[y1:y2, x1:x2]

            if self.save_crop:
                os.makedirs(self.crop_dir, exist_ok=True)
                save_path = os.path.join(self.crop_dir, filename)
                cv2.imwrite(save_path, crop,[cv2.IMWRITE_JPEG_QUALITY, 75])
            else:
                print(f"[INFO] skip saving: save_crop=False (file: ({filename})")
        except Exception as e:
            print(f"[ERROR] processing failed: {img_path},: {e}")

            return filename

    def run(self) -> List[Tuple[str, any]]:
        
        files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(self.supported_ext)
        ]

        workers = min(self.num_workers, cpu_count())

        if workers <= 1: 
            for f in tqdm(files, desc="Processing Cropper"):
                self.crop_all_images(f)
            return

        with ThreadPool(workers) as pool:
            for _ in tqdm(
                     pool.imap_unordered(self.crop_all_images, files, chunksize=50),
                     total=len(files),
                      desc="Processing Cropper"
             ):
                pass
            
        #with Pool(workers) as pool:
        #     for _ in tqdm(
        #         pool.imap_unordered(self.crop_all_images, files, chunksize=10),
        #         total= len(files),
        #         desc="Processing Cropper"
        #     ):
        #         pass
