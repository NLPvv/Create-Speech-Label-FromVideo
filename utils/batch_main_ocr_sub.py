import os
import multiprocessing as mp
from queue import Empty
from ..core.ocr_module import OCRProcessor
from ..core.subtitle_generator import SubtitleGenerator

BASE_DIR = "/mnt/nvme_workspace1/liziwei/ocr/2025_11_12_1920_1080/"
OUTPUT_DIR = "./output/"
SAVE_JSON = False
VERBOSE = True

GPUS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
HPI = True
BATCH_SIZE = 10
SCORE = 0.9


def gpu_worker(task_queue, gpu_id):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "gpu:0"

    print(f"----------[Worker GPU {gpu_id}] Start-----------")

    while True:
        try:
            crop_dir_path = task_queue.get_nowait()
        except Empty:
            break

        folder_name = os.path.basename(crop_dir_path)
        base_name = folder_name.replace("_crop", "")
        srt_path = f"2025_11_12_1920_1080/{base_name}.srt"

        print(f"\n----- {crop_dir_path}: {device} -----")

        # 1. OCR
        ocrer = OCRProcessor(
            input_dir=crop_dir_path,
            output_dir=OUTPUT_DIR,
            device=device,
            enable_hpi=HPI,
            batch_size=BATCH_SIZE,
            score=SCORE,
            save_json=SAVE_JSON,
            verbose=VERBOSE
        )
        ocr_results = ocrer.run()

        # 2. SRT
        sub = SubtitleGenerator(
            ocr_results,
            fps=25,
            sim_threshold=70
        )
        sub.generate_srt(srt_path)

        print(f"[GPU {gpu_id}]  -----→ {srt_path}")



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    folders = sorted([
        os.path.join(BASE_DIR, f)
        for f in os.listdir(BASE_DIR)
        if f.endswith("_crop") and os.path.isdir(os.path.join(BASE_DIR, f))
    ])

    if not folders:
        exit(0)

    for f in folders:
        print(" -", f)

    
    task_queue = mp.Queue()
    for f in folders:
        task_queue.put(f)


    processes = []
    for gpu_id in GPUS:
        p = mp.Process(target=gpu_worker, args=(task_queue, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



