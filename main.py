from core.frame_skip import BatchFFmpeg 
from core.cropper import Cropper
from core.ocr_module import OCRProcessor
from core.subtitle_generator import SubtitleGenerator

if __name__ == "__main__":

    # ===== Basic Config =====
    VIDEO_PATH = '/mnt/nas4_workspace/spmiData/asr/guangxi/ASR/zhuang/zhuang_news/gxtv/2020/gxtv_20200108.mp4'
    INPUT_DIR = "./image/"
    CROP_DIR = "./crop/"
    SRT_DIR = "./subtitle.srt"

    # FRAME
    NJ = 16

    # CUT
    YMIN, YMAX = 590, 711
    XMIN, XMAX = 320, 1280
    NUM_WORKERS = 10


    # OCR
    DEVICE = "gpu:0"
    HPI = True
    BATCH_SIZE = 5
    SCORE = 0.9  # OCR结果置信度

    # SUB
    FPS = 25 # 帧率
    SIM_THRESHOLD = 80 # 相似度得分 


    # Save
    SAVE_CROP = True
    SAVE_JSON = False
    OUTPUT_DIR = "json_rec/output/" # OCR 识别结果json文件
    VERBOSE = True

    print("\n===== Step 1: Video frame extraction =====")
    extractor = BatchFFmpeg(
        output_root=INPUT_DIR,
        fps=FPS,
        quality=2,
        nj=NJ
    )
    extractor.process_video(VIDEO_PATH)
    
    print("\n===== Step 2: Crop all images =====")
    cropper = Cropper(
        input_dir=INPUT_DIR,
        crop_dir=CROP_DIR,
        ymin=YMIN, ymax=YMAX,
        xmin=XMIN, xmax=XMAX,
        save_crop=SAVE_CROP,
        num_workers=NUM_WORKERS
    )
    crop_results = cropper.run() 

    print("\n===== Step 3: OCR Recognition =====")
    ocrer = OCRProcessor(
        input_dir=CROP_DIR,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        enable_hpi=HPI,
        batch_size=BATCH_SIZE,
        score=SCORE,
        save_json=SAVE_JSON,
        verbose=VERBOSE
    )
    ocr_results = ocrer.run()
    
    print("\n====== Step 4: Generate subtitle ===")
    sub = SubtitleGenerator(
        ocr_results, 
        fps=FPS, 
        sim_threshold=SIM_THRESHOLD)
    sub.generate_srt(SRT_DIR)

