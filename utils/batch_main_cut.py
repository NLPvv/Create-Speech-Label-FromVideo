import os
import shutil
from ..core.cropper import Cropper

YMIN, YMAX = 911, 992
XMIN, XMAX = 504, 1910
NUM_WORKERS = 16

ROOT_DIR = "/mnt/nvme_workspace1/liziwei/ocr/2025_11_12_1920_1080"

def run_crop_for_all_folders(root_dir):
    """
    遍历 root_dir 下的所有文件夹，
    对每个文件夹执行 Cropper，
    输出到 新文件夹名 + '_crop'
    并删除原文件夹
    """

    all_folders = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    print("Found sub-folders:", all_folders)

    for folder_path in all_folders:
        folder_name = os.path.basename(folder_path)
        crop_dir = os.path.join(root_dir, folder_name + "_crop")

        print(f"\n=== Processing Folder: {folder_name} ===")
        print(f"Input:  {folder_path}")
        print(f"Output: {crop_dir}")

        cropper = Cropper(
            input_dir=folder_path,
            crop_dir=crop_dir,
            ymin=YMIN, ymax=YMAX,
            xmin=XMIN, xmax=XMAX,
            save_crop=True,
            num_workers=NUM_WORKERS
        )

        cropper.run()

        try:
            shutil.rmtree(folder_path)
            print(f"[INFO] Deleted original folder: {folder_path}")
        except Exception as e:
            print(f"[ERROR] Failed to delete folder {folder_path}: {e}")


if __name__ == "__main__":
    run_crop_for_all_folders(ROOT_DIR)

