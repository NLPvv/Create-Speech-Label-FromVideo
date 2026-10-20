import os
import subprocess
from multiprocessing import Pool
from functools import partial

# ---- 可配置参数 ----
txt_file = "list"
output_root = "/mnt/nvme_workspace1/liziwei/ocr/2025_11_12_1920_1080"
fps = 25
quality = 2
nj = 20              
skip_if_exists = True   
# -------------------


def process_video(video_path, fps, quality, skip_if_exists):
    video_path = video_path.strip()
    if not video_path:
        return

    if not os.path.exists(video_path):
        print(f"Not exist : {video_path}")
        return

    base_name = os.path.basename(video_path)
    name_no_ext = os.path.splitext(base_name)[0]

    out_dir = os.path.join(output_root, name_no_ext)
    
    if skip_if_exists and os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"pass {name_no_ext}")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    output_pattern = os.path.join(out_dir, f"{name_no_ext}_%04d.jpg")

    # ffmpeg 命令
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-qscale:v", str(quality),
        output_pattern
    ]

    print(f"[开始] 抽帧 {base_name}")
    subprocess.run(cmd)
    print(f"[完成] {base_name}")


def main():
    with open(txt_file, "r", encoding="utf-8") as f:
        video_list = [line.strip() for line in f if line.strip()]

    print(f"总文件数: {len(video_list)}, 并发进程数: {nj}")

    worker = partial(process_video, fps=fps, quality=quality, skip_if_exists=skip_if_exists)

    # Pool
    with Pool(nj) as pool:
        pool.map(worker, video_list)


if __name__ == "__main__":
    main()

