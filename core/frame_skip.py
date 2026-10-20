import os
import subprocess
from multiprocessing import Pool
from functools import partial


class BatchFFmpeg:
    def __init__(self, output_root, fps=25, quality=2, skip_if_exists=True, nj=16):
        self.output_root = output_root
        self.fps = fps
        self.quality = quality
        self.skip_if_exists = skip_if_exists
        self.nj = nj

    def process_video(self, video_path):
        video_path = video_path.strip()
        if not video_path:
            return

        if not os.path.exists(video_path):
            print(f"Not exist : {video_path}")
            return

        base_name = os.path.basename(video_path)
        name_no_ext = os.path.splitext(base_name)[0]

        out_dir = os.path.join(self.output_root, name_no_ext)

        if self.skip_if_exists and os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
            print(f"pass {name_no_ext}")
            return

        os.makedirs(out_dir, exist_ok=True)

        output_pattern = os.path.join(out_dir, f"{name_no_ext}_%04d.jpg")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={self.fps}",
            "-qscale:v", str(self.quality),
            output_pattern
        ]

        subprocess.run(cmd)

    def run_batch(self, list_file):
        with open(list_file, "r", encoding="utf-8") as f:
            video_list = [line.strip() for line in f if line.strip()]


        with Pool(self.nj) as pool:
            pool.map(self.process_video, video_list)


def main():
    extractor = BatchFFmpeg(
        output_root="./fram_output",
        fps=25,
        quality=2,
        skip_if_exists=True,
        nj=20
    )
    extractor.run_batch("mp4.lsit")


if __name__ == "__main__":
    main()
