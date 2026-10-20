import re
import unicodedata
from collections import Counter
from typing import List, Tuple
from thefuzz import fuzz

OCRItem = Tuple[Tuple[str], Tuple[str], float]


class SubtitleGenerator:

    def __init__(self, ocr_results: List[OCRItem], fps: int = 25,
                 sim_threshold: int = 90, use_vsf=False):
        self.ocr_results = ocr_results
        self.fps = fps
        self.sim_threshold = sim_threshold
        self.use_vsf = use_vsf

        self.raw_list = []


    # ========= Basic tools ========= #

    def _extract_frame(self, path: str) -> int:
        m = re.findall(r"(\d+)\.jpg$", path)
        if not m:
            m = re.findall(r"(\d+)", path)
        return int(m[-1])

    def _frame_to_tc(self, frame: int) -> str:
        sec = frame / self.fps
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _convert_to_raw_list(self):
        raw = []
        for item in self.ocr_results:
            img = item[0][0]
            text = item[1][0].replace("\n", " ")
            frame = self._extract_frame(img)

            raw.append([frame, "0,0,0,0", text])

        raw.sort(key=lambda x: x[0])
        self.raw_list = raw

    def _remove_duplicate_subtitle(self):
        """
        _remove_duplicate_subtitle():
        https://github.com/YaoFANGUK/video-subtitle-extractor/blob/main/backend/main.py#L734
        output(start_frame, end_frame, text)
        """
        content_list = self.raw_list
        content_list_len = len(content_list)
        result = []

        idx_i = 0

        while idx_i < content_list_len:
            start_frame = content_list[idx_i][0]
            base_text = content_list[idx_i][2].replace(" ", "")

            idx_j = idx_i
            while idx_j < content_list_len:
                curr_text = content_list[idx_j][2].replace(" ", "")

                # 如果下一行不存在 或与 base_text 不相似 → 分段结束
                if (idx_j + 1 == content_list_len) or \
                   (fuzz.ratio(base_text, content_list[idx_j + 1][2].replace(" ", "")) < self.sim_threshold):

                    end_frame = content_list[idx_j][0]

                    # 找出最长文本
                    '''
                    segment = content_list[idx_i: idx_j + 1]
                    seg_clean = [s[2].replace(" ", "") for s in segment]
                    max_index, _ = max(enumerate(seg_clean), key=lambda x: len(x[1]))
                    best_text = segment[max_index][2].strip()

                    result.append((int(start_frame), int(end_frame), best_text))
                    idx_i = idx_j + 1
                    break
                    '''
                    # 这里稍微做些修改下找出出现次数最多句子
                    segment = content_list[idx_i: idx_j + 1]
                    counter = Counter([s[2].strip() for s in segment])
                    best_text = counter.most_common(1)[0][0]
                    result.append((int(start_frame), int(end_frame), best_text))
                    idx_i = idx_j + 1
                    break
                
                idx_j += 1

        return result


    # ========= export SRT ========= #

    def generate_srt(self, save_path="subtitle.srt"):
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

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"ubtitle save: {save_path}")

