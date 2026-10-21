import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.subtitle_generator import SubtitleGenerator, _similarity_ratio


class SubtitleGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep generated artifacts: this project forbids automatic file deletion.
        cls.output_dir = Path(tempfile.mkdtemp(prefix="subtitle-regression-"))

    def render(self, items, filename="subtitle.srt", **options):
        output = self.output_dir / self.id().rsplit(".", 1)[-1] / filename
        generator = SubtitleGenerator(items, **options)
        self.assertEqual(generator.generate_srt(output), str(output))
        return output.read_text(encoding="utf-8")

    def test_first_frame_starts_at_zero_and_end_is_exclusive(self):
        content = self.render([
            ("frame_000002.jpg", "字幕", 0.99),
            ("frame_000001.jpg", "字幕", 0.99),
        ])
        self.assertEqual(content, "1\n00:00:00,000 --> 00:00:00,080\n字幕\n")

    def test_exact_milliseconds_and_rollovers(self):
        generator = SubtitleGenerator([], fps=25)
        self.assertEqual(generator._frame_to_tc(30), "00:00:01,160")
        self.assertEqual(generator._frame_to_tc(1501), "00:01:00,000")
        self.assertEqual(generator._frame_to_tc(90001), "01:00:00,000")

    def test_fractional_fps_rounds_to_nearest_millisecond(self):
        generator = SubtitleGenerator([], fps=29.97)
        self.assertEqual(generator._frame_to_tc(2), "00:00:00,033")
        self.assertEqual(generator._frame_to_tc(31), "00:00:01,001")

    def test_zero_based_frame_indices_remain_supported(self):
        content = self.render([("frame_0.png", "零基帧", 1)], frame_index_base=0)
        self.assertIn("00:00:00,000 --> 00:00:00,040", content)

    def test_filtered_frames_leave_a_gap_between_identical_subtitles(self):
        content = self.render([
            ("frame_1.jpg", "相同字幕", 0.99),
            ("frame_4.jpg", "相同字幕", 0.99),
        ])
        self.assertEqual(
            content,
            "1\n00:00:00,000 --> 00:00:00,040\n相同字幕\n\n"
            "2\n00:00:00,120 --> 00:00:00,160\n相同字幕\n",
        )

    def test_gap_tolerance_is_configurable(self):
        content = self.render([
            ("frame_1.jpg", "相同字幕", 1),
            ("frame_3.jpg", "相同字幕", 1),
        ], max_gap_frames=2)
        self.assertEqual(content, "1\n00:00:00,000 --> 00:00:00,120\n相同字幕\n")

    def test_legacy_and_flat_ocr_records_are_compatible(self):
        content = self.render([
            (("frame_1.jpeg",), ("完整字幕\n第二行",), 0.99),
            (Path("frame_2.PNG"), "完整字幕 第二行", 0.99),
        ])
        self.assertIn("完整字幕 第二行", content)
        self.assertIn("00:00:00,000 --> 00:00:00,080", content)

    def test_empty_text_is_ignored_and_breaks_continuity(self):
        content = self.render([
            ("frame_1.jpg", "文本", 1),
            ("frame_2.jpg", " \n\t ", 1),
            ("frame_3.jpg", "文本", 1),
        ])
        self.assertEqual(content.count(" --> "), 2)
        self.assertEqual(self.render([], filename="empty.srt"), "")

    def test_similar_segment_uses_most_frequent_complete_text(self):
        content = self.render([
            ("frame_1.jpg", "hello wor1d", 1),
            ("frame_2.jpg", "hello world", 1),
            ("frame_3.jpg", "hello world", 1),
            ("frame_4.jpg", "other text", 1),
        ], sim_threshold=80)
        self.assertIn("00:00:00,000 --> 00:00:00,120\nhello world", content)
        self.assertIn("2\n00:00:00,120 --> 00:00:00,160\nother text", content)

    def test_frame_number_comes_from_image_basename_only(self):
        generator = SubtitleGenerator([])
        self.assertEqual(generator._extract_frame("/video123/frame_42.JPEG"), 42)
        self.assertEqual(generator._extract_frame(r"C:\video123\frame_42.png"), 42)
        for path in ("/video123/image.jpg", "/video123/frame_4.txt", "", None, "frame_0.jpg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                generator._extract_frame(path)

    def test_invalid_parameters_fail_before_export(self):
        for options in (
            {"fps": 0}, {"fps": -25}, {"fps": float("nan")},
            {"fps": float("inf")}, {"fps": True}, {"fps": "25"},
            {"sim_threshold": -1}, {"sim_threshold": 101},
            {"sim_threshold": float("nan")}, {"max_gap_frames": 0},
            {"max_gap_frames": 1.5}, {"max_gap_frames": True},
            {"frame_index_base": 2}, {"frame_index_base": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SubtitleGenerator([], **options)
        for records in (None, "not OCR records"):
            with self.subTest(records=records), self.assertRaises(ValueError):
                SubtitleGenerator(records)

    def test_malformed_records_and_output_paths_fail_clearly(self):
        for item in (("frame_1.jpg", "missing score"), ("frame_1.jpg", None, 1)):
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.render([item])
        for path in ("", None, "\x00.srt"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                SubtitleGenerator([]).generate_srt(path)

    def test_export_creates_nested_output_directories(self):
        content = self.render([("frame_1.jpg", "内容", 1)], filename="nested/output.srt")
        self.assertIn("内容", content)

    def test_similarity_fallback_matches_normalized_indel_score(self):
        with patch("core.subtitle_generator.fuzz", None):
            self.assertEqual(_similarity_ratio("同一字幕", "同一字幕"), 100)
            self.assertEqual(_similarity_ratio("abc", "ac"), 80)
            self.assertEqual(_similarity_ratio("abc", "def"), 0)
            self.assertEqual(_similarity_ratio("", "abc"), 0)
            # An exact common subsequence is needed; greedy matching is wrong here.
            self.assertEqual(_similarity_ratio("tide", "diet"), 50)


if __name__ == "__main__":
    unittest.main()
