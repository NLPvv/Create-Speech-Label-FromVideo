import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from core.frame_skip import BatchFFmpeg


class FrameExtractionTests(unittest.TestCase):
    def setUp(self):
        # Preserve all generated files, matching the project's no-deletion rule.
        self.directory = Path(tempfile.mkdtemp(prefix="video-frame-tests-"))
        self.source = self.directory / "news.mp4"
        self.source.write_bytes(b"source video")
        self.output = self.directory / "frames"

    def extractor(self, **options):
        return BatchFFmpeg(self.output, **options)

    @staticmethod
    def fake_run(command, **kwargs):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
                stderr="",
            )
        output = command[-1]
        for index in range(1, 4):
            Path(output % index).write_bytes(b"frame data")
        return subprocess.CompletedProcess(command, 0, stderr="")

    def test_checked_noninteractive_command_and_actual_output_path(self):
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run) as run:
            result = self.extractor().process_video(self.source)
        self.assertEqual(result, self.output / "news")
        self.assertTrue((result / ".frames-complete.json").is_file())
        command = run.call_args.args[0]
        self.assertIn("-nostdin", command)
        self.assertIn("-n", command)
        self.assertIn("-xerror", command)
        self.assertEqual(command[command.index("-start_number") + 1], "1")
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_completed_matching_run_is_reused(self):
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run) as run:
            extractor = self.extractor()
            first = extractor.process_video(self.source)
            second = extractor.process_video(self.source)
        self.assertEqual(first, second)
        self.assertEqual(run.call_count, 1)

    def test_legacy_frames_are_preserved_and_not_reused(self):
        legacy = self.output / "news"
        legacy.mkdir(parents=True)
        original = legacy / "news_9999.jpg"
        original.write_bytes(b"legacy frame")
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run):
            result = self.extractor().process_video(self.source)
        self.assertNotEqual(result, legacy)
        self.assertTrue(result.name.startswith("news.run-"))
        self.assertEqual(original.read_bytes(), b"legacy frame")
        self.assertFalse((result / "news_9999.jpg").exists())

    def test_empty_existing_directory_is_not_shared_with_new_run(self):
        existing = self.output / "news"
        existing.mkdir(parents=True)
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run):
            result = self.extractor().process_video(self.source)
        self.assertNotEqual(result, existing)
        self.assertEqual(list(existing.iterdir()), [])

    def test_changed_settings_create_new_run_and_matching_run_can_be_reused(self):
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run) as run:
            first = self.extractor(fps=25).process_video(self.source)
            second = self.extractor(fps=5).process_video(self.source)
            third = self.extractor(fps=5).process_video(self.source)
        self.assertNotEqual(first, second)
        self.assertEqual(second, third)
        self.assertEqual(run.call_count, 2)

    def test_source_changes_invalidate_completed_output(self):
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run):
            first = self.extractor().process_video(self.source)
            self.source.write_bytes(b"different source video")
            second = self.extractor().process_video(self.source)
        self.assertNotEqual(first, second)
        self.assertTrue((first / "news_0001.jpg").exists())

    def test_same_stem_from_other_source_is_not_reused(self):
        other_source = self.directory / "other" / "news.mp4"
        other_source.parent.mkdir()
        other_source.write_bytes(b"source video")
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run):
            first = self.extractor().process_video(self.source)
            second = self.extractor().process_video(other_source)
        self.assertNotEqual(first, second)

    def test_percent_in_source_stem_is_escaped_in_ffmpeg_pattern(self):
        source = self.directory / "news%04d.mp4"
        source.write_bytes(b"source video")
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run):
            result = self.extractor().process_video(source)
        self.assertTrue((result / "news%04d_0001.jpg").is_file())

    def test_modified_frame_sequence_is_not_reused(self):
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run):
            first = self.extractor().process_video(self.source)
            (first / "news_0002.jpg").write_bytes(b"")
            second = self.extractor().process_video(self.source)
        self.assertNotEqual(first, second)
        self.assertEqual((first / "news_0002.jpg").stat().st_size, 0)

    def test_no_skip_preserves_prior_run_and_creates_new_directory(self):
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run):
            extractor = self.extractor(skip_if_exists=False)
            first = extractor.process_video(self.source)
            second = extractor.process_video(self.source)
        self.assertNotEqual(first, second)
        self.assertEqual(len(list(first.glob("*.jpg"))), 3)

    def test_failed_ffmpeg_retains_partial_frames_without_completion_marker(self):
        def failure(command, **kwargs):
            Path(command[-1] % 1).write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, command, stderr="decode failed")

        with patch("core.frame_skip.subprocess.run", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                self.extractor().process_video(self.source)
        failed = self.output / "news"
        self.assertEqual((failed / "news_0001.jpg").read_bytes(), b"partial")
        self.assertFalse((failed / ".frames-complete.json").exists())
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run):
            recovered = self.extractor().process_video(self.source)
        self.assertNotEqual(recovered, failed)

    def test_successful_exit_without_frames_is_rejected(self):
        with patch("core.frame_skip.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaisesRegex(ValueError, "no frames"):
                self.extractor().process_video(self.source)
        self.assertFalse((self.output / "news" / ".frames-complete.json").exists())

    def test_missing_frame_numbers_are_rejected(self):
        def noncontiguous(command, **kwargs):
            for index in (1, 3):
                Path(command[-1] % index).write_bytes(b"frame")
            return subprocess.CompletedProcess(command, 0)

        with patch("core.frame_skip.subprocess.run", side_effect=noncontiguous):
            with self.assertRaisesRegex(ValueError, "Incomplete frame sequence"):
                self.extractor().process_video(self.source)

    def test_source_change_during_extraction_is_not_marked_complete(self):
        def source_changed(command, **kwargs):
            result = self.fake_run(command, **kwargs)
            self.source.write_bytes(b"updated while ffmpeg was reading")
            return result

        with patch("core.frame_skip.subprocess.run", side_effect=source_changed):
            with self.assertRaisesRegex(RuntimeError, "changed during extraction"):
                self.extractor().process_video(self.source)
        self.assertFalse((self.output / "news" / ".frames-complete.json").exists())

    def test_roi_is_checked_and_added_to_ffmpeg_filter(self):
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run) as run:
            extractor = self.extractor(roi=(320, 590, 1280, 711))
            result = extractor.process_video(self.source)
            self.assertEqual(extractor.process_video(self.source), result)
        self.assertEqual(run.call_count, 2)
        ffmpeg = run.call_args.args[0]
        self.assertEqual(ffmpeg[ffmpeg.index("-vf") + 1], "fps=25,crop=960:121:320:590:exact=1")

    def test_out_of_bounds_roi_fails_before_creating_frame_directory(self):
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run) as run:
            with self.assertRaisesRegex(ValueError, "exceeds video dimensions"):
                self.extractor(roi=(1800, 100, 2000, 200)).process_video(self.source)
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.output / "news").exists())

    def test_public_roi_validation_works_without_enabling_extractor_crop(self):
        extractor = self.extractor()
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run) as run:
            self.assertEqual(extractor.validate_video_roi(self.source, (10, 20, 200, 121)),
                             (1920, 1080))
            result = extractor.process_video(self.source)
        self.assertIsNone(extractor.roi)
        ffmpeg = run.call_args.args[0]
        self.assertEqual(ffmpeg[ffmpeg.index("-vf") + 1], "fps=25")
        self.assertTrue((result / "news_0001.jpg").is_file())

    def test_public_roi_validation_rejects_invalid_explicit_coordinates(self):
        with patch("core.frame_skip.subprocess.run") as run:
            with self.assertRaises(ValueError):
                self.extractor().validate_video_roi(self.source, (0, 0, 0, 10))
        run.assert_not_called()

    def test_rotation_side_data_swaps_displayed_dimensions(self):
        for rotation in (90, -90, 270, 450, 89.5):
            stream = {"width": 1920, "height": 1080,
                      "side_data_list": [{"rotation": rotation}]}
            completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({"streams": [stream]}))
            with self.subTest(rotation=rotation):
                with patch("core.frame_skip.subprocess.run", return_value=completed):
                    extractor = self.extractor()
                    self.assertEqual(
                        extractor.validate_video_roi(self.source, (0, 1200, 1000, 1800)),
                        (1080, 1920),
                    )
                    with self.assertRaisesRegex(ValueError, "1080x1920"):
                        extractor.validate_video_roi(self.source, (1000, 0, 1800, 500))

    def test_legacy_rotate_tag_is_used_when_display_matrix_is_absent(self):
        stream = {"width": 1920, "height": 1080, "tags": {"rotate": "-90"}}
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({"streams": [stream]}))
        with patch("core.frame_skip.subprocess.run", return_value=completed):
            self.assertEqual(
                self.extractor().validate_video_roi(self.source, (0, 1000, 1000, 1800)),
                (1080, 1920),
            )

    def test_display_matrix_rotation_takes_precedence_over_legacy_rotate_tag(self):
        stream = {"width": 1920, "height": 1080, "tags": {"rotate": "90"},
                  "side_data_list": [{"rotation": 180}]}
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({"streams": [stream]}))
        with patch("core.frame_skip.subprocess.run", return_value=completed):
            self.assertEqual(
                self.extractor().validate_video_roi(self.source, (1000, 0, 1800, 500)),
                (1920, 1080),
            )

    def test_non_quarter_turn_rotation_preserves_default_ffmpeg_canvas(self):
        stream = {"width": 1920, "height": 1080, "side_data_list": [{"rotation": 45}]}
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({"streams": [stream]}))
        with patch("core.frame_skip.subprocess.run", return_value=completed):
            self.assertEqual(
                self.extractor().validate_video_roi(self.source, (1000, 0, 1800, 500)),
                (1920, 1080),
            )

    def test_missing_executables_raise_clear_errors(self):
        with patch("core.frame_skip.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "Cannot find ffmpeg"):
                self.extractor().process_video(self.source)
            with self.assertRaisesRegex(RuntimeError, "Cannot find ffprobe"):
                self.extractor(roi=(0, 0, 100, 100)).process_video(self.source)

    def test_missing_input_fails_before_running_ffmpeg(self):
        with patch("core.frame_skip.subprocess.run") as run:
            with self.assertRaises(FileNotFoundError):
                self.extractor().process_video(self.directory / "missing.mp4")
            with self.assertRaises(ValueError):
                self.extractor().process_video("  ")
        run.assert_not_called()

    def test_invalid_configuration_is_rejected(self):
        invalid = [
            {"fps": 0}, {"fps": float("nan")}, {"fps": float("inf")},
            {"fps": True}, {"quality": 0}, {"quality": 32}, {"quality": 1.5},
            {"nj": 0}, {"nj": 2.5}, {"roi": (0, 0, 0, 20)},
            {"roi": (-1, 0, 10, 20)}, {"roi": (0, 0, 10)},
            {"roi": (0, 0, 10.5, 20)}, {"skip_if_exists": "yes"},
        ]
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.extractor(**options)

    def test_batch_uses_shared_implementation_and_preserves_order(self):
        other = self.directory / "other.mp4"
        other.write_bytes(b"other video")
        listing = self.directory / "videos.txt"
        listing.write_text(f"{self.source}\n\n{other}\n", encoding="utf-8")
        with patch("core.frame_skip.subprocess.run", side_effect=self.fake_run):
            results = self.extractor(nj=2).run_batch(listing)
        self.assertEqual([path.name for path in results], ["news", "other"])


if __name__ == "__main__":
    unittest.main()
