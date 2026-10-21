"""Pipeline and OCR regressions without downloading an inference runtime."""

import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main
from core.ocr_module import OCRProcessor
from utils import batch_main_ocr_sub as batch


class OCRProcessorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="subtitle-ocr-test-"))
        self.images = self.root / "images"
        self.images.mkdir()
        for name in ("news_0001.jpg", "news_0002.jpg", "news_0003.png"):
            (self.images / name).write_bytes(b"image")
        (self.images / "ignored.txt").write_text("ignored", encoding="utf-8")

    def results(self):
        return [
            {"input_path": str(self.images / "news_0001.jpg"), "rec_text": " hello ", "rec_score": 0.99},
            {"input_path": str(self.images / "news_0002.jpg"), "rec_text": "uncertain", "rec_score": 0.5},
            {"input_path": str(self.images / "news_0003.png"), "rec_text": "   ", "rec_score": 0.99},
        ]

    def test_filters_confidence_and_empty_text_and_unwraps_output(self):
        processor = OCRProcessor(str(self.images), batch_size=2, score=0.9)
        processor.ocr = Mock()
        processor.ocr.predict_iter.return_value = self.results()
        self.assertEqual(processor.run(), [(str(self.images / "news_0001.jpg"), "hello", 0.99)])
        self.assertEqual(len(processor.ocr.predict_iter.call_args.kwargs["input"]), 3)
        self.assertEqual(processor.ocr.predict_iter.call_args.kwargs["batch_size"], 2)

    def test_model_is_loaded_once_and_reused_for_different_directories(self):
        model = Mock()
        model.predict_iter.return_value = self.results()
        factory = Mock(return_value=model)
        with patch.dict(sys.modules, {"paddleocr": types.SimpleNamespace(TextRecognition=factory)}):
            processor = OCRProcessor(str(self.images), device="cpu")
            factory.assert_not_called()
            processor.run()
            processor.run(str(self.images))
        factory.assert_called_once()
        self.assertEqual(model.predict_iter.call_count, 2)

    def test_json_names_use_image_names_and_caller_output_directory(self):
        class Result(dict):
            save_to_json = Mock()
        results = [Result(result) for result in self.results()]
        processor = OCRProcessor(str(self.images), save_json=True)
        processor.ocr = Mock()
        processor.ocr.predict_iter.return_value = results
        destination = self.root / "json" / "video"
        processor.run(output_dir=str(destination))
        self.assertTrue(destination.is_dir())
        self.assertEqual(results[0].save_to_json.call_args_list[0].kwargs,
                         {"save_path": str(destination / "news_0001.jpg.json")})

    def test_missing_and_empty_input_fail_before_model_load(self):
        empty = self.root / "empty"
        empty.mkdir()
        for source, error in ((self.root / "missing", FileNotFoundError), (empty, ValueError)):
            processor = OCRProcessor(str(source))
            with patch.object(processor, "_get_model") as load, self.assertRaises(error):
                processor.run()
            load.assert_not_called()

    def test_partial_ocr_output_is_reported(self):
        processor = OCRProcessor(str(self.images))
        processor.ocr = Mock()
        processor.ocr.predict_iter.return_value = self.results()[:1]
        with self.assertRaisesRegex(RuntimeError, "1 results for 3 images"):
            processor.run()

    def test_nonfinite_score_is_discarded(self):
        results = self.results()
        results[0]["rec_score"] = float("nan")
        processor = OCRProcessor(str(self.images))
        processor.ocr = Mock()
        processor.ocr.predict_iter.return_value = results
        self.assertEqual(processor.run(), [])

    def test_invalid_configuration_is_rejected(self):
        for options in ({"batch_size": 0}, {"batch_size": 1.5}, {"score": -0.1},
                        {"score": float("nan")}, {"device": ""}, {"model_name": ""}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                OCRProcessor(str(self.images), **options)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="subtitle-pipeline-test-"))
        self.video = self.root / "news.mp4"
        self.video.write_bytes(b"video")
        self.args = main.build_parser().parse_args([
            str(self.video), "--roi", "10", "20", "200", "140",
            "--work-dir", str(self.root / "work"), "--fps", "5",
        ])

    def test_passes_actual_extraction_directory_to_ocr_and_preserves_fps(self):
        actual = self.root / "work" / "crop" / "news.run-123"
        with patch.object(main, "BatchFFmpeg") as extractor, \
                patch.object(main, "OCRProcessor") as ocr, \
                patch.object(main.importlib.util, "find_spec", return_value=object()):
            extractor.return_value.process_video.return_value = actual
            ocr.return_value.run.return_value = [
                (str(actual / "news_0001.jpg"), "hello", 0.99),
                (str(actual / "news_0002.jpg"), "hello", 0.99),
            ]
            output = main.run_pipeline(self.args)
        self.assertEqual(extractor.call_args.kwargs["roi"], (10, 20, 200, 140))
        self.assertEqual(ocr.return_value.run.call_args.args[0], str(actual))
        self.assertEqual(output, self.video.with_suffix(".srt"))
        self.assertIn("00:00:00,000 --> 00:00:00,400", output.read_text(encoding="utf-8"))

    def test_missing_dependencies_fail_before_extraction(self):
        with patch.object(main.importlib.util, "find_spec", return_value=None), \
                patch.object(main.BatchFFmpeg, "process_video") as extract, \
                self.assertRaisesRegex(RuntimeError, "Missing paddleocr"):
            main.run_pipeline(self.args)
        extract.assert_not_called()

    def test_full_frame_mode_validates_roi_and_scopes_json_by_video(self):
        self.args.keep_full_frames = True
        actual = self.root / "work" / "frames" / "news.run-456"
        cropper = Mock()
        with patch.object(main, "BatchFFmpeg") as extractor, \
                patch.object(main, "OCRProcessor") as ocr, \
                patch.object(main.importlib.util, "find_spec", return_value=object()), \
                patch.dict(sys.modules, {"core.cropper": types.SimpleNamespace(Cropper=cropper)}):
            extractor.return_value.process_video.return_value = actual
            ocr.return_value.run.return_value = [(str(actual / "news_0001.jpg"), "hello", 0.99)]
            main.run_pipeline(self.args)
        extractor.return_value.validate_video_roi.assert_called_once_with(str(self.video), (10, 20, 200, 140))
        self.assertIsNone(extractor.call_args.kwargs["roi"])
        self.assertEqual(cropper.call_args.args[0], str(actual))
        self.assertEqual(ocr.return_value.run.call_args.args[1],
                         str(self.root / "work" / "json" / "news.run-456" / "roi_10_20_200_140"))

    def test_full_frame_mode_rejects_out_of_bounds_roi_before_extraction(self):
        self.args.keep_full_frames = True
        with patch.object(main.importlib.util, "find_spec", return_value=object()), \
                patch.object(main.BatchFFmpeg, "validate_video_roi", side_effect=ValueError("exceeds video dimensions")), \
                patch.object(main.BatchFFmpeg, "process_video") as extract, \
                self.assertRaisesRegex(ValueError, "exceeds video dimensions"):
            main.run_pipeline(self.args)
        extract.assert_not_called()

    def test_extraction_failure_stops_before_ocr(self):
        with patch.object(main.importlib.util, "find_spec", return_value=object()), \
                patch.object(main.BatchFFmpeg, "process_video", side_effect=RuntimeError("FFmpeg failed")), \
                patch.object(main.OCRProcessor, "run") as recognize, \
                self.assertRaisesRegex(RuntimeError, "FFmpeg failed"):
            main.run_pipeline(self.args)
        recognize.assert_not_called()

    def test_invalid_video_returns_nonzero_exit(self):
        with patch.object(main.LOGGER, "error"):
            code = main.main([str(self.root / "missing.mp4"), "--roi", "0", "0", "10", "10"])
        self.assertEqual(code, 1)

    def test_output_cannot_overwrite_video_through_a_hardlink(self):
        import os
        output = self.root / "alias.srt"
        os.link(self.video, output)
        self.args.output = output
        with patch.object(main.BatchFFmpeg, "process_video") as extract, \
                self.assertRaisesRegex(ValueError, "overwrite the input video"):
            main.run_pipeline(self.args)
        extract.assert_not_called()
        self.assertEqual(self.video.read_bytes(), b"video")

    def test_output_must_be_srt(self):
        self.args.output = self.video
        with self.assertRaisesRegex(ValueError, ".srt extension"):
            main.run_pipeline(self.args)


class BatchOCRTests(unittest.TestCase):
    def test_worker_reuses_model_and_keeps_json_per_video(self):
        root = Path(tempfile.mkdtemp(prefix="subtitle-batch-test-"))
        args = batch.build_parser().parse_args([str(root), "--output-dir", str(root / "out")])
        tasks = queue.Queue()
        for name in ("video1_crop", "video2_crop"):
            tasks.put(str(root / name))
        tasks.put(None)
        with patch.object(batch, "OCRProcessor") as processor, \
                patch.object(batch, "SubtitleGenerator") as generator:
            processor.return_value.run.return_value = [("frame_0001.jpg", "text", 1.0)]
            batch.gpu_worker(tasks, None, args)
        processor.assert_called_once()
        self.assertEqual(processor.return_value.run.call_count, 2)
        self.assertEqual(generator.call_count, 2)
        calls = processor.return_value.run.call_args_list
        self.assertEqual(calls[0].args[1], str(root / "out" / "json" / "video1"))
        self.assertEqual(calls[1].args[1], str(root / "out" / "json" / "video2"))

    def test_worker_reports_failure_and_continues_other_tasks(self):
        root = Path(tempfile.mkdtemp(prefix="subtitle-batch-failure-test-"))
        args = batch.build_parser().parse_args([str(root)])
        tasks = queue.Queue()
        tasks.put(str(root / "bad_crop"))
        tasks.put(str(root / "good_crop"))
        tasks.put(None)
        with patch.object(batch, "OCRProcessor") as processor, \
                patch.object(batch, "SubtitleGenerator") as generator, \
                patch.object(batch.LOGGER, "exception"), \
                self.assertRaises(SystemExit) as error:
            processor.return_value.run.side_effect = [ValueError("bad image"), [("f_0001.jpg", "ok", 1.0)]]
            batch.gpu_worker(tasks, None, args)
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(processor.return_value.run.call_count, 2)
        generator.assert_called_once()


if __name__ == "__main__":
    unittest.main()
