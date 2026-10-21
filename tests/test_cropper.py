"""Cropper regressions using small stand-ins; OpenCV is not required."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock, mock_open, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fake_cv2 = types.ModuleType("cv2")
fake_cv2.imread = Mock()
fake_cv2.imwrite = Mock()
fake_cv2.IMWRITE_JPEG_QUALITY = 1
fake_tqdm = types.ModuleType("tqdm")
fake_tqdm.tqdm = lambda iterable, **kwargs: iterable
with patch.dict(sys.modules, {"cv2": fake_cv2, "turbojpeg": None, "tqdm": fake_tqdm}):
    cropper_module = load_module("cropper_under_test", "core/cropper.py")
with patch.dict(sys.modules, {"core.cropper": cropper_module}):
    batch_module = load_module("batch_cropper_under_test", "utils/batch_main_cut.py")


class CropperTests(unittest.TestCase):
    def setUp(self):
        self.image = MagicMock()
        self.image.shape = (100, 200, 3)
        self.crop = MagicMock()
        self.crop.copy.return_value = self.crop
        self.image.__getitem__.return_value = self.crop
        fake_cv2.imread.reset_mock()
        fake_cv2.imread.side_effect = None
        fake_cv2.imread.return_value = self.image
        fake_cv2.imwrite.reset_mock()
        fake_cv2.imwrite.side_effect = None
        fake_cv2.imwrite.return_value = True

    def make_cropper(self, **kwargs):
        return cropper_module.Cropper("/frames", "/crops", **kwargs)

    def run_with_images(self, cropper, filenames, output_filenames=None):
        entries = [Path("/frames") / name for name in filenames]
        output_entries = [Path("/crops") / name for name in (output_filenames or [])]
        with patch.object(Path, "is_dir", return_value=True), \
                patch.object(Path, "exists", return_value=output_filenames is not None), \
                patch.object(Path, "iterdir", side_effect=[iter(entries), iter(output_entries)]), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(cropper_module.os, "makedirs") as makedirs:
            return cropper.run(), makedirs

    def test_rejects_invalid_configuration(self):
        for kwargs in ({"num_workers": 0}, {"ymin": 10, "ymax": 10}, {"xmax": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_cropper(**kwargs)

    def test_jpeg_falls_back_when_optional_library_is_absent(self):
        self.assertIsNone(cropper_module.TurboJPEG)
        self.assertIs(self.make_cropper().fast_read("/frames/a.jpg"), self.image)
        fake_cv2.imread.assert_called_once_with("/frames/a.jpg")

    def test_jpeg_falls_back_when_native_library_cannot_initialize(self):
        with patch.object(cropper_module, "TurboJPEG", side_effect=OSError("missing native library")):
            self.assertIs(self.make_cropper().fast_read("/frames/a.jpg"), self.image)

    def test_jpeg_decode_failure_falls_back_to_opencv(self):
        decoder = Mock()
        decoder.decode.side_effect = ValueError("unsupported JPEG")
        with patch.object(cropper_module, "TurboJPEG", return_value=decoder), \
                patch("builtins.open", mock_open(read_data=b"jpeg")):
            self.assertIs(self.make_cropper().fast_read("/frames/a.jpg"), self.image)
        fake_cv2.imread.assert_called_once()

    def test_failed_read_is_not_silently_skipped(self):
        fake_cv2.imread.return_value = None
        with self.assertRaisesRegex(cropper_module.CropError, "/frames/bad.png"):
            self.make_cropper().crop_all_images("bad.png")

    def test_roi_outside_image_is_reported(self):
        with self.assertRaisesRegex(cropper_module.CropError, "ROI.*a.png"):
            self.make_cropper(ymin=150, ymax=200).crop_all_images("a.png")
        fake_cv2.imwrite.assert_not_called()

    def test_partial_roi_is_clipped_to_image(self):
        cropper = self.make_cropper(ymin=-10, ymax=200, xmin=-20, xmax=500, save_crop=False)
        self.assertEqual(cropper.crop_all_images("a.png"), ("a.png", self.crop))
        self.image.__getitem__.assert_called_once_with((slice(0, 100), slice(0, 200)))
        self.crop.copy.assert_called_once_with()

    def test_failed_write_is_reported(self):
        fake_cv2.imwrite.return_value = False
        with self.assertRaisesRegex(cropper_module.CropError, "/crops/a.png"):
            self.make_cropper().crop_all_images("a.png")

    def test_decoder_exception_keeps_path_and_original_cause(self):
        fake_cv2.imread.side_effect = OSError("read error")
        with self.assertRaisesRegex(cropper_module.CropError, "/frames/a.png") as error:
            self.make_cropper().crop_all_images("a.png")
        self.assertIsInstance(error.exception.__cause__, OSError)

    def test_saved_results_only_retain_output_paths(self):
        result = self.make_cropper().crop_all_images("a.jpg")
        self.assertEqual(result, ("a.jpg", "/crops/a.jpg"))
        fake_cv2.imwrite.assert_called_once_with("/crops/a.jpg", self.crop, [1, 75])

    def test_single_worker_returns_sorted_results_and_creates_output_once(self):
        results, makedirs = self.run_with_images(self.make_cropper(num_workers=1), ["b.png", "a.png"])
        self.assertEqual(results, [("a.png", "/crops/a.png"), ("b.png", "/crops/b.png")])
        makedirs.assert_called_once_with("/crops", exist_ok=True)

    def test_multiple_workers_return_same_sorted_results(self):
        results, _ = self.run_with_images(self.make_cropper(num_workers=2), ["b.png", "a.png"])
        self.assertEqual(results, [("a.png", "/crops/a.png"), ("b.png", "/crops/b.png")])

    def test_in_memory_mode_returns_arrays_without_creating_output(self):
        results, makedirs = self.run_with_images(self.make_cropper(save_crop=False), ["a.png"])
        self.assertEqual(results, [("a.png", self.crop)])
        makedirs.assert_not_called()
        fake_cv2.imwrite.assert_not_called()

    def test_existing_matching_output_images_are_rewritten(self):
        results, _ = self.run_with_images(self.make_cropper(), ["a.png"], ["a.png"])
        self.assertEqual(results, [("a.png", "/crops/a.png")])
        fake_cv2.imwrite.assert_called_once()

    def test_existing_non_image_output_files_are_allowed(self):
        results, _ = self.run_with_images(self.make_cropper(), ["a.png"], ["notes.txt"])
        self.assertEqual(results, [("a.png", "/crops/a.png")])

    def test_stale_output_images_fail_before_any_write(self):
        with patch.object(Path, "is_dir", return_value=True), \
                patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "iterdir", side_effect=[
                    iter([Path("/frames/a.png")]),
                    iter([Path("/crops/a.png"), Path("/crops/old.JPG")]),
                ]), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(cropper_module.os, "makedirs") as makedirs:
            with self.assertRaisesRegex(ValueError, "old.JPG.*Use a new output directory"):
                self.make_cropper().run()
        makedirs.assert_not_called()
        fake_cv2.imread.assert_not_called()
        fake_cv2.imwrite.assert_not_called()

    def test_missing_input_fails_before_creating_output(self):
        with patch.object(Path, "is_dir", return_value=False), \
                patch.object(cropper_module.os, "makedirs") as makedirs:
            with self.assertRaises(FileNotFoundError):
                self.make_cropper().run()
            makedirs.assert_not_called()

    def test_empty_input_is_reported_before_creating_output(self):
        with patch.object(Path, "is_dir", return_value=True), \
                patch.object(Path, "iterdir", return_value=iter([])), \
                patch.object(cropper_module.os, "makedirs") as makedirs:
            with self.assertRaisesRegex(ValueError, "No supported images"):
                self.make_cropper().run()
            makedirs.assert_not_called()

    def test_overlapping_directories_are_rejected(self):
        for destination in ("/frames", "/frames/crops", "/"):
            with self.subTest(destination=destination), \
                    patch.object(Path, "is_dir", return_value=True), \
                    self.assertRaisesRegex(ValueError, "overlap"):
                cropper_module.Cropper("/frames", destination).run()


@patch.dict(sys.modules, {"core.cropper": cropper_module})
class BatchCropperTests(unittest.TestCase):
    def test_existing_crop_folders_are_skipped_and_source_is_preserved(self):
        with patch.object(Path, "is_dir", return_value=True), \
                patch.object(Path, "iterdir", return_value=iter([Path("/root/video"), Path("/root/video_crop")])), \
                patch.object(cropper_module, "Cropper") as cropper:
            cropper.return_value.run.return_value = [("a.png", "/root/video_crop/a.png")]
            result = batch_module.run_crop_for_all_folders("/root", ymin=0, ymax=10, xmin=0, xmax=10)
        self.assertEqual(list(result), ["video"])
        self.assertEqual(cropper.call_count, 1)
        self.assertEqual(cropper.call_args.kwargs["input_dir"], "/root/video")
        self.assertEqual(cropper.call_args.kwargs["crop_dir"], "/root/video_crop")

    def test_explicit_output_root_is_not_processed_as_input(self):
        with patch.object(Path, "is_dir", return_value=True), \
                patch.object(Path, "iterdir", return_value=iter([Path("/root/video"), Path("/root/results")])), \
                patch.object(cropper_module, "Cropper") as cropper:
            batch_module.run_crop_for_all_folders(
                "/root", output_root="/root/results", ymin=0, ymax=10, xmin=0, xmax=10,
            )
        self.assertEqual(cropper.call_count, 1)
        self.assertEqual(cropper.call_args.kwargs["crop_dir"], "/root/results/video_crop")

    def test_nested_output_is_rejected_before_any_processing(self):
        with patch.object(Path, "is_dir", return_value=True), \
                patch.object(Path, "iterdir", return_value=iter([Path("/root/video")])), \
                patch.object(cropper_module, "Cropper") as cropper:
            with self.assertRaisesRegex(ValueError, "overlap"):
                batch_module.run_crop_for_all_folders(
                    "/root", output_root="/root/video/results", ymin=0, ymax=10, xmin=0, xmax=10,
                )
            cropper.assert_not_called()

    def test_cli_passes_roi_and_worker_count(self):
        with patch.object(batch_module, "run_crop_for_all_folders") as run:
            batch_module.main(["/root", "--output-root", "/results", "--roi", "1", "2", "3", "4", "--num-workers", "3"])
        run.assert_called_once_with("/root", output_root="/results", ymin=2, ymax=4, xmin=1, xmax=3, num_workers=3)

    def test_cli_requires_explicit_roi(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            batch_module.main(["/root"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
