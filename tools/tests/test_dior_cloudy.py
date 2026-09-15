"""Author cloudy equations, MATLAB pixel conventions and TRAIN-only resume."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.dataset.generate_dior_cloudy import (
    CLOUD_ARCHIVE_SHA256, cloud_intensity, generate, matlab_bicubic_uint8,
    resize_weights, synthesize, uint8_round,
)


class CloudyAlgorithmTest(unittest.TestCase):
    def test_uint8_is_saturating_half_up_not_numpy_bankers_rounding(self):
        np.testing.assert_array_equal(uint8_round(np.array([-.5, .5, 2.5, 254.5, 999])),
                                      [0, 1, 3, 255, 255])

    def test_identity_and_constant_resize_preserve_pixels(self):
        rgb = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        np.testing.assert_array_equal(matlab_bicubic_uint8(rgb, (5, 7)), rgb)
        for shape in ((2, 3), (11, 17), (1, 1)):
            actual = matlab_bicubic_uint8(np.full((5, 7, 3), 117, np.uint8), shape)
            np.testing.assert_array_equal(actual, np.full((*shape, 3), 117))

    def test_matlab_pixel_centers_and_symmetric_cubic_stencil(self):
        indices, weights = resize_weights(4, 8)
        # First output is centered at -1/4, mirrored across the -1/2 boundary.
        signal = np.array([0., 64., 128., 192.])
        actual = np.sum(signal[indices] * weights, axis=1)
        np.testing.assert_allclose(actual, [-6, 11.5, 46.5, 80, 112, 145.5, 180.5, 198])
        _, down = resize_weights(16, 4)
        self.assertGreater(np.count_nonzero(down[1]), 4)
        np.testing.assert_allclose(down.sum(axis=1), 1)

    def test_uint8_rounding_between_axes_is_not_float_image_resize(self):
        image = np.repeat(np.array([[0, 1], [2, 3]], np.uint8)[:, :, None], 3, axis=2)
        expected = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 3]], np.uint8)
        np.testing.assert_array_equal(matlab_bicubic_uint8(image, (3, 3))[:, :, 0], expected)

    def test_author_constant_texture_is_rejected_not_substituted(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            cloud_intensity(np.full((2, 2, 3), 17, np.uint8), (2, 2))

    def test_channel_and_image_state_match_literal_author_scalar_equations(self):
        original = np.array([[[40, 170, 240], [250, 8, 15]],
                             [[220, 15, 190], [3, 105, 200]]], dtype=np.uint8)
        cloud = np.array([[[10, 200, 40], [100, 30, 140]],
                          [[240, 60, 250], [90, 220, 60]]], dtype=np.uint8)
        ci = cloud_intensity(cloud, (2, 2))

        def reference(image, atmosphere):
            result = np.empty_like(image)
            for channel in range(3):
                light, mixed = {}, {}
                for column in range(2):
                    for row in range(2):
                        q = (row, column)
                        light[q] = (255 - ci[row, column, channel]) / 255 * image[row, column, channel]
                        mixed[q] = light[q] + atmosphere * ci[row, column, channel]
                position = max(mixed, key=mixed.get)
                denominator = ci[position[0], position[1], channel]
                if denominator:
                    atmosphere = min(atmosphere, (255 - light[position]) / denominator)
                for row, column in mixed:
                    result[row, column, channel] = min(
                        255, max(0, int(np.floor(light[row, column] + atmosphere * ci[row, column, channel] + .5))))
            return result, atmosphere

        a = .95
        for image in (original, 255 - original):
            expected, expected_a = reference(image, a)
            actual, actual_a = synthesize(image, ci, a)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual_a, expected_a)
            self.assertLessEqual(actual_a, a)
            a = actual_a
        self.assertLess(a, .95)


class CloudyTrainTest(unittest.TestCase):
    def test_only_train_ids_lexicographic_cycle_and_stateful_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, clouds, out = (root / p for p in ("DIOR", "clouds", "out"))
            (data / "ImageSets").mkdir(parents=True)
            (data / "ImageSets/train.txt").write_text("002\n001\n")
            annotations = data / "Annotations/Oriented Bounding Boxes"
            annotations.mkdir(parents=True)
            (data / "JPEGImages").mkdir()
            pixels = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3) * 2
            for name in ("001", "002", "TEST-only"):
                Image.fromarray(pixels).save(data / "JPEGImages" / f"{name}.png")
                (annotations / f"{name}.xml").write_text("<annotation/>")
            (clouds / "png").mkdir(parents=True)
            names = sorted(f"{i}.png" for i in range(1, 31))
            for name in names:
                Image.fromarray(pixels).save(clouds / "png" / name)
            (clouds / "NOTICE.md").write_text("fixture attribution")
            (clouds / "order.json").write_text(json.dumps({
                "zip_sha256": CLOUD_ARCHIVE_SHA256, "pinned_matlab_compatible_order": names}))
            with patch("tools.dataset.generate_dior_cloudy.TRAIN_COUNT", 2):
                result = generate(data, clouds, out)
                self.assertEqual([r["image_id"] for r in result["records"]], ["001", "002"])
                self.assertEqual([r["cloud"] for r in result["records"]], ["1.png", "10.png"])
                self.assertEqual({p.stem for p in out.glob("*.png")}, {"001", "002"})
                resumed = generate(data, clouds, out, resume=True)
                self.assertEqual(resumed["records"], result["records"])
                Image.fromarray(np.zeros_like(pixels)).save(out / "001.png")
                with self.assertRaisesRegex(ValueError, "differs from author-state replay"):
                    generate(data, clouds, out, resume=True)


if __name__ == "__main__":
    unittest.main()
