from __future__ import annotations

import unittest

import numpy as np

from tools.analysis.cam_overlay import eigen_cam, overlay_cam

try:
    import cv2  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class EigenCamTests(unittest.TestCase):
    def test_peak_follows_spatial_energy(self):
        feat = np.zeros((6, 8, 8), dtype=np.float32)
        feat[:, 2, 5] = np.arange(1, 7, dtype=np.float32)
        cam = eigen_cam(feat)
        self.assertEqual(cam.shape, (8, 8))
        self.assertGreaterEqual(cam.min(), 0.0)
        self.assertLessEqual(cam.max(), 1.0 + 1e-6)
        peak = np.unravel_index(int(cam.argmax()), cam.shape)
        self.assertEqual(peak, (2, 5))

    def test_torch_peak_matches_numpy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from tools.analysis.cam_overlay import eigen_cam_torch

        feat = np.zeros((6, 8, 8), dtype=np.float32)
        feat[:, 2, 5] = np.arange(1, 7, dtype=np.float32)
        cam_np = eigen_cam(feat)
        cam_t = eigen_cam_torch(torch.from_numpy(feat))
        self.assertEqual(tuple(np.unravel_index(int(cam_t.argmax()), cam_t.shape)), (2, 5))
        self.assertGreater(float(np.corrcoef(cam_np.ravel(), cam_t.ravel())[0, 1]), 0.99)

    @unittest.skipUnless(_HAS_CV2, "cv2 not installed")
    def test_overlay_keeps_rgb_shape(self):
        image = np.full((16, 20, 3), 30, dtype=np.uint8)
        cam = np.zeros((4, 5), dtype=np.float32)
        cam[1, 2] = 1.0
        out = overlay_cam(image, cam)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
