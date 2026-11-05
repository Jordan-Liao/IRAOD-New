"""vis_sem.select: each mode zooms on its own kind of target and modes don't repeat images."""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'lods'))
from vis_sem import select  # noqa: E402


def box(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]


class SelectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.img, self.ann = os.path.join(self.tmp.name, 'img'), os.path.join(self.tmp.name, 'ann')
        os.makedirs(self.img), os.makedirs(self.ann)
        scenes = {
            # many small ships packed in one corner plus one big ship far away
            'a': [box(50 + 12 * i, 50, 8, 8) for i in range(6)] + [box(600, 600, 100, 60)],
            # one long thin ship
            'b': [box(300, 300, 200, 20), box(100, 100, 40, 40)],
            # dense medium ships, none small
            'c': [box(400 + 30 * (i % 4), 400 + 30 * (i // 4), 25, 25) for i in range(12)],
        }
        for name, boxes in scenes.items():
            open(os.path.join(self.img, name + '.jpg'), 'w').close()
            with open(os.path.join(self.ann, name + '.txt'), 'w') as f:
                for b in boxes:
                    f.write(' '.join(f'{v:.1f}' for v in b) + ' ship 0\n')

    def tearDown(self):
        self.tmp.cleanup()

    def pick(self, mode, exclude=()):
        return select(self.img, self.ann, 'rsar', {'ship'}, mode, 1, 0, exclude=set(exclude))[0]

    def test_small_focuses_on_small_cluster(self):
        _, f, objs, focus = self.pick('small')
        self.assertEqual(f, 'a.jpg')
        self.assertEqual(len(focus), 6)  # the big ship is not in the zoom set
        self.assertLess(len(focus), len(objs))

    def test_slender_focuses_on_one_elongated_target(self):
        _, f, _, focus = self.pick('slender')
        self.assertEqual(f, 'b.jpg')
        self.assertEqual(len(focus), 1)
        p = focus[0][1]
        self.assertGreater(np.ptp(p[:, 0]) / np.ptp(p[:, 1]), 3)

    def test_dense_skips_images_used_by_other_modes(self):
        self.assertEqual(self.pick('dense')[1], 'c.jpg')
        self.assertNotEqual(self.pick('dense', exclude=['c.jpg'])[1], 'c.jpg')


if __name__ == '__main__':
    unittest.main()
