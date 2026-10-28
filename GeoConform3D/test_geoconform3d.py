"""Regression tests; run: py -m unittest -v test_geoconform3d.py"""
import math
import unittest
from geoconform3d import estimate, make_target, scenario_data

class GeoConformTests(unittest.TestCase):
    def test_exact_recovery(self):
        src, tgt, _ = scenario_data(1)
        result = estimate(src, tgt)
        for got, want in zip(result['translation'], (10, -5, 8)):
            self.assertAlmostEqual(got, want, places=6)
        for got, want in zip(result['angles'], (1.5, -2, 0.8)):
            self.assertAlmostEqual(got, want, places=5)
        self.assertAlmostEqual((result['scale'] - 1) * 1e6, 2.5, places=5)
        self.assertLess(result['rmse'], 1e-8)

    def test_noisy_data(self):
        src, tgt, _ = scenario_data(2)
        result = estimate(src, tgt)
        self.assertGreater(result['rmse'], 0)
        self.assertLess(result['rmse'], 0.01)
        self.assertAlmostEqual(result['sigma0'] ** 2 * (3 * len(src) - 7),
                               result['rmse'] ** 2 * (3 * len(src)), places=10)

    def test_extra_points(self):
        src, tgt, new = scenario_data(3)
        result = estimate(src, tgt)
        expected = make_target(new)
        for p, q in zip(new, expected):
            for got, want in zip(result['transform'](p), q):
                self.assertAlmostEqual(got, want, places=6)

    def test_collinear_rejected(self):
        src, tgt, _ = scenario_data(4)
        with self.assertRaisesRegex(ValueError, 'collinear'):
            estimate(src, tgt)

    def test_bad_inputs(self):
        src, tgt, _ = scenario_data(1)
        with self.assertRaises(ValueError):
            estimate(src[:2], tgt[:2])
        with self.assertRaises(ValueError):
            estimate(src, tgt[:-1])
        with self.assertRaises(ValueError):
            estimate(src, [(float('nan'), 0, 0)] + tgt[1:])
        with self.assertRaises(ValueError):
            estimate(src, tgt)['transform']((float('inf'), 0, 0))

    def test_arbitrary_rotation_and_scale(self):
        src, _, _ = scenario_data(1)
        tgt = make_target(src, (25, -12, 3), (12, -7, 4), -3)
        result = estimate(src, tgt)
        for got, want in zip(result['translation'], (25, -12, 3)):
            self.assertAlmostEqual(got, want, places=6)
        for got, want in zip(result['angles'], (12, -7, 4)):
            self.assertAlmostEqual(got, want, places=5)
        self.assertAlmostEqual((result['scale'] - 1) * 1e6, -3, places=5)

if __name__ == '__main__':
    unittest.main()
