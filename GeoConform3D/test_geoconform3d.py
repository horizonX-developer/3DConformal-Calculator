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

    def test_additional_point_detailed_substitution(self):
        from computation_steps import computation_steps
        src, tgt, new = scenario_data(1)
        result = estimate(src, tgt)
        working = computation_steps(src, tgt, result, additional_points=[('N01', new[0])])
        self.assertIn('Point N01:', working)
        self.assertIn('R X_new = ', working)
        self.assertIn('X_prime = T + k (R X_new)', working)
        for value in result['transform'](new[0]):
            self.assertIn(f'{value:.10f}', working)

    def test_no_additional_points_explanation(self):
        from computation_steps import computation_steps
        src, tgt, _ = scenario_data(1)
        working = computation_steps(src, tgt, estimate(src, tgt), additional_points=[])
        self.assertIn('No additional points were supplied.', working)

    def test_detailed_working(self):
        from computation_steps import computation_steps
        src, tgt, _ = scenario_data(1)
        working = computation_steps(src, tgt, estimate(src, tgt))
        self.assertIn('STEP 10', working)
        self.assertIn('STEP 5', working)
        self.assertIn('10.000000', working)

if __name__ == '__main__':
    unittest.main()

class LeastSquaresTests(unittest.TestCase):
    def test_exact_lsa(self):
        from least_squares import estimate_lsa
        src, tgt, _ = scenario_data(1)
        r = estimate_lsa(src, tgt)
        self.assertLess(r['rmse'], 1e-8)
        self.assertAlmostEqual(r['translation'][0], 10, places=6)
        self.assertAlmostEqual(r['angles'][0], 1.5, places=5)
        self.assertEqual(len(r['lsa']['N']), 7)
        self.assertEqual(len(r['lsa']['A']), 3*len(src))

    def test_noisy_lsa_matches_horn(self):
        from least_squares import estimate_lsa
        src, tgt, _ = scenario_data(2)
        a, b = estimate_lsa(src, tgt), estimate(src, tgt)
        self.assertLess(abs(a['rmse']-b['rmse']), 1e-8)

    def test_collinear_lsa_rejected(self):
        from least_squares import estimate_lsa
        src, tgt, _ = scenario_data(4)
        with self.assertRaises(ValueError):
            estimate_lsa(src, tgt)
