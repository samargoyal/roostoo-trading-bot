import itertools
import random
import unittest

from bot.optimize import (covariance, erc_weights, erc_weights_fixed, min_variance,
                          project_capped_simplex, risk_contributions)


def random_cov(n, seed):
    rng = random.Random(seed)
    a = [[rng.gauss(0, 0.02) for _ in range(n)] for _ in range(n)]
    cov = [[sum(a[i][k] * a[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    for i in range(n):
        cov[i][i] += rng.uniform(1e-4, 5e-4)
    return cov


def variance(cov, w):
    return sum(w[i] * cov[i][j] * w[j] for i in range(len(w)) for j in range(len(w)))


class OptimizeTest(unittest.TestCase):
    def test_covariance_matches_hand_calculation_and_shrinks_off_diagonal(self):
        x, y = [1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]
        cov = covariance([x, y], shrink=0.0)
        self.assertAlmostEqual(cov[0][0], 5.0 / 3.0)
        self.assertAlmostEqual(cov[0][1], 1.0)
        self.assertAlmostEqual(covariance([x, y], shrink=0.5)[0][1], 0.5)

    def test_erc_gives_equal_risk_contributions(self):
        for seed in range(5):
            cov = random_cov(5, seed)
            w = erc_weights(cov)
            self.assertAlmostEqual(sum(w), 1.0)
            for rc in risk_contributions(cov, w):
                self.assertAlmostEqual(rc, 0.2, places=6)

    def test_capped_simplex_projection(self):
        w = project_capped_simplex([0.9, 0.5, -0.2, 0.1], cap=0.5)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertTrue(all(-1e-12 <= x <= 0.5 + 1e-12 for x in w))

    def test_min_variance_beats_every_point_on_a_fine_grid(self):
        cov = random_cov(3, 7)
        w = min_variance(cov, cap=0.6)
        self.assertAlmostEqual(sum(w), 1.0)
        best = variance(cov, w)
        steps = [i / 50 for i in range(51)]
        for a, b in itertools.product(steps, steps):
            c = 1.0 - a - b
            if 0 <= c <= 0.6 and a <= 0.6 and b <= 0.6:
                self.assertLessEqual(best, variance(cov, [a, b, c]) + 1e-12)

    def test_erc_fixed_without_fixed_weights_is_plain_erc_scaled_to_the_budget(self):
        cov = random_cov(4, 3)
        for a, b in zip(erc_weights_fixed(cov, {}, 0.6), erc_weights(cov)):
            self.assertAlmostEqual(a, 0.6 * b, places=9)

    def test_erc_fixed_gives_the_free_assets_equal_risk_and_the_budget(self):
        for seed in range(5):
            cov = random_cov(5, seed)
            w = erc_weights_fixed(cov, {4: 0.2}, 0.5)
            self.assertEqual(w[4], 0.2)
            self.assertAlmostEqual(sum(w[:4]), 0.5)
            cw = [sum(cov[i][k] * w[k] for k in range(5)) for i in range(5)]
            contributions = [w[i] * cw[i] for i in range(4)]
            for rc in contributions:
                self.assertAlmostEqual(rc / contributions[0], 1.0, places=6)

    def test_erc_fixed_gives_less_to_the_asset_that_moves_with_the_fixed_one(self):
        # Assets 0 and 1 alike, except that 0 moves with asset 2 (held fixed) and 1 does not.
        cov = [[1e-4, 0.0, 0.9e-4], [0.0, 1e-4, 0.0], [0.9e-4, 0.0, 1e-4]]
        w = erc_weights_fixed(cov, {2: 0.3}, 0.4)
        self.assertLess(w[0], w[1])
        self.assertAlmostEqual(w[0] + w[1], 0.4)
        plain = erc_weights([row[:2] for row in cov[:2]])
        self.assertAlmostEqual(plain[0], plain[1])

    def test_min_variance_respects_the_cap(self):
        cov = [[0.01, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # asset 0 far calmer
        w = min_variance(cov, cap=0.4)
        self.assertAlmostEqual(w[0], 0.4, places=6)
        self.assertAlmostEqual(w[1], 0.3, places=6)


if __name__ == "__main__":
    unittest.main()


class Round83OptimisersTest(unittest.TestCase):
    """Mean-variance with momentum alphas and hierarchical risk parity (round 83)."""

    def test_mean_variance_follows_alpha_and_risk(self):
        from bot.optimize import mean_variance
        cov = [[1.0, 0.0], [0.0, 1.0]]
        w = mean_variance([1.0, -1.0], cov, risk_aversion=1.0)
        self.assertGreater(w[0], 0.9)                              # all on the stronger coin
        w = mean_variance([0.0, 0.0], [[1.0, 0.0], [0.0, 4.0]], risk_aversion=10.0)
        self.assertAlmostEqual(w[0], 0.8, places=3)                # no alpha: minimum variance
        w = mean_variance([1.0, -1.0], cov, risk_aversion=1.0, cap=0.6)
        self.assertAlmostEqual(w[0], 0.6, places=6)                # capped

    def test_hrp_is_inverse_variance_for_unrelated_assets(self):
        from bot.optimize import hrp_weights
        w = hrp_weights([[1.0, 0.0], [0.0, 4.0]])
        self.assertAlmostEqual(w[0], 0.8)
        self.assertAlmostEqual(sum(w), 1.0)

    def test_hrp_treats_twins_as_one_cluster(self):
        from bot.optimize import hrp_weights
        cov = [[1.0, 0.99, 0.0], [0.99, 1.0, 0.0], [0.0, 0.0, 1.0]]   # two near-identical coins and one apart
        w = hrp_weights(cov)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertGreater(w[2], w[0] + 0.1)                       # the loner gets more than each twin
