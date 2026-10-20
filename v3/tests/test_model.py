import math
import random
import unittest
from datetime import date, timedelta

from union_lotto_v3.data import Draw
from union_lotto_v3.model import (
    DistributionConfig,
    posterior_alpha,
    posterior_mean,
    predict,
    sample_dirichlet,
    train_and_backtest,
    unordered_pl_probability,
    weighted_without_replacement,
)


def synthetic_draws(count: int = 510) -> list[Draw]:
    draws = []
    current = date(2020, 1, 2)
    serials: dict[int, int] = {}
    while len(draws) < count:
        if current.weekday() in (1, 3, 6):
            serials[current.year] = serials.get(current.year, 0) + 1
            index = len(draws)
            reds = tuple(sorted((((index * 7 + step * 5) % 33) + 1) for step in range(6)))
            draws.append(Draw(f"{current.year}{serials[current.year]:03d}", current, reds, index % 16 + 1, "test"))
        current += timedelta(days=1)
    return draws


class ProbabilityTests(unittest.TestCase):
    def test_uniform_pl_assigns_equal_probability_to_each_set(self):
        probability = unordered_pl_probability([1.0] * 5, (1, 3))
        self.assertAlmostEqual(probability, 1 / math.comb(5, 2), places=12)

    def test_dirichlet_and_sampling_probability_mass(self):
        alpha = [2.0, 3.0, 5.0]
        self.assertAlmostEqual(sum(posterior_mean(alpha)), 1.0, places=12)
        sampled = sample_dirichlet(alpha, random.Random(7))
        self.assertAlmostEqual(sum(sampled), 1.0, places=12)
        self.assertEqual(len(weighted_without_replacement(sampled, 2, random.Random(8))), 2)

    def test_posterior_alpha_is_positive(self):
        alpha = posterior_alpha(synthetic_draws(), "red", DistributionConfig(96, 10.0))
        self.assertEqual(len(alpha), 33)
        self.assertTrue(all(value > 0 for value in alpha))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draws = synthetic_draws()
        cls.model = train_and_backtest(cls.draws)

    def test_model_schema_and_holdout(self):
        self.assertEqual(self.model["schema_version"], 3)
        self.assertGreater(self.model["red"]["test"]["draws"], 0)
        self.assertEqual(len(self.model["red"]["posterior_alpha"]), 33)
        self.assertEqual(len(self.model["blue"]["posterior_alpha"]), 16)

    def test_prediction_is_reproducible(self):
        first = predict(self.draws, self.model, tickets=3, seed=9, simulations=300)
        second = predict(self.draws, self.model, tickets=3, seed=9, simulations=300)
        self.assertEqual(first, second)
        self.assertEqual(len(first["tickets"]), 3)
        self.assertAlmostEqual(
            sum(row["posterior_predictive_inclusion_probability"] for row in first["red_distribution"]),
            6.0,
            places=10,
        )

    def test_stale_data_is_rejected(self):
        with self.assertRaises(ValueError):
            predict(self.draws[:-1], self.model, simulations=50)


if __name__ == "__main__":
    unittest.main()
