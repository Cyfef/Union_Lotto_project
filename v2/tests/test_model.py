import unittest
from datetime import date, timedelta

from union_lotto_v2.data import Draw
from union_lotto_v2.model import predict, probabilities_at, train_and_backtest


def synthetic_draws(count: int = 520) -> list[Draw]:
    draws = []
    current = date(2020, 1, 2)
    serials = {}
    while len(draws) < count:
        if current.weekday() in (1, 3, 6):
            serials[current.year] = serials.get(current.year, 0) + 1
            index = len(draws)
            reds = tuple(sorted({((index * 7 + step * 5) % 33) + 1 for step in range(6)}))
            draws.append(Draw(f"{current.year}{serials[current.year]:03d}", current, reds, index % 16 + 1, "test"))
        current += timedelta(days=1)
    return draws


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draws = synthetic_draws()
        cls.model = train_and_backtest(cls.draws)

    def test_model_schema_and_probability_mass(self):
        self.assertEqual(self.model["schema_version"], 2)
        red = probabilities_at(self.draws, "red", self.model["red"]["weights"])
        blue = probabilities_at(self.draws, "blue", self.model["blue"]["weights"])
        self.assertAlmostEqual(sum(red), 6.0, places=5)
        self.assertAlmostEqual(sum(blue), 1.0, places=5)

    def test_prediction_is_reproducible_and_has_statistics(self):
        first = predict(self.draws, self.model, tickets=3, seed=9, candidate_pool=700)
        second = predict(self.draws, self.model, tickets=3, seed=9, candidate_pool=700)
        self.assertEqual(first["tickets"], second["tickets"])
        self.assertEqual(len(first["tickets"]), 3)
        self.assertIn("span", first["tickets"][0]["statistics"])

    def test_stale_data_is_rejected(self):
        with self.assertRaises(ValueError):
            predict(self.draws[:-1], self.model)


if __name__ == "__main__":
    unittest.main()

