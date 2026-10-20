import unittest

from union_lotto.model import Params, predict, probabilities, train_and_backtest
from test_data import sample_draws


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draws = sample_draws()

    def test_probabilities_sum_to_pick_count(self):
        red = probabilities(self.draws, "red", Params(24, 0.5))
        blue = probabilities(self.draws, "blue", Params(24, 0.5))
        self.assertAlmostEqual(sum(red), 6.0, places=7)
        self.assertAlmostEqual(sum(blue), 1.0, places=7)
        self.assertEqual(len(red), 33)
        self.assertEqual(len(blue), 16)

    def test_train_is_leakage_safe_and_predict_is_reproducible(self):
        model = train_and_backtest(self.draws)
        self.assertLess(int(model["data"]["test_start_issue"]), int(model["data"]["latest_issue"]))
        first = predict(self.draws, model, tickets=3, seed=7)
        second = predict(self.draws, model, tickets=3, seed=7)
        self.assertEqual(first["tickets"], second["tickets"])
        for ticket in first["tickets"]:
            self.assertEqual(len(set(ticket["reds"])), 6)
            self.assertTrue(all(1 <= n <= 33 for n in ticket["reds"]))
            self.assertTrue(1 <= ticket["blue"] <= 16)

    def test_stale_model_is_rejected(self):
        model = train_and_backtest(self.draws)
        with self.assertRaises(ValueError):
            predict(self.draws[:-1], model)


if __name__ == "__main__":
    unittest.main()

