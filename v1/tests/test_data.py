import tempfile
import unittest
from datetime import date
from pathlib import Path

from union_lotto.data import Draw, load_csv, next_draw_date, next_issue, save_csv, validate_draws


def sample_draws(count: int = 320) -> list[Draw]:
    draws = []
    current = date(2023, 1, 1)
    serial = 1
    while len(draws) < count:
        if current.weekday() in (1, 3, 6):
            start = len(draws) % 28 + 1
            reds = tuple(sorted({((start + offset * 5 - 1) % 33) + 1 for offset in range(6)}))
            # The chosen step is coprime to 33, so the set always has six values.
            draws.append(Draw(f"{current.year}{serial:03d}", current, reds, len(draws) % 16 + 1, "test"))
            serial += 1
        current = current.fromordinal(current.toordinal() + 1)
    return draws


class DataTests(unittest.TestCase):
    def test_validation_and_csv_roundtrip(self):
        draws = sample_draws(10)
        validate_draws(draws)
        # Keep the temporary file inside the repository so restricted Windows
        # runners do not redirect it to an inaccessible app-container folder.
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "draws.csv"
            self.assertEqual(save_csv(draws, path), 10)
            self.assertEqual(load_csv(path), draws)

    def test_invalid_duplicate_red_is_rejected(self):
        draw = Draw("2026001", date(2026, 1, 1), (1, 2, 3, 4, 5, 5), 1)
        with self.assertRaises(ValueError):
            draw.validate()

    def test_next_schedule_and_issue(self):
        target = next_draw_date(date(2026, 10, 18))
        self.assertEqual(target, date(2026, 10, 20))
        self.assertEqual(next_issue("2026119", target), "2026120")
        self.assertEqual(next_issue("2025153", date(2026, 1, 1)), "2026001")


if __name__ == "__main__":
    unittest.main()

