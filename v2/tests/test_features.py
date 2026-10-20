import unittest

from union_lotto_v2.features import shape_features


class ShapeFeatureTests(unittest.TestCase):
    def test_shape_statistics(self):
        result = shape_features((1, 2, 11, 17, 22, 33), {2, 8, 17, 20, 25, 30})
        self.assertEqual(result.total, 86)
        self.assertEqual(result.span, 32)
        self.assertEqual(result.odd_count, 4)
        self.assertEqual(result.big_count, 3)
        self.assertEqual((result.zone1, result.zone2, result.zone3), (3, 2, 1))
        self.assertEqual(result.consecutive_pairs, 1)
        self.assertEqual(result.repeated_from_previous, 2)
        self.assertEqual(result.duplicate_tails, 2)

    def test_invalid_combination(self):
        with self.assertRaises(ValueError):
            shape_features((1, 1, 2, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()

