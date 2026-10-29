import csv
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from counting_metrics import PointMetrics, append_history, match_counts


class MetricsTest(unittest.TestCase):
    def test_duplicate_detections_are_false_positives(self):
        self.assertEqual(match_counts([[0, 0], [0, 0]], [[0, 0]], 4), (1, 1, 0))

    def test_empty_cases_and_boundary(self):
        self.assertEqual(match_counts([], [], 4), (0, 0, 0))
        self.assertEqual(match_counts([], [[0, 0]], 4), (0, 0, 1))
        self.assertEqual(match_counts([[0, 0]], [], 4), (0, 1, 0))
        self.assertEqual(match_counts([[0, 4]], [[0, 0]], 4), (1, 0, 0))

    def test_matching_maximizes_valid_matches(self):
        # Exhaustive tiny bipartite matchings, including nontrivial conflicts.
        rng = np.random.default_rng(17)
        for _ in range(80):
            pred = rng.uniform(0, 10, (4, 2))
            gt = rng.uniform(0, 10, (3, 2))
            valid = np.linalg.norm(pred[:, None] - gt[None], axis=-1) <= 4
            best = max(sum(valid[p, g] for g, p in enumerate(perm))
                       for perm in itertools.permutations(range(4), 3))
            self.assertEqual(match_counts(pred, gt, 4)[0], best)

    def test_count_and_localization_can_disagree(self):
        meter = PointMetrics([4])
        meter.update('wrong-location', [[100, 100]], [[0, 0]])
        result = meter.summary()
        self.assertEqual(result['mae'], 0)
        self.assertEqual(result['f1_4px'], 0)

    def test_micro_aggregation_and_export(self):
        meter = PointMetrics([4])
        meter.update('a', [[0, 0], [0, 0]], [[0, 0]])
        meter.update('b', [], [[20, 20]])
        result = meter.summary()
        self.assertEqual(result['mae'], 1)
        self.assertEqual(result['rmse'], 1)
        self.assertEqual(result['precision_4px'], .5)
        self.assertEqual(result['recall_4px'], .5)
        self.assertEqual(result['f1_4px'], .5)
        with tempfile.TemporaryDirectory() as td:
            meter.export(td)
            saved = json.loads((Path(td) / 'metrics.json').read_text())
            self.assertEqual(saved['metrics'], result)
            with (Path(td) / 'per_image.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            history = Path(td) / 'history.csv'
            append_history(history, dict(epoch=1, **result))
            append_history(history, dict(epoch=2, **result))
            with history.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            PointMetrics([0])
        with self.assertRaises(ValueError):
            PointMetrics([4]).summary()
        with self.assertRaises(ValueError):
            match_counts([[float('nan'), 0]], [[0, 0]], 4)


if __name__ == '__main__':
    unittest.main()
