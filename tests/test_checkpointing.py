import copy
import csv
import logging
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from yacs.config import CfgNode as CN

from config import _C
from checkpointing import (atomic_save, find_latest, load_training_state,
                           resume_metric_history, save_training_state,
                           should_validate, trusted_load)


LOGGER = logging.getLogger('checkpoint_test')
LOGGER.addHandler(logging.NullHandler())


def config(root):
    cfg = _C.clone()
    cfg.OUTPUT = str(root)
    cfg.TRAIN.EPOCHS = 4
    cfg.freeze()
    return cfg


def seed():
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)


def build():
    student = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.3),
                                  torch.nn.Linear(5, 1))
    teacher = copy.deepcopy(student)
    optimizer = torch.optim.Adam(student.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
    return [teacher, student], optimizer, scheduler


def epoch(models, optimizer, scheduler):
    teacher, student = models
    for _ in range(3):
        inputs = torch.randn(4, 3) + random.random() + float(np.random.rand())
        loss = student(inputs).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for t, s in zip(teacher.parameters(), student.parameters()):
                t.mul_(.998).add_(s, alpha=.002)
    # Current trainer intentionally uses constant LR; also verify scheduler
    # roundtrip with a nontrivial scheduler state in this synthetic test.
    scheduler.step()


class CheckpointTest(unittest.TestCase):
    def test_full_resume_matches_continuous_student_teacher_and_rng(self):
        seed()
        expected, eo, es = build()
        for _ in range(4):
            epoch(expected, eo, es)
        expected_rng = (random.random(), np.random.rand(), torch.rand(2))
        seed()
        models, optimizer, scheduler = build()
        for _ in range(2):
            epoch(models, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as td:
            cfg = config(td)
            save_training_state(cfg, 'latest', models, [3., 4., .2], LOGGER,
                                optimizer, scheduler, 1, best_epoch=1)
            restored, ro, rs = build()
            cfg.defrost()
            cfg.MODEL.RESUME = find_latest(td)
            cfg.freeze()
            self.assertEqual(load_training_state(cfg, restored, ro, rs, LOGGER), [3., 4., .2])
            self.assertEqual(cfg.TRAIN.START_EPOCH, 2)
            self.assertEqual(cfg.TRAIN.BEST_EPOCH, 1)
            self.assertEqual(scheduler.state_dict(), rs.state_dict())
            for _ in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
                epoch(restored, ro, rs)
            for e, r in zip(expected, restored):
                for key, value in e.state_dict().items():
                    torch.testing.assert_close(value, r.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(expected_rng[0], random.random())
            self.assertEqual(expected_rng[1], np.random.rand())
            torch.testing.assert_close(expected_rng[2], torch.rand(2), rtol=0, atol=0)

    def test_failed_atomic_write_keeps_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'ckpt_epoch_latest.pth'
            atomic_save({'old': 1}, path)
            before = path.read_bytes()
            def fail(state, stream):
                stream.write(b'partial')
                raise OSError('disk full')
            with patch('checkpointing.torch.save', side_effect=fail):
                with self.assertRaises(OSError):
                    atomic_save({'new': 2}, path)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(list(Path(td).glob('*.tmp')), [])

    def test_auto_resume_prefers_latest_over_newer_best_and_ignores_temps(self):
        with tempfile.TemporaryDirectory() as td:
            latest = Path(td) / 'ckpt_epoch_latest.pth'
            latest.touch()
            (Path(td) / 'ckpt_epoch_best.pth').touch()
            (Path(td) / 'ckpt_epoch_latest.pth.incomplete.tmp').touch()
            self.assertEqual(find_latest(td), str(latest))

    def test_legacy_weights_allow_eval_but_reject_false_resume(self):
        with tempfile.TemporaryDirectory() as td:
            models, opt, sched = build()
            path = Path(td) / 'old.pth'
            atomic_save(dict(teacher=models[0].state_dict(), student=models[1].state_dict()), path)
            cfg = config(td)
            cfg.defrost(); cfg.MODEL.RESUME = str(path); cfg.freeze()
            with self.assertRaisesRegex(ValueError, 'Old weights-only'):
                load_training_state(cfg, models, opt, sched, LOGGER)
            cfg.defrost(); cfg.EVAL_MODE = True; cfg.freeze()
            load_training_state(cfg, models, opt, sched, LOGGER)
            self.assertEqual(cfg.TRAIN.START_EPOCH, 0)

    def test_changed_total_epochs_rejected_and_batch_size_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = config(td)
            models, opt, sched = build()
            save_training_state(cfg, 'latest', models, [3., 4., .2], LOGGER, opt, sched, 0)
            cfg.defrost(); cfg.MODEL.RESUME = find_latest(td); cfg.TRAIN.EPOCHS = 5; cfg.freeze()
            with self.assertRaisesRegex(ValueError, 'TRAIN.EPOCHS'):
                load_training_state(cfg, models, opt, sched, LOGGER)
            cfg.defrost(); cfg.TRAIN.EPOCHS = 4; cfg.DATA.BATCH_SIZE = 2; cfg.freeze()
            load_training_state(cfg, models, opt, sched, LOGGER)

    def test_history_keeps_completed_rows_and_restores_curves(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'metrics_history.csv'
            path.write_text('epoch,mae,mse,loss\n5,3,4,0.2\n10,2,3,0.1\n')
            curves = resume_metric_history(path, 5)
            self.assertEqual(curves, ([5], [3.], [4.], [.2]))
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertIn('rmse', rows[0])
            self.assertNotIn('mse', rows[0])

    def test_validation_rounds_include_last_without_duplicate(self):
        self.assertEqual([e + 1 for e in range(50) if should_validate(e, 5, 50)],
                         list(range(5, 51, 5)))
        self.assertEqual([e + 1 for e in range(12) if should_validate(e, 5, 12)], [5, 10, 12])
        with self.assertRaises(ValueError):
            should_validate(0, 0, 50)


if __name__ == '__main__':
    unittest.main()
