"""Full teacher/student state saved atomically at completed epoch boundaries."""
import csv
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def atomic_save(state, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp',
                                     dir=str(destination.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def should_validate(epoch, interval, total_epochs):
    if interval < 1:
        raise ValueError('SAVE_FREQ (validation interval) must be positive')
    return (epoch + 1) % interval == 0 or epoch == total_epochs - 1


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from checkpoint; use the original visible devices')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_training_state(config, label, models, best_metrics, logger,
                        optimizer=None, lr_scheduler=None, epoch_index=None,
                        metrics=None, best_epoch=-1):
    if optimizer is None or lr_scheduler is None or epoch_index is None:
        raise ValueError('Saving a training checkpoint requires optimizer, scheduler and completed epoch')
    teacher, student = models
    state = dict(checkpoint_version=2, teacher=teacher.state_dict(),
                 student=student.state_dict(), optimizer=optimizer.state_dict(),
                 lr_scheduler=lr_scheduler.state_dict(), epoch=int(epoch_index),
                 max_accuracy=list(best_metrics), best_epoch=int(best_epoch),
                 rng_state=capture_rng(), config=json.loads(json.dumps(config)),
                 selection_metric='mae', completed_epoch=True)
    if metrics is not None:
        state['metrics'] = metrics
    destination = Path(config.OUTPUT) / f'ckpt_epoch_{label}.pth'
    atomic_save(state, destination)
    logger.info(f'Checkpoint saved: {destination} | completed epoch {epoch_index + 1}')


def trusted_load(path):
    # Python/NumPy RNG state requires pickle: only load trusted checkpoints.
    if str(path).startswith('https://'):
        try:
            return torch.hub.load_state_dict_from_url(path, map_location='cpu',
                                                       check_hash=True, weights_only=False)
        except TypeError:
            return torch.hub.load_state_dict_from_url(path, map_location='cpu', check_hash=True)
    path = Path(path).expanduser()
    if path.is_dir():
        path /= 'ckpt_epoch_latest.pth'
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def nested(mapping, path):
    value = mapping
    for key in path.split('.'):
        if key not in value:
            return None
        value = value[key]
    return value


def load_training_state(config, models, optimizer, lr_scheduler, logger):
    state = trusted_load(config.MODEL.RESUME)
    if not config.EVAL_MODE:
        missing = [key for key in ('teacher', 'student', 'optimizer', 'lr_scheduler', 'epoch')
                   if key not in state]
        if missing:
            raise ValueError(f'Checkpoint lacks training state {missing}. Old weights-only files support '
                             '--eval, but cannot resume training. Start a new --tag for a fresh run.')
        saved = state.get('config', {})
        fixed = ('MODEL.NAME', 'MODEL.FACTOR', 'MODEL.LOSS', 'DATA.DATASET',
                 'DATA.LABEL_PERCENT', 'DATA.LABEL_PROTOCOL', 'TRAIN.EPOCHS',
                 'TRAIN.BASE_LR', 'TRAIN.BACKBONE_LR', 'TRAIN.WEIGHT_DECAY',
                 'TRAIN.LR_SCHEDULER', 'SEED')
        for key in fixed:
            before, after = nested(saved, key), nested(config, key)
            if before is not None and before != after:
                raise ValueError(f'Resume configuration differs at {key}: {before!r} -> {after!r}; '
                                 'keep the original training settings (EPOCHS is the total count)')
        for key in ('DATA.BATCH_SIZE', 'DATA.NUM_WORKERS', 'DATA.DATA_PATH'):
            before, after = nested(saved, key), nested(config, key)
            if before is not None and before != after:
                logger.warning(f'Resume changed {key}: {before!r} -> {after!r}; trajectory may differ')
    teacher, student = models
    teacher.load_state_dict(state['teacher'], strict=True)
    student.load_state_dict(state['student'], strict=True)
    best = state.get('max_accuracy', [1e6] * 3)
    if not config.EVAL_MODE:
        optimizer.load_state_dict(state['optimizer'])
        lr_scheduler.load_state_dict(state['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = int(state['epoch']) + 1
        config.TRAIN.BEST_EPOCH = int(state.get('best_epoch', -1))
        config.freeze()
        if 'rng_state' in state:
            restore_rng(state['rng_state'])
        else:
            logger.warning('Checkpoint has no RNG state; exact continuation is not guaranteed')
        logger.info(f'Resumed full state: completed {config.TRAIN.START_EPOCH} epochs; '
                    f'next epoch {config.TRAIN.START_EPOCH + 1}/{config.TRAIN.EPOCHS}')
    else:
        logger.info('Loaded teacher/student weights for evaluation')
    return best


def find_latest(output_dir):
    directory = Path(output_dir)
    latest = directory / 'ckpt_epoch_latest.pth'
    if latest.is_file():
        return str(latest)
    # Legacy files will be checked explicitly on load rather than silently
    # presented as full resume. Never select incomplete atomic-save temp files.
    candidates = list(directory.glob('*.pth'))
    return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None


def resume_metric_history(path, completed_epochs):
    """Trim evaluation rows newer than the last durable checkpoint; return curves."""
    path = Path(path)
    if not path.is_file():
        return [], [], [], []
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or 'epoch' not in fields:
        raise ValueError(f'Missing epoch column in {path}')
    kept = [row for row in rows if int(row['epoch']) <= completed_epochs]
    migrated = 'mse' in fields
    if migrated:
        fields = [field for field in fields if field != 'mse'] if 'rmse' in fields else [
            'rmse' if field == 'mse' else field for field in fields]
        for row in kept:
            old = row.pop('mse')
            row.setdefault('rmse', old)
    if migrated or len(kept) != len(rows):
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
        try:
            with os.fdopen(fd, 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return ([int(row['epoch']) for row in kept], [float(row['mae']) for row in kept],
            [float(row['rmse']) for row in kept], [float(row['loss']) for row in kept])
