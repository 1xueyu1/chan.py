from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


class PurgedKFold:
    def __init__(self, n_splits: int, embargo_bars: int = 0):
        self.n_splits = max(2, int(n_splits))
        self.embargo_bars = max(0, int(embargo_bars))

    def split(
        self,
        t0_pos: np.ndarray,
        t1_pos: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = int(len(t0_pos))
        if n_samples == 0:
            return

        order = np.argsort(t0_pos)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = order[start:stop]
            current = stop
            if len(test_idx) == 0:
                continue

            test_t0_min = int(np.min(t0_pos[test_idx]))
            test_t1_max = int(np.max(t1_pos[test_idx]))

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False

            overlap = (t1_pos >= test_t0_min) & (t0_pos <= test_t1_max)
            train_mask[overlap] = False

            if self.embargo_bars > 0:
                emb = (t0_pos > test_t1_max) & (t0_pos <= test_t1_max + self.embargo_bars)
                train_mask[emb] = False

            train_idx = np.where(train_mask)[0]
            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx
