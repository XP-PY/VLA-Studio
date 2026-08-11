#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    """Sampler over episode frames that stores only per-episode boundaries.

    Logical positions map to frame indices on the fly (O(num_episodes) construction memory)
    instead of materializing a Python list of every frame index.

    Each epoch is shuffled with a `torch.randperm` seeded from `(seed, epoch)`, so the data order
    is a pure function of `(seed, epoch)`: it reproduces on every rank without synchronizing the
    global RNG (no `generator` to sync across distributed ranks), and `state_dict` /
    `load_state_dict` resume a run sample-exactly by regenerating the epoch's permutation and
    continuing from the saved offset. Each call to `__iter__` advances the epoch. During a
    resumed epoch, `__len__` still reports the full length.

    Epoch advancement: `__iter__` eagerly advances the epoch, and `set_epoch` / `load_state_dict`
    set it explicitly. Within a single run callers should rely on exactly one of these mechanisms,
    not both: advancing the epoch by hand *and* letting `__iter__` auto-advance over the same
    iterations would skip or repeat epochs. The training loop drives it purely through `__iter__`
    (via `cycle`); `set_epoch` / `load_state_dict` are used only to (re)position before iteration
    starts (e.g. on resume or in tests).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Seed the permutation is derived from (together with the epoch).
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # Derive a per-epoch seed from (seed, epoch) so the permutation is a pure function of both
        # and reproduces identically on every rank without touching the global RNG.
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        return int(self._starts[episode]) + position_in_episode

    def __iter__(self) -> Iterator[int]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._frame_index(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._frame_index(k)

    def __len__(self) -> int:
        return self._num_frames


class KeyframeAwareSampler(EpisodeAwareSampler):
    """Mix uniform episode frames with oversampled supervised keyframes.

    Every epoch still contains ``num_frames`` samples. ``keyframe_fraction`` of
    those samples are drawn from the supplied keyframes with replacement; the
    remainder are uniformly shuffled eligible frames without replacement.
    The sequence remains a pure function of ``(seed, epoch)`` and therefore
    retains the resume/distributed behavior of :class:`EpisodeAwareSampler`.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        keyframe_indices: list[int] | np.ndarray,
        *,
        keyframe_fraction: float,
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        seed: int = 0,
    ) -> None:
        if not 0.0 < keyframe_fraction < 1.0:
            raise ValueError("keyframe_fraction must be in (0, 1).")
        super().__init__(
            dataset_from_indices,
            dataset_to_indices,
            episode_indices_to_use=episode_indices_to_use,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=True,
            seed=seed,
        )
        lengths = np.diff(np.concatenate(([0], self._cum_lengths))).astype(np.int64)
        self._uniform_indices = np.concatenate(
            [np.arange(start, start + length, dtype=np.int64) for start, length in zip(self._starts, lengths)]
        )
        candidate_keyframes = np.asarray(keyframe_indices, dtype=np.int64).reshape(-1)
        if candidate_keyframes.size == 0:
            raise ValueError("At least one keyframe index is required for keyframe-aware sampling.")
        positions = np.searchsorted(self._starts, candidate_keyframes, side="right") - 1
        valid_position = positions >= 0
        clipped = positions.clip(min=0)
        valid = valid_position & (
            candidate_keyframes < self._starts[clipped] + lengths[clipped]
        )
        self._keyframe_indices = np.unique(candidate_keyframes[valid])
        if self._keyframe_indices.size == 0:
            raise ValueError("No supplied keyframes remain after episode/frame filtering.")
        self.keyframe_fraction = float(keyframe_fraction)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        generator = self._epoch_generator(epoch)
        keyframe_count = min(
            self._num_frames,
            max(1, round(self._num_frames * self.keyframe_fraction)),
        )
        uniform_count = self._num_frames - keyframe_count
        uniform_positions = torch.randperm(self._num_frames, generator=generator)[:uniform_count]
        uniform_indices = torch.from_numpy(self._uniform_indices)[uniform_positions]
        sampled_keyframe_positions = torch.randint(
            len(self._keyframe_indices),
            (keyframe_count,),
            generator=generator,
        )
        sampled_keyframes = torch.from_numpy(self._keyframe_indices)[sampled_keyframe_positions]
        mixed = torch.cat([uniform_indices, sampled_keyframes])
        mixed = mixed[torch.randperm(self._num_frames, generator=generator)]
        for position in range(start, self._num_frames):
            yield int(mixed[position])


def get_feature_indices(dataset, feature_name: str) -> np.ndarray:
    """Return global row indices where a scalar dataset feature is truthy."""

    if feature_name not in dataset.features:
        raise KeyError(f"Dataset does not contain keyframe feature {feature_name!r}.")
    if feature_name == "planning.box_keyframe":
        sidecar = Path(dataset.root) / "annotations/stage_keyframes/keyframes.parquet"
        if sidecar.is_file():
            import pyarrow.parquet as pq

            indices = np.asarray(
                pq.read_table(sidecar, columns=["dataset_index"])["dataset_index"].to_numpy(),
                dtype=np.int64,
            )
            if indices.size and (indices.min() < 0 or indices.max() >= dataset.num_frames):
                raise ValueError(f"Keyframe sidecar contains indices outside the dataset: {sidecar}.")
            return np.unique(indices)
    values = np.asarray(dataset.hf_dataset[feature_name], dtype=np.uint8).reshape(-1)
    if values.shape[0] != dataset.num_frames:
        raise ValueError(
            f"Feature {feature_name!r} has {values.shape[0]} rows; expected {dataset.num_frames}."
        )
    return np.flatnonzero(values)


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off.
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}
