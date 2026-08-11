#!/usr/bin/env python

from __future__ import annotations

from lerobot.datasets.sampler import KeyframeAwareSampler


def test_keyframe_sampler_is_deterministic_and_oversamples_keyframes():
    kwargs = {
        "dataset_from_indices": [0, 5],
        "dataset_to_indices": [5, 10],
        "keyframe_indices": [2, 7],
        "keyframe_fraction": 0.4,
        "seed": 17,
    }
    first = list(KeyframeAwareSampler(**kwargs))
    second = list(KeyframeAwareSampler(**kwargs))

    assert first == second
    assert len(first) == 10
    assert sum(index in {2, 7} for index in first) >= 4


def test_keyframe_sampler_resume_offset_matches_full_epoch_suffix():
    kwargs = {
        "dataset_from_indices": [0],
        "dataset_to_indices": [8],
        "keyframe_indices": [1, 4],
        "keyframe_fraction": 0.25,
        "seed": 3,
    }
    full = list(KeyframeAwareSampler(**kwargs))
    resumed = KeyframeAwareSampler(**kwargs)
    resumed.load_state_dict({"epoch": 0, "start_index": 3})

    assert list(resumed) == full[3:]
