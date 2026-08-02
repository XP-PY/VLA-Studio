#!/usr/bin/env python

from __future__ import annotations

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.action_flow_v2_2.configuration_action_flow_v2_2 import (
    ActionFlowV22Config,
)
from lerobot.policies.action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    BOX_FEATURE_KEYS,
    GROUNDING_TOKEN_LOSS_WEIGHT,
    GROUNDING_TOKEN_MASK,
    GROUNDING_TOKENS,
    TARGET_BOX_BINS,
    TARGET_BOXES,
    TARGET_BOX_LOSS_WEIGHT,
    TARGET_BOX_VISIBILITY,
    TARGET_ENTITY_IDS,
    TARGET_ENTITY_ROLES,
    TARGET_ENTITY_VALID,
    ActionFlowV22DatasetMetadata,
    TargetBoxRecord,
    TargetRole,
    TargetVisibility,
)
from lerobot.policies.action_flow_v2_2.modeling_action_flow_v2_2 import (
    _copy_smolvla_tensor,
    _smolvla_tensor_copy_mode,
)
from lerobot.policies.factory import (
    get_policy_class,
    make_policy_config,
)
from lerobot.utils.constants import ACTION, ACTION_TOKEN_MASK, ACTION_TOKENS


def make_features(
    metadata: ActionFlowV22DatasetMetadata,
    *,
    action_names: list[str] | None = None,
) -> dict:
    if action_names is None:
        action_names = [
            "joint_1.pos",
            "joint_2.pos",
            "joint_3.pos",
            "joint_4.pos",
            "joint_5.pos",
            "joint_6.pos",
            "gripper.pos",
        ]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": action_names,
        },
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            }
            for camera in metadata.camera_names
        },
        **metadata.feature_metadata(),
        ACTION: {
            "dtype": "float32",
            "shape": [7],
            "names": action_names,
        },
    }


def make_policy_features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        TARGET_BOXES: PolicyFeature(FeatureType.STATE, (2, 8, 4)),
        TARGET_BOX_BINS: PolicyFeature(FeatureType.STATE, (2, 8, 4)),
        TARGET_BOX_VISIBILITY: PolicyFeature(FeatureType.STATE, (2, 8)),
        TARGET_BOX_LOSS_WEIGHT: PolicyFeature(FeatureType.STATE, (2, 8)),
        TARGET_ENTITY_IDS: PolicyFeature(FeatureType.STATE, (8,)),
        TARGET_ENTITY_ROLES: PolicyFeature(FeatureType.STATE, (8,)),
        TARGET_ENTITY_VALID: PolicyFeature(FeatureType.STATE, (8,)),
    }
    output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (7,))}
    return input_features, output_features


def test_config_and_policy_registration_are_available():
    config = make_policy_config("action_flow_v2_2", device="cpu")

    assert isinstance(config, ActionFlowV22Config)
    assert get_policy_class("action_flow_v2_2").__name__ == "ActionFlowV22Policy"


def test_token_ranges_are_deterministic_and_non_overlapping():
    config = ActionFlowV22Config(device="cpu")

    assert config.structural_token_id("<grounding>") == 49_280
    assert config.location_token(0) == "<loc_000>"
    assert config.location_token_id(0) == 49_292
    assert config.location_token(255) == "<loc_255>"
    assert config.fast_token(0) == "<fast_0000>"
    assert config.fast_token_id(0) == 49_548
    assert config.fast_token(2_047) == "<fast_2047>"
    assert config.extended_vlm_vocab_size == 51_596
    assert config.token_contract()["location_token_range"] == [49_292, 49_548]
    assert config.token_contract()["fast_token_range"] == [49_548, 51_596]


def test_config_checkpoint_reload_preserves_contracts(tmp_path):
    config = ActionFlowV22Config(
        device="cpu",
        camera_names=("wrist", "top"),
        relative_exclude_joints=["gripper"],
        smolvla_pretrained_path="lerobot/smolvla_base",
    )
    config.save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(loaded, ActionFlowV22Config)
    assert loaded.smolvla_pretrained_path == "lerobot/smolvla_base"
    assert loaded.token_contract() == config.token_contract()
    assert loaded.architecture_contract() == config.architecture_contract()
    assert loaded.model_batch_contract() == config.model_batch_contract()
    assert loaded.dataset_contract(("tomato", "pink_plate")).feature_metadata() == config.dataset_contract(
        ("tomato", "pink_plate")
    ).feature_metadata()


def test_smolvla_vocabulary_initialization_preserves_added_rows():
    source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    target = torch.full((6, 3), -1.0)
    key = "model.vlm_with_expert.vlm.lm_head.weight"

    mode = _smolvla_tensor_copy_mode(
        key,
        tuple(target.shape),
        tuple(source.shape),
        base_vocab_size=4,
    )
    with torch.no_grad():
        _copy_smolvla_tensor(target, source, mode)

    assert mode == "vocabulary_prefix"
    torch.testing.assert_close(target[:4], source)
    torch.testing.assert_close(target[4:], torch.full((2, 3), -1.0))


def test_smolvla_initialization_rejects_non_vocabulary_shape_mismatch():
    assert (
        _smolvla_tensor_copy_mode(
            "model.action_in_proj.weight",
            (8, 4),
            (6, 4),
            base_vocab_size=4,
        )
        is None
    )


def test_sidecar_record_distinguishes_visible_absent_and_invalid_labels():
    visible = TargetBoxRecord(
        episode_index=3,
        frame_index=17,
        camera="wrist",
        entity_id=1,
        entity_name="tomato",
        entity_role=TargetRole.SOURCE,
        visibility=TargetVisibility.VISIBLE,
        box_xyxy=(0.1, 0.2, 0.4, 0.7),
    )
    absent = TargetBoxRecord(
        episode_index=3,
        frame_index=17,
        camera="top",
        entity_id=2,
        entity_name="pink_plate",
        entity_role=TargetRole.DESTINATION,
        visibility=TargetVisibility.NOT_VISIBLE,
        box_xyxy=None,
    )
    invalid = TargetBoxRecord(
        episode_index=3,
        frame_index=18,
        camera="wrist",
        entity_id=1,
        entity_name="tomato",
        entity_role=TargetRole.SOURCE,
        visibility=TargetVisibility.VISIBLE,
        box_xyxy=(0.1, 0.2, 0.4, 0.7),
        loss_weight=0.0,
    )

    assert TargetBoxRecord.from_dict(visible.to_dict()) == visible
    assert TargetBoxRecord.from_dict(absent.to_dict()) == absent
    assert invalid.loss_weight == 0
    with pytest.raises(ValueError, match="must not contain box"):
        TargetBoxRecord(
            episode_index=0,
            frame_index=0,
            camera="top",
            entity_id=0,
            entity_name="tomato",
            entity_role=TargetRole.SOURCE,
            visibility=TargetVisibility.NOT_VISIBLE,
            box_xyxy=(0.1, 0.2, 0.3, 0.4),
        )


def test_dataset_metadata_round_trip_and_feature_validation(tmp_path):
    metadata = ActionFlowV22DatasetMetadata(
        camera_names=("wrist", "top"),
        entity_names=("tomato", "pink_plate"),
        max_entities=8,
    )
    path = tmp_path / "action_flow_v2_2_boxes.json"
    metadata.save(path)
    loaded = ActionFlowV22DatasetMetadata.load(path)

    assert loaded == metadata
    loaded.validate_feature_metadata(make_features(metadata))


def test_global_entity_vocabulary_may_exceed_per_frame_slot_count():
    metadata = ActionFlowV22DatasetMetadata(
        camera_names=("wrist", "top"),
        entity_names=tuple(f"entity_{index}" for index in range(11)),
        max_entities=8,
    )

    assert len(metadata.entity_names) == 11
    assert metadata.feature_metadata()[TARGET_BOX_BINS]["coordinate_bins"] == 256


def test_dataset_metadata_is_validated_and_box_labels_are_not_model_inputs():
    input_features, output_features = make_policy_features()
    config = ActionFlowV22Config(
        device="cpu",
        input_features=input_features,
        output_features=output_features,
    )
    metadata = config.dataset_contract(("tomato", "pink_plate"))

    config.set_dataset_feature_metadata(make_features(metadata))
    config.validate_features()

    assert config.action_feature_names[-1] == "gripper.pos"
    assert not set(BOX_FEATURE_KEYS).intersection(config.input_features)
    assert list(config.image_features) == [
        "observation.images.wrist",
        "observation.images.top",
    ]


def test_dataset_metadata_rejects_wrong_camera_order():
    input_features, output_features = make_policy_features()
    config = ActionFlowV22Config(
        device="cpu",
        input_features=input_features,
        output_features=output_features,
    )
    wrong_order = ActionFlowV22DatasetMetadata(
        camera_names=("top", "wrist"),
        entity_names=("tomato",),
        max_entities=8,
    )

    with pytest.raises(ValueError, match="camera order"):
        config.set_dataset_feature_metadata(make_features(wrong_order))


def test_relative_action_contract_requires_six_joint_dimensions():
    input_features, output_features = make_policy_features()
    config = ActionFlowV22Config(
        device="cpu",
        input_features=input_features,
        output_features=output_features,
    )
    metadata = config.dataset_contract(("tomato",))
    bad_names = [f"joint_{index}.pos" for index in range(7)]

    with pytest.raises(ValueError, match="produced 7 dimensions"):
        config.set_dataset_feature_metadata(make_features(metadata, action_names=bad_names))


def test_model_batch_contract_exposes_box_grounding_and_fast_fields():
    contract = {
        field.key: field
        for field in ActionFlowV22Config(device="cpu").model_batch_contract()
    }

    assert contract[TARGET_BOXES].shape == ("batch_size", 2, 8, 4)
    assert contract[TARGET_BOX_BINS].shape == ("batch_size", 2, 8, 4)
    assert contract[GROUNDING_TOKENS].shape == ("batch_size", 192)
    assert contract[GROUNDING_TOKEN_MASK].dtype == "bool"
    assert contract[GROUNDING_TOKEN_LOSS_WEIGHT].dtype == "float32"
    assert contract[ACTION_TOKENS].shape == ("batch_size", 256)
    assert contract[ACTION_TOKEN_MASK].dtype == "bool"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extended_vlm_vocab_size": 50_000}, "deterministic vocabulary size"),
        ({"max_grounding_tokens": 32}, "too small"),
        (
            {"knowledge_insulation": True, "detach_expert_prefix_kv": False},
            "detached expert prefix",
        ),
        (
            {"knowledge_insulation": True, "allow_discrete_continuous_attention": True},
            "forbids attention",
        ),
        (
            {"train_expert_only": True, "box_loss_weight": 1.0},
            "incompatible",
        ),
    ],
)
def test_incompatible_configuration_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        ActionFlowV22Config(device="cpu", **overrides)
