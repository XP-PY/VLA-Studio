#!/usr/bin/env python

"""Configuration contract for Action Flow V2.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION

from ..smolvla.configuration_smolvla import SmolVLAConfig
from .data_contracts_action_flow_v2_2 import (
    BOX_FEATURE_KEYS,
    ActionFlowV22DatasetMetadata,
    make_model_batch_contract,
)

DEFAULT_STRUCTURAL_TOKENS = (
    "<grounding>",
    "</grounding>",
    "<camera>",
    "</camera>",
    "<source>",
    "<destination>",
    "<source_container>",
    "<destination_container>",
    "<box>",
    "<not_visible>",
    "<action>",
    "<action_end>",
)


@PreTrainedConfig.register_subclass("action_flow_v2_2")
@dataclass
class ActionFlowV22Config(SmolVLAConfig):
    """Configuration for knowledge-insulated Action Flow V2.2.

    The token ranges are deterministic and non-overlapping:

    1. Existing SmolVLM tokens occupy ``[0, base_vlm_vocab_size)``.
    2. Structural tokens follow in ``structural_tokens`` order.
    3. Coordinate tokens follow in ascending bin order.
    4. FAST tokens follow in ascending FAST-symbol order.

    Later stages must extend the tokenizer in exactly this order and resize both
    input embeddings and the language-model head to
    ``extended_vlm_vocab_size``.
    """

    # V2.2 trains the VLM with grounding and FAST cross-entropy losses.
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    load_vlm_weights: bool = True
    # Optional warm start for the shared SmolVLA architecture. None preserves
    # the original V2.2 initialization (pretrained SmolVLM plus random expert).
    smolvla_pretrained_path: str | None = None
    n_action_steps: int = 10

    # Relative actions: six RM65 joints are relative while gripper stays absolute.
    use_relative_actions: bool = True
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None
    expected_state_dim: int = 7
    expected_action_dim: int = 7
    expected_relative_action_dim: int = 6

    # Ordered target-box data contract.
    camera_names: tuple[str, ...] = ("wrist", "top")
    max_cameras: int = 8
    max_box_entities: int = 8
    box_schema_version: int = 1

    # Tokenizer sources and fixed sequence budgets.
    vlm_tokenizer_name: str | None = None
    action_tokenizer_name: str = "lerobot/fast-action-tokenizer"
    tokenizer_artifact_path: str = "artifacts/tokenizers/action_flow_v2_2_smolvlm_fast"
    checkpoint_tokenizer_dirname: str = "action_flow_v2_2_tokenizer"
    max_grounding_tokens: int = 192
    max_fast_action_tokens: int = 256

    # Deterministic extension of the local SmolVLM vocabulary.
    token_contract_version: int = 1
    base_vlm_vocab_size: int = 49_280
    structural_tokens: tuple[str, ...] = DEFAULT_STRUCTURAL_TOKENS
    coordinate_token_count: int = 256
    fast_token_count: int = 2_048
    location_token_prefix: str = "<loc_"
    location_token_suffix: str = ">"
    location_token_width: int = 3
    fast_token_prefix: str = "<fast_"
    fast_token_suffix: str = ">"
    fast_token_width: int = 4
    extended_vlm_vocab_size: int | None = None

    # Independently reduced training objectives.
    box_loss_weight: float = 1.0
    fast_loss_weight: float = 1.0
    flow_loss_weight: float = 1.0

    # The continuous expert sees detached VLM prefix K/V and no AR target K/V.
    knowledge_insulation: bool = True
    detach_expert_prefix_kv: bool = True
    allow_discrete_continuous_attention: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.token_contract_version != 1:
            raise ValueError(
                f"Unsupported Action Flow V2.2 token contract version: {self.token_contract_version}."
            )
        if self.box_schema_version != 1:
            raise ValueError(
                f"Unsupported Action Flow V2.2 box schema version: {self.box_schema_version}."
            )
        if self.expected_state_dim <= 0 or self.expected_action_dim <= 0:
            raise ValueError("Expected state and action dimensions must be positive.")
        if self.smolvla_pretrained_path is not None and not self.smolvla_pretrained_path.strip():
            raise ValueError("`smolvla_pretrained_path` must be None or a non-empty path/repo ID.")
        if not 0 < self.expected_relative_action_dim <= self.expected_action_dim:
            raise ValueError(
                "`expected_relative_action_dim` must be in [1, expected_action_dim]."
            )
        if self.use_relative_actions and not self.relative_exclude_joints:
            raise ValueError("Relative RM65 actions must keep at least the gripper absolute.")

        if not self.camera_names or len(self.camera_names) > self.max_cameras:
            raise ValueError("`camera_names` must be non-empty and fit within `max_cameras`.")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("`camera_names` must be unique and ordered.")
        if any(not name.strip() for name in self.camera_names):
            raise ValueError("`camera_names` cannot contain empty values.")
        if self.max_box_entities <= 0:
            raise ValueError("`max_box_entities` must be positive.")

        if not self.vlm_tokenizer_path:
            raise ValueError("A SmolVLM tokenizer path must be provided.")
        if not self.action_tokenizer_name:
            raise ValueError("`action_tokenizer_name` must be non-empty.")
        if not self.tokenizer_artifact_path:
            raise ValueError("`tokenizer_artifact_path` must be non-empty.")
        if not self.checkpoint_tokenizer_dirname or "/" in self.checkpoint_tokenizer_dirname:
            raise ValueError(
                "`checkpoint_tokenizer_dirname` must be one non-empty directory name."
            )
        if self.max_grounding_tokens < self.minimum_grounding_tokens:
            raise ValueError(
                f"`max_grounding_tokens={self.max_grounding_tokens}` is too small for the configured "
                f"camera/entity contract; it must be at least {self.minimum_grounding_tokens}."
            )
        if self.max_fast_action_tokens <= 0:
            raise ValueError("`max_fast_action_tokens` must be positive.")

        self._validate_token_contract()
        required_vocab_size = self.required_vlm_vocab_size
        if self.extended_vlm_vocab_size is None:
            self.extended_vlm_vocab_size = required_vocab_size
        elif self.extended_vlm_vocab_size != required_vocab_size:
            raise ValueError(
                f"`extended_vlm_vocab_size={self.extended_vlm_vocab_size}` does not match the "
                f"required deterministic vocabulary size {required_vocab_size}."
            )

        for name, weight in (
            ("box_loss_weight", self.box_loss_weight),
            ("fast_loss_weight", self.fast_loss_weight),
            ("flow_loss_weight", self.flow_loss_weight),
        ):
            if weight < 0:
                raise ValueError(f"`{name}` must be non-negative.")
        if self.box_loss_weight == self.fast_loss_weight == self.flow_loss_weight == 0:
            raise ValueError("At least one Action Flow V2.2 loss must be enabled.")
        if self.train_expert_only and (self.box_loss_weight > 0 or self.fast_loss_weight > 0):
            raise ValueError(
                "`train_expert_only=True` is incompatible with nonzero box or FAST loss weights."
            )
        if not self.use_cache:
            raise ValueError("Joint Action Flow V2.2 training requires `use_cache=True`.")
        if self.knowledge_insulation:
            if not self.detach_expert_prefix_kv:
                raise ValueError("Knowledge insulation requires detached expert prefix K/V tensors.")
            if self.allow_discrete_continuous_attention:
                raise ValueError(
                    "Knowledge insulation forbids attention between autoregressive targets and "
                    "continuous expert tokens."
                )

    @property
    def vlm_tokenizer_path(self) -> str:
        """Tokenizer source, defaulting to the configured SmolVLM model."""

        return self.vlm_tokenizer_name or self.vlm_model_name

    @property
    def structural_token_start_id(self) -> int:
        return self.base_vlm_vocab_size

    @property
    def location_token_start_id(self) -> int:
        return self.structural_token_start_id + len(self.structural_tokens)

    @property
    def fast_token_start_id(self) -> int:
        return self.location_token_start_id + self.coordinate_token_count

    @property
    def required_vlm_vocab_size(self) -> int:
        return self.fast_token_start_id + self.fast_token_count

    @property
    def minimum_grounding_tokens(self) -> int:
        """Conservative lower bound excluding variable-length entity-name text."""

        tokens_per_visible_entity = 7  # role, entity name, box, and four locations
        camera_delimiters = 2 * len(self.camera_names)
        return (
            2
            + camera_delimiters
            + len(self.camera_names) * self.max_box_entities * tokens_per_visible_entity
        )

    def structural_token_id(self, token: str) -> int:
        """Return the deterministic ID of a configured structural token."""

        try:
            index = self.structural_tokens.index(token)
        except ValueError as error:
            raise KeyError(f"Unknown Action Flow V2.2 structural token: {token!r}.") from error
        return self.structural_token_start_id + index

    def location_token(self, coordinate_bin: int) -> str:
        """Return the string form of one quantized coordinate token."""

        if not 0 <= coordinate_bin < self.coordinate_token_count:
            raise ValueError(
                f"Coordinate bin {coordinate_bin} is outside [0, {self.coordinate_token_count})."
            )
        return (
            f"{self.location_token_prefix}{coordinate_bin:0{self.location_token_width}d}"
            f"{self.location_token_suffix}"
        )

    def location_token_id(self, coordinate_bin: int) -> int:
        self.location_token(coordinate_bin)
        return self.location_token_start_id + coordinate_bin

    def fast_token(self, fast_symbol: int) -> str:
        """Return the string form of one dedicated FAST symbol token."""

        if not 0 <= fast_symbol < self.fast_token_count:
            raise ValueError(f"FAST symbol {fast_symbol} is outside [0, {self.fast_token_count}).")
        return (
            f"{self.fast_token_prefix}{fast_symbol:0{self.fast_token_width}d}"
            f"{self.fast_token_suffix}"
        )

    def fast_token_id(self, fast_symbol: int) -> int:
        self.fast_token(fast_symbol)
        return self.fast_token_start_id + fast_symbol

    def token_contract(self) -> dict[str, Any]:
        """Return the complete deterministic token allocation for checkpoint checks."""

        return {
            "version": self.token_contract_version,
            "base_vlm_vocab_size": self.base_vlm_vocab_size,
            "extended_vlm_vocab_size": self.extended_vlm_vocab_size,
            "structural_token_ids": {
                token: self.structural_token_id(token) for token in self.structural_tokens
            },
            "location_token_range": [
                self.location_token_start_id,
                self.fast_token_start_id,
            ],
            "fast_token_range": [
                self.fast_token_start_id,
                self.required_vlm_vocab_size,
            ],
        }

    def architecture_contract(self) -> dict[str, Any]:
        """Return Stage 1 architecture switches that a checkpoint must preserve."""

        return {
            "vlm_backbone": "smolvlm",
            "autoregressive_targets": ["target_boxes", "fast_actions"],
            "continuous_action_head": "flow_matching_expert",
            "knowledge_insulation": self.knowledge_insulation,
            "detach_expert_prefix_kv": self.detach_expert_prefix_kv,
            "allow_discrete_continuous_attention": self.allow_discrete_continuous_attention,
            "use_relative_actions": self.use_relative_actions,
            "relative_exclude_joints": list(self.relative_exclude_joints),
        }

    def dataset_contract(
        self, entity_names: tuple[str, ...] = ()
    ) -> ActionFlowV22DatasetMetadata:
        """Build dataset metadata using the configuration's fixed dimensions."""

        return ActionFlowV22DatasetMetadata(
            camera_names=tuple(self.camera_names),
            entity_names=entity_names,
            max_entities=self.max_box_entities,
            schema_version=self.box_schema_version,
            coordinate_bins=self.coordinate_token_count,
        )

    def model_batch_contract(self):
        """Return model-facing tensor keys, dtypes, and symbolic shapes."""

        return make_model_batch_contract(
            camera_count=len(self.camera_names),
            max_entities=self.max_box_entities,
            max_grounding_tokens=self.max_grounding_tokens,
            max_fast_action_tokens=self.max_fast_action_tokens,
        )

    def validate_features(self) -> None:
        """Validate RM65 state/action dimensions and configured camera order."""

        super().validate_features()

        if self.image_features:
            image_camera_names = tuple(key.rsplit(".", 1)[-1] for key in self.image_features)
            if image_camera_names != tuple(self.camera_names):
                raise ValueError(
                    "Policy image order does not match `camera_names`: "
                    f"{image_camera_names} != {self.camera_names}."
                )
        if self.robot_state_feature is not None:
            state_dim = self.robot_state_feature.shape[-1]
            if state_dim != self.expected_state_dim:
                raise ValueError(
                    f"RM65 state dimension is {state_dim}, expected {self.expected_state_dim}."
                )
        if self.action_feature is not None:
            action_dim = self.action_feature.shape[-1]
            if action_dim != self.expected_action_dim:
                raise ValueError(
                    f"RM65 action dimension is {action_dim}, expected {self.expected_action_dim}."
                )

    def set_dataset_feature_metadata(self, features: dict[str, dict[str, Any]]) -> None:
        """Validate box/action metadata and exclude labels from normalized inputs."""

        entity_names = tuple(features.get("observation.target_boxes", {}).get("entity_names", ()))
        self.dataset_contract(entity_names).validate_feature_metadata(features)

        action_metadata = features.get(ACTION)
        if action_metadata is None:
            raise ValueError("Action Flow V2.2 requires dataset action metadata.")
        action_shape = tuple(action_metadata.get("shape", ()))
        if action_shape != (self.expected_action_dim,):
            raise ValueError(
                f"Dataset action shape is {action_shape}, expected {(self.expected_action_dim,)}."
            )

        action_names = list(action_metadata.get("names") or [])
        if len(action_names) != self.expected_action_dim:
            raise ValueError(
                "Dataset action names must identify every action dimension; "
                f"got {len(action_names)} names for {self.expected_action_dim} dimensions."
            )
        self.action_feature_names = action_names

        if self.use_relative_actions:
            exclude_tokens = [name.lower() for name in self.relative_exclude_joints]
            excluded = [
                name
                for name in action_names
                if any(token == name.lower() or token in name.lower() for token in exclude_tokens)
            ]
            relative_dim = len(action_names) - len(excluded)
            if relative_dim != self.expected_relative_action_dim:
                raise ValueError(
                    f"Relative-action selection produced {relative_dim} dimensions, expected "
                    f"{self.expected_relative_action_dim}. Action names: {action_names}; "
                    f"excluded: {excluded}."
                )

        if self.input_features is not None:
            for key in BOX_FEATURE_KEYS:
                self.input_features.pop(key, None)

    def _validate_token_contract(self) -> None:
        if self.base_vlm_vocab_size <= 0:
            raise ValueError("`base_vlm_vocab_size` must be positive.")
        if not self.structural_tokens or len(set(self.structural_tokens)) != len(
            self.structural_tokens
        ):
            raise ValueError("`structural_tokens` must be non-empty and unique.")
        if any(
            not token or not token.startswith("<") or not token.endswith(">")
            for token in self.structural_tokens
        ):
            raise ValueError("Every structural token must be a non-empty angle-bracket token.")
        if self.coordinate_token_count < 2 or self.fast_token_count <= 0:
            raise ValueError("Coordinate and FAST token counts must be at least 2 and 1 respectively.")
        if self.location_token_width <= 0 or self.fast_token_width <= 0:
            raise ValueError("Token numeric widths must be positive.")

        generated_tokens = {
            self.location_token(0),
            self.location_token(self.coordinate_token_count - 1),
            self.fast_token(0),
            self.fast_token(self.fast_token_count - 1),
        }
        overlap = generated_tokens.intersection(self.structural_tokens)
        if overlap:
            raise ValueError(f"Generated location/FAST tokens overlap structural tokens: {overlap}.")
