"""Dataset-specific semantics for Action Flow V2 grounding annotation.

The Grounding DINO runner owns video decoding, model inference, ranking, and
artifact generation. This module only describes how a dataset's task prompts
map to semantic entities and detector queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Mapping


WORKSPACE = Path(__file__).resolve().parents[2]

Cardinality = Literal["single", "all"]
CandidateOrder = Literal[
    "confidence",
    "top_to_bottom",
    "bottom_to_top",
    "left_to_right",
    "right_to_left",
]


@dataclass(frozen=True)
class EntitySpec:
    """One task-relevant entity or one ordered instance of an entity group."""

    object_id: int
    object_name: str
    object_role: str
    query: str
    task_step: int
    cardinality: Cardinality = "single"
    instance_index: int = 0
    candidate_order: CandidateOrder = "confidence"
    relation_container_query: str | None = None
    relation_member_queries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.object_id < 1:
            raise ValueError("object_id must be positive")
        if self.task_step < 1:
            raise ValueError("task_step must be positive")
        if self.instance_index < 0:
            raise ValueError("instance_index must be non-negative")
        if self.cardinality == "single" and self.instance_index != 0:
            raise ValueError("single entities must use instance_index=0")


TaskParser = Callable[[str, Callable[[str], str]], list[EntitySpec]]


@dataclass(frozen=True)
class GroundingDatasetConfig:
    """Grounding semantics and default inputs for one LeRobot dataset family."""

    name: str
    default_dataset_root: Path
    cameras: tuple[str, ...]
    detector_queries: tuple[str, ...]
    task_parser: TaskParser = field(repr=False)
    query_aliases: Mapping[str, str] = field(default_factory=dict)
    camera_entity_names: Mapping[str, frozenset[str]] = field(default_factory=dict)
    box_threshold: float = 0.4
    text_threshold: float = 0.4
    min_quality: float = 0.5

    def __post_init__(self) -> None:
        unknown_cameras = set(self.camera_entity_names).difference(self.cameras)
        if unknown_cameras:
            raise ValueError(
                f"Dataset config {self.name!r} restricts unknown cameras: "
                f"{sorted(unknown_cameras)}"
            )

    def grounding_query(self, object_name: str) -> str:
        """Return the detector phrase without changing the semantic object name."""
        return self.query_aliases.get(object_name, object_name)

    def entities_from_task(self, task: str) -> list[EntitySpec]:
        """Convert one dataset task prompt into deterministic entity specs."""
        entities = self.task_parser(task, self.grounding_query)
        object_ids = [entity.object_id for entity in entities]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(f"Dataset config {self.name!r} produced duplicate object IDs")
        return entities

    def entities_for_camera(self, entities: list[EntitySpec], camera: str) -> list[EntitySpec]:
        """Return only the task entities relevant to a particular camera."""
        allowed_names = self.camera_entity_names.get(camera)
        if allowed_names is None:
            return entities
        return [entity for entity in entities if entity.object_name in allowed_names]


TABLE_VEGETABLE_NAMES = ("tomato", "garlic", "corn", "cucumber")
TABLE_PLATE_COLORS = ("pink", "yellow", "blue")
TABLE_QUERY_ALIASES = {
    "cucumber": "green cucumber",
    "garlic": "white garlic",
    "pink plate": "pink square bowl",
    "yellow plate": "cream square bowl",
    "blue plate": "blue square bowl",
    "collection box": "brown cardboard box",
}

PLACE_PATTERN = re.compile(
    r"place the (tomato|garlic|corn|cucumber) on the (pink|yellow|blue) plate",
    flags=re.IGNORECASE,
)
CLEAR_PATTERN = re.compile(
    r"put the vegetable from the (pink|yellow|blue) plate into the collection box",
    flags=re.IGNORECASE,
)


def parse_table_preparation_task(
    task: str,
    query_for: Callable[[str], str],
) -> list[EntitySpec]:
    """Parse the table-preparation and table-bussing prompt templates."""
    placements = PLACE_PATTERN.findall(task)
    if placements:
        entities: list[EntitySpec] = []
        for step, (vegetable, plate_color) in enumerate(placements, start=1):
            vegetable = vegetable.lower()
            plate_name = f"{plate_color.lower()} plate"
            entities.append(
                EntitySpec(
                    object_id=len(entities) + 1,
                    object_name=vegetable,
                    object_role="source",
                    query=query_for(vegetable),
                    task_step=step,
                )
            )
            entities.append(
                EntitySpec(
                    object_id=len(entities) + 1,
                    object_name=plate_name,
                    object_role="destination",
                    query=query_for(plate_name),
                    task_step=step,
                )
            )
        return entities

    clear_match = CLEAR_PATTERN.fullmatch(task.rstrip("."))
    if clear_match:
        plate_name = f"{clear_match.group(1).lower()} plate"
        source_name = f"vegetable on {plate_name}"
        return [
            EntitySpec(
                object_id=1,
                object_name=source_name,
                object_role="source",
                query=f"vegetable on the {plate_name}",
                task_step=1,
                relation_container_query=query_for(plate_name),
                relation_member_queries=tuple(query_for(name) for name in TABLE_VEGETABLE_NAMES),
            ),
            EntitySpec(2, plate_name, "source container", query_for(plate_name), 1),
            EntitySpec(3, "collection box", "destination", query_for("collection box"), 1),
        ]

    if task.rstrip(".").lower() == "put all vegetables into the collection box":
        entities = [
            EntitySpec(index + 1, name, "source", query_for(name), 1)
            for index, name in enumerate(TABLE_VEGETABLE_NAMES)
        ]
        entities.append(
            EntitySpec(
                len(entities) + 1,
                "collection box",
                "destination",
                query_for("collection box"),
                1,
            )
        )
        return entities

    raise ValueError(f"Unsupported table-preparation task prompt: {task!r}")


ASSEMBLY_TASK = (
    "grasp the visible assembly parts from farthest to closest relative to the front camera, "
    "and place each pair in the center"
)
ASSEMBLY_QUERY_ALIASES = {
    "large assembly part": "metal cylinder",
    "small assembly part": "small metal block",
    "center assembly area": "white rectangular paper",
}


def _ordered_entity_group(
    *,
    first_object_id: int,
    object_name: str,
    object_role: str,
    query: str,
    count: int,
) -> list[EntitySpec]:
    return [
        EntitySpec(
            object_id=first_object_id + index,
            object_name=object_name,
            object_role=object_role,
            query=query,
            task_step=index + 1,
            cardinality="all",
            instance_index=index,
            candidate_order="top_to_bottom",
        )
        for index in range(count)
    ]


def parse_assembly_parts_task(
    task: str,
    query_for: Callable[[str], str],
) -> list[EntitySpec]:
    """Parse the fixed bimanual farthest-to-closest assembly task."""
    if " ".join(task.rstrip(".").lower().split()) != ASSEMBLY_TASK:
        raise ValueError(f"Unsupported assembly-parts task prompt: {task!r}")

    entities = _ordered_entity_group(
        first_object_id=1,
        object_name="large assembly part",
        object_role="source",
        query=query_for("large assembly part"),
        count=3,
    )
    entities.extend(
        _ordered_entity_group(
            first_object_id=4,
            object_name="small assembly part",
            object_role="source",
            query=query_for("small assembly part"),
            count=3,
        )
    )
    entities.append(
        EntitySpec(
            object_id=7,
            object_name="center assembly area",
            object_role="destination",
            query=query_for("center assembly area"),
            task_step=1,
        )
    )
    return entities


TABLE_PREPARATION_CONFIG = GroundingDatasetConfig(
    name="table_preparation",
    default_dataset_root=(
        WORKSPACE / "runs" / "dataset" / "rm65_single_table_preparation_and_bussing"
    ),
    cameras=("top", "wrist"),
    detector_queries=tuple(
        [TABLE_QUERY_ALIASES.get(name, name) for name in TABLE_VEGETABLE_NAMES]
        + [
            TABLE_QUERY_ALIASES.get(f"{color} plate", f"{color} plate")
            for color in TABLE_PLATE_COLORS
        ]
        + [TABLE_QUERY_ALIASES["collection box"]]
    ),
    task_parser=parse_table_preparation_task,
    query_aliases=TABLE_QUERY_ALIASES,
)

ASSEMBLY_PARTS_CONFIG = GroundingDatasetConfig(
    name="assembly_parts",
    default_dataset_root=(
        WORKSPACE
        / "runs"
        / "dataset"
        / "rm65_bimanual_farthest_to_closest_assembly_parts_to_center_no_top"
    ),
    cameras=("left_wrist", "right_wrist", "front"),
    detector_queries=tuple(ASSEMBLY_QUERY_ALIASES.values()),
    task_parser=parse_assembly_parts_task,
    query_aliases=ASSEMBLY_QUERY_ALIASES,
    camera_entity_names={
        "left_wrist": frozenset({"large assembly part", "center assembly area"}),
        "right_wrist": frozenset({"small assembly part", "center assembly area"}),
        "front": frozenset(
            {"large assembly part", "small assembly part", "center assembly area"}
        ),
    },
    min_quality=0.4,
)

DATASET_CONFIGS = {
    config.name: config
    for config in (
        TABLE_PREPARATION_CONFIG,
        ASSEMBLY_PARTS_CONFIG,
    )
}


def get_dataset_config(name: str) -> GroundingDatasetConfig:
    """Return a registered dataset config by CLI name."""
    try:
        return DATASET_CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset config {name!r}; choose one of: {available}") from exc
