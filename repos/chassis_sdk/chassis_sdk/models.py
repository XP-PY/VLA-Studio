"""Typed models for PLNS map and pose data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math


@dataclass(frozen=True)
class Pose2D:
    """2D pose in SI units."""

    x_m: float
    y_m: float
    yaw_rad: float = 0.0

    @classmethod
    def from_proto_units(cls, x_mm: int, y_mm: int, yaw_mrad: int = 0) -> "Pose2D":
        """Function: Convert PLNS protocol units to the SDK SI-unit pose.
        
        Args:
            x_mm: X coordinate in mm.
            y_mm: Y coordinate in mm.
            yaw_mrad: Yaw angle in 0.001 rad.
        
        Returns:
            2D pose expressed in meters and radians."""

        return cls(x_mm / 1000.0, y_mm / 1000.0, yaw_mrad / 1000.0)

    def to_proto_units(self) -> tuple[int, int, int]:
        """Function: Convert the SDK SI-unit pose to PLNS protocol units.
        
        Returns:
            Integer (x_mm, y_mm, yaw_mrad) tuple."""

        return (
            int(round(self.x_m * 1000)),
            int(round(self.y_m * 1000)),
            int(round(self.yaw_rad * 1000)),
        )

    @property
    def yaw_deg(self) -> float:
        """Function: Get the yaw angle in degrees.
        
        Returns:
            Yaw angle in degrees."""

        return math.degrees(self.yaw_rad)


@dataclass(frozen=True)
class Node:
    id: int
    x_mm: int
    y_mm: int
    yaw_mrad: int = 0

    @property
    def pose(self) -> Pose2D:
        """Function: Get the 2D pose for the node.
        
        Returns:
            Node pose expressed in meters and radians."""

        return Pose2D.from_proto_units(self.x_mm, self.y_mm, self.yaw_mrad)


@dataclass(frozen=True)
class Station:
    id: int
    name: str
    node_id: int
    type: int
    x_mm: int
    y_mm: int
    yaw_mrad: int
    direction: int
    is_rotate: bool
    obst_stop_mm: int

    @property
    def pose(self) -> Pose2D:
        """Function: Get the 2D pose for the station.
        
        Returns:
            Station pose expressed in meters and radians."""

        return Pose2D.from_proto_units(self.x_mm, self.y_mm, self.yaw_mrad)

    @property
    def is_charger(self) -> bool:
        """Function: Check whether the station is a charger station.
        
        Returns:
            True for charger stations; otherwise False."""

        return self.type == 2


@dataclass(frozen=True)
class Edge:
    id: int
    type: int
    s_node: int
    e_node: int
    sx_mm: int
    sy_mm: int
    ex_mm: int
    ey_mm: int
    cost_mm: int
    direction: int
    limit_v_mm_s: int
    limit_w_mrad_s: int
    limit_acc_mm_s2: int
    limit_dec_mm_s2: int
    obst_stop_mm: int
    s_facing_mrad: int = 0
    e_facing_mrad: int = 0
    cx_mm: int = 0
    cy_mm: int = 0
    dx_mm: int = 0
    dy_mm: int = 0


@dataclass(frozen=True)
class RoadMap:
    nodes: dict[int, Node]
    stations: dict[int, Station]
    edges: dict[int, Edge]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoadMap":
        """Function: Build a RoadMap from a road-network JSON dictionary.
        
        Args:
            data: Dictionary data parsed from map.json.
        
        Returns:
            Road map containing nodes, stations, and edges."""

        nodes = {
            int(item["id"]): Node(
                id=int(item["id"]),
                x_mm=int(item["x"]),
                y_mm=int(item["y"]),
                yaw_mrad=int(item.get("yaw", 0)),
            )
            for item in data.get("node", [])
        }
        stations = {
            int(item["id"]): Station(
                id=int(item["id"]),
                name=str(item.get("name", item["id"])),
                node_id=int(item["node_id"]),
                type=int(item.get("type", 1)),
                x_mm=int(item["x"]),
                y_mm=int(item["y"]),
                yaw_mrad=int(item.get("yaw", 0)),
                direction=int(item.get("direction", 0)),
                is_rotate=bool(item.get("is_rotate", False)),
                obst_stop_mm=int(item.get("obst_stop", 0)),
            )
            for item in data.get("station", [])
        }
        edges = {
            int(item["id"]): Edge(
                id=int(item["id"]),
                type=int(item.get("type", 1)),
                s_node=int(item["s_node"]),
                e_node=int(item["e_node"]),
                sx_mm=int(item["sx"]),
                sy_mm=int(item["sy"]),
                ex_mm=int(item["ex"]),
                ey_mm=int(item["ey"]),
                cost_mm=int(item.get("cost", 0)),
                direction=int(item.get("direction", 1)),
                limit_v_mm_s=int(item.get("limit_v", 0)),
                limit_w_mrad_s=int(item.get("limit_w", 0)),
                limit_acc_mm_s2=int(item.get("limit_acc", 0)),
                limit_dec_mm_s2=int(item.get("limit_dec", 0)),
                obst_stop_mm=int(item.get("obst_stop", 0)),
                s_facing_mrad=int(item.get("s_facing", 0)),
                e_facing_mrad=int(item.get("e_facing", 0)),
                cx_mm=int(item.get("cx", 0)),
                cy_mm=int(item.get("cy", 0)),
                dx_mm=int(item.get("dx", 0)),
                dy_mm=int(item.get("dy", 0)),
            )
            for item in data.get("edge", [])
        }
        return cls(nodes=nodes, stations=stations, edges=edges)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "RoadMap":
        """Function: Load a road map from a map.json file.
        
        Args:
            path: Path to the map.json file.
        
        Returns:
            Parsed road map."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def require_station(self, station_id: int) -> Station:
        """Function: Get a station by ID and raise an error when it does not exist.
        
        Args:
            station_id: Station ID to look up.
        
        Returns:
            Matched station object."""

        try:
            return self.stations[station_id]
        except KeyError as exc:
            raise ValueError(f"unknown station id: {station_id}") from exc


@dataclass(frozen=True)
class MapMetadata:
    image: str
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    origin_yaw_rad: float

    def world_to_pixel(self, pose: Pose2D) -> tuple[float, float]:
        """Function: Convert map world coordinates to pixel coordinates.
        
        Args:
            pose: Map world pose expressed in meters and radians.
        
        Returns:
            (pixel_x, pixel_y) pixel coordinates."""

        return (
            (pose.x_m - self.origin_x_m) / self.resolution_m,
            (pose.y_m - self.origin_y_m) / self.resolution_m,
        )

    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> Pose2D:
        """Function: Convert pixel coordinates to map world coordinates.
        
        Args:
            pixel_x: Pixel coordinate in the X direction.
            pixel_y: Pixel coordinate in the Y direction.
        
        Returns:
            2D pose expressed in meters and radians, with yaw fixed to 0."""

        return Pose2D(
            self.origin_x_m + pixel_x * self.resolution_m,
            self.origin_y_m + pixel_y * self.resolution_m,
            0.0,
        )


@dataclass(frozen=True)
class MapBundle:
    road_map: RoadMap
    metadata: MapMetadata | None = None
    image_path: Path | None = None
