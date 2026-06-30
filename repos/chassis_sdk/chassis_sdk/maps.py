"""Map loading and optional SFTP synchronization."""

from __future__ import annotations

from pathlib import Path
import re

from .config import SFTPConfig
from .models import MapBundle, MapMetadata, RoadMap


def parse_map_yaml(path: str | Path) -> MapMetadata:
    """Function: Parse ROS-style map.yaml metadata used by PLNS.
    
    Args:
        path: Path to the map.yaml file.
    
    Returns:
        Map image, resolution, and origin metadata."""

    values: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    origin_raw = values.get("origin", "[0, 0, 0]")
    origin = [float(part) for part in re.findall(r"[-+]?\d+(?:\.\d+)?", origin_raw)]
    while len(origin) < 3:
        origin.append(0.0)

    return MapMetadata(
        image=values.get("image", ""),
        resolution_m=float(values.get("resolution", "0.05")),
        origin_x_m=origin[0],
        origin_y_m=origin[1],
        origin_yaw_rad=origin[2],
    )


def load_map_bundle(directory: str | Path) -> MapBundle:
    """Function: Load map.json plus optional map.yaml and map.pgm from a directory.
    
    Args:
        directory: Directory containing map files.
    
    Returns:
        Map bundle containing the road map, metadata, and image path."""

    root = Path(directory)
    road_map = RoadMap.from_json_file(root / "map.json")
    metadata = parse_map_yaml(root / "map.yaml") if (root / "map.yaml").exists() else None
    image_path = root / "map.pgm" if (root / "map.pgm").exists() else None
    return MapBundle(road_map=road_map, metadata=metadata, image_path=image_path)


def sync_maps_via_sftp(config: SFTPConfig, files: tuple[str, ...] = ("map.json", "map.yaml", "map.pgm")) -> None:
    """Function: Download PLNS map-related files over SFTP.
    
    Args:
        config: SFTP host, credentials, remote directory, and local directory config.
        files: Filenames to synchronize."""

    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError("install the 'sftp' extra or paramiko to sync maps") from exc

    config.local_dir.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            config.host,
            config.port,
            config.username,
            config.password,
            timeout=10,
        )
        sftp = client.open_sftp()
        try:
            for filename in files:
                remote = config.remote_dir.rstrip("/") + "/" + filename
                local = config.local_dir / filename
                sftp.get(remote, str(local))
        finally:
            sftp.close()
    finally:
        client.close()
