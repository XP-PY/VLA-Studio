"""Configuration objects for the chassis SDK."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class PLNSConfig:
    """TCP settings for the PLNS command channel."""

    host: str = "192.168.1.91"
    port: int = 5005
    username: str = "admin"
    password: str = "admin"
    heartbeat_interval_s: float = 10.0
    max_frame_size: int = 128 * 1024
    file_max_frame_size: int = 16 * 1024 * 1024
    mapping_file_heartbeat_interval_s: float = 1.0
    connect_timeout_s: float = 5.0

    @classmethod
    def from_env(cls) -> "PLNSConfig":
        """Function: Read PLNS TCP and login config from environment variables.
        
        Returns:
            PLNS config object merged from environment variables and defaults."""

        return cls(
            host=os.getenv("PLNS_HOST", cls.host),
            port=int(os.getenv("PLNS_PORT", str(cls.port))),
            username=os.getenv("PLNS_USERNAME", cls.username),
            password=os.getenv("PLNS_PASSWORD", cls.password),
            heartbeat_interval_s=float(
                os.getenv("PLNS_HEARTBEAT_INTERVAL_S", str(cls.heartbeat_interval_s))
            ),
            max_frame_size=int(os.getenv("PLNS_MAX_FRAME_SIZE", str(cls.max_frame_size))),
            file_max_frame_size=int(os.getenv("PLNS_FILE_MAX_FRAME_SIZE", str(cls.file_max_frame_size))),
            mapping_file_heartbeat_interval_s=float(
                os.getenv(
                    "PLNS_MAPPING_FILE_HEARTBEAT_INTERVAL_S",
                    str(cls.mapping_file_heartbeat_interval_s),
                )
            ),
            connect_timeout_s=float(os.getenv("PLNS_CONNECT_TIMEOUT_S", str(cls.connect_timeout_s))),
        )


@dataclass(frozen=True)
class SFTPConfig:
    """SFTP settings for map synchronization."""

    host: str = "192.168.1.91"
    port: int = 22
    username: str = "nvidia"
    password: str = "bjs1234"
    remote_dir: str = "/home/nvidia/maps/"
    local_dir: Path = Path("maps")

    @classmethod
    def from_env(cls) -> "SFTPConfig":
        """Function: Read PLNS SFTP map sync config from environment variables.
        
        Returns:
            SFTP config object merged from environment variables and defaults."""

        return cls(
            host=os.getenv("PLNS_SFTP_HOST", cls.host),
            port=int(os.getenv("PLNS_SFTP_PORT", str(cls.port))),
            username=os.getenv("PLNS_SFTP_USERNAME", cls.username),
            password=os.getenv("PLNS_SFTP_PASSWORD", cls.password),
            remote_dir=os.getenv("PLNS_SFTP_REMOTE_DIR", cls.remote_dir),
            local_dir=Path(os.getenv("PLNS_SFTP_LOCAL_DIR", str(cls.local_dir))),
        )


@dataclass(frozen=True)
class PeripheralConfig:
    """Best-known settings for the peripheral/controller channel."""

    host: str = "192.168.1.100"
    port: int = 60000
    timeout_s: float = 3.0

    @classmethod
    def from_env(cls) -> "PeripheralConfig":
        """Function: Read peripheral controller probe config from environment variables.
        
        Returns:
            Peripheral config object merged from environment variables and defaults."""

        return cls(
            host=os.getenv("PERIPHERAL_HOST", cls.host),
            port=int(os.getenv("PERIPHERAL_PORT", str(cls.port))),
            timeout_s=float(os.getenv("PERIPHERAL_TIMEOUT_S", str(cls.timeout_s))),
        )
