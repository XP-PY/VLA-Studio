"""Read-only helper for the peripheral/controller TCP endpoint."""

from __future__ import annotations

from .config import PeripheralConfig
from .transport import probe_tcp_endpoint


async def probe_peripheral(config: PeripheralConfig | None = None) -> bytes:
    """Function: Connect to the peripheral controller endpoint and read only initial data without sending commands.
    
    Args:
        config: Peripheral endpoint config. Uses environment-based defaults when omitted.
    
    Returns:
        Initial bytes read from the endpoint; empty bytes when no data is read before timeout."""

    cfg = config or PeripheralConfig.from_env()
    return await probe_tcp_endpoint(cfg.host, cfg.port, timeout_s=cfg.timeout_s)
