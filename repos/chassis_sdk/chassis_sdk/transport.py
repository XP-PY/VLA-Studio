"""Async TCP transport for framed PLNS messages."""

from __future__ import annotations

import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import BinaryIO


class FrameError(RuntimeError):
    """Raised when TCP frame data is malformed."""


@dataclass(frozen=True)
class FileFramePayload:
    """Payload read from the PLNS File channel."""

    payload: bytes
    is_mapping_frame: bool


class AsyncFrameTransport:
    """Read and write 4-byte big-endian length-prefixed frames."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_frame_size: int = 128 * 1024,
        connect_timeout_s: float = 5.0,
        raw_log_path: str | Path | None = None,
    ) -> None:
        """Function: Initialize the PLNS TCP frame transport.
        
        Args:
            host: TCP service host address.
            port: TCP service port.
            max_frame_size: Maximum payload size allowed for reads or writes, in bytes.
            connect_timeout_s: Connection timeout in seconds.
            raw_log_path: Optional raw frame log file path."""

        self.host = host
        self.port = port
        self.max_frame_size = max_frame_size
        self.connect_timeout_s = connect_timeout_s
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._raw_log: BinaryIO | None = None
        if raw_log_path:
            path = Path(raw_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._raw_log = path.open("ab")

    @property
    def is_connected(self) -> bool:
        """Function: Check whether the current TCP writer is still connected.
        
        Returns:
            True when connected and not closing; otherwise False."""

        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> None:
        """Function: Open an asynchronous TCP connection to the target host and port."""

        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=self.connect_timeout_s,
        )

    async def close(self) -> None:
        """Function: Close the TCP connection and raw frame log file."""

        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._reader = None
        self._writer = None
        if self._raw_log:
            self._raw_log.close()
            self._raw_log = None

    async def write_frame(self, frame: bytes) -> None:
        """Function: Validate and send one PLNS TCP frame with a 4-byte length header.
        
        Args:
            frame: Complete frame containing a 4-byte big-endian length header and protobuf payload."""

        if not self._writer:
            raise FrameError("transport is not connected")
        if len(frame) < 4:
            raise FrameError("frame missing 4-byte header")
        declared = int.from_bytes(frame[:4], byteorder="big")
        if declared != len(frame) - 4:
            raise FrameError(f"frame length mismatch: header={declared} actual={len(frame) - 4}")
        if declared > self.max_frame_size:
            raise FrameError(f"frame exceeds max size: {declared}")
        self._writer.write(frame)
        await self._writer.drain()
        self._log_raw(b">", frame)

    async def read_payload(self) -> bytes:
        """Function: Read one PLNS TCP frame and return the protobuf payload.
        
        Returns:
            Payload bytes without the 4-byte length header."""

        if not self._reader:
            raise FrameError("transport is not connected")
        header = await self._reader.readexactly(4)
        frame_size = int.from_bytes(header, byteorder="big")
        if frame_size <= 0:
            raise FrameError(f"invalid frame size: {frame_size}")
        if frame_size > self.max_frame_size:
            raise FrameError(f"frame exceeds max size: {frame_size}")
        payload = await self._reader.readexactly(frame_size)
        self._log_raw(b"<", header + payload)
        return payload

    def _log_raw(self, direction: bytes, frame: bytes) -> None:
        """Function: Write a raw send/receive frame to the log file.
        
        Args:
            direction: Frame direction marker, usually b">" or b"<".
            frame: Complete raw frame data."""

        if not self._raw_log:
            return
        self._raw_log.write(direction + len(frame).to_bytes(4, "big") + frame + b"\n")
        self._raw_log.flush()


class AsyncFileFrameTransport(AsyncFrameTransport):
    """Read and write PLNS File-channel frames.

    The File channel writes normal length-prefixed protobuf frames, but reads
    both normal frames and mapping point-cloud frames whose header is
    ``AA 55 BB 66`` followed by a 4-byte big-endian protobuf body size.
    """

    MAPPING_MAGIC = b"\xaa\x55\xbb\x66"

    async def read_file_payload(self) -> FileFramePayload:
        """Function: Read one File-channel frame and return the protobuf payload.

        Returns:
            FileFramePayload containing the protobuf body and whether the frame
            used the special mapping-data header.
        """

        if not self._reader:
            raise FrameError("transport is not connected")

        header = await self._reader.readexactly(4)
        if header == self.MAPPING_MAGIC:
            size_bytes = await self._reader.readexactly(4)
            frame_size = int.from_bytes(size_bytes, byteorder="big")
            if frame_size <= 0:
                raise FrameError(f"invalid file mapping frame size: {frame_size}")
            if frame_size > self.max_frame_size:
                raise FrameError(f"file mapping frame exceeds max size: {frame_size}")
            payload = await self._reader.readexactly(frame_size)
            self._log_raw(b"F", header + size_bytes + payload)
            return FileFramePayload(payload=payload, is_mapping_frame=True)

        frame_size = int.from_bytes(header, byteorder="big")
        if frame_size <= 0:
            raise FrameError(f"invalid file frame size: {frame_size}")
        if frame_size > self.max_frame_size:
            raise FrameError(f"file frame exceeds max size: {frame_size}")
        payload = await self._reader.readexactly(frame_size)
        self._log_raw(b"f", header + payload)
        return FileFramePayload(payload=payload, is_mapping_frame=False)


async def probe_tcp_endpoint(host: str, port: int, timeout_s: float = 3.0, read_bytes: int = 1024) -> bytes:
    """Function: Connect to a TCP endpoint and read initial data without sending anything.
    
    Args:
        host: TCP service host address.
        port: TCP service port.
        timeout_s: Connect and read timeout in seconds.
        read_bytes: Maximum number of bytes to read.
    
    Returns:
        Initial bytes read from the endpoint; empty bytes when no data is read before timeout."""

    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_s)
    try:
        try:
            return await asyncio.wait_for(reader.read(read_bytes), timeout=timeout_s)
        except asyncio.TimeoutError:
            return b""
    finally:
        writer.close()
        await writer.wait_closed()
