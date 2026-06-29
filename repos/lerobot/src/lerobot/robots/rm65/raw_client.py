#!/usr/bin/env python

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from json import JSONDecodeError
from typing import Any


class RM65RawClient:
    """Small JSON/TCP client for the RealMan RM65 controller."""

    def __init__(self, ip: str, port: int, timeout_s: float = 3.0) -> None:
        self.ip = ip
        self.port = port
        self.timeout_s = timeout_s
        self._request_sock: socket.socket | None = None
        self._arm_sock: socket.socket | None = None
        self._gripper_sock: socket.socket | None = None
        self._request_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._drain_threads: list[threading.Thread] = []

    @property
    def is_connected(self) -> bool:
        return self._request_sock is not None

    def connect(self) -> None:
        self._stop_event.clear()
        self._request_sock = self._connect_socket()
        self._arm_sock = self._connect_socket()
        self._gripper_sock = self._connect_socket()
        self._drain_threads = [
            self._start_drain_thread(self._arm_sock),
            self._start_drain_thread(self._gripper_sock),
        ]

    def close(self) -> None:
        self._stop_event.set()
        for sock in (self._request_sock, self._arm_sock, self._gripper_sock):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        for thread in self._drain_threads:
            thread.join(timeout=0.5)
        self._request_sock = None
        self._arm_sock = None
        self._gripper_sock = None
        self._drain_threads = []

    def _connect_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect((self.ip, self.port))
        return sock

    def _start_drain_thread(self, sock: socket.socket) -> threading.Thread:
        thread = threading.Thread(target=self._drain_socket, args=(sock,), daemon=True)
        thread.start()
        return thread

    def _drain_socket(self, sock: socket.socket) -> None:
        while not self._stop_event.is_set():
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break

    @staticmethod
    def _encode(command: dict[str, Any]) -> bytes:
        return (json.dumps(command, separators=(",", ":")) + "\r\n").encode("utf-8")

    @staticmethod
    def _decode_response(buffer: bytearray) -> dict[str, Any] | None:
        raw_text = buffer.decode("utf-8", errors="ignore")
        text = raw_text.lstrip()
        if not text:
            return None
        leading_chars = len(raw_text) - len(text)
        decoder = json.JSONDecoder()
        try:
            obj, end = decoder.raw_decode(text)
        except JSONDecodeError:
            return None
        del buffer[: len(raw_text[: leading_chars + end].encode("utf-8"))]
        if not isinstance(obj, dict):
            raise RuntimeError(f"Expected JSON object from RM65, got {obj!r}")
        return obj

    def request(
        self,
        command: dict[str, Any],
        expected: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        if self._request_sock is None:
            raise RuntimeError("RM65RawClient is not connected")

        with self._request_lock:
            sock = self._request_sock
            sock.sendall(self._encode(command))
            buffer = bytearray()
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("RM65 TCP connection closed while waiting for response")
                buffer.extend(chunk)
                while True:
                    response = self._decode_response(buffer)
                    if response is None:
                        break
                    if expected is None or expected(response):
                        return response

    def send_arm_command(self, command: dict[str, Any]) -> None:
        if self._arm_sock is None:
            raise RuntimeError("RM65RawClient arm command socket is not connected")
        self._arm_sock.sendall(self._encode(command))

    def send_gripper_command(self, command: dict[str, Any]) -> None:
        if self._gripper_sock is None:
            raise RuntimeError("RM65RawClient gripper command socket is not connected")
        self._gripper_sock.sendall(self._encode(command))

    @staticmethod
    def _deg_to_mdeg(values: list[float]) -> list[int]:
        return [int(round(value * 1000.0)) for value in values]

    @staticmethod
    def _mdeg_to_deg(values: list[int | float]) -> list[float]:
        return [float(value) / 1000.0 for value in values]

    def get_current_arm_state(self) -> dict[str, Any]:
        response = self.request(
            {"command": "get_current_arm_state"},
            expected=lambda data: data.get("state") == "current_arm_state"
            and isinstance(data.get("arm_state"), dict),
        )
        state = response.get("arm_state")
        if not isinstance(state, dict):
            raise RuntimeError(f"RM65 current arm state response did not include arm_state: {response}")
        return state

    def get_joint_positions(self) -> list[float]:
        state = self.get_current_arm_state()
        joints = state.get("joint")
        if not isinstance(joints, list):
            raise RuntimeError(f"RM65 arm state did not include joint list: {state}")
        return self._mdeg_to_deg(joints)

    def get_rm_plus_state_info(self) -> dict[str, Any]:
        response = self.request(
            {"command": "get_rm_plus_state_info"},
            expected=lambda data: data.get("command") == "get_rm_plus_state_info",
        )
        state = response.get("state_info")
        if not isinstance(state, dict):
            raise RuntimeError(f"RM65 RM Plus response did not include state_info: {response}")
        return state

    def movej_canfd(
        self,
        joint_degrees: list[float],
        follow: bool,
        expand: int = 1000,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> None:
        self.send_arm_command(
            {
                "command": "movej_canfd",
                "joint": self._deg_to_mdeg(joint_degrees),
                "follow": bool(follow),
                "expand": int(expand),
                "trajectory_mode": int(trajectory_mode),
                "radio": int(radio),
            }
        )

    def movej(
        self,
        joint_degrees: list[float],
        speed: int,
        blend_radius: int,
        trajectory_connect: int,
    ) -> None:
        self.send_arm_command(
            {
                "command": "movej",
                "joint": self._deg_to_mdeg(joint_degrees),
                "v": int(speed),
                "r": int(blend_radius),
                "trajectory_connect": int(trajectory_connect),
            }
        )

    def set_gripper_route(self, min_position: int, max_position: int) -> None:
        response = self.request(
            {"command": "set_gripper_route", "min": int(min_position), "max": int(max_position)},
            expected=lambda data: data.get("command") == "set_gripper_route",
        )
        if response.get("set_state") is False:
            raise RuntimeError(f"RM65 set_gripper_route failed: {response}")

    def set_gripper_position(self, position: int, block: bool = False) -> None:
        self.send_gripper_command(
            {"command": "set_gripper_position", "position": int(position), "block": bool(block)}
        )

    def set_arm_stop(self) -> None:
        response = self.request(
            {"command": "set_arm_stop"},
            expected=lambda data: data.get("command") == "set_arm_stop",
        )
        if response.get("arm_stop") is False:
            raise RuntimeError(f"RM65 set_arm_stop failed: {response}")

    def set_joint_clear_err(self, joint_num: int) -> None:
        response = self.request(
            {"command": "set_joint_clear_err", "joint_clear_err": int(joint_num)},
            expected=lambda data: data.get("command") == "set_joint_clear_err",
        )
        if response.get("joint_clear_err") is not True:
            raise RuntimeError(f"RM65 set_joint_clear_err({joint_num}) failed: {response}")

    def set_io_mode(self, io_num: int, io_mode: int) -> None:
        response = self.request(
            {"command": "set_IO_mode", "IO_Num": int(io_num), "IO_mode": int(io_mode)},
            expected=lambda data: data.get("command") == "set_IO_mode",
        )
        if response.get("set_state") is False:
            raise RuntimeError(f"RM65 set_IO_mode({io_num}, {io_mode}) failed: {response}")

    def get_io_state(self, io_num: int) -> dict[str, Any]:
        response = self.request(
            {"command": "get_IO_state", "IO_Num": int(io_num)},
            expected=lambda data: data.get("state") == "IO_state",
        )
        if response.get("state") != "IO_state":
            raise RuntimeError(f"RM65 get_IO_state({io_num}) failed: {response}")
        return response

    def get_io_input(self) -> list[int]:
        response = self.request(
            {"command": "get_IO_input"},
            expected=lambda data: data.get("state") == "IO_input_state",
        )
        values = response.get("DI")
        if not isinstance(values, list):
            raise RuntimeError(f"RM65 get_IO_input response did not include DI list: {response}")
        return [int(value) for value in values]

    def get_io_output(self) -> list[int]:
        response = self.request(
            {"command": "get_IO_output"},
            expected=lambda data: data.get("state") == "IO_output_state",
        )
        values = response.get("DO")
        if not isinstance(values, list):
            raise RuntimeError(f"RM65 get_IO_output response did not include DO list: {response}")
        return [int(value) for value in values]

    def get_tool_io_state(self) -> dict[str, Any]:
        response = self.request(
            {"command": "get_tool_IO_state"},
            expected=lambda data: data.get("state") == "tool_IO_state",
        )
        if response.get("state") != "tool_IO_state":
            raise RuntimeError(f"RM65 get_tool_IO_state failed: {response}")
        return response

    def set_realtime_push(
        self,
        enable: bool,
        target_ip: str,
        port: int,
        cycle: int,
        custom: dict[str, bool],
        force_coordinate: int | None = 2,
    ) -> dict[str, Any]:
        command: dict[str, Any] = {
            "command": "set_realtime_push",
            "cycle": int(cycle),
            "enable": bool(enable),
            "port": int(port),
            "ip": target_ip,
            "custom": custom,
        }
        if force_coordinate is not None:
            command["force_coordinate"] = int(force_coordinate)

        response = self.request(
            command,
            expected=lambda data: data.get("command") == "set_realtime_push",
        )
        if response.get("state") is False:
            raise RuntimeError(f"RM65 set_realtime_push failed: {response}")
        return response
