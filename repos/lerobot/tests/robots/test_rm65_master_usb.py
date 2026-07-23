#!/usr/bin/env python

"""Probe the RealMan master arm over USB without moving the RM65 follower arm."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import serial

QUERY = bytes.fromhex("55 AA 02 00 00 67")
MIN_RESPONSE_BYTES = 41
JOINT_OFFSETS = (7, 12, 17, 22, 27, 32)
GRIPPER_OFFSET = 37


def read_response(ser: serial.Serial, timeout_s: float) -> bytes:
    """Read one query response, waiting briefly after the first received byte."""
    deadline = time.monotonic() + timeout_s
    last_data_time = None
    response = bytearray()

    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            response.extend(ser.read(waiting))
            last_data_time = time.monotonic()
        elif last_data_time is not None and time.monotonic() - last_data_time >= 0.003:
            break
        else:
            time.sleep(0.001)

    return bytes(response)


def extract_packet(response: bytes) -> bytes:
    """Return the first complete vendor packet beginning with ``55 AA``."""
    header_index = response.find(b"\x55\xaa")
    if header_index < 0:
        raise ValueError("response does not contain the 55 AA header")
    packet = response[header_index:]
    if len(packet) < MIN_RESPONSE_BYTES:
        raise ValueError(
            f"incomplete response: expected at least {MIN_RESPONSE_BYTES} bytes, got {len(packet)}"
        )
    return packet


def decode_master_state(packet: bytes) -> np.ndarray:
    """Decode six joint angles and the gripper using the RealMan reference offsets."""
    joints = [
        int.from_bytes(packet[offset : offset + 4], byteorder="little", signed=True) / 10000.0
        for offset in JOINT_OFFSETS
    ]
    gripper = int.from_bytes(
        packet[GRIPPER_OFFSET : GRIPPER_OFFSET + 4],
        byteorder="little",
        signed=True,
    )
    return np.asarray([*joints, gripper], dtype=np.float64)


def configure_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def probe(args: argparse.Namespace) -> int:
    if not os.path.exists(args.port):
        logging.error("Serial device does not exist: %s", args.port)
        return 2
    if not os.access(args.port, os.R_OK | os.W_OK):
        logging.error(
            "No read/write permission for %s. Add the user to the device group or install a udev rule.",
            args.port,
        )
        return 2

    states = []
    errors = 0
    interval_s = 1.0 / args.hz
    logging.info("Opening %s at %d baud.", args.port, args.baudrate)
    logging.info("Move every master-arm joint and open/close the gripper for %.1f seconds.", args.duration_s)
    logging.info("This test does not connect to or move the RM65 follower arm.")

    with serial.Serial(args.port, args.baudrate, timeout=0) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(args.startup_delay_s)
        deadline = time.monotonic() + args.duration_s
        next_sample = time.monotonic()

        while time.monotonic() < deadline:
            next_sample += interval_s
            ser.write(QUERY)
            ser.flush()
            response = read_response(ser, args.response_timeout_s)
            try:
                packet = extract_packet(response)
                state = decode_master_state(packet)
            except ValueError as exc:
                errors += 1
                raw = response.hex(" ").upper() if response else "<empty>"
                logging.warning("Invalid response: %s; raw=%s", exc, raw)
            else:
                states.append(state)
                logging.info(
                    "joints_deg=%s gripper=%d bytes=%d%s",
                    np.round(state[:6], 4).tolist(),
                    int(state[6]),
                    len(response),
                    f" raw={response.hex(' ').upper()}" if args.show_raw else "",
                )

            sleep_s = next_sample - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    if not states:
        logging.error("No valid master-arm packets were decoded. Invalid responses: %d.", errors)
        return 1

    values = np.stack(states)
    ranges = values.max(axis=0) - values.min(axis=0)
    changed = [*list(ranges[:6] >= args.min_joint_change_deg), ranges[6] >= args.min_gripper_change]
    logging.info("Decoded %d valid packets; invalid responses=%d.", len(states), errors)
    logging.info(
        "Observed ranges: joints_deg=%s gripper=%.0f",
        np.round(ranges[:6], 4).tolist(),
        ranges[6],
    )
    logging.info("Changed channels: J1..J6,gripper=%s", changed)

    if all(changed):
        logging.info("PASS: all six joints and the gripper produced changing data.")
        return 0

    logging.error("FAIL: move every channel through a larger range and run the probe again.")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=460800)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--startup-delay-s", type=float, default=1.0)
    parser.add_argument("--response-timeout-s", type=float, default=0.05)
    parser.add_argument("--min-joint-change-deg", type=float, default=0.1)
    parser.add_argument("--min-gripper-change", type=float, default=5.0)
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--log-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_path)
    raise SystemExit(probe(args))


if __name__ == "__main__":
    main()
