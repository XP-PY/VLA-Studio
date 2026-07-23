#!/usr/bin/env python

"""Inspect the RealMan master-arm USB serial transport without moving an RM65."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import serial
from serial.tools import list_ports

QUERY = bytes.fromhex("55 AA 02 00 00 67")


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


def log_port_metadata(port: str) -> None:
    matching = [info for info in list_ports.comports() if info.device == port]
    if not matching:
        logging.warning("%s was not returned by serial.tools.list_ports.", port)
        return

    info = matching[0]
    logging.info(
        "Device metadata: device=%s description=%s hwid=%s",
        info.device,
        info.description,
        info.hwid,
    )
    logging.info(
        "USB metadata: vid=%s pid=%s serial=%s manufacturer=%s product=%s location=%s",
        f"0x{info.vid:04X}" if info.vid is not None else None,
        f"0x{info.pid:04X}" if info.pid is not None else None,
        info.serial_number,
        info.manufacturer,
        info.product,
        info.location,
    )


def read_for(ser: serial.Serial, duration_s: float) -> bytes:
    deadline = time.monotonic() + duration_s
    received = bytearray()
    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            chunk = ser.read(waiting)
            received.extend(chunk)
            logging.info(
                "RX +%.3fs: %d bytes: %s",
                duration_s - (deadline - time.monotonic()),
                len(chunk),
                chunk.hex(" ").upper(),
            )
        else:
            time.sleep(0.001)
    return bytes(received)


def log_modem_lines(ser: serial.Serial) -> None:
    try:
        logging.info(
            "Control lines: DTR=%s RTS=%s CTS=%s DSR=%s RI=%s CD=%s",
            ser.dtr,
            ser.rts,
            ser.cts,
            ser.dsr,
            ser.ri,
            ser.cd,
        )
    except (OSError, serial.SerialException) as exc:
        logging.warning("Could not read all modem-control lines: %s", exc)


def inspect_transport(args: argparse.Namespace) -> int:
    if not os.path.exists(args.port):
        logging.error("Serial device does not exist: %s", args.port)
        return 2
    if not os.access(args.port, os.R_OK | os.W_OK):
        logging.error("No read/write permission for %s.", args.port)
        return 2

    log_port_metadata(args.port)
    logging.info(
        "Opening %s: baud=%d data=8 parity=N stop=1 xonxoff=False rtscts=False dsrdtr=False",
        args.port,
        args.baudrate,
    )

    total_received = 0
    with serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0,
        write_timeout=1.0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    ) as ser:
        log_modem_lines(ser)
        logging.info(
            "Passive phase: listening for %.1fs. Move the master arm and gripper now.",
            args.listen_s,
        )
        passive_data = read_for(ser, args.listen_s)
        total_received += len(passive_data)
        logging.info("Passive phase complete: received %d bytes.", len(passive_data))

        ser.reset_input_buffer()
        logging.info(
            "Query phase: sending '%s' %d times with a %.2fs response window.",
            QUERY.hex(" ").upper(),
            args.query_count,
            args.response_timeout_s,
        )
        for index in range(args.query_count):
            written = ser.write(QUERY)
            ser.flush()
            logging.info(
                "TX query %d/%d: requested=%d written=%d bytes: %s",
                index + 1,
                args.query_count,
                len(QUERY),
                written,
                QUERY.hex(" ").upper(),
            )
            response = read_for(ser, args.response_timeout_s)
            total_received += len(response)
            if not response:
                logging.warning("RX query %d/%d: <empty>", index + 1, args.query_count)
            if index + 1 < args.query_count:
                time.sleep(args.query_interval_s)

    if total_received:
        logging.info("PASS: received %d serial bytes in total.", total_received)
        return 0

    logging.error(
        "FAIL: Linux accepted the transmitted query, but the FTDI serial port returned no bytes."
    )
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=460800)
    parser.add_argument("--listen-s", type=float, default=10.0)
    parser.add_argument("--query-count", type=int, default=5)
    parser.add_argument("--query-interval-s", type=float, default=0.5)
    parser.add_argument("--response-timeout-s", type=float, default=1.0)
    parser.add_argument("--log-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_path)
    raise SystemExit(inspect_transport(args))


if __name__ == "__main__":
    main()
