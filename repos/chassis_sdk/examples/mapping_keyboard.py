"""Qt mapping demo for the PLNS Python SDK.

This example provides a graphical mapping tool built on ``PLNSClient``. It can
connect to the chassis, query common device states, switch operation modes,
send hold-to-move manual velocity commands, control emergency stop, navigate to
a station, start/save/cancel mapping, and display live mapping data from the
PLNS File channel.

Install UI dependencies before running:
    pip install .[ui]

Keyboard shortcuts:
- Hold Arrow Up/Down: move forward/backward.
- Hold Arrow Left/Right: rotate left/right.
- Space: send zero velocity.
- s: stop mapping and save.
- q: cancel mapping.

Mapping behavior:
- Starting mapping prepares manual-control mode and stops localization.
- During mapping, the SDK receives File-channel map-data notifications,
  requests mapping point-cloud data, decodes it, and displays the live map
  points and mapping track in the UI.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from qasync import QEventLoop
except ImportError as exc:  # pragma: no cover - depends on optional UI deps.
    raise SystemExit(
        "Qt UI dependencies are missing. Install them with: pip install .[ui]"
    ) from exc

from google.protobuf.json_format import MessageToDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chassis_sdk import PLNSClient, PLNSConfig


Point = tuple[float, float]


def format_proto(message: Any) -> str:
    if message is None:
        return "OK"
    try:
        data = MessageToDict(message, preserving_proto_field_name=True)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return str(message)


def enum_name(owner: object, value: int) -> str:
    try:
        return f"{owner.Name(value)}({value})"  # type: ignore[attr-defined]
    except Exception:
        return str(value)


def response_name(client: PLNSClient, response: Any) -> str:
    if response is None:
        return "OK"
    try:
        response_type = response.response.type
        return enum_name(client.pb2.Response.ResponseType, response_type)
    except Exception:
        return response.__class__.__name__


def tail_sample(points: list[Point], limit: int) -> list[Point]:
    if limit <= 0 or len(points) <= limit:
        return points
    step = max(1, math.ceil(len(points) / limit))
    return points[::step][-limit:]


class MapWidget(QtWidgets.QWidget):
    def __init__(self, args: argparse.Namespace, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.args = args
        self.cloud_points: list[Point] = []
        self.track_points: list[Point] = []
        self.laser_points: list[Point] = []
        self.pose: tuple[float, float, float] | None = None
        self.setMinimumSize(700, 520)
        self.setAutoFillBackground(False)

    def set_data(
        self,
        *,
        cloud_points: list[Point] | None = None,
        track_points: list[Point] | None = None,
        laser_points: list[Point] | None = None,
        pose: tuple[float, float, float] | None | object = None,
    ) -> None:
        if cloud_points is not None:
            self.cloud_points = cloud_points
        if track_points is not None:
            self.track_points = track_points
        if laser_points is not None:
            self.laser_points = laser_points
        if pose is not None:
            self.pose = pose if isinstance(pose, tuple) else None
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#0f1720"))
        self._draw_grid(painter)

        cloud = tail_sample(self.cloud_points, self.args.max_map_points)
        track = tail_sample(self.track_points, self.args.max_track_points)
        laser = tail_sample(self.laser_points, self.args.max_laser_points)
        pose_points = [(self.pose[0], self.pose[1])] if self.pose else []
        transform = self._make_transform(cloud + track + laser + pose_points)

        self._draw_points(painter, cloud, transform, QtGui.QColor("#4ecdc4"), 2)
        self._draw_points(painter, laser, transform, QtGui.QColor("#70a5ff"), 3)
        self._draw_track(painter, track, transform)
        if self.pose:
            self._draw_pose(painter, self.pose, transform)

        if not (cloud or track or laser or pose_points):
            painter.setPen(QtGui.QColor("#8ea0b8"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No map data yet")

    def _draw_grid(self, painter: QtGui.QPainter) -> None:
        pen = QtGui.QPen(QtGui.QColor("#172231"))
        pen.setWidth(1)
        painter.setPen(pen)
        for x in range(0, self.width(), 50):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 50):
            painter.drawLine(0, y, self.width(), y)

    def _make_transform(self, points: list[Point]):
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
        else:
            min_x, max_x, min_y, max_y = -5.0, 5.0, -5.0, 5.0

        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        margin = 40.0
        scale = min((self.width() - margin * 2) / span_x, (self.height() - margin * 2) / span_y)
        if not math.isfinite(scale) or scale <= 0:
            scale = 40.0
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        def transform(point: Point) -> QtCore.QPointF:
            x, y = point
            return QtCore.QPointF(
                self.width() / 2 + (x - center_x) * scale,
                self.height() / 2 - (y - center_y) * scale,
            )

        return transform

    def _draw_points(
        self,
        painter: QtGui.QPainter,
        points: list[Point],
        transform: Any,
        color: QtGui.QColor,
        size: int,
    ) -> None:
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        for point in points:
            screen = transform(point)
            painter.drawEllipse(screen, size, size)

    def _draw_track(self, painter: QtGui.QPainter, points: list[Point], transform: Any) -> None:
        if not points:
            return
        pen = QtGui.QPen(QtGui.QColor("#ffd166"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        if len(points) == 1:
            screen = transform(points[0])
            painter.drawEllipse(screen, 4, 4)
            return
        path = QtGui.QPainterPath(transform(points[0]))
        for point in points[1:]:
            path.lineTo(transform(point))
        painter.drawPath(path)

    def _draw_pose(
        self,
        painter: QtGui.QPainter,
        pose: tuple[float, float, float],
        transform: Any,
    ) -> None:
        screen = transform((pose[0], pose[1]))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#ff6b6b"))
        painter.drawEllipse(screen, 8, 8)

        pen = QtGui.QPen(QtGui.QColor("#ff6b6b"))
        pen.setWidth(3)
        painter.setPen(pen)
        yaw = pose[2]
        painter.drawLine(
            screen,
            QtCore.QPointF(screen.x() + math.cos(yaw) * 28, screen.y() - math.sin(yaw) * 28),
        )


class MappingWindow(QtWidgets.QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.client: PLNSClient | None = None
        self.current_velocity = (0, 0, 0)
        self.current_pose: tuple[float, float, float] | None = None
        self.mapping_updates = 0
        self.track_updates = 0
        self.tasks: set[asyncio.Task[Any]] = set()
        self.closing = False
        self.active_motion_key: int | None = None

        self.setWindowTitle("PLNS Mapping Demo")
        self.resize(args.width, args.height)
        self._build_ui()
        self._bind_shortcuts()
        QtWidgets.QApplication.instance().installEventFilter(self)

        self.sample_timer = QtCore.QTimer(self)
        self.sample_timer.setInterval(max(100, int(args.sample_period * 1000)))
        self.sample_timer.timeout.connect(lambda: self.create_task(self.sample_state()))
        self.sample_timer.start()

        self.velocity_timer = QtCore.QTimer(self)
        self.velocity_timer.setInterval(max(50, int(args.velocity_period * 1000)))
        self.velocity_timer.timeout.connect(lambda: self.create_task(self.resend_velocity()))
        self.velocity_timer.start()

        self.mapping_timer = QtCore.QTimer(self)
        self.mapping_timer.setInterval(100)
        self.mapping_timer.timeout.connect(self.refresh_mapping_buffers)
        self.mapping_timer.start()

        self.set_controls_enabled(False)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QGridLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setHorizontalSpacing(10)
        root.setVerticalSpacing(8)
        root.setColumnStretch(1, 1)
        root.setRowStretch(1, 1)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top, 0, 0, 1, 2)
        self.edit_user = self._readonly_line("admin")
        self.edit_password = self._readonly_line("admin")
        self.edit_ip = QtWidgets.QLineEdit(self.args.host)
        self.edit_port = QtWidgets.QLineEdit(str(self.args.port))
        top.addWidget(QtWidgets.QLabel("用户名"))
        top.addWidget(self.edit_user)
        top.addWidget(QtWidgets.QLabel("密码"))
        top.addWidget(self.edit_password)
        top.addWidget(QtWidgets.QLabel("IP"))
        top.addWidget(self.edit_ip)
        top.addWidget(QtWidgets.QLabel("端口"))
        top.addWidget(self.edit_port)
        top.addStretch(1)
        self.button_connect = QtWidgets.QPushButton("连接")
        self.button_connect.clicked.connect(lambda: self.create_task(self.connect_or_logout()))
        top.addWidget(self.button_connect)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(6)
        controls_layout.setVerticalSpacing(6)
        root.addWidget(controls, 1, 0)

        row = 0
        row = self._add_section(controls_layout, "状态查询", row)
        self._add_button(controls_layout, "获取车辆信息", self.request_info, row, 0)
        self._add_button(controls_layout, "获取系统状态", self.request_system_state, row, 1)
        row += 1
        self._add_button(controls_layout, "获取硬件状态", self.request_hardware_state, row, 0)
        self._add_button(controls_layout, "获取地图列表", self.request_map_list, row, 1)
        row += 1
        self._add_button(controls_layout, "获取雷达点云", self.request_laser_points, row, 0, 2)
        row += 1

        row = self._add_section(controls_layout, "控制模式", row)
        self._add_button(controls_layout, "手动", self.enable_manual, row, 0)
        self._add_button(controls_layout, "自动", self.disable_manual, row, 1)
        row += 1

        row = self._add_section(controls_layout, "手动运动", row)
        self._add_hold_button(controls_layout, "前进", self.args.linear_speed, 0, 0, row, 0)
        self._add_hold_button(controls_layout, "后退", -self.args.linear_speed, 0, 0, row, 1)
        row += 1
        self._add_hold_button(controls_layout, "左转", 0, 0, self.args.angular_speed, row, 0)
        self._add_hold_button(controls_layout, "右转", 0, 0, -self.args.angular_speed, row, 1)
        row += 1
        self._add_button(controls_layout, "停止", lambda: self.set_velocity(0, 0, 0), row, 0, 2)
        row += 1

        row = self._add_section(controls_layout, "急停", row)
        self._add_button(controls_layout, "急停", self.trigger_emergency, row, 0)
        self._add_button(controls_layout, "解除", self.cancel_emergency, row, 1)
        row += 1

        row = self._add_section(controls_layout, "导航", row)
        self.edit_station = QtWidgets.QLineEdit(str(self.args.station_id))
        controls_layout.addWidget(self.edit_station, row, 0)
        self._add_button(controls_layout, "导航到站点", self.go_station, row, 1)
        row += 1

        self.button_path_left = self._add_disabled_path_button(controls_layout, "左转20°", row, 0)
        self.button_path_right = self._add_disabled_path_button(controls_layout, "右转20°", row, 1)
        row += 1
        self.button_path_back = self._add_disabled_path_button(controls_layout, "后退0.2", row, 0)
        self.button_path_front = self._add_disabled_path_button(controls_layout, "前进0.2", row, 1)
        row += 1

        row = self._add_section(controls_layout, "建图", row)
        self._add_button(controls_layout, "关闭定位", self.stop_location, row, 0, 2)
        row += 1
        self._add_button(controls_layout, "启动建图", self.start_mapping, row, 0, 2)
        row += 1
        self._add_button(controls_layout, "取消建图", self.cancel_mapping, row, 0)
        self._add_button(controls_layout, "停止建图", self.stop_mapping, row, 1)
        row += 1
        controls_layout.setRowStretch(row, 1)

        self.map_widget = MapWidget(self.args)
        root.addWidget(self.map_widget, 1, 1)

        bottom = QtWidgets.QGridLayout()
        root.addLayout(bottom, 2, 0, 1, 2)
        self.text_send = self._text_box()
        self.text_recv = self._text_box()
        self.text_log = self._text_box()
        self._add_text_panel(bottom, "发送", self.text_send, 0)
        self._add_text_panel(bottom, "接收", self.text_recv, 1)
        self._add_text_panel(bottom, "日志", self.text_log, 2)

        self.label_status = QtWidgets.QLabel("disconnected")
        self.statusBar().addPermanentWidget(self.label_status, 1)

    def _readonly_line(self, value: str) -> QtWidgets.QLineEdit:
        line = QtWidgets.QLineEdit(value)
        line.setReadOnly(True)
        line.setEnabled(False)
        return line

    def _add_section(self, layout: QtWidgets.QGridLayout, text: str, row: int) -> int:
        label = QtWidgets.QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addWidget(label, row, 0, 1, 2)
        return row + 1

    def _add_button(
        self,
        layout: QtWidgets.QGridLayout,
        text: str,
        handler: Any,
        row: int,
        col: int,
        colspan: int = 1,
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        if asyncio.iscoroutinefunction(handler):
            button.clicked.connect(lambda _checked=False, h=handler: self.create_task(h()))
        else:
            button.clicked.connect(lambda _checked=False, h=handler: h())
        layout.addWidget(button, row, col, 1, colspan)
        return button

    def _add_hold_button(
        self,
        layout: QtWidgets.QGridLayout,
        text: str,
        vx: int,
        vy: int,
        w: int,
        row: int,
        col: int,
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.pressed.connect(lambda: self.set_velocity(vx, vy, w))
        button.released.connect(lambda: self.set_velocity(0, 0, 0))
        layout.addWidget(button, row, col)
        return button

    def _add_disabled_path_button(
        self,
        layout: QtWidgets.QGridLayout,
        text: str,
        row: int,
        col: int,
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setEnabled(False)
        button.setToolTip("MT_MOVE_FOLLOW_PATH is not exposed by the customer SDK.")
        layout.addWidget(button, row, col)
        return button

    def _text_box(self) -> QtWidgets.QTextEdit:
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setMinimumHeight(150)
        return text

    def _add_text_panel(
        self,
        layout: QtWidgets.QGridLayout,
        title: str,
        text: QtWidgets.QTextEdit,
        col: int,
    ) -> None:
        frame = QtWidgets.QWidget()
        frame_layout = QtWidgets.QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(QtWidgets.QLabel(title))
        frame_layout.addWidget(text)
        layout.addWidget(frame, 0, col)
        layout.setColumnStretch(col, 1)

    def _bind_shortcuts(self) -> None:
        shortcuts = [
            ("S", lambda: self.create_task(self.stop_mapping())),
            ("Q", lambda: self.create_task(self.cancel_mapping())),
        ]
        for key, handler in shortcuts:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(handler)

    def eventFilter(self, obj: object, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease):
            key_event = event  # type: ignore[assignment]
            key = key_event.key()  # type: ignore[attr-defined]
            velocity = self.velocity_for_key(key)
            if velocity is not None:
                if key_event.isAutoRepeat():  # type: ignore[attr-defined]
                    return True
                if event.type() == QtCore.QEvent.KeyPress:
                    self.active_motion_key = key
                    self.set_velocity(*velocity)
                elif self.active_motion_key == key:
                    self.active_motion_key = None
                    self.set_velocity(0, 0, 0)
                return True
            if event.type() == QtCore.QEvent.KeyPress and key == QtCore.Qt.Key_Space:
                if not key_event.isAutoRepeat():  # type: ignore[attr-defined]
                    self.active_motion_key = None
                    self.set_velocity(0, 0, 0)
                return True
        return super().eventFilter(obj, event)

    def velocity_for_key(self, key: int) -> tuple[int, int, int] | None:
        if key == QtCore.Qt.Key_Up:
            return (self.args.linear_speed, 0, 0)
        if key == QtCore.Qt.Key_Down:
            return (-self.args.linear_speed, 0, 0)
        if key == QtCore.Qt.Key_Left:
            return (0, 0, self.args.angular_speed)
        if key == QtCore.Qt.Key_Right:
            return (0, 0, -self.args.angular_speed)
        return None

    def set_controls_enabled(self, enabled: bool) -> None:
        for button in self.findChildren(QtWidgets.QPushButton):
            if button is self.button_connect:
                continue
            if button in (
                self.button_path_left,
                self.button_path_right,
                self.button_path_back,
                self.button_path_front,
            ):
                button.setEnabled(False)
                continue
            button.setEnabled(enabled)
        self.edit_station.setEnabled(enabled)

    def create_task(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self.tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.log(f"error: {exc}")
            self.label_status.setText(f"error: {exc}")

    def log(self, text: str) -> None:
        self.text_log.insertPlainText(f"{time.strftime('%H:%M:%S')} {text.strip()}\n")
        self.text_log.moveCursor(QtGui.QTextCursor.End)

    def set_send(self, text: str) -> None:
        self.text_send.setPlainText(text)
        self.log(f"send {text.splitlines()[0]}")

    def set_recv(self, text: str) -> None:
        self.text_recv.setPlainText(text)

    def require_client(self) -> PLNSClient:
        if not self.client or not self.client.is_logged_in:
            raise RuntimeError("not connected")
        return self.client

    async def connect_or_logout(self) -> None:
        if self.client and self.client.is_logged_in:
            await self.logout()
        else:
            await self.connect()

    async def connect(self) -> None:
        if self.client and self.client.is_logged_in:
            return
        raw_log_path = Path(self.args.raw_log_path)
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        raw_log_path.write_bytes(b"")

        config = PLNSConfig.from_env()
        config = replace(config, host=self.edit_ip.text().strip(), port=int(self.edit_port.text()))
        self.client = PLNSClient(config=config, raw_log_path=raw_log_path)
        self.set_send("REQUEST_LOGIN + File REQUEST_LOGIN")
        self.label_status.setText("connecting")
        await self.client.connect(login=True)
        self.button_connect.setText("退出")
        self.set_controls_enabled(True)
        self.label_status.setText(
            f"connected session={self.client.session_id} file={'ok' if self.client.is_file_logged_in else '-'}"
        )
        self.set_recv("login ok")
        self.log("login ok")

    async def logout(self) -> None:
        if not self.client:
            return
        self.set_send("REQUEST_LOGOUT")
        await self.client.close()
        self.client = None
        self.current_velocity = (0, 0, 0)
        self.button_connect.setText("连接")
        self.set_controls_enabled(False)
        self.label_status.setText("disconnected")
        self.set_recv("logout ok")
        self.log("logout ok")

    async def sdk_call(self, send_text: str, func: Any, *args: Any) -> Any:
        client = self.require_client()
        self.set_send(send_text)
        result = await func(*args)
        self.set_recv(f"{response_name(client, result)}\n{format_proto(result)}")
        return result

    async def request_info(self) -> None:
        await self.sdk_call("REQUEST_INFO", self.require_client().get_info)

    async def request_system_state(self) -> None:
        result = await self.sdk_call("REQUEST_SYSTEM_STATE", self.require_client().get_system_state)
        self.update_pose_from_system_state(result)

    async def request_hardware_state(self) -> None:
        await self.sdk_call("REQUEST_HARDWARE_STATE", self.require_client().get_hardware_state)

    async def request_map_list(self) -> None:
        await self.sdk_call("REQUEST_MAP_LIST", self.require_client().get_map_list)

    async def request_laser_points(self) -> None:
        result = await self.sdk_call("REQUEST_LASER_POINTS", self.require_client().get_laser_points)
        self.map_widget.set_data(laser_points=self.laser_points_from_response(result))

    async def enable_manual(self) -> None:
        await self.sdk_call("CMD_ENABLE_MANUAL_CONTROL", self.require_client().prepare_manual_control)

    async def disable_manual(self) -> None:
        await self.sdk_call("CMD_DISABLE_MANUAL_CONTROL", self.require_client().disable_manual)

    async def trigger_emergency(self) -> None:
        await self.sdk_call("CMD_TRIGGER_EMERGENCY", self.require_client().trigger_emergency)

    async def cancel_emergency(self) -> None:
        await self.sdk_call("CMD_CANCEL_EMERGENCY", self.require_client().cancel_emergency)

    async def go_station(self) -> None:
        station_id = int(self.edit_station.text())
        await self.sdk_call(
            f"CMD_NEW_MOVEMENT_TASK MT_MOVE_TO_STATION station={station_id}",
            self.require_client().go_station,
            station_id,
        )

    async def stop_location(self) -> None:
        await self.sdk_call("CMD_STOP_LOCATION", self.require_client().stop_localization)

    async def start_mapping(self) -> None:
        await self.sdk_call("CMD_NEW_MAP_START", self.require_client().start_mapping)
        self.label_status.setText("mapping")

    async def stop_mapping(self) -> None:
        if self.client and self.client.is_logged_in:
            await self.send_velocity_once(0, 0, 0)
            await self.sdk_call("CMD_NEW_MAP_STOP", self.client.stop_mapping)
            self.label_status.setText("connected")

    async def cancel_mapping(self) -> None:
        if self.client and self.client.is_logged_in:
            await self.send_velocity_once(0, 0, 0)
            await self.sdk_call("CMD_NEW_MAP_CANCEL", self.client.cancel_mapping)
            self.label_status.setText("connected")

    def set_velocity(self, vx: int, vy: int, w: int) -> None:
        self.current_velocity = (vx, vy, w)
        self.create_task(self.send_velocity_once(vx, vy, w))

    async def send_velocity_once(self, vx: int, vy: int, w: int) -> None:
        if not self.client or not self.client.is_logged_in:
            return
        self.set_send(f"CMD_SET_MANUAL_SPEED vx={vx} vy={vy} w={w}")
        await self.client.set_manual_velocity(vx, w, vy, wait=False, prepare=False)

    async def resend_velocity(self) -> None:
        vx, vy, w = self.current_velocity
        if vx or vy or w:
            await self.send_velocity_once(vx, vy, w)

    async def sample_state(self) -> None:
        if not self.client or not self.client.is_logged_in:
            return
        try:
            system_state = await self.client.get_system_state()
            self.update_pose_from_system_state(system_state)
            if self.args.auto_laser:
                laser_response = await self.client.get_laser_points()
                self.map_widget.set_data(laser_points=self.laser_points_from_response(laser_response))
        except Exception as exc:
            self.log(f"sample error: {exc}")

    def refresh_mapping_buffers(self) -> None:
        if not self.client:
            return
        changed = False
        while True:
            try:
                self.client.mapping_cloud_updates.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.mapping_updates += 1
            cloud = [(float(x), float(y)) for x, y, _z in self.client.mapping_cloud_points]
            self.map_widget.set_data(cloud_points=cloud)
            self.set_recv(
                "RESPONSE_GET_MAPPING_DATA\n"
                f"cloud points: {len(cloud)}\n"
                f"updates: {self.mapping_updates}"
            )
            changed = True

        while True:
            try:
                self.client.mapping_track_updates.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.track_updates += 1
            track = [(float(x), float(y)) for x, y, _z in self.client.mapping_track_points]
            self.map_widget.set_data(track_points=track)
            self.set_recv(
                "RESPONSE_GET_MAPPING_TRACK\n"
                f"track points: {len(track)}\n"
                f"updates: {self.track_updates}"
            )
            changed = True

        if changed:
            self.log(
                "mapping data updated: cloud={} track={}".format(
                    len(self.map_widget.cloud_points),
                    len(self.map_widget.track_points),
                )
            )

    def update_pose_from_system_state(self, response: Any) -> None:
        try:
            system_state = response.response.system_state
            pose = system_state.location_pose
            self.current_pose = (pose.x / 1000.0, pose.y / 1000.0, pose.yaw / 1000.0)
            self.map_widget.set_data(pose=self.current_pose)
            if self.client:
                self.label_status.setText(
                    "session={} file={} map={} progress={} station={}".format(
                        self.client.session_id,
                        "ok" if self.client.is_file_logged_in else "-",
                        system_state.map_name,
                        system_state.map_saving_progress,
                        system_state.station_no,
                    )
                )
        except Exception:
            return

    def laser_points_from_response(self, response: Any) -> list[Point]:
        try:
            laser = response.response.laser_points
        except Exception:
            return []
        points = [(x / 1000.0, y / 1000.0) for x, y in zip(laser.xs, laser.ys)]
        if not self.current_pose:
            return points
        px, py, yaw = self.current_pose
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return [
            (px + x * cos_yaw - y * sin_yaw, py + x * sin_yaw + y * cos_yaw)
            for x, y in points
        ]

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.closing:
            event.accept()
            return
        self.closing = True
        event.ignore()
        self.create_task(self.close_async())

    async def close_async(self) -> None:
        try:
            if self.client:
                await self.client.close()
        finally:
            self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.91", help="PLNS Cmd-channel IP address.")
    parser.add_argument("--port", type=int, default=5005, help="PLNS Cmd-channel TCP port.")
    parser.add_argument(
        "--raw-log-path",
        default=str(PROJECT_ROOT / "logs" / "mapping_keyboard_frames.bin"),
        help="Binary raw frame log path. Cmd and File channel frames are both recorded.",
    )
    parser.add_argument("--width", type=int, default=1500, help="Window width.")
    parser.add_argument("--height", type=int, default=1000, help="Window height.")
    parser.add_argument("--linear-speed", type=int, default=300, help="Manual linear speed in mm/s.")
    parser.add_argument("--angular-speed", type=int, default=300, help="Manual angular speed in mrad/s.")
    parser.add_argument(
        "--velocity-period",
        type=float,
        default=0.1,
        help="Manual velocity resend period in seconds. It must be less than 0.2.",
    )
    parser.add_argument("--sample-period", type=float, default=0.5, help="System-state sample period.")
    parser.add_argument("--station-id", type=int, default=1, help="Default station ID.")
    parser.add_argument("--auto-laser", action="store_true", help="Continuously refresh laser points.")
    parser.add_argument("--max-map-points", type=int, default=30000, help="Maximum map points drawn.")
    parser.add_argument("--max-track-points", type=int, default=8000, help="Maximum mapping track points drawn.")
    parser.add_argument("--max-laser-points", type=int, default=3000, help="Maximum laser points drawn.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.velocity_period >= 0.2:
        raise SystemExit("--velocity-period must be less than 0.2 seconds")

    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = MappingWindow(args)
    window.show()
    app.aboutToQuit.connect(loop.stop)
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
