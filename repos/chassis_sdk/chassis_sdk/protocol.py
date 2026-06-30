"""Protocol helpers around the generated PLNS protobuf module."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from .models import Pose2D


class ProtocolError(RuntimeError):
    """Raised when a PLNS protobuf message cannot be built or decoded."""


def md5_string(value: str) -> str:
    """Function: Compute the UTF-8 MD5 hexadecimal digest for a string.
    
    Args:
        value: String to digest.
    
    Returns:
        Lowercase 32-character MD5 string."""

    return hashlib.md5(value.encode("utf-8")).hexdigest()


def _find_default_pb2_path() -> Path | None:
    """Function: Find the default available PLNS protobuf Python module path.
    
    Returns:
        Module path when found; otherwise None."""

    env_path = os.getenv("PLNS_PROTO_PB2")
    if env_path:
        return Path(env_path)

    packaged = Path(__file__).resolve().parent / "proto" / "PlnsProto_pb2.py"
    return packaged if packaged.exists() else None


def load_pb2(path: str | Path | None = None) -> ModuleType:
    """Function: Load the generated PLNS protobuf Python module.
    
    Args:
        path: Optional PlnsProto_pb2.py path. Loads the SDK bundled module when omitted.
    
    Returns:
        Loaded Python module object."""

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    pb2_path = Path(path) if path else _find_default_pb2_path()
    if not pb2_path or not pb2_path.exists():
        raise ProtocolError(
            "PlnsProto_pb2.py was not found; set PLNS_PROTO_PB2 or regenerate it from chassis_sdk/proto/main.proto"
        )

    module_name = "_chassis_sdk_plns_pb2"
    spec = importlib.util.spec_from_file_location(module_name, pb2_path)
    if not spec or not spec.loader:
        raise ProtocolError(f"cannot load protobuf module from {pb2_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except TypeError as exc:
        if "Descriptors cannot" not in str(exc):
            raise
        raise ProtocolError(
            "the bundled PlnsProto_pb2.py requires protobuf 3.x; install 'protobuf<4' "
            "or regenerate from chassis_sdk/proto/main.proto"
        ) from exc
    return module


def enum_value(owner: Any, name: str) -> int:
    """Function: Read an enum value from a protobuf enum container, including valid zero-valued enums.
    
    Args:
        owner: Protobuf-generated enum container.
        name: Enum member name.
    
    Returns:
        Integer value of the enum member."""

    if not hasattr(owner, name):
        raise ProtocolError(f"unknown enum value {name!r} on {owner!r}")
    return int(getattr(owner, name))


@dataclass
class MessageFactory:
    """Factory for PLNS request and command messages."""

    pb2: ModuleType

    def request(self, request_name: str, seq: int, session_id: int = 0) -> Any:
        """Function: Build a PLNS Request message.
        
        Args:
            request_name: RequestType name defined in main.proto.
            seq: Message sequence number.
            session_id: Current session ID. Login requests may use 0.
        
        Returns:
            Serializable protobuf Message."""

        msg = self.pb2.Message()
        msg.type = enum_value(self.pb2.MessageType, "MSG_REQUEST")
        msg.seq = seq
        if session_id:
            msg.session_id = int(session_id)

        request = self.pb2.Request()
        request.request_type = enum_value(self.pb2.Request.RequestType, request_name)
        msg.request.CopyFrom(request)
        return msg

    def login(self, username: str, password: str, seq: int) -> Any:
        """Function: Build a PLNS login request message.
        
        Args:
            username: Login username.
            password: Plain login password. It is converted to MD5 inside the function.
            seq: Message sequence number.
        
        Returns:
            Serializable protobuf Message."""

        msg = self.request("REQUEST_LOGIN", seq=seq)
        msg.request.login_request.username = username
        msg.request.login_request.password = md5_string(password)
        return msg

    def change_password(self, old_password: str, new_password: str, seq: int, session_id: int) -> Any:
        """Function: Build a password change request message.
        
        Args:
            old_password: Current plain password. It is converted to MD5 inside the function.
            new_password: New plain password. It is converted to MD5 inside the function.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        msg = self.request("REQUEST_CHANGE_PW", seq=seq, session_id=session_id)
        msg.request.param_str = md5_string(old_password)
        msg.request.param_str1 = md5_string(new_password)
        return msg

    def file_operate(
        self,
        file_op: str,
        source: str,
        target: str,
        extra: str,
        seq: int,
        session_id: int,
    ) -> Any:
        """Function: Build a file operation request message.
        
        Args:
            file_op: FileOperateType enum name defined in main.proto.
            source: Source path or operation input written to `Request.param_str`.
            target: Target path or operation output written to `Request.param_str1`.
            extra: Optional extra operation parameter written to `Request.param_str2`.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        msg = self.request("REQUEST_FILE_OPERATE", seq=seq, session_id=session_id)
        msg.request.file_op = enum_value(self.pb2.Request.FileOperateType, file_op)
        msg.request.param_str = source
        msg.request.param_str1 = target
        msg.request.param_str2 = extra
        return msg

    def command(self, command_name: str, seq: int, session_id: int) -> Any:
        """Function: Build a PLNS Command message without extra parameters.
        
        Args:
            command_name: CommandType name defined in main.proto.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, command_name)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def int_command(self, command_name: str, value: int, seq: int, session_id: int) -> Any:
        """Function: Build a PLNS Command message with `param_int` set.
        
        Args:
            command_name: CommandType name defined in main.proto.
            value: Integer value to write into `Command.param_int`.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, command_name)
        cmd.param_int = int(value)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def string_command(self, command_name: str, value: str, seq: int, session_id: int) -> Any:
        """Function: Build a PLNS Command message with `param_str` set.

        Args:
            command_name: CommandType name defined in main.proto.
            value: String value to write into `Command.param_str`.
            seq: Message sequence number.
            session_id: Current session ID.

        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, command_name)
        cmd.param_str = str(value)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def location_mode_command(self, mode: int, seq: int, session_id: int) -> Any:
        """Function: Build a command message that sets `Command.location_mode`.

        Args:
            mode: LocationMode enum integer value.
            seq: Message sequence number.
            session_id: Current session ID.

        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, "CMD_SET_LOCATION_MODE")
        cmd.location_mode = int(mode)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def general_io_output(self, output_mask: int, seq: int, session_id: int) -> Any:
        """Function: Build a command message that sets the general IO output bitmask.
        
        Args:
            output_mask: Unsigned 64-bit output state bitmask.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, "CMD_SET_GENERAL_IO_OUPUT")
        cmd.general_io_ouput = int(output_mask)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def extend_control(self, extend_cmd: int, seq: int, session_id: int) -> Any:
        """Function: Build a peripheral extension control command message.
        
        Args:
            extend_cmd: Custom peripheral extension command value.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, "CMD_EXTEND_CONTROL")
        cmd.extend_cmd = int(extend_cmd)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def command_message(self, command: Any, seq: int, session_id: int) -> Any:
        """Function: Wrap a Command submessage in a full PLNS Message.
        
        Args:
            command: Populated protobuf Command submessage.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        msg = self.pb2.Message()
        msg.type = enum_value(self.pb2.MessageType, "MSG_COMMAND")
        msg.seq = seq
        msg.session_id = int(session_id)
        msg.command.CopyFrom(command)
        return msg

    def manual_speed(self, vx_mm_s: int, w_mrad_s: int, vy_mm_s: int, seq: int, session_id: int) -> Any:
        """Function: Build a manual velocity command message.
        
        Args:
            vx_mm_s: Linear velocity in the X direction, in mm/s.
            w_mrad_s: Angular velocity, in 0.001 rad/s.
            vy_mm_s: Linear velocity in the Y direction, in mm/s.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, "CMD_SET_MANUAL_SPEED")
        cmd.param_int = int(vx_mm_s)
        cmd.param_int1 = int(w_mrad_s)
        cmd.param_int2 = int(vy_mm_s)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def pose_command(self, command_name: str, pose: Pose2D, seq: int, session_id: int) -> Any:
        """Function: Build a command message that carries a 2D pose parameter.
        
        Args:
            command_name: CommandType name defined in main.proto.
            pose: 2D pose expressed in meters and radians.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, command_name)
        x_mm, y_mm, yaw_mrad = pose.to_proto_units()
        cmd.pose.x = x_mm
        cmd.pose.y = y_mm
        cmd.pose.yaw = yaw_mrad
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def move_to_station(self, station_id: int, seq: int, session_id: int) -> Any:
        """Function: Build a movement task command that navigates to a station.
        
        Args:
            station_id: Target station ID.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, "CMD_NEW_MOVEMENT_TASK")
        task = self.pb2.MovementTask()
        task.type = enum_value(self.pb2.MovementTask.TaskType, "MT_MOVE_TO_STATION")
        # Current firmware ignores this task field setting; avoid mode is controlled by CMD_ENABLE/CMD_DISABLE_OBSTACLE_AVOID.
        task.avoid_policy = enum_value(self.pb2.MovementTask.AvoidPolicy, "OBSTACLE_AVOID_WAIT")
        task.dst_station_type = enum_value(self.pb2.MovementTask.DstStationType, "DS_DEFAULT")
        task.stations.append(int(station_id))
        cmd.movement_task.CopyFrom(task)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def move_to_pose(self, pose: Pose2D, seq: int, session_id: int) -> Any:
        """Function: Build a movement task command that navigates to a pose.
        
        Args:
            pose: Target 2D pose expressed in meters and radians.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, "CMD_NEW_MOVEMENT_TASK")
        task = self.pb2.MovementTask()
        task.type = enum_value(self.pb2.MovementTask.TaskType, "MT_MOVE_TO_POSE")
        # Current firmware ignores this task field setting; avoid mode is controlled by CMD_ENABLE/CMD_DISABLE_OBSTACLE_AVOID.
        task.avoid_policy = enum_value(self.pb2.MovementTask.AvoidPolicy, "OBSTACLE_AVOID_WAIT")
        x_mm, y_mm, yaw_mrad = pose.to_proto_units()
        pose_msg = task.poses.add()
        pose_msg.x = x_mm
        pose_msg.y = y_mm
        pose_msg.yaw = yaw_mrad
        cmd.movement_task.CopyFrom(task)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def action_task(
        self,
        task_type: str,
        seq: int,
        session_id: int,
        *,
        timeout_s: int = 0,
        task_id: int = 0,
    ) -> Any:
        """Function: Build an action task command message.
        
        Args:
            task_type: ActionTask.TaskType enum name defined in main.proto.
            seq: Message sequence number.
            session_id: Current session ID.
            timeout_s: Optional task timeout or wait duration in seconds.
            task_id: Optional action task ID written into the protobuf task.
        
        Returns:
            Serializable protobuf Message."""

        cmd = self.pb2.Command()
        cmd.command = enum_value(self.pb2.CommandType, "CMD_NEW_ACTION_TASK")
        task = self.pb2.ActionTask()
        task.type = enum_value(self.pb2.ActionTask.TaskType, task_type)
        task.timeout = int(timeout_s)
        task.id = int(task_id)
        cmd.action_task.CopyFrom(task)
        return self.command_message(cmd, seq=seq, session_id=session_id)

    def wait_action(self, timeout_s: int, task_id: int, seq: int, session_id: int) -> Any:
        """Function: Build an action task command that waits for a fixed duration.
        
        Args:
            timeout_s: Wait duration in seconds.
            task_id: Optional action task ID written into the protobuf task.
            seq: Message sequence number.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        return self.action_task(
            "AT_WAITING_TIMEOUT",
            seq=seq,
            session_id=session_id,
            timeout_s=timeout_s,
            task_id=task_id,
        )

    def notify_ack(self, notification: Any, seq: int, session_id: int) -> Any:
        """Function: Build a notification acknowledgement response message.
        
        Args:
            notification: Notification submessage sent by PLNS.
            seq: Sequence number of the notification being acknowledged.
            session_id: Current session ID.
        
        Returns:
            Serializable protobuf Message."""

        msg = self.pb2.Message()
        msg.type = enum_value(self.pb2.MessageType, "MSG_RESPONSE")
        msg.seq = seq
        msg.session_id = int(session_id)
        msg.response.response_type = enum_value(self.pb2.Response.ResponseType, "RESPONSE_NOTIFY")
        msg.response.notify_response.ack = True
        msg.response.notify_response.type = int(notification.notify_type)
        return msg


def pack_message(message: Any, max_frame_size: int = 128 * 1024) -> bytes:
    """Function: Serialize a protobuf message and wrap it as a PLNS TCP frame.
    
    Args:
        message: Protobuf Message to send.
        max_frame_size: Maximum allowed payload size in bytes.
    
    Returns:
        Binary frame containing a 4-byte big-endian length header and payload."""

    payload = message.SerializeToString()
    if len(payload) > max_frame_size:
        raise ProtocolError(f"payload too large: {len(payload)} bytes")
    return len(payload).to_bytes(4, byteorder="big") + payload


def parse_payload(pb2: ModuleType, payload: bytes) -> Any:
    """Function: Deserialize a PLNS frame payload into a protobuf Message.
    
    Args:
        pb2: Loaded PLNS protobuf Python module.
        payload: Protobuf binary payload without the 4-byte length header.
    
    Returns:
        Parsed protobuf Message."""

    message = pb2.Message()
    message.ParseFromString(payload)
    return message
