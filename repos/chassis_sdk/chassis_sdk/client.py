"""High-level async client for PLNS."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
import zlib

from .config import PLNSConfig, SFTPConfig
from .maps import load_map_bundle, sync_maps_via_sftp
from .models import MapBundle, Pose2D, RoadMap
from .protocol import MessageFactory, enum_value, load_pb2, pack_message, parse_payload
from .transport import AsyncFileFrameTransport, AsyncFrameTransport


class PLNSError(RuntimeError):
    """Raised when PLNS returns an error or the client is used incorrectly."""


class PLNSResultError(PLNSError):
    """Raised when PLNS returns a failed Result message."""


class PLNSSessionError(PLNSError):
    """Raised when PLNS returns a response for a different session."""


class PLNSClient:
    """UI-free async SDK client for the PLNS TCP protocol."""

    def __init__(
        self,
        config: PLNSConfig | None = None,
        *,
        pb2_path: str | Path | None = None,
        raw_log_path: str | Path | None = None,
    ) -> None:
        """Function: Initialize the PLNS client, protocol factory, TCP transport, and runtime state.
        
        Args:
            config: PLNS connection, login, heartbeat, and frame-size settings. Uses environment defaults when omitted.
            pb2_path: Optional protobuf Python module path. Uses the bundled module when omitted.
            raw_log_path: Optional raw send/receive frame log path."""

        self.config = config or PLNSConfig.from_env()
        self.pb2: ModuleType = load_pb2(pb2_path)
        self.factory = MessageFactory(self.pb2)
        self.transport = AsyncFrameTransport(
            self.config.host,
            self.config.port,
            max_frame_size=self.config.max_frame_size,
            connect_timeout_s=self.config.connect_timeout_s,
            raw_log_path=raw_log_path,
        )
        self.file_transport = AsyncFileFrameTransport(
            self.config.host,
            self.config.port + 1,
            max_frame_size=self.config.file_max_frame_size,
            connect_timeout_s=self.config.connect_timeout_s,
            raw_log_path=raw_log_path,
        )
        self.session_id = 0
        self.is_logged_in = False
        self.is_file_logged_in = False
        self.system_state: Any | None = None
        self.hardware_state: Any | None = None
        self.info: Any | None = None
        self.task_state: tuple[Any | None, Any | None] = (None, None)
        self.mapping_cloud: Any | None = None
        self.mapping_cloud_points: list[tuple[float, float, float]] = []
        self.mapping_track: Any | None = None
        self.mapping_track_points: list[tuple[float, float, float]] = []
        self.notifications: asyncio.Queue[Any] = asyncio.Queue()
        self.mapping_cloud_updates: asyncio.Queue[Any] = asyncio.Queue()
        self.mapping_track_updates: asyncio.Queue[Any] = asyncio.Queue()
        self._seq = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._file_pending: dict[int, asyncio.Future[Any]] = {}
        self._file_login_future: asyncio.Future[Any] | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._file_recv_task: asyncio.Task[None] | None = None
        self._file_heartbeat_task: asyncio.Task[None] | None = None
        self._manual_task: asyncio.Task[None] | None = None
        self._is_mapping = False

    async def __aenter__(self) -> "PLNSClient":
        """Function: Connect to and log in to PLNS when entering the async context.
        
        Returns:
            The connected client instance."""

        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Function: Close the client and release the connection when exiting the async context.
        
        Args:
            exc_type: Exception type from the context, or None when no exception occurred.
            exc: Exception instance from the context, or None when no exception occurred.
            tb: Exception traceback from the context, or None when no exception occurred."""

        await self.close()

    def _next_seq(self) -> int:
        """Function: Generate the next PLNS session message sequence number.
        
        Returns:
            The incremented message sequence number."""

        self._seq += 1
        return self._seq

    async def connect(self, *, login: bool = True) -> None:
        """Function: Connect to the PLNS TCP service and start the receive task, with optional login and heartbeat.
        
        Args:
            login: Whether to log in and start the heartbeat task after connecting."""

        await self.transport.connect()
        self._recv_task = asyncio.create_task(self._recv_loop())
        if login:
            await self.login(self.config.username, self.config.password)
            await self._connect_file_channel()
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def close(self) -> None:
        """Function: Stop active movement, log out, cancel background tasks, and close the TCP connection."""

        await self._cleanup_motion_before_close()
        if self.is_logged_in:
            try:
                await self.logout()
            except Exception:
                pass
        await self._close_file_channel()
        for task in (self._heartbeat_task, self._recv_task):
            if task:
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._heartbeat_task, self._recv_task) if task),
            return_exceptions=True,
        )
        self._heartbeat_task = None
        self._recv_task = None
        self.is_logged_in = False
        self.is_file_logged_in = False
        self._is_mapping = False
        self.session_id = 0
        await self.transport.close()

    async def _cleanup_motion_before_close(self) -> None:
        """Function: Stop manual control or cancel navigation tasks before closing the client."""

        if not self.is_logged_in:
            return
        try:
            state = await self.get_system_state()
            if self._operation_is_manual(state):
                await self.stop_manual_velocity(disable_manual=False)
            elif self._movement_needs_cancel(state):
                await self.cancel_movement_task()
            elif self._manual_task:
                await self.stop_manual_velocity(disable_manual=False)
        except Exception:
            pass

    async def login(self, username: str = "admin", password: str = "admin") -> Any:
        """Function: Send a PLNS login request and store the returned session ID.
        
        Args:
            username: PLNS login username.
            password: Plain PLNS login password. It is converted to MD5 before sending.
        
        Returns:
            The PLNS login response message."""

        seq = self._next_seq()
        msg = self.factory.login(username, password, seq)
        response = await self._send_and_wait(msg, seq)
        result = response.response.result
        if result.result_state != enum_value(self.pb2.Result.ResultState, "RESPONSE_OK"):
            raise PLNSError(f"login failed: result_code={result.result_code}")
        self.session_id = int(response.response.login_info.session_id)
        self.is_logged_in = True
        return response

    async def logout(self) -> Any:
        """Function: Send a PLNS logout request and clear the local login state.
        
        Returns:
            The PLNS logout response message."""

        response = await self.request("REQUEST_LOGOUT")
        self.is_logged_in = False
        self.is_file_logged_in = False
        self._is_mapping = False
        self.session_id = 0
        await self._close_file_channel()
        return response

    async def request(self, request_name: str, *, wait: bool = True) -> Any | None:
        """Function: Send a generic PLNS Request message.
        
        Args:
            request_name: RequestType name defined in main.proto.
            wait: Whether to wait for and return the matching response.
        
        Returns:
            The PLNS response message when waiting; otherwise None."""

        self._require_login()
        seq = self._next_seq()
        msg = self.factory.request(request_name, seq=seq, session_id=self.session_id)
        return await self._send_and_wait(msg, seq) if wait else await self._send_no_wait(msg)

    async def command(self, command_name: str, *, wait: bool = True) -> Any | None:
        """Function: Send a generic PLNS Command message.
        
        Args:
            command_name: CommandType name defined in main.proto.
            wait: Whether to wait for and return the matching response.
        
        Returns:
            The PLNS response message when waiting; otherwise None."""

        self._require_login()
        seq = self._next_seq()
        msg = self.factory.command(command_name, seq=seq, session_id=self.session_id)
        return await self._send_and_wait(msg, seq) if wait else await self._send_no_wait(msg)

    async def get_info(self) -> Any:
        """Function: Request vehicle or firmware information.
        
        Returns:
            The PLNS RESPONSE_INFO response message."""

        return await self.request("REQUEST_INFO")

    async def get_system_state(self) -> Any:
        """Function: Request one PLNS system state snapshot.
        
        Returns:
            The PLNS RESPONSE_SYSTEM_STATE response message."""

        return await self.request("REQUEST_SYSTEM_STATE")

    async def get_hardware_state(self) -> Any:
        """Function: Request one PLNS hardware state snapshot.
        
        Returns:
            The PLNS RESPONSE_HARDWARE_STATE response message."""

        return await self.request("REQUEST_HARDWARE_STATE")

    async def get_map_list(self) -> Any:
        """Function: Request the list of map files known by PLNS.
        
        Returns:
            The PLNS RESPONSE_MAP_LIST response message."""

        return await self.request("REQUEST_MAP_LIST")

    async def get_laser_points(self) -> Any:
        """Function: Request one snapshot of laser point data.
        
        Returns:
            The PLNS RESPONSE_LASER_POINTS response message."""

        return await self.request("REQUEST_LASER_POINTS")

    async def get_task_state(self) -> Any:
        """Function: Request current movement and action task state.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.
            You can get state through the `system_state` and `hardware_state` attributes.

        Returns:
            The PLNS RESPONSE_TASK_STATE response message."""

        return await self.request("REQUEST_TASK_STATE")

    async def get_current_map(self) -> Any:
        """Function: Request the current map after PLNS publishes a new map notification.

        Returns:
            The PLNS RESPONSE_GET_CUR_MAP response message."""

        return await self.request("REQUEST_GET_CUR_MAP")

    async def get_timestamp(self) -> Any:
        """Function: Request the current PLNS system timestamp.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Returns:
            The PLNS RESPONSE_TIMESTAMP response message."""

        return await self.request("REQUEST_TIMESTAMP")

    async def get_mapping_data(self, *, wait: bool = True) -> Any | None:
        """Function: Request mapping point cloud data through the File channel during map drawing.

        Args:
            wait: Whether to wait for and return the matching File-channel response.
        
        Returns:
            The PLNS RESPONSE_GET_MAPPING_DATA response message when waiting; otherwise None."""

        return await self._file_request("REQUEST_GET_MAPPING_DATA", wait=wait)

    async def get_mapping_track(self, timeout_s: float = 5.0) -> Any:
        """Function: Wait for the next File-channel mapping trajectory response.

        The firmware sends RESPONSE_GET_MAPPING_TRACK actively on the File
        channel during mapping; the upper computer does not request it.

        Args:
            timeout_s: Maximum time to wait for the next trajectory message.

        Returns:
            The PLNS RESPONSE_GET_MAPPING_TRACK response message."""

        return await asyncio.wait_for(self.mapping_track_updates.get(), timeout=timeout_s)

    async def change_password(self, old_password: str, new_password: str) -> Any:
        """Function: Request a PLNS login password change.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Args:
            old_password: Current plain password. The SDK sends its MD5 digest.
            new_password: New plain password. The SDK sends its MD5 digest.
        
        Returns:
            The PLNS RESPONSE_CHANGE_PW response message."""

        if not old_password:
            raise ValueError("old password must not be empty")
        if not new_password:
            raise ValueError("new password must not be empty")
        self._require_login()
        seq = self._next_seq()
        msg = self.factory.change_password(
            old_password=old_password,
            new_password=new_password,
            seq=seq,
            session_id=self.session_id,
        )
        return await self._send_and_wait(msg, seq)

    async def file_operate(
        self,
        file_op: str,
        source: str = "",
        target: str = "",
        extra: str = "",
    ) -> Any:
        """Function: Request a PLNS file operation.

        TODO: Unverified

        Args:
            file_op: FileOperateType enum name defined in main.proto.
            source: Source path or operation input.
            target: Target path or operation output.
            extra: Optional extra operation parameter.
        
        Returns:
            The PLNS RESPONSE_FILE_OPERATE response message."""

        self._require_login()
        seq = self._next_seq()
        msg = self.factory.file_operate(
            file_op=file_op,
            source=source,
            target=target,
            extra=extra,
            seq=seq,
            session_id=self.session_id,
        )
        return await self._send_and_wait(msg, seq)

    async def import_map_file(self, source: str, target: str = "", extra: str = "") -> Any:
        """Function: Request PLNS to import map.pgm or map.yaml.
        
        TODO: Unverified
        
        Args:
            source: Source map file path.
            target: Optional target path or name.
            extra: Optional extra operation parameter.
        
        Returns:
            The PLNS RESPONSE_FILE_OPERATE response message."""

        return await self.file_operate("IMPORT_MAP_FILE", source=source, target=target, extra=extra)

    async def import_config_file(self, source: str, target: str = "", extra: str = "") -> Any:
        """Function: Request PLNS to import map.json.
        
        TODO: Unverified
        
        Args:
            source: Source configuration file path.
            target: Optional target path or name.
            extra: Optional extra operation parameter.
        
        Returns:
            The PLNS RESPONSE_FILE_OPERATE response message."""

        return await self.file_operate("IMPORT_CONFIG_FILE", source=source, target=target, extra=extra)

    async def import_system_update_file(self, source: str, target: str = "", extra: str = "") -> Any:
        """Function: Request PLNS to import a system update file.
        
        TODO: Unverified
        
        Args:
            source: Source system update file path.
            target: Optional target path or name.
            extra: Optional extra operation parameter.
        
        Returns:
            The PLNS RESPONSE_FILE_OPERATE response message."""

        return await self.file_operate("IMPORT_SYSTEM_UPDATE_FILE", source=source, target=target, extra=extra)

    async def restore_factory_settings(self) -> Any:
        """Function: Request PLNS to restore factory settings.

        TODO: Unverified
        
        Returns:
            The PLNS RESPONSE_FILE_OPERATE response message."""

        return await self.file_operate("RESTORE_FACTORY_SETTINGS")

    async def subscribe_state(self) -> None:
        """Function: Subscribe to 10 Hz system state, hardware state, and laser data updates."""

        await self.request("REQUEST_STATE", wait=False)

    async def enable_manual(self) -> Any:
        """Function: Switch PLNS to manual control mode.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_ENABLE_MANUAL_CONTROL")

    async def disable_manual(self) -> Any:
        """Function: Exit PLNS manual control mode.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_DISABLE_MANUAL_CONTROL")

    async def slow_down(self) -> Any:
        """Function: Send the PLNS slow-down command.
        
       Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Returns:
            The PLNS command response message."""

        return await self.command("CMD_SPEED_SLOW_DOWN")

    async def enable_obstacle_avoid(self) -> Any:
        """Function: Enable PLNS obstacle avoidance.

        Note:
            Real-device testing shows this switches movement_state.avoid_policy to
            OBSTACLE_AVOID_REPLAN on the current firmware.
            And OBSTACLE_AVOID_REPLAN is not implemented on the current firmware.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_ENABLE_OBSTACLE_AVOID")

    async def disable_obstacle_avoid(self) -> Any:
        """Function: Disable PLNS obstacle avoidance.

        Note:
            Real-device testing shows this switches movement_state.avoid_policy to
            OBSTACLE_AVOID_WAIT on the current firmware
            This is the usable mode for navigation in the tests.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_DISABLE_OBSTACLE_AVOID")

    async def set_speed_level(self, level: int) -> Any:
        """Function: Set the PLNS automatic navigation speed level.
        
        Args:
            level: Speed level from 0 to 100.
        
        Returns:
            The PLNS command response message."""

        if not 0 <= level <= 100:
            raise ValueError("speed level must be between 0 and 100")
        self._require_login()
        seq = self._next_seq()
        msg = self.factory.int_command(
            "CMD_SET_SPEED_LEVEL",
            value=level,
            seq=seq,
            session_id=self.session_id,
        )
        return await self._send_and_wait(msg, seq)

    async def prepare_manual_control(self, *, cancel_existing: bool = True) -> None:
        """Function: Prepare PLNS state before manual velocity control.

        Manual velocity requires manual-control mode. When requested, this also
        clears any active or failed movement task before switching modes.

        Args:
            cancel_existing: Whether to cancel an active or failed movement task."""

        self._require_login()
        state = await self.get_system_state()
        if cancel_existing and self._movement_needs_cancel(state):
            await self.cancel_movement_task()
        if not self._operation_is_manual(state):
            await self.enable_manual()

    async def set_manual_velocity(
        self,
        vx_mm_s: int,
        w_mrad_s: int = 0,
        vy_mm_s: int = 0,
        *,
        wait: bool = False,
        prepare: bool = False,
    ) -> Any | None:
        """Function: Send one manual velocity command.
        
        Args:
            vx_mm_s: Linear velocity in the X direction, in mm/s.
            w_mrad_s: Angular velocity, in 0.001 rad/s.
            vy_mm_s: Linear velocity in the Y direction, in mm/s.
            wait: Whether to wait for and return the velocity command response.
            prepare: Whether to prepare manual-control mode before sending this
                standalone speed command. Keep this false inside high-rate loops.
        
        Returns:
            The PLNS response message when waiting; otherwise None."""

        self._require_login()
        self._validate_manual_velocity(vx_mm_s, w_mrad_s, vy_mm_s)
        if prepare:
            await self.prepare_manual_control()
        seq = self._next_seq()
        msg = self.factory.manual_speed(vx_mm_s, w_mrad_s, vy_mm_s, seq=seq, session_id=self.session_id)
        return await self._send_and_wait(msg, seq) if wait else await self._send_no_wait(msg)

    async def start_manual_velocity(
        self,
        vx_mm_s: int,
        w_mrad_s: int = 0,
        vy_mm_s: int = 0,
        *,
        period_s: float = 0.1,
        prepare: bool = False,
    ) -> None:
        """Function: Enter manual mode and continuously send velocity commands at a fixed period.
        
        Args:
            vx_mm_s: Linear velocity in the X direction, in mm/s.
            w_mrad_s: Angular velocity, in 0.001 rad/s.
            vy_mm_s: Linear velocity in the Y direction, in mm/s.
            period_s: Velocity command refresh period. It must be less than 0.2 seconds.
            prepare: Whether to prepare manual-control mode before sending this
                standalone speed command. Keep this false inside high-rate loops."""

        if period_s >= 0.2:
            raise ValueError("manual velocity period must be less than 0.2 seconds")
        self._validate_manual_velocity(vx_mm_s, w_mrad_s, vy_mm_s)
        await self.stop_manual_velocity(disable_manual=False)
        if prepare:
            await self.prepare_manual_control()
        self._manual_task = asyncio.create_task(
            self._manual_velocity_loop(vx_mm_s, w_mrad_s, vy_mm_s, period_s)
        )

    async def stop_manual_velocity(self, *, disable_manual: bool = True) -> None:
        """Function: Stop the manual velocity loop, send zero velocity and stop commands, and optionally exit manual mode.
        
        Args:
            disable_manual: Whether to switch back to automatic mode after stopping."""

        if self._manual_task:
            self._manual_task.cancel()
            await asyncio.gather(self._manual_task, return_exceptions=True)
            self._manual_task = None
        if self.is_logged_in:
            try:
                await self.set_manual_velocity(0, 0, 0, wait=False, prepare=False)
                if disable_manual:
                    await self.command("CMD_DISABLE_MANUAL_CONTROL", wait=False)
            except Exception:
                if disable_manual:
                    raise

    async def start_localization(self) -> Any:
        """Function: Start PLNS localization.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_START_LOCATION")

    async def stop_localization(self) -> Any:
        """Function: Stop PLNS localization.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_STOP_LOCATION")

    async def set_location_mode(self, mode: int | str) -> Any:
        """Function: Set the PLNS localization mode.
        
        Args:
            mode: Location mode integer or LocationMode enum name such as `LOCATION_MODE_LIDAR`.
        
        Returns:
            The PLNS command response message."""

        if isinstance(mode, str):
            mode_value = enum_value(self.pb2.LocationMode, mode)
        else:
            mode_value = int(mode)
        valid_modes = {
            self.pb2.LocationMode.LOCATION_MODE_NONE,
            self.pb2.LocationMode.LOCATION_MODE_LIDAR,
            self.pb2.LocationMode.LOCATION_MODE_GNSS,
            self.pb2.LocationMode.LOCATION_MODE_MIX2,
            self.pb2.LocationMode.LOCATION_MODE_INS,
        }
        if mode_value not in valid_modes:
            raise ValueError("location mode must be a valid LocationMode value")
        self._require_login()
        seq = self._next_seq()
        msg = self.factory.location_mode_command(mode=mode_value, seq=seq, session_id=self.session_id)
        return await self._send_and_wait(msg, seq)

    async def set_initial_pose(self, pose: Pose2D) -> Any:
        """Function: Set the initial PLNS localization pose.
        
        Args:
            pose: Initial 2D pose expressed in meters and radians.
        
        Returns:
            The PLNS command response message."""

        self._require_login()
        seq = self._next_seq()
        msg = self.factory.pose_command(
            "CMD_SET_LOCATION_INITIAL_POSE",
            pose=pose,
            seq=seq,
            session_id=self.session_id,
        )
        return await self._send_and_wait(msg, seq)

    async def prepare_mapping(self, *, stop_localization: bool = True) -> None:
        """Function: Prepare PLNS state before starting map drawing.

        Real-device mapping requires manual-control mode and localization
        stopped. This method enters manual mode and optionally stops
        localization before CMD_NEW_MAP_START is sent.

        Args:
            stop_localization: Whether to stop localization before mapping."""

        self._require_login()
        await self.prepare_manual_control()
        if not stop_localization:
            return

        state = await self.get_system_state()
        if not self._localization_is_stopped(state):
            await self.stop_localization()

    async def start_mapping(self, *, prepare: bool = True) -> Any:
        """Function: Start drawing a new map.

        Args:
            prepare: Whether to enter manual-control mode and stop localization
                before sending the mapping command.
        
        Returns:
            The PLNS command response message."""

        if prepare:
            await self.prepare_mapping()
        response = await self.command("CMD_NEW_MAP_START")
        self._is_mapping = True
        await self._send_file_heartbeat()
        return response

    async def stop_mapping(self) -> Any:
        """Function: Stop and save the new map drawing.
        
        Returns:
            The PLNS command response message."""

        try:
            return await self.command("CMD_NEW_MAP_STOP")
        finally:
            self._is_mapping = False

    async def cancel_mapping(self) -> Any:
        """Function: Cancel the new map drawing without saving.
        
        Returns:
            The PLNS command response message."""

        try:
            return await self.command("CMD_NEW_MAP_CANCEL")
        finally:
            self._is_mapping = False

    async def sync_time(self, time_value: datetime | str | None = None) -> Any:
        """Function: Synchronize the PLNS system time.

        Note: 
            Current firmware ignores the protocol PDF's `param_int` description and 
            expects `Command.param_str` in `yyyy-MM-dd hh:mm:ss` format.
        
        Args:
            time_value: Local time to send. Uses current local time when omitted.
                A string must already use `yyyy-MM-dd hh:mm:ss`.
        
        Returns:
            The PLNS command response message."""

        if time_value is None:
            time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(time_value, datetime):
            time_text = time_value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(time_value, str):
            time_text = time_value
        else:
            raise TypeError("time_value must be datetime, string, or None")
        self._require_login()
        seq = self._next_seq()
        msg = self.factory.string_command(
            "CMD_SYNC_TIME",
            value=time_text,
            seq=seq,
            session_id=self.session_id,
        )
        return await self._send_and_wait(msg, seq)

    async def pause_movement(self) -> Any:
        """Function: Pause the current movement task.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_PAUSE_MOVEMENT")

    async def continue_movement(self) -> Any:
        """Function: Continue the current movement task.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_CONTINUE_MOVEMENT")

    async def stop_movement(self) -> Any:
        """Function: Stop the current movement.

        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_STOP_MOVEMENT")

    async def cancel_movement_task(self) -> Any:
        """Function: Cancel the current movement task.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_CANCEL_MOVEMENT_TASK")

    async def cancel_action_task(self) -> Any:
        """Function: Cancel the current action task.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Returns:
            The PLNS command response message."""

        return await self.command("CMD_CANCEL_ACTION_TASK")

    async def set_speaker_volume(self, volume: int) -> Any:
        """Function: Set the VSC speaker volume.
        
        Args:
            volume: Speaker volume from 0 to 100.
        
        Returns:
            The PLNS command response message."""

        if not 0 <= volume <= 100:
            raise ValueError("speaker volume must be between 0 and 100")
        self._require_login()
        seq = self._next_seq()
        msg = self.factory.int_command(
            "CMD_SET_SPEAKER_VOLUME",
            value=volume,
            seq=seq,
            session_id=self.session_id,
        )
        return await self._send_and_wait(msg, seq)

    async def set_general_io_output(self, output_mask: int) -> Any:
        """Function: Set the general IO output state bitmask.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Args:
            output_mask: Unsigned 64-bit bitmask representing the desired IO output state.
        
        Returns:
            The PLNS command response message."""

        if not 0 <= output_mask <= 2**64 - 1:
            raise ValueError("general IO output mask must fit in uint64 range")
        self._require_login()
        seq = self._next_seq()
        msg = self.factory.general_io_output(
            output_mask=output_mask,
            seq=seq,
            session_id=self.session_id,
        )
        return await self._send_and_wait(msg, seq)

    async def extend_control(self, extend_cmd: int) -> Any:
        """Function: Send a custom peripheral extension control command.
        
        TODO: Unverified
        
        Args:
            extend_cmd: Custom extension command value to write into `Command.extend_cmd`.
        
        Returns:
            The PLNS command response message."""

        if not -(2**31) <= extend_cmd <= 2**31 - 1:
            raise ValueError("extension command must fit in int32 range")
        self._require_login()
        seq = self._next_seq()
        msg = self.factory.extend_control(
            extend_cmd=extend_cmd,
            seq=seq,
            session_id=self.session_id,
        )
        return await self._send_and_wait(msg, seq)

    async def prepare_navigation(
        self,
        *,
        location_mode: int | str | None = "LOCATION_MODE_LIDAR",
        cancel_existing: bool = True,
        enable_localization: bool = True,
        switch_to_auto: bool = True,
        use_obstacle_wait: bool = True,
    ) -> None:
        """Function: Prepare PLNS state before sending a go-pose or go-station task.

        Real-device navigation requires localization, automatic operation mode,
        no active movement task, and OBSTACLE_AVOID_WAIT on the current firmware.

        Args:
            location_mode: Optional localization mode to set before navigation.
            cancel_existing: Whether to cancel an active or failed movement task.
            enable_localization: Whether to start localization when it is not running.
            switch_to_auto: Whether to leave manual-control mode before navigation.
            use_obstacle_wait: Whether to send CMD_DISABLE_OBSTACLE_AVOID, which
                selects OBSTACLE_AVOID_WAIT on the current firmware."""

        before = await self.get_system_state()
        if cancel_existing and self._movement_needs_cancel(before):
            await self.cancel_movement_task()

        if enable_localization and not self._localization_is_running(before):
            await self.start_localization()

        if location_mode is not None:
            await self.set_location_mode(location_mode)

        after = await self.get_system_state()
        if switch_to_auto and self._operation_is_manual(after):
            await self.disable_manual()

        if use_obstacle_wait:
            await self.disable_obstacle_avoid()

    async def go_station(
        self,
        station_id: int,
        road_map: RoadMap | None = None,
        *,
        prepare: bool = True,
    ) -> Any:
        """Function: Create a movement task that navigates to a station.

        Note:
            Go-station can use the stop-obstacle threshold configured in the
            road-map station/edge data. Real-device navigation must still run
            in automatic mode and OBSTACLE_AVOID_WAIT mode; the default
            preparation step enters those modes before sending the movement task.
        
        Args:
            station_id: Target station ID.
            road_map: Optional local road map. When provided, the station is validated before sending.
            prepare: Whether to prepare localization, auto mode, obstacle wait,
                and active movement cleanup before sending the task.
        
        Returns:
            The PLNS command response message."""

        if road_map:
            road_map.require_station(station_id)
        self._require_login()
        if prepare:
            await self.prepare_navigation()
        seq = self._next_seq()
        msg = self.factory.move_to_station(station_id, seq=seq, session_id=self.session_id)
        return await self._send_and_wait(msg, seq)

    async def go_pose(self, pose: Pose2D, *, prepare: bool = True) -> Any:
        """Function: Create a movement task that navigates to a map pose.
        
        Note:
            Current firmware does not expose a usable stop-obstacle threshold
            for go-pose tasks. Real-device navigation must run in automatic
            mode and OBSTACLE_AVOID_WAIT mode; the default preparation step
            enters those modes before sending the movement task.

        Args:
            pose: Target 2D pose expressed in meters and radians.
            prepare: Whether to prepare localization, auto mode, obstacle wait,
                and active movement cleanup before sending the task.
        
        Returns:
            The PLNS command response message."""

        self._require_login()
        if prepare:
            await self.prepare_navigation()
        seq = self._next_seq()
        msg = self.factory.move_to_pose(pose, seq=seq, session_id=self.session_id)
        return await self._send_and_wait(msg, seq)

    async def wait_action(self, seconds: int, task_id: int = 0) -> Any:
        """Function: Create an action task that waits for a fixed duration.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Args:
            seconds: Wait duration in seconds.
            task_id: Optional action task ID.
        
        Returns:
            The PLNS command response message."""

        if seconds < 0:
            raise ValueError("wait action seconds must be non-negative")
        if not -(2**31) <= task_id <= 2**31 - 1:
            raise ValueError("action task ID must fit in int32 range")
        self._require_login()
        seq = self._next_seq()
        msg = self.factory.wait_action(
            timeout_s=seconds,
            task_id=task_id,
            seq=seq,
            session_id=self.session_id,
        )
        return await self._send_and_wait(msg, seq)

    async def jack_up_action(self, task_id: int = 0) -> Any:
        """Function: Create a jack-up action task.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Args:
            task_id: Optional action task ID.
        
        Returns:
            The PLNS command response message."""

        return await self._simple_action("AT_JACK_UP", task_id=task_id)

    async def jack_down_action(self, task_id: int = 0) -> Any:
        """Function: Create a jack-down action task.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Args:
            task_id: Optional action task ID.
        
        Returns:
            The PLNS command response message."""

        return await self._simple_action("AT_JACK_DOWN", task_id=task_id)

    async def take_photo_action(self, task_id: int = 0) -> Any:
        """Function: Create a take-photo action task.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Args:
            task_id: Optional action task ID.
        
        Returns:
            The PLNS command response message."""

        return await self._simple_action("AT_TAKE_PHOTOS", task_id=task_id)

    async def _simple_action(self, task_type: str, *, task_id: int = 0) -> Any:
        """Function: Create an action task that only needs a task type and optional ID.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Args:
            task_type: ActionTask.TaskType enum name defined in main.proto.
            task_id: Optional action task ID.
        
        Returns:
            The PLNS command response message."""

        if not -(2**31) <= task_id <= 2**31 - 1:
            raise ValueError("action task ID must fit in int32 range")
        self._require_login()
        seq = self._next_seq()
        msg = self.factory.action_task(
            task_type,
            seq=seq,
            session_id=self.session_id,
            task_id=task_id,
        )
        return await self._send_and_wait(msg, seq)

    async def trigger_emergency(self) -> Any:
        """Function: Trigger PLNS emergency stop.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_TRIGGER_EMERGENCY")

    async def cancel_emergency(self) -> Any:
        """Function: Clear PLNS emergency stop state.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_CANCEL_EMERGENCY")

    async def enable_auto_charge(self) -> Any:
        """Function: Start PLNS automatic charging.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Returns:
            The PLNS command response message."""

        return await self.command("CMD_ENABLE_AUTO_CHARGE")

    async def stop_charge(self) -> Any:
        """Function: Stop PLNS charging.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Returns:
            The PLNS command response message."""

        return await self.command("CMD_STOP_CHARGE")

    async def calibrate(self) -> Any:
        """Function: Start one PLNS sensor calibration.
        
        Firmware limitation:
            Vendor confirmed this command is not implemented on the current firmware.
            This method is kept for protocol completeness only.

        Returns:
            The PLNS command response message."""

        return await self.command("CMD_CALIBRATION")

    async def reboot_plns(self) -> Any:
        """Function: Reboot the PLNS module.
        
        Returns:
            The PLNS command response message."""

        return await self.command("CMD_REBOOT_PLNS")

    def load_local_map(self, directory: str | Path) -> MapBundle:
        """Function: Load a map bundle from a local directory.
        
        Args:
            directory: Directory containing map.json and optional map.yaml and map.pgm files.
        
        Returns:
            The parsed map bundle."""

        return load_map_bundle(directory)

    def sync_maps(self, config: SFTPConfig | None = None) -> None:
        """Function: Synchronize map files from PLNS to local storage over SFTP.
        
        Args:
            config: SFTP sync config. Uses environment-based defaults when omitted."""

        sync_maps_via_sftp(config or SFTPConfig.from_env())

    async def _send_and_wait(self, message: Any, seq: int, timeout_s: float = 5.0) -> Any:
        """Function: Send one PLNS message frame and wait for the response with the specified sequence number.
        
        Args:
            message: Protobuf message to send.
            seq: Message sequence number used to match the response.
            timeout_s: Response wait timeout in seconds.
        
        Returns:
            The matched PLNS response message."""

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[seq] = future
        try:
            await self.transport.write_frame(pack_message(message, self.config.max_frame_size))
            response = await asyncio.wait_for(future, timeout=timeout_s)
            self._validate_response(response)
            return response
        finally:
            self._pending.pop(seq, None)

    async def _send_no_wait(self, message: Any) -> None:
        """Function: Send one PLNS message frame without waiting for a response.
        
        Args:
            message: Protobuf message to send."""

        await self.transport.write_frame(pack_message(message, self.config.max_frame_size))
        return None

    async def _connect_file_channel(self) -> None:
        """Function: Connect and log in to the PLNS File channel."""

        if self.is_file_logged_in and self.file_transport.is_connected:
            return
        if not self.is_logged_in or not self.session_id:
            raise PLNSError("cmd channel must be logged in before connecting the File channel")

        await self.file_transport.connect()
        self._file_recv_task = asyncio.create_task(self._file_recv_loop())
        await self._file_login()
        if not self._file_heartbeat_task or self._file_heartbeat_task.done():
            self._file_heartbeat_task = asyncio.create_task(self._file_heartbeat_loop())

    async def _close_file_channel(self) -> None:
        """Function: Stop File-channel background tasks and close the File TCP connection."""

        for task in (self._file_heartbeat_task, self._file_recv_task):
            if task:
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._file_heartbeat_task, self._file_recv_task) if task),
            return_exceptions=True,
        )
        self._file_heartbeat_task = None
        self._file_recv_task = None
        self.is_file_logged_in = False
        await self.file_transport.close()

    async def _file_login(self) -> Any:
        """Function: Log in to the File channel using the cmd-channel session ID."""

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._file_login_future = future
        try:
            msg = self.factory.request("REQUEST_LOGIN", seq=0, session_id=self.session_id)
            await self._send_file_no_wait(msg)
            return await asyncio.wait_for(future, timeout=self.config.connect_timeout_s)
        finally:
            self._file_login_future = None

    async def _file_request(self, request_name: str, *, wait: bool = True, timeout_s: float = 5.0) -> Any | None:
        """Function: Send a Request message through the File channel."""

        self._require_login()
        if not self.is_file_logged_in:
            await self._connect_file_channel()

        seq = self._next_seq()
        msg = self.factory.request(request_name, seq=seq, session_id=self.session_id)
        if not wait:
            await self._send_file_no_wait(msg)
            return None

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._file_pending[seq] = future
        try:
            await self._send_file_no_wait(msg)
            return await asyncio.wait_for(future, timeout=timeout_s)
        finally:
            self._file_pending.pop(seq, None)

    async def _send_file_no_wait(self, message: Any) -> None:
        """Function: Send one normal framed protobuf message through the File channel."""

        await self.file_transport.write_frame(pack_message(message, self.config.file_max_frame_size))

    async def _send_file_heartbeat(self) -> None:
        """Function: Send one heartbeat command through the File channel."""

        if not self.is_file_logged_in:
            await self._connect_file_channel()
        seq = self._next_seq()
        msg = self.factory.command("CMD_HEARTBEATS_UPDATE", seq=seq, session_id=self.session_id)
        await self._send_file_no_wait(msg)

    def _validate_response(self, message: Any) -> None:
        """Function: Validate response session ID and PLNS Result fields.

        Args:
            message: Parsed PLNS response message.
        """

        if self.session_id and int(message.session_id) != int(self.session_id):
            raise PLNSSessionError(
                f"response session mismatch: expected={self.session_id} actual={int(message.session_id)}"
            )

        result = message.response.result
        result_state = int(result.result_state)
        result_code = int(result.result_code)
        failed = enum_value(self.pb2.Result.ResultState, "RESPONSE_FAILED")
        no_error = enum_value(self.pb2.Result.ResultCode, "RESULT_CODE_NONE")
        if result_state == failed or result_code != no_error:
            state_name = self.pb2.Result.ResultState.Name(result_state)
            code_name = self.pb2.Result.ResultCode.Name(result_code)
            raise PLNSResultError(f"PLNS response failed: result_state={state_name} result_code={code_name}")

    async def _recv_loop(self) -> None:
        """Function: Continuously receive PLNS frames in the background, parse messages, and dispatch them."""

        while True:
            payload = await self.transport.read_payload()
            message = parse_payload(self.pb2, payload)
            await self._handle_message(message)

    async def _file_recv_loop(self) -> None:
        """Function: Continuously receive and dispatch PLNS File-channel frames."""

        while True:
            file_payload = await self.file_transport.read_file_payload()
            message = parse_payload(self.pb2, file_payload.payload)
            await self._handle_file_message(message)

    async def _handle_message(self, message: Any) -> None:
        """Function: Handle a received PLNS response or notification message.
        
        Args:
            message: Parsed PLNS protobuf message."""

        if message.type == enum_value(self.pb2.MessageType, "MSG_RESPONSE"):
            self._cache_response(message)
            future = self._pending.get(int(message.seq))
            if future and not future.done():
                future.set_result(message)
        elif message.type == enum_value(self.pb2.MessageType, "MSG_NOTIFICATION"):
            await self.notifications.put(message.notification)
            await self._ack_notification(message)

    async def _handle_file_message(self, message: Any) -> None:
        """Function: Handle a received PLNS File-channel message."""

        if self.session_id and int(message.session_id) != int(self.session_id):
            return

        if message.type == enum_value(self.pb2.MessageType, "MSG_REQUEST"):
            request_type = int(message.request.request_type)
            if request_type == enum_value(self.pb2.Request.RequestType, "REQUEST_GET_CUR_MAP"):
                await self.get_mapping_data(wait=False)
            return

        if message.type != enum_value(self.pb2.MessageType, "MSG_RESPONSE"):
            return

        response_type = int(message.response.response_type)
        if response_type == enum_value(self.pb2.Response.ResponseType, "RESPONSE_LOGIN"):
            await self._handle_file_login_response(message)
            return

        future = self._file_pending.get(int(message.seq))
        error: Exception | None = None
        try:
            self._validate_response(message)
            await self._cache_file_response(message)
        except Exception as exc:
            error = exc

        if future and not future.done():
            if error:
                future.set_exception(error)
            else:
                future.set_result(message)

    async def _handle_file_login_response(self, message: Any) -> None:
        """Function: Validate and complete the File-channel login response."""

        future = self._file_login_future
        try:
            self._validate_response(message)
        except Exception as exc:
            self.is_file_logged_in = False
            if future and not future.done():
                future.set_exception(exc)
            return

        self.is_file_logged_in = True
        if future and not future.done():
            future.set_result(message)

    async def _cache_file_response(self, message: Any) -> None:
        """Function: Cache File-channel mapping data and notify mapping queues."""

        response_type = int(message.response.response_type)
        if response_type == enum_value(self.pb2.Response.ResponseType, "RESPONSE_GET_MAPPING_DATA"):
            self.mapping_cloud = message.response.mapping_cloud
            self.mapping_cloud_points = self._decode_mapping_cloud_points(self.mapping_cloud)
            await self.mapping_cloud_updates.put(message)
        elif response_type == enum_value(self.pb2.Response.ResponseType, "RESPONSE_GET_MAPPING_TRACK"):
            self.mapping_track = message.response.mapping_trajectory
            self.mapping_track_points = self._decode_mapping_track_points(self.mapping_track)
            await self.mapping_track_updates.put(message)

    def _cache_response(self, message: Any) -> None:
        """Function: Cache common state, hardware, information, and task data based on response type.
        
        Args:
            message: Parsed PLNS response message."""

        response_type = int(message.response.response_type)
        response = message.response
        if response_type in (
            enum_value(self.pb2.Response.ResponseType, "RESPONSE_SYSTEM_STATE"),
            enum_value(self.pb2.Response.ResponseType, "RESPONSE_STATE"),
        ):
            self.system_state = response.system_state
            self.hardware_state = response.hardware_state
        elif response_type == enum_value(self.pb2.Response.ResponseType, "RESPONSE_HARDWARE_STATE"):
            self.hardware_state = response.hardware_state
        elif response_type == enum_value(self.pb2.Response.ResponseType, "RESPONSE_INFO"):
            self.info = response.info
        elif response_type == enum_value(self.pb2.Response.ResponseType, "RESPONSE_TASK_STATE"):
            self.task_state = (response.movement_task, response.action_task)

    async def _ack_notification(self, message: Any) -> None:
        """Function: Send an acknowledgement response for a PLNS notification.
        
        Args:
            message: PLNS message containing the notification field."""

        if not self.is_logged_in:
            return
        ack = self.factory.notify_ack(message.notification, seq=message.seq, session_id=self.session_id)
        await self._send_no_wait(ack)

    async def _heartbeat_loop(self) -> None:
        """Function: Periodically send heartbeat commands in the background to keep the PLNS session alive."""

        while True:
            await asyncio.sleep(self.config.heartbeat_interval_s)
            if self.is_logged_in:
                await self.command("CMD_HEARTBEATS_UPDATE", wait=False)

    async def _file_heartbeat_loop(self) -> None:
        """Function: Periodically send File-channel heartbeat commands."""

        while True:
            interval = (
                self.config.mapping_file_heartbeat_interval_s
                if self._is_mapping
                else self.config.heartbeat_interval_s
            )
            await asyncio.sleep(interval)
            if self.is_logged_in and self.is_file_logged_in:
                await self._send_file_heartbeat()

    async def _manual_velocity_loop(
        self,
        vx_mm_s: int,
        w_mrad_s: int,
        vy_mm_s: int,
        period_s: float,
    ) -> None:
        """Function: Continuously send manual velocity commands at a fixed period in the background.
        
        Args:
            vx_mm_s: Linear velocity in the X direction, in mm/s.
            w_mrad_s: Angular velocity, in 0.001 rad/s.
            vy_mm_s: Linear velocity in the Y direction, in mm/s.
            period_s: Velocity command refresh period in seconds."""

        while True:
            await self.set_manual_velocity(vx_mm_s, w_mrad_s, vy_mm_s, prepare=False)
            await asyncio.sleep(period_s)

    def _require_login(self) -> None:
        """Function: Verify that the client is logged in to PLNS."""

        if not self.is_logged_in:
            raise PLNSError("PLNS client is not logged in")

    def _localization_is_running(self, response: Any) -> bool:
        """Function: Check whether a system-state response reports running localization."""

        state = response.response.system_state.location_state
        return int(state) == int(enum_value(self.pb2.SystemState.LocationState, "LOCATION_STATE_RUNNING"))

    def _localization_is_stopped(self, response: Any) -> bool:
        """Function: Check whether a system-state response reports stopped localization."""

        state = int(response.response.system_state.location_state)
        stopped_states = {
            int(enum_value(self.pb2.SystemState.LocationState, "LOCATION_STATE_ZERO")),
            int(enum_value(self.pb2.SystemState.LocationState, "LOCATION_STATE_NONE")),
            int(enum_value(self.pb2.SystemState.LocationState, "LOCATION_STATE_INITIALING")),
        }
        return state in stopped_states

    def _operation_is_manual(self, response: Any) -> bool:
        """Function: Check whether a system-state response reports manual operation mode."""

        state = response.response.system_state.operation_state
        return int(state) == int(enum_value(self.pb2.SystemState.OperationState, "OPERATION_MANUAL"))

    def _movement_needs_cancel(self, response: Any) -> bool:
        """Function: Check whether a system-state response reports an active or failed movement task."""

        state = response.response.system_state
        task_state = int(state.movement_state.state)
        active_task_states = {
            int(enum_value(self.pb2.MovementTask.TaskState, "MT_WAIT_FOR_START")),
            int(enum_value(self.pb2.MovementTask.TaskState, "MT_RUNNING")),
            int(enum_value(self.pb2.MovementTask.TaskState, "MT_PAUSED")),
        }
        nav_error_states = {
            int(enum_value(self.pb2.SystemState.SysState, "SYS_STATE_TASK_NAV_NO_WAY")),
            int(enum_value(self.pb2.SystemState.SysState, "SYS_STATE_TASK_NAV_NO_STATION")),
            int(enum_value(self.pb2.SystemState.SysState, "SYS_STATE_TASK_NAV_PATH_ERROR")),
        }
        return task_state in active_task_states or int(state.sys_state) in nav_error_states

    @staticmethod
    def _validate_manual_velocity(vx_mm_s: int, w_mrad_s: int, vy_mm_s: int) -> None:
        """Function: Validate that manual velocity is within the SDK default safety limits.
        
        Args:
            vx_mm_s: Linear velocity in the X direction, in mm/s.
            w_mrad_s: Angular velocity, in 0.001 rad/s.
            vy_mm_s: Linear velocity in the Y direction, in mm/s."""

        pass
        # if abs(vx_mm_s) > 300 or abs(vy_mm_s) > 300:
        #     raise ValueError("manual linear velocity is limited to 300 mm/s by default")
        # if abs(w_mrad_s) > 500:
        #     raise ValueError("manual angular velocity is limited to 0.5 rad/s by default")

    @staticmethod
    def _uint32_list_to_bytes(values: Any, expected_len: int = 0) -> bytes:
        """Function: Convert PLNS repeated uint32 byte chunks to bytes."""

        data = bytearray()
        for value in values:
            data.extend(int(value).to_bytes(4, byteorder="big", signed=False))
        if expected_len:
            return bytes(data[:expected_len])
        return bytes(data)

    @classmethod
    def _decode_mapping_cloud_points(cls, cloud_map: Any) -> list[tuple[float, float, float]]:
        """Function: Decode a File-channel CloudMap payload into meter-unit points."""

        compressed_len = int(cloud_map.compressed_char_len)
        original_len = int(cloud_map.org_char_len)
        cloud_size = int(cloud_map.cloud_size)
        compressed = cls._uint32_list_to_bytes(cloud_map.data, compressed_len)
        try:
            raw = zlib.decompress(compressed)
        except zlib.error as exc:
            raise PLNSError(f"mapping cloud zlib decompress failed: {exc}") from exc

        if original_len and len(raw) != original_len:
            raise PLNSError(f"mapping cloud length mismatch: expected={original_len} actual={len(raw)}")
        if cloud_size and len(raw) != cloud_size * 12:
            raise PLNSError(f"mapping cloud point length mismatch: cloud_size={cloud_size} bytes={len(raw)}")

        points: list[tuple[float, float, float]] = []
        for offset in range(0, len(raw), 12):
            x = int.from_bytes(raw[offset : offset + 4], byteorder="big", signed=True)
            y = int.from_bytes(raw[offset + 4 : offset + 8], byteorder="big", signed=True)
            z = int.from_bytes(raw[offset + 8 : offset + 12], byteorder="big", signed=True)
            points.append((x / 1000.0, y / 1000.0, z / 1000.0))
        return points

    @staticmethod
    def _decode_mapping_track_points(track: Any) -> list[tuple[float, float, float]]:
        """Function: Decode File-channel mapping trajectory points into meter-unit points."""

        if len(track.xs) != len(track.ys) or len(track.ys) != len(track.zs):
            raise PLNSError(
                f"mapping track coordinate length mismatch: xs={len(track.xs)} ys={len(track.ys)} zs={len(track.zs)}"
            )
        return [
            (int(track.xs[i]) / 1000.0, int(track.ys[i]) / 1000.0, int(track.zs[i]) / 1000.0)
            for i in range(len(track.xs))
        ]
