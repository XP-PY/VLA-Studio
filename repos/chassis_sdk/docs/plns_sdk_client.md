# PLNS Python (暨科创平台底盘) SDK Client API

This SDK document describes the public `PLNSClient` APIs implemented in `chassis_sdk/client.py`.

This SDK is implemented based on the PLNS protocol document: `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`.
All PLNS protocol interface meanings can be looked up in this document.
All PLNS protocol message, enum, request, response, command, and notification definitions can be found in `chassis_sdk/proto/main.proto`.

The layout follows a simple SDK reference style: each method includes the method prototype, function description, corresponding PLNS protocol interface when available, parameter description, return value, and usage example.

Firmware-limited or unverified methods in `chassis_sdk/client.py` are intentionally not included in this document.
These methods are defined in the protocol or SDK for completeness, but the current firmware does not implement them.

## Table Of Contents

- [Basic Import](#basic-import)
- [Connection And Session](#connection-and-session)
  - [Initialize Client `__init__()`](#initialize-client-__init__)
  - [Connect `connect()`](#connect-connect)
  - [Close `close()`](#close-close)
  - [Async Context Enter `__aenter__()`](#async-context-enter-__aenter__)
  - [Async Context Exit `__aexit__()`](#async-context-exit-__aexit__)
  - [Login `login()`](#login-login)
  - [Logout `logout()`](#logout-logout)
- [Generic Protocol APIs](#generic-protocol-apis)
  - [Generic Request `request()`](#generic-request-request)
  - [Generic Command `command()`](#generic-command-command)
- [State Query APIs](#state-query-apis)
  - [Get Info `get_info()`](#get-info-get_info)
  - [Get System State `get_system_state()`](#get-system-state-get_system_state)
  - [Get Hardware State `get_hardware_state()`](#get-hardware-state-get_hardware_state)
  - [Get Map List `get_map_list()`](#get-map-list-get_map_list)
  - [Get Laser Points `get_laser_points()`](#get-laser-points-get_laser_points)
  - [Subscribe State `subscribe_state()`](#subscribe-state-subscribe_state)
- [Manual Control APIs](#manual-control-apis)
  - [Enable Manual `enable_manual()`](#enable-manual-enable_manual)
  - [Disable Manual `disable_manual()`](#disable-manual-disable_manual)
  - [Prepare Manual Control `prepare_manual_control()`](#prepare-manual-control-prepare_manual_control)
  - [Set Manual Velocity `set_manual_velocity()`](#set-manual-velocity-set_manual_velocity)
  - [Start Manual Velocity `start_manual_velocity()`](#start-manual-velocity-start_manual_velocity)
  - [Stop Manual Velocity `stop_manual_velocity()`](#stop-manual-velocity-stop_manual_velocity)
- [Obstacle Avoidance And Speed APIs](#obstacle-avoidance-and-speed-apis)
  - [Enable Obstacle Avoid `enable_obstacle_avoid()`](#enable-obstacle-avoid-enable_obstacle_avoid)
  - [Disable Obstacle Avoid `disable_obstacle_avoid()`](#disable-obstacle-avoid-disable_obstacle_avoid)
  - [Set Speed Level `set_speed_level()`](#set-speed-level-set_speed_level)
- [Localization And Mapping APIs](#localization-and-mapping-apis)
  - [Start Localization `start_localization()`](#start-localization-start_localization)
  - [Stop Localization `stop_localization()`](#stop-localization-stop_localization)
  - [Set Location Mode `set_location_mode()`](#set-location-mode-set_location_mode)
  - [Set Initial Pose `set_initial_pose()`](#set-initial-pose-set_initial_pose)
  - [Prepare Mapping `prepare_mapping()`](#prepare-mapping-prepare_mapping)
  - [Start Mapping `start_mapping()`](#start-mapping-start_mapping)
  - [Stop Mapping `stop_mapping()`](#stop-mapping-stop_mapping)
  - [Cancel Mapping `cancel_mapping()`](#cancel-mapping-cancel_mapping)
  - [Get Current Map `get_current_map()`](#get-current-map-get_current_map)
  - [Get Mapping Data `get_mapping_data()`](#get-mapping-data-get_mapping_data)
  - [Get Mapping Track `get_mapping_track()`](#get-mapping-track-get_mapping_track)
  - [Sync Maps `sync_maps()`](#sync-maps-sync_maps)
  - [Load Local Map `load_local_map()`](#load-local-map-load_local_map)
  - [Sync Time `sync_time()`](#sync-time-sync_time)
- [Movement Task APIs](#movement-task-apis)
  - [Pause Movement `pause_movement()`](#pause-movement-pause_movement)
  - [Continue Movement `continue_movement()`](#continue-movement-continue_movement)
  - [Cancel Movement Task `cancel_movement_task()`](#cancel-movement-task-cancel_movement_task)
  - [Prepare Navigation `prepare_navigation()`](#prepare-navigation-prepare_navigation)
  - [Go Station `go_station()`](#go-station-go_station)
  - [Go Pose `go_pose()`](#go-pose-go_pose)
- [Device Command APIs](#device-command-apis)
  - [Set Speaker Volume `set_speaker_volume()`](#set-speaker-volume-set_speaker_volume)
  - [Trigger Emergency `trigger_emergency()`](#trigger-emergency-trigger_emergency)
  - [Cancel Emergency `cancel_emergency()`](#cancel-emergency-cancel_emergency)
  - [Reboot PLNS `reboot_plns()`](#reboot-plns-reboot_plns)

## Basic Import

```python
import asyncio

from chassis_sdk import PLNSClient, PLNSConfig
from chassis_sdk.models import Pose2D
```

## Connection And Session

### Initialize Client `__init__()`

#### Method prototype

```python
PLNSClient(
    config: PLNSConfig | None = None,
    *,
    pb2_path: str | Path | None = None,
    raw_log_path: str | Path | None = None,
) -> None
```

#### Function description

Creates an async PLNS client. This method only creates local SDK objects and
runtime state; it does not connect to the chassis.

Corresponding PLNS protocol interface: none.

Initialized components:
- TCP configuration: stores the chassis IP, Cmd port, login account, heartbeat
  periods, frame-size limits, and timeout settings.
- Protobuf module: provides the generated PLNS message, enum, request,
  response, command, and notification classes used to encode and decode
  protocol messages.
- Message factory: builds common PLNS request and command protobuf messages
  from Python method arguments.
- Cmd-channel transport: implements the Cmd-channel TCP communication mechanism described in section `1. 通信机制` of the PLNS protocol document `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`. It sends and receives normal length-prefixed protobuf frames for login, requests, commands, movement tasks, and normal responses.
- File-channel transport: manages the secondary TCP channel at `config.port + 1`. This channel is used for File-channel login, mapping data, mapping track, and File-channel heartbeat. The File channel login must be performed after Cmd-channel login, using the Cmd-channel session ID. During mapping, the File channel uses a 1-second heartbeat interval. The File channel normally uses the same length-prefixed protobuf frame format as the Cmd channel, but mapping point-cloud frames use a special frame format: `AA 55 BB 66` magic bytes, followed by a 4-byte big-endian protobuf body size, followed by the protobuf message body. This File-channel mapping behavior was provided by the vendor engineer and is not described in the current PLNS protocol PDF.
- Session state: records login state, File-channel login state, session ID, and
  whether a mapping task is active.
- Cached state fields: stores the latest known system state, hardware state,
  vehicle information, task state, mapping cloud, and mapping track.
- Notification queues: buffer asynchronous PLNS notifications such as movement
  task completion messages.
- Mapping data queues: buffer File-channel mapping cloud and mapping track
  updates so applications can consume live mapping data without blocking the
  receive loop.
- Background task handles: track Cmd/File receive loops, Cmd/File heartbeat
  loops, and continuous manual-velocity sending tasks so they can be stopped
  cleanly when the client closes.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `config` | `PLNSConfig \| None` | PLNS connection, login, heartbeat, frame-size, mapping file heartbeat and timeout settings. If omitted, `PLNSConfig.from_env()` is used. Users usually do not need to modify these default settings. |
| `pb2_path` | `str \| Path \| None` | Optional protobuf Python module path. If omitted, the bundled protobuf module `chassis_sdk/proto/PlnsProto_pb2.py` is used. This bundled module is generated from the PLNS protocol `chassis_sdk/proto/main.proto` file, and can be regenerated with `examples/generate_plns_proto.py`. |
| `raw_log_path` | `str \| Path \| None` | Optional `.bin` file path for recording raw send and receive PLNS frames from both Cmd and File channels. The file stores the exact binary frame bytes sent to and received from the chassis, including frame headers and serialized protobuf message bodies, so it can be used later for protocol debugging and replay analysis. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | The initialized client instance is returned by the constructor itself. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient, PLNSConfig


async def main():
    config = PLNSConfig.from_env()
    # local log path
    raw_log_path = "logs/example_client_frames.bin"
    client = PLNSClient(config, raw_log_path=raw_log_path)
    try:
        print(f"Client target: {client.config.host}:{client.config.port}")
        print(f"Raw frame log: {raw_log_path}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### Connect `connect()`

#### Method prototype

```python
async def connect(self, *, login: bool = True) -> None
```

#### Function description

Connects to the PLNS Cmd TCP channel and starts the Cmd-channel background receive loop. When `login=True`, it also logs in to the Cmd channel, connects to the File channel on `config.port + 1`, logs in to the File channel with the Cmd session ID, and starts heartbeat loops.

Corresponding PLNS protocol interface: Cmd-channel `REQUEST_LOGIN` when login is enabled, File-channel `REQUEST_LOGIN` after Cmd login succeeds, and periodic `CMD_HEARTBEATS_UPDATE` heartbeat commands. During mapping, File-channel heartbeat uses `PLNSConfig.mapping_file_heartbeat_interval_s`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `login` | `bool` | Whether to log in, connect the File channel, and start heartbeat commands after the Cmd TCP connection. If `False`, only the Cmd TCP connection and receive loop are started. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    client = PLNSClient()
    try:
        await client.connect(login=True)
        print(f"Connected and logged in, session_id={client.session_id}")
        print(f"File channel logged in: {client.is_file_logged_in}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### Close `close()`

#### Method prototype

```python
async def close(self) -> None
```

#### Function description

Closes the PLNS client and releases the Cmd and File TCP connections.

Corresponding PLNS protocol interface: if the client is logged in, the SDK first sends `REQUEST_SYSTEM_STATE` to get system state information. 
When the chassis is in manual operation mode, it sends `CMD_SET_MANUAL_SPEED` with zero speed.
When an active navigation movement task is detected, it sends `CMD_CANCEL_MOVEMENT_TASK`. 
After motion cleanup, it sends `REQUEST_LOGOUT`.
Finally, the SDK closes the File channel, cancels local Cmd/File receive and heartbeat tasks, clears local session and mapping state, and closes the Cmd transport.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    client = PLNSClient()
    try:
        await client.connect(login=True)
        await client.get_system_state()
        print("Client is connected. Closing now.")
    finally:
        await client.close()
        print("Client closed.")


if __name__ == "__main__":
    asyncio.run(main())
```

### Async Context Enter `__aenter__()`

#### Method prototype

```python
async def __aenter__(self) -> PLNSClient
```

#### Function description

Connects to PLNS and logs in when entering an `async with` block.

Corresponding PLNS protocol interface: calls `connect()`, which logs in to the Cmd channel, connects and logs in to the File channel, and starts heartbeat commands.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `PLNSClient` | The connected client instance. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.get_info()
        print(f"Login session: {client.session_id}")
        print(f"File channel logged in: {client.is_file_logged_in}")
        print(f"Vehicle serial: {response.response.info.vehicle_serial_no}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Async Context Exit `__aexit__()`

#### Method prototype

```python
async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None
```

#### Function description

Closes the client when leaving an `async with` block by calling `close()`.

Corresponding PLNS protocol interface: the actual commands are the same as `close()`.
If the client is logged in, `close()` first sends `REQUEST_SYSTEM_STATE` to get system state information.
When the chassis is in manual operation mode, it sends `CMD_SET_MANUAL_SPEED` with zero speed.
When an active navigation movement task is detected, it sends `CMD_CANCEL_MOVEMENT_TASK`. 
After motion cleanup, it sends `REQUEST_LOGOUT`, closes the File channel, cancels local Cmd/File background tasks, clears local session and mapping state, and closes the Cmd transport.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `exc_type` | `object` | Exception type from the context, or `None` when no exception occurred. |
| `exc` | `object` | Exception instance from the context, or `None` when no exception occurred. |
| `tb` | `object` | Exception traceback from the context, or `None` when no exception occurred. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        state = await client.get_system_state()
        print(f"Session before exit: {client.session_id}")
        print(f"System state value: {int(state.response.system_state.sys_state)}")
    print("Client closed by __aexit__().")


if __name__ == "__main__":
    asyncio.run(main())
```

### Login `login()`

#### Method prototype

```python
async def login(self, username: str = "admin", password: str = "admin") -> Any
```

#### Function description

Sends a PLNS Cmd-channel login request and stores the returned session ID in the client.
The plain password is converted to MD5 before being sent.

Corresponding PLNS protocol interface: `REQUEST_LOGIN`.

This method only performs Cmd-channel login. It does not connect the File channel or start heartbeat tasks by itself. For normal use, call `connect(login=True)` or use `async with PLNSClient()`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `username` | `str` | PLNS login username. |
| `password` | `str` | Plain PLNS login password. The SDK converts it to MD5 before sending. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS `RESPONSE_LOGIN` protobuf message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    client = PLNSClient()
    try:
        await client.connect(login=False)
        response = await client.login("admin", "admin")
        print(f"Login session_id: {response.response.login_info.session_id}")
        print(f"Login result: {response.response.login_info.result_str}")
        print(f"File channel logged in: {client.is_file_logged_in}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### Logout `logout()`

#### Method prototype

```python
async def logout(self) -> Any
```

#### Function description

Sends a PLNS Cmd-channel logout request, closes the File channel, and clears local login and mapping state.

Corresponding PLNS protocol interface: `REQUEST_LOGOUT`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS `RESPONSE_LOGOUT` protobuf message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    client = PLNSClient()
    try:
        await client.connect(login=True)
        response = await client.logout()
        print(f"Logout response type: {int(response.response.response_type)}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Generic Protocol APIs

### Generic Request `request()`

#### Method prototype

```python
async def request(self, request_name: str, *, wait: bool = True) -> Any | None
```

#### Function description

Sends a generic PLNS `Request` message by enum name.

Corresponding PLNS protocol interface: any `Request.RequestType` name defined in `chassis_sdk/proto/main.proto` and each type meaning can be found in section `3. 请求消息说明` of protocol document `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`, such as `REQUEST_INFO` or `REQUEST_SYSTEM_STATE`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `request_name` | `str` | `RequestType` enum name defined in `main.proto`. |
| `wait` | `bool` | Whether to wait for and return the matching response. |

#### Return value

| Type | Description |
| --- | --- |
| `Any \| None` | PLNS response message when `wait=True`; otherwise `None`. Specific meanings of response message types can be found in section `4. 响应消息说明` of protocol document `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.request("REQUEST_INFO")
        print(f"Response type: {int(response.response.response_type)}")
        print(f"Vehicle serial: {response.response.info.vehicle_serial_no}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Generic Command `command()`

#### Method prototype

```python
async def command(self, command_name: str, *, wait: bool = True) -> Any | None
```

#### Function description

Sends a generic PLNS `Command` message by enum name without extra parameters.

Corresponding PLNS protocol interface: any parameterless `CommandType` name defined in `chassis_sdk/proto/main.proto` and each type meaning can be found in section `5. 命令消息说明` of protocol document `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`, such as `CMD_ENABLE_MANUAL_CONTROL`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `command_name` | `str` | `CommandType` enum name defined in `main.proto`. |
| `wait` | `bool` | Whether to wait for and return the matching response. |

#### Return value

| Type | Description |
| --- | --- |
| `Any \| None` | PLNS response message when `wait=True`; otherwise `None`. Specific meanings of response message types can be found in section `4. 响应消息说明` of protocol document `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.command("CMD_ENABLE_MANUAL_CONTROL")
        print(f"Enable manual result: {int(response.response.result.result_state)}")
        await client.command("CMD_DISABLE_MANUAL_CONTROL")
        print("Manual control disabled.")


if __name__ == "__main__":
    asyncio.run(main())
```

## State Query APIs

### Get Info `get_info()`

#### Method prototype

```python
async def get_info(self) -> Any
```

#### Function description

Requests vehicle or firmware information and caches the response in `client.info`.

Corresponding PLNS protocol interface: `REQUEST_INFO`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS `RESPONSE_INFO` protobuf message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.get_info()
        info = response.response.info
        print(f"PLNS serial: {info.plns_serial_no}")
        print(f"Vehicle serial: {info.vehicle_serial_no}")
        print(f"Hardware version: {info.hardware_version}")
        print(f"TC version: {info.tc_version}")
        print(f"Base version: {info.base_version}")
        print(f"Licence valid: {info.licence_valid}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Get System State `get_system_state()`

#### Method prototype

```python
async def get_system_state(self) -> Any
```

#### Function description

Requests one PLNS system-state snapshot and caches it in `client.system_state`.

Corresponding PLNS protocol interface: `REQUEST_SYSTEM_STATE`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS `RESPONSE_SYSTEM_STATE` protobuf message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.get_system_state()
        state = response.response.system_state
        pose = state.location_pose
        print(f"System state: {client.pb2.SystemState.SysState.Name(int(state.sys_state))}")
        print(f"Location state: {client.pb2.SystemState.LocationState.Name(int(state.location_state))}")
        print(f"Operation state: {client.pb2.SystemState.OperationState.Name(int(state.operation_state))}")
        # Note:
        # `state.map_name` is hardcoded in the firmware 
        # and does not reflect the map actually in use;
        # map information can be obtained through get_map_list().
        print(f"Map name: {state.map_name}")
        print(f"Pose: x={pose.x} mm, y={pose.y} mm, yaw={pose.yaw} mrad")
        print(f"Station no: {state.station_no}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Get Hardware State `get_hardware_state()`

#### Method prototype

```python
async def get_hardware_state(self) -> Any
```

#### Function description

Requests one PLNS hardware-state snapshot and caches it in
`client.hardware_state`.

Corresponding PLNS protocol interface: `REQUEST_HARDWARE_STATE`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS `RESPONSE_HARDWARE_STATE` protobuf message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.get_hardware_state()
        state = response.response.hardware_state
        print(f"Hardware state: {client.pb2.HardwareState.HState.Name(int(state.hardware_state))}")
        print(f"Battery: {state.battery_percentage}%")
        print(f"Battery voltage: {state.battery_voltage}")
        print(f"Battery current: {state.battery_current}")
        print(f"Laser state: {client.pb2.HardwareState.LaserState.Name(int(state.laser_state))}")
        print(f"WiFi state: {client.pb2.HardwareState.WiFiState.Name(int(state.wifi_state))}")
        print(f"Disk usage: {state.disk_usage}%")
        print(f"Remain disk space: {state.remain_disk_space}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Get Map List `get_map_list()`

#### Method prototype

```python
async def get_map_list(self) -> Any
```

#### Function description

Requests the list of map files known by PLNS.

Corresponding PLNS protocol interface: `REQUEST_MAP_LIST`.

Real-device testing showed that this request may only list `map.pgm` and `map.yaml`; `map.json` can still exist and should be handled through the SFTP map file workflow by `sync_maps()`.

Map file meanings:
- `map.pgm`: Grid map image file. It stores the 2D map pixels used for map visualization and lidar-based localization. Each pixel represents an area in the real world, usually meaning occupied, free, or unknown space.
- `map.yaml`: Map metadata file. It stores parameters such as map image path, resolution, origin, occupied threshold, free threshold, and negate flag.
- `map.json`: Road-network file. In the tested map files, it stores `edge`, `node`, and `station` data, including station poses, road connections, speed limits, acceleration/deceleration limits, rotation limits, and obstacle stop thresholds such as `obst_stop`.

More detailed map file information can be found in section `8. 文件 SFTP 服务通信协议` of the documentation `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS `RESPONSE_MAP_LIST` protobuf message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.get_map_list()
        files = response.response.map_list.list
        print(f"Map file count from REQUEST_MAP_LIST: {len(files)}")
        for item in files:
            print(
                f"name={item.name}, size={item.size}, "
                f"last_write_time={item.last_write_time}, md5={item.md5sum}"
            )
        print("Note: map.json may exist on SFTP even when it is not listed here.")


if __name__ == "__main__":
    asyncio.run(main())
```

### Get Laser Points `get_laser_points()`

#### Method prototype

```python
async def get_laser_points(self) -> Any
```

#### Function description

Requests one snapshot of laser point data.

Corresponding PLNS protocol interface: `REQUEST_LASER_POINTS`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS `RESPONSE_LASER_POINTS` protobuf message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.get_laser_points()
        points = response.response.laser_points
        print(f"Laser point count: {len(points.xs)}")
        # Get current 5 laser points information
        for index, xyz in enumerate(zip(points.xs[:5], points.ys[:5], points.zs[:5])):
            x, y, z = xyz
            print(f"point[{index}]: x={x}, y={y}, z={z}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Subscribe State `subscribe_state()`

#### Method prototype

```python
async def subscribe_state(self) -> None
```

#### Function description

Subscribes to the PLNS 10 Hz state stream.

Corresponding PLNS protocol interface: sends `REQUEST_STATE` without waiting for a response. Incoming `RESPONSE_STATE` frames are cached in `client.system_state` and `client.hardware_state`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.subscribe_state()
        await asyncio.sleep(0.5)
        system = client.system_state
        hardware = client.hardware_state
        if system is not None:
            print(f"Cached system state: {client.pb2.SystemState.SysState.Name(int(system.sys_state))}")
            # Note:
            # `system.map_name` is hardcoded in the firmware 
            # and does not reflect the map actually in use;
            # map information can be obtained through get_map_list().
            print(f"Cached map name: {system.map_name}")
        if hardware is not None:
            print(f"Cached battery: {hardware.battery_percentage}%")
            print(f"Cached disk usage: {hardware.disk_usage}%")


if __name__ == "__main__":
    asyncio.run(main())
```

## Manual Control APIs

### Enable Manual `enable_manual()`

#### Method prototype

```python
async def enable_manual(self) -> Any
```

#### Function description

Switches PLNS to manual-control mode.

Corresponding PLNS protocol interface: `CMD_ENABLE_MANUAL_CONTROL`.

PLNS commonly uses three operation modes, which correspond to `SystemState.operation_state`:
- `OPERATION_MANUAL`: Manual-control mode. The SDK directly sends velocity commands to control chassis movement, such as forward, backward, rotation, and stop.
- `OPERATION_AUTO`: Automatic navigation mode. The chassis executes navigation tasks by itself, such as going to a target pose or station, based on localization, map data, and obstacle-avoidance state.
- `OPERATION_NONE`: No explicit operation mode is active. This usually means the chassis is idle or no operation mode has been selected.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.enable_manual()
        print("Manual control mode enabled.")
        await asyncio.sleep(0.5)
        await client.disable_manual()
        print("Manual control mode disabled.")


if __name__ == "__main__":
    asyncio.run(main())
```

### Disable Manual `disable_manual()`

#### Method prototype

```python
async def disable_manual(self) -> Any
```

#### Function description

Exits manual-control mode and switches PLNS back to automatic navigation mode..

Corresponding PLNS protocol interface: `CMD_DISABLE_MANUAL_CONTROL`.

PLNS commonly uses three operation modes, which correspond to `SystemState.operation_state`:
- `OPERATION_MANUAL`: Manual-control mode. The SDK directly sends velocity commands to control chassis movement, such as forward, backward, rotation, and stop.
- `OPERATION_AUTO`: Automatic navigation mode. The chassis executes navigation tasks by itself, such as going to a target pose or station, based on localization, map data, and obstacle-avoidance state.
- `OPERATION_NONE`: No explicit operation mode is active. This usually means the chassis is idle or no operation mode has been selected.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.enable_manual()
        print("Manual control mode enabled first.")
        await client.disable_manual()
        print("Manual control mode disabled, chassis is back to auto mode.")


if __name__ == "__main__":
    asyncio.run(main())
```

### Prepare Manual Control `prepare_manual_control()`

#### Method prototype

```python
async def prepare_manual_control(self, *, cancel_existing: bool = True) -> None
```

#### Function description

Prepares PLNS before manual velocity control in manual-control mode. It reads system state, optionally cancels an active or failed movement task, and switches to manual-control mode when the chassis is not already in manual-control mode.

Corresponding PLNS protocol interfaces: `REQUEST_SYSTEM_STATE`, `CMD_CANCEL_MOVEMENT_TASK` when cleanup is needed, and `CMD_ENABLE_MANUAL_CONTROL` when the operation mode is not manual-control mode.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `cancel_existing` | `bool` | Whether to cancel an active or failed movement task before entering manual control. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


def print_manual_state(client: PLNSClient, label: str, response) -> None:
    state = response.response.system_state
    sys_state = client.pb2.SystemState.SysState.Name(int(state.sys_state))
    operation = client.pb2.SystemState.OperationState.Name(int(state.operation_state))
    movement = client.pb2.MovementTask.TaskState.Name(int(state.movement_state.state))
    print(f"{label}: sys={sys_state}, operation={operation}, movement={movement}")


async def main():
    async with PLNSClient() as client:
        before = await client.get_system_state()
        print_manual_state(client, "Before prepare_manual_control", before)

        await client.prepare_manual_control()

        after = await client.get_system_state()
        print_manual_state(client, "After prepare_manual_control", after)
        await client.disable_manual()


if __name__ == "__main__":
    asyncio.run(main())
```

### Set Manual Velocity `set_manual_velocity()`

#### Method prototype

```python
async def set_manual_velocity(
    self,
    vx_mm_s: int,
    w_mrad_s: int = 0,
    vy_mm_s: int = 0,
    *,
    wait: bool = False,
    prepare: bool = False,
) -> Any | None
```

#### Function description

Sends one manual velocity command.

Corresponding PLNS protocol interface: `CMD_SET_MANUAL_SPEED`.

The command writes `vx_mm_s` to `Command.param_int`, `w_mrad_s` to `Command.param_int1`, and `vy_mm_s` to `Command.param_int2`. When `prepare=True`, `prepare_manual_control()` is called first. Keep `prepare=False` inside high-rate loops.

For continuous manual movement, this command must be sent repeatedly with an interval shorter than 200 ms, as required by the protocol. A single `set_manual_velocity()` call should be treated as one speed update, not as a long-running motion command.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `vx_mm_s` | `int` | Linear velocity in the X direction, in mm/s. |
| `w_mrad_s` | `int` | Angular velocity, in 0.001 rad/s. |
| `vy_mm_s` | `int` | Linear velocity in the Y direction, in mm/s. |
| `wait` | `bool` | Whether to wait for and return the command response. |
| `prepare` | `bool` | Whether to prepare manual mode before sending this one command. |

#### Return value

| Type | Description |
| --- | --- |
| `Any \| None` | PLNS response message when `wait=True`; otherwise `None`. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def send_velocity_for(
    client: PLNSClient,
    vx_mm_s: int,
    w_mrad_s: int,
    duration_s: float,
    period_s: float = 0.1,
) -> None:
    end_time = asyncio.get_running_loop().time() + duration_s
    while asyncio.get_running_loop().time() < end_time:
        await client.set_manual_velocity(vx_mm_s, w_mrad_s)

        # Sending delay, not exceeding 200ms for continuous movement
        await asyncio.sleep(period_s)


async def main():
    async with PLNSClient() as client:
        await client.prepare_manual_control()
        try:
            # Continuously move forward at 120 mm/s for 1 second.
            await send_velocity_for(client, 120, 0, duration_s=1.0)

            # Continuously rotate left at 300 mrad/s for 1 second.
            await send_velocity_for(client, 0, 300, duration_s=1.0)

            # Stop the chassis.
            await client.set_manual_velocity(0, 0)
        finally:
            await client.stop_manual_velocity(disable_manual=True)


if __name__ == "__main__":
    asyncio.run(main())
```

#### Actual Effect
![manual_control](./asset/manual_control_ev.gif)

### Start Manual Velocity `start_manual_velocity()`

#### Method prototype

```python
async def start_manual_velocity(
    self,
    vx_mm_s: int,
    w_mrad_s: int = 0,
    vy_mm_s: int = 0,
    *,
    period_s: float = 0.1,
    prepare: bool = False,
) -> None
```

#### Function description

Starts a background loop that continuously sends manual velocity command `CMD_SET_MANUAL_SPEED`.

Corresponding PLNS protocol interface: repeatedly sends `CMD_SET_MANUAL_SPEED`. If `prepare=True`, it first prepares manual-control mode. The refresh period must be less than 0.2 seconds.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `vx_mm_s` | `int` | Linear velocity in the X direction, in mm/s. |
| `w_mrad_s` | `int` | Angular velocity, in 0.001 rad/s. |
| `vy_mm_s` | `int` | Linear velocity in the Y direction, in mm/s. |
| `period_s` | `float` | Velocity command refresh period in seconds. Must be less than 0.2. |
| `prepare` | `bool` | Whether to prepare manual-control mode before starting the loop. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        try:
            # Continuously move backward at 120 mm/s for 2 second.
            await client.start_manual_velocity(-120, 0, period_s=0.1, prepare=True)
            await asyncio.sleep(2.0)
            await client.stop_manual_velocity(disable_manual=False)

            # Continuously rotate right at 250 mrad/s for 2 second.
            await client.start_manual_velocity(0, -300, period_s=0.1, prepare=False)
            await asyncio.sleep(2.0)
        finally:
            # Stop motion and exit manual mode.
            await client.stop_manual_velocity(disable_manual=True)


if __name__ == "__main__":
    asyncio.run(main())
```

#### Actual Effect
![start_manual_velocity](./asset/start_manual_velocity_ev.gif)

### Stop Manual Velocity `stop_manual_velocity()`

#### Method prototype

```python
async def stop_manual_velocity(self, *, disable_manual: bool = True) -> None
```

#### Function description

Stops the background manual velocity loop, sends zero velocity and optionally exits manual-control mode.

Corresponding PLNS protocol interfaces: `CMD_SET_MANUAL_SPEED` with zero velocity and optionally `CMD_DISABLE_MANUAL_CONTROL`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `disable_manual` | `bool` | Whether to switch back to automatic navigation mode after stopping. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        try:
            # Continuously move backward at 120 mm/s for 2 second.
            await client.start_manual_velocity(-120, 0, period_s=0.1, prepare=True)
            await asyncio.sleep(2.0)
            await client.stop_manual_velocity(disable_manual=False)

            # Continuously rotate right at 250 mrad/s for 2 second.
            await client.start_manual_velocity(0, -300, period_s=0.1, prepare=False)
            await asyncio.sleep(2.0)
        finally:
            # Stop motion and exit manual mode.
            await client.stop_manual_velocity(disable_manual=True)


if __name__ == "__main__":
    asyncio.run(main())
```

#### Actual Effect
![stop_manual_velocity](./asset/start_manual_velocity_ev.gif)

## Obstacle Avoidance And Speed APIs

### Enable Obstacle Avoid `enable_obstacle_avoid()`

#### Method prototype

```python
async def enable_obstacle_avoid(self) -> Any
```

#### Function description

Sends the obstacle-avoid enable command.

Corresponding PLNS protocol interface: `CMD_ENABLE_OBSTACLE_AVOID`.

PLNS navigation currently involves two obstacle-avoidance policies:
- `OBSTACLE_AVOID_WAIT`: Stop-and-wait obstacle avoidance. The chassis stops when an obstacle is detected and continues after the obstacle is cleared. This is the only usable obstacle-avoidance mode on the current firmware.
- `OBSTACLE_AVOID_REPLAN`: Replanning obstacle avoidance. The chassis is expected to replan a new path around the obstacle, but this mode is not implemented on the current firmware.

Important real-device behavior: testing showed this command switches `movement_state.avoid_policy` to `OBSTACLE_AVOID_REPLAN`, so it should not be used for current navigation tests. Use `disable_obstacle_avoid()` instead to switch back to `OBSTACLE_AVOID_WAIT`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.enable_obstacle_avoid()
        state = await client.get_system_state()
        avoid_policy = state.response.system_state.movement_state.avoid_policy
        print(f"Avoid policy after enable: {int(avoid_policy)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Disable Obstacle Avoid `disable_obstacle_avoid()`

#### Method prototype

```python
async def disable_obstacle_avoid(self) -> Any
```

#### Function description

Sends the obstacle-avoid disable command.

Corresponding PLNS protocol interface: `CMD_DISABLE_OBSTACLE_AVOID`.

PLNS navigation currently involves two obstacle-avoidance policies:
- `OBSTACLE_AVOID_WAIT`: Stop-and-wait obstacle avoidance. The chassis stops when an obstacle is detected and continues after the obstacle is cleared. This is the only usable obstacle-avoidance mode on the current firmware.
- `OBSTACLE_AVOID_REPLAN`: Replanning obstacle avoidance. The chassis is expected to replan a new path around the obstacle, but this mode is not implemented on the current firmware.

Important real-device behavior: testing showed this switches `movement_state.avoid_policy` to `OBSTACLE_AVOID_WAIT`. On the current firmware, `OBSTACLE_AVOID_WAIT` is the only usable navigation obstacle-avoid mode. This method is therefore used by `prepare_navigation()`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.disable_obstacle_avoid()
        state = await client.get_system_state()
        avoid_policy = state.response.system_state.movement_state.avoid_policy
        print(f"Avoid policy after disable: {int(avoid_policy)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Set Speed Level `set_speed_level()`

#### Method prototype

```python
async def set_speed_level(self, level: int) -> Any
```

#### Function description

Sets the PLNS automatic navigation speed level.

Corresponding PLNS protocol interface: `CMD_SET_SPEED_LEVEL`, with the speed level in `Command.param_int`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `level` | `int` | Speed level from 0 to 100. Values outside this range raise `ValueError`. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.set_speed_level(50)
        print(f"Set speed level result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Localization And Mapping APIs

### Start Localization `start_localization()`

#### Method prototype

```python
async def start_localization(self) -> Any
```

#### Function description

Starts PLNS localization.

Corresponding PLNS protocol interface: `CMD_START_LOCATION`.

Localization must be running before using automatic navigation mode, such as `go_pose()` and `go_station()`, because the chassis needs its current pose on the map to plan and execute navigation tasks.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.start_localization()
        state = await client.get_system_state()
        print(f"Location state: {int(state.response.system_state.location_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Stop Localization `stop_localization()`

#### Method prototype

```python
async def stop_localization(self) -> Any
```

#### Function description

Stops PLNS localization.

Corresponding PLNS protocol interface: `CMD_STOP_LOCATION`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.stop_localization()
        state = await client.get_system_state()
        print(f"Location state after stop: {int(state.response.system_state.location_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Set Location Mode `set_location_mode()`

#### Method prototype

```python
async def set_location_mode(self, mode: int | str) -> Any
```

#### Function description

Sets the PLNS localization mode.

Corresponding PLNS protocol interface: `CMD_SET_LOCATION_MODE`, with `Command.location_mode` set to the selected mode.

Optional modes are `LOCATION_MODE_NONE`, `LOCATION_MODE_LIDAR`, `LOCATION_MODE_GNSS`, `LOCATION_MODE_MIX2`, and `LOCATION_MODE_INS`.

If localization fails or the chassis pose jumps to an incorrect position, manually set the initial pose with `set_initial_pose()` first, then start localization again.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `mode` | `int \| str` | Location mode integer or enum name, for example `LOCATION_MODE_LIDAR`. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.set_location_mode("LOCATION_MODE_LIDAR")
        print(f"Set location mode result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Set Initial Pose `set_initial_pose()`

#### Method prototype

```python
async def set_initial_pose(self, pose: Pose2D) -> Any
```

#### Function description

Sets the initial PLNS localization pose.

Corresponding PLNS protocol interface: `CMD_SET_LOCATION_INITIAL_POSE`, with `Command.pose` populated from `Pose2D`. The SDK converts meters to millimeters and radians to 0.001 rad units.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `pose` | `Pose2D` | Initial 2D pose expressed in meters and radians. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient
from chassis_sdk.models import Pose2D


async def main():
    async with PLNSClient() as client:
        initial_pose = Pose2D(0.0, 0.0, 0.0)
        response = await client.set_initial_pose(initial_pose)
        print(f"Set initial pose result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Prepare Mapping `prepare_mapping()`

#### Method prototype

```python
async def prepare_mapping(self, *, stop_localization: bool = True) -> None
```

#### Function description

Prepares PLNS state before starting map drawing.

Corresponding PLNS protocol interfaces: may call `REQUEST_SYSTEM_STATE`, `CMD_CANCEL_MOVEMENT_TASK`, `CMD_ENABLE_MANUAL_CONTROL`, and `CMD_STOP_LOCATION`, depending on the current chassis state.

Real-device mapping requires manual-control mode and localization stopped. This method calls `prepare_manual_control()` first, then calls `stop_localization()` when localization is still running and `stop_localization=True`.

Specific usage mapping examples can be found in `examples/mapping_keyboard.py`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `stop_localization` | `bool` | Whether to stop localization before mapping. Keep this `True` for real-device mapping unless a vendor workflow explicitly says otherwise. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.prepare_mapping()
        state = await client.get_system_state()
        print(f"Operation state: {int(state.response.system_state.operation_state)}")
        print(f"Location state: {int(state.response.system_state.location_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Start Mapping `start_mapping()`

#### Method prototype

```python
async def start_mapping(self, *, prepare: bool = True) -> Any
```

#### Function description

Starts drawing a new map.

Corresponding PLNS protocol interface: `CMD_NEW_MAP_START`.

By default, this method calls `prepare_mapping()` before sending the mapping command. 
After `CMD_NEW_MAP_START` succeeds, the SDK marks the client as being in mapping mode and immediately sends one File-channel heartbeat. 
While mapping is active, File-channel heartbeat uses `PLNSConfig.mapping_file_heartbeat_interval_s`.

Specific usage mapping examples can be found in `examples/mapping_keyboard.py`.
Although this SDK provides a mapping example script, it is recommended to use the vendor-provided CassieOne application for actual map creation: `asset/科创平台_灵童L1_资料_v0502/CassieOne_V3.0.23d/CassieOne_V3.0.23d.exe`.
The CassieOne application provides clearer map visualization and more complete map-editing features, such as visual map correction, station creation, road-network editing, and station navigation stop-obstacle threshold configuration.
The CassieOne application mapping user guide can be found in section `4.4 建图` and `4.5 修图` of the documentation `asset/科创平台_灵童L1_资料_v0502/灵童-L1 用户手册（单手遥控器版）.pdf`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `prepare` | `bool` | Whether to enter manual-control mode and stop localization before sending `CMD_NEW_MAP_START`. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        try:
            response = await client.start_mapping()
            print(f"Start mapping result: {int(response.response.result.result_state)}")
            await asyncio.sleep(1.0)
        finally:
            # Cancel this short example map so it is not saved.
            await client.cancel_mapping()


if __name__ == "__main__":
    asyncio.run(main())
```

### Stop Mapping `stop_mapping()`

#### Method prototype

```python
async def stop_mapping(self) -> Any
```

#### Function description

Stops and saves the current new-map drawing task.

Corresponding PLNS protocol interface: `CMD_NEW_MAP_STOP`.

Specific usage mapping examples can be found in `examples/mapping_keyboard.py`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.start_mapping()
        print("Mapping started. Move the chassis manually if needed.")
        await asyncio.sleep(2.0)
        response = await client.stop_mapping()
        print(f"Stop and save mapping result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Cancel Mapping `cancel_mapping()`

#### Method prototype

```python
async def cancel_mapping(self) -> Any
```

#### Function description

Cancels the current new-map drawing task without saving it.

Corresponding PLNS protocol interface: `CMD_NEW_MAP_CANCEL`.

Specific usage mapping examples can be found in `examples/mapping_keyboard.py`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.start_mapping()
        print("Mapping started for cancel test.")
        await asyncio.sleep(1.0)
        response = await client.cancel_mapping()
        print(f"Cancel mapping result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Get Current Map `get_current_map()`

#### Method prototype

```python
async def get_current_map(self) -> Any
```

#### Function description

Requests the current map through the Cmd channel.

Corresponding PLNS protocol interface: `REQUEST_GET_CUR_MAP`.

Important real-device behavior: in the verified mapping workflow, this request
is not the normal way to obtain live mapping point-cloud data. During mapping,
the chassis sends a File-channel `MSG_REQUEST` with
`Request.request_type = REQUEST_GET_CUR_MAP` to notify that new map data is
available. The SDK then calls `get_mapping_data(wait=False)` automatically to
request `REQUEST_GET_MAPPING_DATA` through the File channel.

For live mapping visualization, use `get_mapping_data()`, `get_mapping_track()`,
`client.mapping_cloud_updates`, `client.mapping_cloud_points`,
`client.mapping_track_updates`, and `client.mapping_track_points`.

Specific usage mapping examples can be found in `examples/mapping_keyboard.py`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS `RESPONSE_GET_CUR_MAP` protobuf message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.get_current_map()
        print(f"Response type: {int(response.response.response_type)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Get Mapping Data `get_mapping_data()`

#### Method prototype

```python
async def get_mapping_data(self, *, wait: bool = True) -> Any | None
```

#### Function description

Requests live mapping point-cloud data through the File channel while a new map
is being drawn.

Corresponding PLNS protocol interface: `REQUEST_GET_MAPPING_DATA` sent through
the File channel. The response is `RESPONSE_GET_MAPPING_DATA`.

Important real-device behavior: during mapping, the chassis first sends a
File-channel `REQUEST_GET_CUR_MAP` notification. The SDK handles that
notification automatically and calls `get_mapping_data(wait=False)`. When the
File-channel response arrives, the SDK decodes the compressed `CloudMap` data
with zlib and updates:

- `client.mapping_cloud`: raw protobuf `CloudMap`.
- `client.mapping_cloud_points`: decoded `(x, y, z)` points in meters.
- `client.mapping_cloud_updates`: async queue containing each
  `RESPONSE_GET_MAPPING_DATA` message.

The File channel is connected after Cmd-channel login. During mapping, the SDK
sends File-channel heartbeat every
`PLNSConfig.mapping_file_heartbeat_interval_s` seconds.

Specific usage mapping examples can be found in `examples/mapping_keyboard.py`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `wait` | `bool` | Whether to wait for and return the matching File-channel response. Use `False` when sending the request from notification-handling code. |

#### Return value

| Type | Description |
| --- | --- |
| `Any \| None` | PLNS `RESPONSE_GET_MAPPING_DATA` protobuf message when `wait=True`; otherwise `None`. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.start_mapping()
        try:
            response = await client.get_mapping_data()
            cloud = response.response.mapping_cloud
            print(f"Cloud size from response: {cloud.cloud_size}")
            print(f"Decoded point count: {len(client.mapping_cloud_points)}")
            print(f"First point: {client.mapping_cloud_points[:1]}")
        finally:
            await client.cancel_mapping()


if __name__ == "__main__":
    asyncio.run(main())
```

### Get Mapping Track `get_mapping_track()`

#### Method prototype

```python
async def get_mapping_track(self, timeout_s: float = 5.0) -> Any
```

#### Function description

Waits for the next live mapping trajectory message from the File channel.

Corresponding PLNS protocol interface: `RESPONSE_GET_MAPPING_TRACK` received
through the File channel.

Important real-device behavior: the upper computer does not request mapping
track data. The chassis actively pushes `RESPONSE_GET_MAPPING_TRACK` while
mapping. The SDK caches the latest track data in:

- `client.mapping_track`: raw protobuf `LaserPoints`.
- `client.mapping_track_points`: decoded `(x, y, z)` points in meters.
- `client.mapping_track_updates`: async queue containing each pushed
  `RESPONSE_GET_MAPPING_TRACK` message.

Specific usage mapping examples can be found in `examples/mapping_keyboard.py`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `timeout_s` | `float` | Maximum time to wait for the next pushed mapping-track response. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS `RESPONSE_GET_MAPPING_TRACK` protobuf message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.start_mapping()
        try:
            response = await client.get_mapping_track(timeout_s=5.0)
            track = response.response.mapping_trajectory
            print(f"Track point count from response: {len(track.xs)}")
            print(f"Decoded track point count: {len(client.mapping_track_points)}")
            print(f"First track point: {client.mapping_track_points[:1]}")
        finally:
            await client.cancel_mapping()


if __name__ == "__main__":
    asyncio.run(main())
```

### Sync Maps `sync_maps()`

#### Method prototype

```python
def sync_maps(self, config: SFTPConfig | None = None) -> None
```

#### Function description

Synchronizes map files, including `map.pgm`, `map.yaml` and `map.json`, from PLNS to local storage over SFTP.

Corresponding PLNS protocol interface: none on the TCP command channel. This method uses the SFTP map-file workflow through `sync_maps_via_sftp()`.

Map file meanings:
- `map.pgm`: Grid map image file. It stores the 2D map pixels used for map visualization and lidar-based localization. Each pixel represents an area in the real world, usually meaning occupied, free, or unknown space.
- `map.yaml`: Map metadata file. It stores parameters such as map image path, resolution, origin, occupied threshold, free threshold, and negate flag.
- `map.json`: Road-network file. In the tested map files, it stores `edge`, `node`, and `station` data, including station poses, road connections, speed limits, acceleration/deceleration limits, rotation limits, and obstacle stop thresholds such as `obst_stop`.

More detailed map file information can be found in section `8. 文件 SFTP 服务通信协议` of the documentation `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `config` | `SFTPConfig \| None` | SFTP sync configuration. If omitted, `SFTPConfig.from_env()` is used. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
from chassis_sdk import PLNSClient, SFTPConfig


def main():
    client = PLNSClient()
    config = SFTPConfig.from_env()
    client.sync_maps(config)
    print(f"Map files synced to: {config.local_dir}")


if __name__ == "__main__":
    main()
```

### Load Local Map `load_local_map()`

#### Method prototype

```python
def load_local_map(self, directory: str | Path) -> MapBundle
```

#### Function description

Loads a local map bundle from disk.

Corresponding PLNS protocol interface: none. This is a local utility that calls `load_map_bundle()`.

The directory should contain `map.json` and can also contain `map.yaml` and `map.pgm`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `directory` | `str \| Path` | Directory containing `map.json` and optional `map.yaml` and `map.pgm` files. |

#### Return value

| Type | Description |
| --- | --- |
| `MapBundle` | Parsed local map bundle. |

#### Usage example

```python
from pathlib import Path

from chassis_sdk import PLNSClient


def main():
    client = PLNSClient()
    map_dir = Path("maps")
    bundle = client.load_local_map(map_dir)
    print(f"Nodes: {len(bundle.road_map.nodes)}")
    print(f"Stations: {len(bundle.road_map.stations)}")
    print(f"Edges: {len(bundle.road_map.edges)}")
    if bundle.metadata:
        print(f"Resolution: {bundle.metadata.resolution_m}")
        print(f"Image path: {bundle.image_path}")


if __name__ == "__main__":
    main()
```

### Sync Time `sync_time()`

#### Method prototype

```python
async def sync_time(self, time_value: datetime | str | None = None) -> Any
```

#### Function description

Synchronizes PLNS system time.

Corresponding PLNS protocol interface: `CMD_SYNC_TIME`.

Important real-device behavior: the current firmware expects `Command.param_str` in `yyyy-MM-dd HH:mm:ss` format. This differs from the protocol, `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`, description that used an integer timestamp.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `time_value` | `datetime \| str \| None` | Time to send. If omitted, the current local time is used. A string must already be formatted as `yyyy-MM-dd HH:mm:ss`. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio
from datetime import datetime

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.sync_time(datetime.now())
        print(f"Sync time result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Movement Task APIs

Movement Task APIs are used in automatic navigation mode. Before calling `go_station()` or `go_pose()`, localization must be running and PLNS should be in `OPERATION_AUTO`; `prepare_navigation()` is provided to set the required navigation state.

### Pause Movement `pause_movement()`

#### Method prototype

```python
async def pause_movement(self) -> Any
```

#### Function description

Pauses the current movement task.

Corresponding PLNS protocol interface: `CMD_PAUSE_MOVEMENT`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient
from chassis_sdk.models import Pose2D


async def main():
    async with PLNSClient() as client:
        try:
            await client.go_pose(Pose2D(1.0, 0.0, 0.0))
            await asyncio.sleep(1.0)
            response = await client.pause_movement()
            print(f"Pause movement result: {int(response.response.result.result_state)}")
        finally:
            await client.cancel_movement_task()


if __name__ == "__main__":
    asyncio.run(main())
```

### Continue Movement `continue_movement()`

#### Method prototype

```python
async def continue_movement(self) -> Any
```

#### Function description

Continues the current movement task.

Corresponding PLNS protocol interface: `CMD_CONTINUE_MOVEMENT`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient
from chassis_sdk.models import Pose2D


async def main():
    async with PLNSClient() as client:
        try:
            await client.go_pose(Pose2D(1.0, 0.0, 0.0))
            await asyncio.sleep(1.0)
            await client.pause_movement()
            await asyncio.sleep(0.5)
            response = await client.continue_movement()
            print(f"Continue movement result: {int(response.response.result.result_state)}")
        finally:
            await client.cancel_movement_task()


if __name__ == "__main__":
    asyncio.run(main())
```

### Cancel Movement Task `cancel_movement_task()`

#### Method prototype

```python
async def cancel_movement_task(self) -> Any
```

#### Function description

Cancels the current movement task.

Corresponding PLNS protocol interface: `CMD_CANCEL_MOVEMENT_TASK`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient
from chassis_sdk.models import Pose2D


async def main():
    async with PLNSClient() as client:
        await client.go_pose(Pose2D(1.0, 0.0, 0.0))
        await asyncio.sleep(1.0)
        response = await client.cancel_movement_task()
        print(f"Cancel movement task result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Prepare Navigation `prepare_navigation()`

#### Method prototype

```python
async def prepare_navigation(
    self,
    *,
    location_mode: int | str | None = "LOCATION_MODE_LIDAR",
    cancel_existing: bool = True,
    enable_localization: bool = True,
    switch_to_auto: bool = True,
    use_obstacle_wait: bool = True,
) -> None
```

#### Function description

Prepares PLNS before sending a go-pose or go-station movement task.

Corresponding PLNS protocol interfaces: `REQUEST_SYSTEM_STATE`, `CMD_CANCEL_MOVEMENT_TASK` when cleanup is needed, `CMD_START_LOCATION`, `CMD_SET_LOCATION_MODE`, `CMD_DISABLE_MANUAL_CONTROL`, and `CMD_DISABLE_OBSTACLE_AVOID`.

Important real-device behavior: navigation requires localization, automatic operation mode, no active movement task, and `OBSTACLE_AVOID_WAIT` on the current firmware. The SDK uses `CMD_DISABLE_OBSTACLE_AVOID` to select `OBSTACLE_AVOID_WAIT`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `location_mode` | `int \| str \| None` | Optional localization mode to set before navigation. Use `None` to skip mode setting. |
| `cancel_existing` | `bool` | Whether to cancel an active or failed movement task. |
| `enable_localization` | `bool` | Whether to start localization when it is not running. |
| `switch_to_auto` | `bool` | Whether to leave manual-control mode before navigation. |
| `use_obstacle_wait` | `bool` | Whether to send `CMD_DISABLE_OBSTACLE_AVOID` to select `OBSTACLE_AVOID_WAIT`. |

#### Return value

| Type | Description |
| --- | --- |
| `None` | No value is returned. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        await client.prepare_navigation()
        state = await client.get_system_state()
        print(f"Operation state: {int(state.response.system_state.operation_state)}")
        print(f"Location state: {int(state.response.system_state.location_state)}")
        print(f"Avoid policy: {int(state.response.system_state.movement_state.avoid_policy)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Go Station `go_station()`

#### Method prototype

```python
async def go_station(
    self,
    station_id: int,
    road_map: RoadMap | None = None,
    *,
    prepare: bool = True,
) -> Any
```

#### Function description

Creates a movement task that navigates to a station.

Corresponding PLNS protocol interface: `CMD_NEW_MOVEMENT_TASK` with `MovementTask.type = MT_MOVE_TO_STATION`.

Go-station can use the stop-obstacle threshold configured in road-map station or edge data by configing it using the vendor-provided CassieOne application `asset/科创平台_灵童L1_资料_v0502/CassieOne_V3.0.23d/CassieOne_V3.0.23d.exe`. The CassieOne application stop-obstacle threshold configuration guide can be found in section `4.7 常见问题` of the documentation `asset/科创平台_灵童L1_资料_v0502/灵童-L1 用户手册（单手遥控器版）.pdf`. Or you can modify road-network file `map.json` in the PLNS system directory `/home/nvidia/maps/` that more detailed information can be found in section `8. 文件 SFTP 服务通信协议` of the documentation `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`.

Real-device navigation must still run in automatic navigation mode and `OBSTACLE_AVOID_WAIT`; the default `prepare=True` performs that preparation through `prepare_navigation()`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `station_id` | `int` | Target station ID. If the target `station_id` does not exist in that road map, the SDK raises an error instead of sending an invalid navigation task to the chassis. |
| `road_map` | `RoadMap \| None` | Optional local road map for station validation. If `road_map` is provided, the SDK checks the local road-network data loaded from `map.json` before sending the task. |
| `prepare` | `bool` | Whether to run `prepare_navigation()` before sending the task. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def wait_for_movement_finished(client: PLNSClient, timeout_s: float = 120.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Movement did not finish within {timeout_s} seconds.")

        notification = await asyncio.wait_for(client.notifications.get(), timeout=remaining)
        notify_type = int(notification.notify_type)
        notify_name = client.pb2.Notification.NotifyType.Name(notify_type)
        print(f"Navigation notification: {notify_name}")

        # Specific meanings of notify message types can be found in section `6. 通知消息说明` of protocol document `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`.
        if notify_name == "NOTIFY_MOVE_PATH_SENT":
            # Some firmware states wait for a continue command after path planning.
            await client.continue_movement()

        if notify_name == "NOTIFY_MOVE_TASK_FINISHED":
            result = notification.task_result
            result_state = client.pb2.Result.ResultState.Name(int(result.result_state))
            result_code = client.pb2.Result.ResultCode.Name(int(result.result_code))
            print(f"Movement finished: {result_state}/{result_code}")
            return


async def main():
    # Navigate from station 2 to station 3
    station_id = 3
    async with PLNSClient() as client:
        finished = False
        try:
            response = await client.go_station(station_id)
            print(f"Go station result: {int(response.response.result.result_state)}")
            await wait_for_movement_finished(client)
            finished = True
        finally:
            if not finished:
                await client.cancel_movement_task()


if __name__ == "__main__":
    asyncio.run(main())
```

#### Actual Effect
![Station screenshot](./asset/go_station.png)
![go_station](./asset/go_station_ev.gif)

### Go Pose `go_pose()`

#### Method prototype

```python
async def go_pose(self, pose: Pose2D, *, prepare: bool = True) -> Any
```

#### Function description

Creates a movement task that navigates to a map pose.

Corresponding PLNS protocol interface: `CMD_NEW_MOVEMENT_TASK` with `MovementTask.type = MT_MOVE_TO_POSE`.

Important real-device behavior: the current firmware does not expose a configurable stop-obstacle threshold for go-pose tasks.
Real-device navigation must run in automatic navigation mode and stop-obstacle mode `OBSTACLE_AVOID_WAIT`; the default `prepare=True` performs that preparation.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `pose` | `Pose2D` | Target 2D pose expressed in meters and radians. |
| `prepare` | `bool` | Whether to run `prepare_navigation()` before sending the task. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient
from chassis_sdk.models import Pose2D


async def wait_for_movement_finished(client: PLNSClient, timeout_s: float = 120.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Movement did not finish within {timeout_s} seconds.")

        notification = await asyncio.wait_for(client.notifications.get(), timeout=remaining)
        notify_type = int(notification.notify_type)
        notify_name = client.pb2.Notification.NotifyType.Name(notify_type)
        print(f"Navigation notification: {notify_name}")

        # Specific meanings of notify message types can be found in section `6. 通知消息说明` of protocol document `asset/科创平台_灵童L1_资料_v0502/PLNS网络通信协议V2.1.pdf`.
        if notify_name == "NOTIFY_MOVE_PATH_SENT":
            # Some firmware states wait for a continue command after path planning.
            await client.continue_movement()

        if notify_name == "NOTIFY_MOVE_TASK_FINISHED":
            result = notification.task_result
            result_state = client.pb2.Result.ResultState.Name(int(result.result_state))
            result_code = client.pb2.Result.ResultCode.Name(int(result.result_code))
            print(f"Movement finished: {result_state}/{result_code}")
            return


async def main():
    # Navigate from station 3 to target pose (x=6.5 m, y=-0.1 m, yaw=0.0 rad).
    target_pose = Pose2D(6.5, -0.1, 0.0)
    async with PLNSClient() as client:
        finished = False
        try:
            response = await client.go_pose(target_pose)
            print(f"Go pose result: {int(response.response.result.result_state)}")
            await wait_for_movement_finished(client)
            finished = True
        finally:
            if not finished:
                await client.cancel_movement_task()


if __name__ == "__main__":
    asyncio.run(main())
```

#### Actual Effect
![go_pose](./asset/go_pose_ev.gif)

## Device Command APIs

### Set Speaker Volume `set_speaker_volume()`

#### Method prototype

```python
async def set_speaker_volume(self, volume: int) -> Any
```

#### Function description

Sets the VSC speaker volume.

Corresponding PLNS protocol interface: `CMD_SET_SPEAKER_VOLUME`, with the volume in `Command.param_int`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| `volume` | `int` | Speaker volume from 0 to 100. Values outside this range raise `ValueError`. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.set_speaker_volume(50)
        print(f"Set speaker volume result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Trigger Emergency `trigger_emergency()`

#### Method prototype

```python
async def trigger_emergency(self) -> Any
```

#### Function description

Triggers PLNS emergency stop.

Corresponding PLNS protocol interface: `CMD_TRIGGER_EMERGENCY`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.trigger_emergency()
        print(f"Trigger emergency result: {int(response.response.result.result_state)}")
        await asyncio.sleep(0.5)
        await client.cancel_emergency()
        print("Emergency state cleared.")


if __name__ == "__main__":
    asyncio.run(main())
```

### Cancel Emergency `cancel_emergency()`

#### Method prototype

```python
async def cancel_emergency(self) -> Any
```

#### Function description

Clears PLNS emergency stop state.

Corresponding PLNS protocol interface: `CMD_CANCEL_EMERGENCY`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    async with PLNSClient() as client:
        response = await client.cancel_emergency()
        print(f"Cancel emergency result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Reboot PLNS `reboot_plns()`

#### Method prototype

```python
async def reboot_plns(self) -> Any
```

#### Function description

Reboots the PLNS module.

Corresponding PLNS protocol interface: `CMD_REBOOT_PLNS`.

#### Parameter description

| Parameter | Type | Description |
| --- | --- | --- |
| None | - | This method has no user parameters. |

#### Return value

| Type | Description |
| --- | --- |
| `Any` | PLNS command response message. |

#### Usage example

```python
import asyncio

from chassis_sdk import PLNSClient


async def main():
    confirm = input("Type REBOOT to reboot PLNS: ")
    if confirm != "REBOOT":
        print("Reboot skipped.")
        return

    async with PLNSClient() as client:
        response = await client.reboot_plns()
        print(f"Reboot PLNS result: {int(response.response.result.result_state)}")


if __name__ == "__main__":
    asyncio.run(main())
```
