from lerobot.configs import parser
from dataclasses import dataclass
from lerobot.configs import (
    PreTrainedConfig
)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

@dataclass
class TestConfig():
    policy: PreTrainedConfig | None = None
    test_name: str = 'TEST'

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]

    def validate(self) -> None:
        print(f"[INFO] policy: {self.policy}")
        print(f"[INFO] policy type: {self.policy.type}")
        print(f"[INFO] policy device: {self.policy.device}")

@parser.wrap()
def test_config(cfg: TestConfig):
    cfg.validate()

import serial
import time
import os
import binascii
import numpy as np

MASTER_PORT = '/dev/ttyACM0' 
GHT_INFO_RATE = 50

def arm_conf():
    l_port = MASTER_PORT
    baudrate = 460800
    hex_data = "55 AA 02 00 00 67"
    #

    l_ser = serial.Serial(l_port, baudrate, timeout=0)

    bytes_to_send = binascii.unhexlify(hex_data.replace(" ", ""))
    l_ser.write(bytes_to_send)
    time.sleep(1)
    return l_ser, bytes_to_send

def bytes_to_signed_int(byte_data):
    value = int.from_bytes(byte_data, byteorder='little', signed=True)
    return value

def get_Joint(hex_received):
    J1 = hex_received[14:22]
    J1_byte_data = bytearray.fromhex(J1)
    Joint1 = bytes_to_signed_int(J1_byte_data) / 10000.0

    J2 = hex_received[24:32]
    J2_byte_data = bytearray.fromhex(J2)
    Joint2 = bytes_to_signed_int(J2_byte_data) / 10000.0

    J3 = hex_received[34:42]
    J3_byte_data = bytearray.fromhex(J3)
    Joint3 = bytes_to_signed_int(J3_byte_data) / 10000.0

    J4 = hex_received[44:52]
    J4_byte_data = bytearray.fromhex(J4)
    Joint4 = bytes_to_signed_int(J4_byte_data) / 10000.0

    J5 = hex_received[54:62]
    J5_byte_data = bytearray.fromhex(J5)
    Joint5 = bytes_to_signed_int(J5_byte_data) / 10000.0

    J6 = hex_received[64:72]
    J6_byte_data = bytearray.fromhex(J6)
    Joint6 = bytes_to_signed_int(J6_byte_data) / 10000.0

    G7 = hex_received[74:82]
    G7_byte_data = bytearray.fromhex(G7)
    Grasp = bytes_to_signed_int(G7_byte_data)

    return Joint1, Joint2, Joint3, Joint4, Joint5, Joint6, Grasp

def get_master_act(l_ser, byte_send):
    l_ser.write(byte_send)
    l_bytes_received = l_ser.read(l_ser.inWaiting())
    l_hex_received = binascii.hexlify(l_bytes_received).decode('utf-8').upper()
    l_actions = get_Joint(l_hex_received)
    return np.array((l_actions))

def thread_get_master_joints():
    l_ser, byte_send = arm_conf()
    interval = 1.0 / GHT_INFO_RATE  # 每次迭代之间应该等待的时间（秒）
    next_time = time.time()  # 初始化下一个动作的时间点

    while True:
        # 更新下一个动作的时间点
        next_time += interval
        joints = get_master_act(l_ser, byte_send).tolist()
        # if not sum([x == 0 and x == 0.0 for x in joints]) >= 3:
        #     if all(1001 >= y > -1001 for y in joints):
        #         real_time_data.now_master_joints = joints
        #         print("Master arm joints=%s gripper=%s" % (
        #             [round(v, 3) for v in joints[:-1]],
        #             round(float(joints[-1]), 3),
        #         ))

        print("Master arm joints=%s gripper=%s" % (
            [round(v, 3) for v in joints[:-1]],
            round(float(joints[-1]), 3),
        ))

        while time.time() < next_time:
            time.sleep(0.0001)

def check_master_arm():
    if not os.path.exists(MASTER_PORT):
        print(f"错误：设备 {MASTER_PORT} 不存在")

    # 检查当前用户是否有读写权限
    if not os.access(MASTER_PORT, os.R_OK | os.W_OK):
        print(f"警告：当前用户对 {MASTER_PORT} 缺少读写权限（需 sudo 或用户组配置）")
    else:
        print(f"设备 {MASTER_PORT} 存在且可读写")

        thread_get_master_joints()

def test_port():
    check_master_arm()

def main():
    test_port()


if __name__ == "__main__":
    main()
