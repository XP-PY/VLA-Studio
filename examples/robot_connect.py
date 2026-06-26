from Robotic_Arm.rm_robot_interface import *

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
print("Handle: ", handle.id)

# print("Gripper open/close: ")
# print(arm.rm_set_gripper_position(200, True, 3))

print("Gripper state: ")
print(arm.rm_get_rm_plus_state_info())

# 获取IO 1通道状态
print("IO state: ")
# print(arm.rm_get_io_state(1))
# print(arm.rm_get_io_state(2))

# print(arm.rm_get_io_input())
print(arm.rm_get_io_output())

arm.rm_delete_robot_arm()