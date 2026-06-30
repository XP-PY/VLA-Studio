#!/usr/bin/python3
# coding=utf-8
###################################################################
  #  _____   _   _____       ___   _____   _____   _____
  # |  _  \ | | /  ___|     /   | /  ___/ /  ___/ /  _  \
  # | |_| | | | | |        / /| | | |___  | |___  | | | |
  # |  ___/ | | | |       / / | | \___  \ \___  \ | | | |
  # | |     | | | |___   / /  | |  ___| |  ___| | | |_| |
  # |_|     |_| \_____| /_/   |_| /_____/ /_____/ \_____/
  #
  # Copyright (c) 2024 Picasso Robot Technology Co., Ltd.
  # All rights reserved.
  # @author shun
  # @brief   Generated PLNS protocol file. Do not modify.
###################################################################

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='plns.proto',
  package='proto',
  syntax='proto3',
  serialized_pb=_b('\n\nplns.proto\x12\x05proto\"\xdc\x01\n\x07Message\x12 \n\x04type\x18\x01 \x01(\x0e\x32\x12.proto.MessageType\x12\x0b\n\x03seq\x18\x02 \x01(\x07\x12\x12\n\nsession_id\x18\x03 \x01(\x06\x12\x1f\n\x07request\x18\x04 \x01(\x0b\x32\x0e.proto.Request\x12!\n\x08response\x18\x05 \x01(\x0b\x32\x0f.proto.Response\x12\x1f\n\x07\x63ommand\x18\x06 \x01(\x0b\x32\x0e.proto.Command\x12)\n\x0cnotification\x18\x07 \x01(\x0b\x32\x13.proto.Notification\"\xe0\x05\n\x07Request\x12\x30\n\x0crequest_type\x18\x01 \x01(\x0e\x32\x1a.proto.Request.RequestType\x12*\n\rlogin_request\x18\x02 \x01(\x0b\x32\x13.proto.LoginRequest\x12/\n\x07\x66ile_op\x18\x03 \x01(\x0e\x32\x1e.proto.Request.FileOperateType\x12\x11\n\tparam_str\x18\x04 \x01(\t\x12\x12\n\nparam_str1\x18\x05 \x01(\t\x12\x12\n\nparam_str2\x18\x06 \x01(\t\"\xf5\x02\n\x0bRequestType\x12\x11\n\rREQUEST_LOGIN\x10\x00\x12\x10\n\x0cREQUEST_INFO\x10\x01\x12\x11\n\rREQUEST_STATE\x10\x02\x12\x18\n\x14REQUEST_SYSTEM_STATE\x10\x03\x12\x1a\n\x16REQUEST_HARDWARE_STATE\x10\x04\x12\x14\n\x10REQUEST_MAP_LIST\x10\x05\x12\x18\n\x14REQUEST_LASER_POINTS\x10\x06\x12\x16\n\x12REQUEST_TASK_STATE\x10\x07\x12\x18\n\x14REQUEST_FILE_OPERATE\x10\x08\x12\x12\n\x0eREQUEST_LOGOUT\x10\t\x12\x15\n\x11REQUEST_CHANGE_PW\x10\n\x12\x15\n\x11REQUEST_TIMESTAMP\x10\x0b\x12\x17\n\x13REQUEST_GET_CUR_MAP\x10\x0c\x12\x1c\n\x18REQUEST_GET_MAPPING_DATA\x10\r\x12\x1d\n\x19REQUEST_GET_MAPPING_TRACK\x10\x0e\"\x92\x01\n\x0f\x46ileOperateType\x12\x15\n\x11\x46ILE_OPERATE_ZERO\x10\x00\x12\x13\n\x0fIMPORT_MAP_FILE\x10\x01\x12\x16\n\x12IMPORT_CONFIG_FILE\x10\x02\x12\x1d\n\x19IMPORT_SYSTEM_UPDATE_FILE\x10\x03\x12\x1c\n\x18RESTORE_FACTORY_SETTINGS\x10\x04\"\xca\x08\n\x08Response\x12\x33\n\rresponse_type\x18\x01 \x01(\x0e\x32\x1c.proto.Response.ResponseType\x12$\n\nlogin_info\x18\x02 \x01(\x0b\x32\x10.proto.LoginInfo\x12\x19\n\x04info\x18\x03 \x01(\x0b\x32\x0b.proto.Info\x12(\n\x0csystem_state\x18\x04 \x01(\x0b\x32\x12.proto.SystemState\x12,\n\x0ehardware_state\x18\x05 \x01(\x0b\x32\x14.proto.HardwareState\x12 \n\x08map_list\x18\x06 \x01(\x0b\x32\x0e.proto.MapList\x12(\n\x0claser_points\x18\x07 \x01(\x0b\x32\x12.proto.LaserPoints\x12*\n\rmovement_task\x18\x08 \x01(\x0b\x32\x13.proto.MovementTask\x12&\n\x0b\x61\x63tion_task\x18\t \x01(\x0b\x32\x11.proto.ActionTask\x12\x33\n\x12occupancy_grid_map\x18\n \x01(\x0b\x32\x17.proto.OccupancyGridMap\x12&\n\rmapping_cloud\x18\x0b \x01(\x0b\x32\x0f.proto.CloudMap\x12.\n\x12mapping_trajectory\x18\x0c \x01(\x0b\x32\x12.proto.LaserPoints\x12\x1d\n\x06result\x18\r \x01(\x0b\x32\r.proto.Result\x12.\n\x0fnotify_response\x18\x0e \x01(\x0b\x32\x15.proto.NotifyResponse\x12\x1d\n\x04list\x18\x14 \x01(\x0b\x32\x0f.proto.FileList\x12\x11\n\ttimestamp\x18\x15 \x01(\x04\x12\x0f\n\x07retInfo\x18\x16 \x01(\t\"\xb0\x03\n\x0cResponseType\x12\x12\n\x0eRESPONSE_LOGIN\x10\x00\x12\x11\n\rRESPONSE_INFO\x10\x01\x12\x12\n\x0eRESPONSE_STATE\x10\x02\x12\x19\n\x15RESPONSE_SYSTEM_STATE\x10\x03\x12\x1b\n\x17RESPONSE_HARDWARE_STATE\x10\x04\x12\x15\n\x11RESPONSE_MAP_LIST\x10\x05\x12\x19\n\x15RESPONSE_LASER_POINTS\x10\x06\x12\x17\n\x13RESPONSE_TASK_STATE\x10\x07\x12\x19\n\x15RESPONSE_FILE_OPERATE\x10\x08\x12\x13\n\x0fRESPONSE_LOGOUT\x10\t\x12\x16\n\x12RESPONSE_CHANGE_PW\x10\n\x12\x16\n\x12RESPONSE_TIMESTAMP\x10\x0b\x12\x18\n\x14RESPONSE_GET_CUR_MAP\x10\x0c\x12\x1d\n\x19RESPONSE_GET_MAPPING_DATA\x10\r\x12\x1e\n\x1aRESPONSE_GET_MAPPING_TRACK\x10\x0e\x12\x14\n\x10RESPONSE_COMMAND\x10\x14\x12\x13\n\x0fRESPONSE_NOTIFY\x10\x15\"\xd9\x02\n\x07\x43ommand\x12#\n\x07\x63ommand\x18\x01 \x01(\x0e\x32\x12.proto.CommandType\x12\x11\n\tparam_int\x18\x02 \x01(\x05\x12\x12\n\nparam_int1\x18\x03 \x01(\x05\x12\x12\n\nparam_int2\x18\x04 \x01(\x05\x12\x19\n\x04pose\x18\x05 \x01(\x0b\x32\x0b.proto.Pose\x12\x11\n\tparam_str\x18\x06 \x01(\t\x12\x12\n\nparam_str1\x18\x07 \x01(\t\x12*\n\rlocation_mode\x18\x08 \x01(\x0e\x32\x13.proto.LocationMode\x12*\n\rmovement_task\x18\t \x01(\x0b\x32\x13.proto.MovementTask\x12&\n\x0b\x61\x63tion_task\x18\n \x01(\x0b\x32\x11.proto.ActionTask\x12\x18\n\x10general_io_ouput\x18\x0b \x01(\x04\x12\x12\n\nextend_cmd\x18\x0c \x01(\x05\"\xc7\x04\n\x0cNotification\x12\x33\n\x0bnotify_type\x18\x01 \x01(\x0e\x32\x1e.proto.Notification.NotifyType\x12\x16\n\x0ereserved_field\x18\x02 \x01(\r\x12\x16\n\x0e\x63ur_station_no\x18\x03 \x01(\r\x12\x1d\n\x08\x63ur_pose\x18\x04 \x01(\x0b\x32\x0b.proto.Pose\x12\"\n\x0btask_result\x18\x05 \x01(\x0b\x32\r.proto.Result\x12*\n\rmovement_task\x18\n \x01(\x0b\x32\x13.proto.MovementTask\x12&\n\x0b\x61\x63tion_task\x18\x0b \x01(\x0b\x32\x11.proto.ActionTask\x12,\n\ncal_result\x18\x0c \x01(\x0b\x32\x18.proto.CalibrationResult\x12*\n\rupdate_result\x18\r \x01(\x0b\x32\x13.proto.UpdateResult\x12\x19\n\x11pass_station_type\x18\x0e \x01(\r\"\xc5\x01\n\nNotifyType\x12\x1d\n\x19NOTIFY_MOVE_TASK_FINISHED\x10\x00\x12\x1f\n\x1bNOTIFY_ACTION_TASK_FINISHED\x10\x01\x12\x1f\n\x1bNOTIFY_CALIBRATION_FINISHED\x10\x02\x12\x1a\n\x16NOTIFY_UPDATE_FINISHED\x10\x03\x12\x19\n\x15NOTIFY_MOVE_PATH_SENT\x10\x04\x12\x1f\n\x1bNOTIFY_PASS_THROUGH_STATION\x10\x05\"\x95\x02\n\x0eNotifyResponse\x12\x0b\n\x03\x61\x63k\x18\x01 \x01(\x08\x12.\n\x04type\x18\x02 \x01(\x0e\x32 .proto.NotifyResponse.NotifyType\"\xc5\x01\n\nNotifyType\x12\x1d\n\x19NOTIFY_MOVE_TASK_FINISHED\x10\x00\x12\x1f\n\x1bNOTIFY_ACTION_TASK_FINISHED\x10\x01\x12\x1f\n\x1bNOTIFY_CALIBRATION_FINISHED\x10\x02\x12\x1a\n\x16NOTIFY_UPDATE_FINISHED\x10\x03\x12\x19\n\x15NOTIFY_MOVE_PATH_SENT\x10\x04\x12\x1f\n\x1bNOTIFY_PASS_THROUGH_STATION\x10\x05\"H\n\x11\x43\x61librationResult\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\r\n\x05theta\x18\x03 \x01(\x01\x12\x0e\n\x06status\x18\x04 \x01(\x05\"x\n\x0cUpdateResult\x12\x33\n\x0bresult_code\x18\x01 \x01(\x0e\x32\x1e.proto.UpdateResult.ResultCode\"3\n\nResultCode\x12\x12\n\x0eUPDATE_SUCCESS\x10\x00\x12\x11\n\rUPDATE_FAILED\x10\x01\"2\n\x0cLoginRequest\x12\x10\n\x08username\x18\x01 \x01(\t\x12\x10\n\x08password\x18\x02 \x01(\t\"3\n\tLoginInfo\x12\x12\n\nsession_id\x18\x01 \x01(\x06\x12\x12\n\nresult_str\x18\x02 \x01(\t\"\x94\x01\n\x04Info\x12\x15\n\rlicence_valid\x18\x01 \x01(\t\x12\x16\n\x0eplns_serial_no\x18\x02 \x01(\t\x12\x18\n\x10hardware_version\x18\x03 \x01(\t\x12\x12\n\ntc_version\x18\x04 \x01(\t\x12\x14\n\x0c\x62\x61se_version\x18\x05 \x01(\t\x12\x19\n\x11vehicle_serial_no\x18\x06 \x01(\t\"h\n\x08\x46ileInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04size\x18\x02 \x01(\x05\x12\x17\n\x0flast_write_time\x18\x03 \x01(\x04\x12\x17\n\x0f\x66ile_compressed\x18\x04 \x01(\x08\x12\x0e\n\x06md5sum\x18\x05 \x01(\t\"9\n\x07MapList\x12\x0f\n\x07req_seq\x18\x01 \x01(\r\x12\x1d\n\x04list\x18\x02 \x03(\x0b\x32\x0f.proto.FileInfo\":\n\x08\x46ileList\x12\x0f\n\x07req_seq\x18\x01 \x01(\r\x12\x1d\n\x04list\x18\x02 \x03(\x0b\x32\x0f.proto.FileInfo\"e\n\x04Pose\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\x12\t\n\x01z\x18\x03 \x01(\x05\x12\x0c\n\x04roll\x18\x04 \x01(\x05\x12\r\n\x05pitch\x18\x05 \x01(\x05\x12\x0b\n\x03yaw\x18\x06 \x01(\x05\x12\x12\n\nconfidence\x18\x07 \x01(\x05\"\xda\x02\n\x04Path\x12\"\n\x04type\x18\x01 \x01(\x0e\x32\x14.proto.Path.PathType\x12\n\n\x02sx\x18\x02 \x01(\x05\x12\n\n\x02sy\x18\x03 \x01(\x05\x12\n\n\x02\x65x\x18\x04 \x01(\x05\x12\n\n\x02\x65y\x18\x05 \x01(\x05\x12\n\n\x02\x63x\x18\x06 \x01(\x05\x12\n\n\x02\x63y\x18\x07 \x01(\x05\x12\n\n\x02\x64x\x18\x08 \x01(\x05\x12\n\n\x02\x64y\x18\t \x01(\x05\x12\x14\n\x0crotate_angle\x18\n \x01(\x05\x12\x11\n\tdirection\x18\x0b \x01(\x05\x12\x11\n\tobst_stop\x18\x0c \x01(\x05\x12\x0f\n\x07limit_v\x18\r \x01(\x05\x12\x0f\n\x07limit_w\x18\x0e \x01(\x05\x12\x11\n\tlimit_acc\x18\x0f \x01(\x05\x12\x11\n\tlimit_dec\x18\x10 \x01(\x05\"J\n\x08PathType\x12\r\n\tPATH_ZERO\x10\x00\x12\r\n\tPATH_LINE\x10\x01\x12\x0f\n\x0bPATH_BEZIER\x10\x02\x12\x0f\n\x0bPATH_ROTATE\x10\x03\"\xe2\x01\n\x12MotionControlState\x12\x30\n\x05state\x18\x01 \x01(\x0e\x32!.proto.MotionControlState.MCState\x12\x0b\n\x03v_x\x18\x02 \x01(\x05\x12\x0b\n\x03v_y\x18\x03 \x01(\x05\x12\t\n\x01w\x18\x04 \x01(\x05\x12\x0f\n\x07path_no\x18\x05 \x01(\x05\"d\n\x07MCState\x12\x0b\n\x07MC_ZERO\x10\x00\x12\x11\n\rMC_INITIALING\x10\x01\x12\x0b\n\x07MC_IDLE\x10\x02\x12\x13\n\x0fMC_PATH_RUNNING\x10\x03\x12\x17\n\x13MC_VELOCITY_RUNNING\x10\x04\"\xda\x05\n\x0cMovementTask\x12,\n\x05state\x18\x01 \x01(\x0e\x32\x1d.proto.MovementTask.TaskState\x12<\n\x10\x64st_station_type\x18\x03 \x01(\x0e\x32\".proto.MovementTask.DstStationType\x12*\n\x04type\x18\x04 \x01(\x0e\x32\x1c.proto.MovementTask.TaskType\x12\x10\n\x08stations\x18\x05 \x03(\r\x12\x1a\n\x05poses\x18\x06 \x03(\x0b\x32\x0b.proto.Pose\x12\x1a\n\x05paths\x18\x07 \x03(\x0b\x32\x0b.proto.Path\x12\x35\n\x0c\x61void_policy\x18\x08 \x01(\x0e\x32\x1f.proto.MovementTask.AvoidPolicy\x12\x13\n\x0bremain_time\x18\t \x01(\x05\x12\x17\n\x0fremain_distance\x18\n \x01(\x05\x12\x16\n\x0etotal_distance\x18\x0b \x01(\x05\"]\n\tTaskState\x12\t\n\x05MT_NA\x10\x00\x12\x15\n\x11MT_WAIT_FOR_START\x10\x01\x12\x0e\n\nMT_RUNNING\x10\x02\x12\r\n\tMT_PAUSED\x10\x03\x12\x0f\n\x0bMT_FINISHED\x10\x04\"2\n\x0e\x44stStationType\x12\x0e\n\nDS_DEFAULT\x10\x00\x12\x10\n\x0c\x44S_NO_ROTATE\x10\x01\"o\n\x08TaskType\x12\x0b\n\x07MT_ZERO\x10\x00\x12\x17\n\x13MT_MOVE_FOLLOW_PATH\x10\x01\x12\x16\n\x12MT_MOVE_TO_STATION\x10\x02\x12\x13\n\x0fMT_MOVE_TO_POSE\x10\x03\x12\x10\n\x0cMT_MOVE_MIX2\x10\x04\"g\n\x0b\x41voidPolicy\x12\x0b\n\x07\x41T_ZERO\x10\x00\x12\x17\n\x13OBSTACLE_AVOID_WAIT\x10\x01\x12\x19\n\x15OBSTACLE_AVOID_REPLAN\x10\x02\x12\x17\n\x13OBSTACLE_AVOID_NONE\x10\x03\"\xc5\x02\n\nActionTask\x12*\n\x05state\x18\x01 \x01(\x0e\x32\x1b.proto.ActionTask.TaskState\x12(\n\x04type\x18\x03 \x01(\x0e\x32\x1a.proto.ActionTask.TaskType\x12\x0f\n\x07timeout\x18\x04 \x01(\x05\x12\n\n\x02id\x18\x05 \x01(\x05\"]\n\tTaskState\x12\t\n\x05\x41T_NA\x10\x00\x12\x15\n\x11\x41T_WAIT_FOR_START\x10\x01\x12\x0e\n\nAT_RUNNING\x10\x02\x12\r\n\tAT_PAUSED\x10\x03\x12\x0f\n\x0b\x41T_FINISHED\x10\x04\"e\n\x08TaskType\x12\x0b\n\x07\x41T_ZERO\x10\x00\x12\x16\n\x12\x41T_WAITING_TIMEOUT\x10\x01\x12\x0e\n\nAT_JACK_UP\x10\x02\x12\x10\n\x0c\x41T_JACK_DOWN\x10\x03\x12\x12\n\x0e\x41T_TAKE_PHOTOS\x10\x04\"\xd8\x12\n\x0bSystemState\x12\x0f\n\x07req_seq\x18\x01 \x01(\r\x12.\n\tsys_state\x18\x02 \x01(\x0e\x32\x1b.proto.SystemState.SysState\x12\x38\n\x0elocation_state\x18\x03 \x01(\x0e\x32 .proto.SystemState.LocationState\x12*\n\rlocation_mode\x18\x04 \x01(\x0e\x32\x13.proto.LocationMode\x12\"\n\rlocation_pose\x18\x05 \x01(\x0b\x32\x0b.proto.Pose\x12:\n\x0f\x65mergency_state\x18\x06 \x01(\x0e\x32!.proto.SystemState.EmergencyState\x12+\n\x08mc_state\x18\x07 \x01(\x0b\x32\x19.proto.MotionControlState\x12+\n\x0emovement_state\x18\x08 \x01(\x0b\x32\x13.proto.MovementTask\x12\'\n\x0c\x61\x63tion_state\x18\t \x01(\x0b\x32\x11.proto.ActionTask\x12\x12\n\nstation_no\x18\n \x01(\r\x12\x10\n\x08map_name\x18\x0b \x01(\t\x12\x1b\n\x13map_saving_progress\x18\x0c \x01(\x05\x12:\n\x0foperation_state\x18\r \x01(\x0e\x32!.proto.SystemState.OperationState\x12;\n\x10oba_enable_state\x18\x0e \x01(\x0e\x32!.proto.SystemState.ObaEnableState\x12.\n\trun_state\x18\x0f \x01(\x0e\x32\x1b.proto.SystemState.RunState\x12\x30\n\nload_state\x18\x10 \x01(\x0e\x32\x1c.proto.SystemState.LoadState\x12\x37\n\x0egnss_msg_state\x18\x12 \x01(\x0e\x32\x1f.proto.SystemState.GnssMsgState\x12\x1e\n\tgnss_pose\x18\x13 \x01(\x0b\x32\x0b.proto.Pose\x12\x15\n\rsatellite_num\x18\x14 \x01(\r\x12\x13\n\x0bspeed_level\x18\x1f \x01(\r\x12\x12\n\nfault_code\x18  \x01(\r\x12\x15\n\rtotal_mileage\x18! \x01(\r\x12\x1f\n\x07uiState\x18\" \x01(\x0b\x32\x0e.proto.UiState\"\x86\x06\n\x08SysState\x12\x12\n\x0eSYS_STATE_ZERO\x10\x00\x12\x18\n\x14SYS_STATE_INITIALING\x10\x01\x12\x12\n\x0eSYS_STATE_IDLE\x10\x02\x12\x13\n\x0fSYS_STATE_ERROR\x10\x03\x12\x1c\n\x18SYS_STATE_HARDWARE_ERROR\x10\x04\x12\x1c\n\x18SYS_STATE_START_LOCATING\x10\x05\x12!\n\x1dSYS_STATE_TASK_NEWMAP_DRAWING\x10\x06\x12 \n\x1cSYS_STATE_TASK_NEWMAP_SAVING\x10\x07\x12 \n\x1cSYS_STATE_TASK_MANUAL_PAUSED\x10\x08\x12!\n\x1dSYS_STATE_TASK_NAV_INITIALING\x10\t\x12#\n\x1fSYS_STATE_TASK_NAV_FINDING_PATH\x10\n\x12%\n!SYS_STATE_TASK_NAV_WAITING_FINISH\x10\x0b\x12*\n&SYS_STATE_TASK_NAV_WAITING_FINISH_SLOW\x10\x0c\x12%\n!SYS_STATE_TASK_NAV_REFINDING_PATH\x10\r\x12\x1d\n\x19SYS_STATE_TASK_NAV_PAUSED\x10\x0e\x12\x1d\n\x19SYS_STATE_TASK_NAV_NO_WAY\x10\x0f\x12!\n\x1dSYS_STATE_TASK_NAV_NO_STATION\x10\x10\x12!\n\x1dSYS_STATE_TASK_NAV_PATH_ERROR\x10\x11\x12&\n\"SYS_STATE_TASK_PATH_NAV_INITIALING\x10\x12\x12&\n\"SYS_STATE_TASK_PATH_WAITING_FINISH\x10\x13\x12+\n\'SYS_STATE_TASK_PATH_WAITING_FINISH_SLOW\x10\x14\x12\x1e\n\x1aSYS_STATE_TASK_PATH_PAUSED\x10\x15\x12\x1d\n\x19SYS_STATE_TASK_PATH_ERROR\x10\x16\"\xb5\x01\n\rLocationState\x12\x17\n\x13LOCATION_STATE_ZERO\x10\x00\x12\x17\n\x13LOCATION_STATE_NONE\x10\x01\x12\x1d\n\x19LOCATION_STATE_INITIALING\x10\x02\x12\x1a\n\x16LOCATION_STATE_RUNNING\x10\x03\x12\x1d\n\x19LOCATION_STATE_RELOCATING\x10\x04\x12\x18\n\x14LOCATION_STATE_ERROR\x10\x05\"\x7f\n\x0e\x45mergencyState\x12\x16\n\x12STATE_EMERGENCY_NA\x10\x00\x12\x18\n\x14STATE_EMERGENCY_NONE\x10\x01\x12\x1a\n\x16STATE_EMERGENCY_TRIGER\x10\x02\x12\x1f\n\x1bSTATE_EMERGENCY_RECOVERABLE\x10\x03\"N\n\x0eOperationState\x12\x12\n\x0eOPERATION_NONE\x10\x00\x12\x12\n\x0eOPERATION_AUTO\x10\x01\x12\x14\n\x10OPERATION_MANUAL\x10\x02\"?\n\x0eObaEnableState\x12\n\n\x06OBA_NA\x10\x00\x12\x0f\n\x0bOBA_ENABLED\x10\x01\x12\x10\n\x0cOBA_DISABLED\x10\x02\"a\n\x08RunState\x12\n\n\x06RUN_NA\x10\x00\x12\x0c\n\x08RUN_IDLE\x10\x01\x12\x0f\n\x0bRUN_RUNNING\x10\x02\x12\x16\n\x12RUN_BLOCK_SLOWDOWN\x10\x03\x12\x12\n\x0eRUN_BLOCK_STOP\x10\x04\"8\n\tLoadState\x12\r\n\tLOAD_NONE\x10\x00\x12\r\n\tLOAD_FREE\x10\x01\x12\r\n\tLOAD_FULL\x10\x02\"`\n\x0cGnssMsgState\x12\x14\n\x10GNSS_MSG_INVALID\x10\x00\x12\x13\n\x0fGNSS_MSG_SINGLE\x10\x01\x12\x11\n\rGNSS_MSG_DIFF\x10\x04\x12\x12\n\x0eGNSS_MSG_FLOAT\x10\x05\"\xc7\x0b\n\rHardwareState\x12\x0f\n\x07req_seq\x18\x01 \x01(\r\x12\x11\n\tcpu_usage\x18\x02 \x01(\r\x12\x14\n\x0cmemory_usage\x18\x03 \x01(\r\x12\x12\n\ndisk_usage\x18\x04 \x01(\r\x12\x19\n\x11remain_disk_space\x18\x05 \x01(\r\x12\x38\n\rbattery_state\x18\t \x01(\x0e\x32!.proto.HardwareState.BatteryState\x12\x1a\n\x12\x62\x61ttery_percentage\x18\n \x01(\r\x12\x17\n\x0f\x62\x61ttery_voltage\x18\x0b \x01(\r\x12\x17\n\x0f\x62\x61ttery_current\x18\x0c \x01(\x05\x12\x1b\n\x13\x62\x61ttery_tempreature\x18\r \x01(\x05\x12\x1b\n\x13\x62\x61ttery_remain_time\x18\x0e \x01(\r\x12\x34\n\x0blaser_state\x18\x10 \x01(\x0e\x32\x1f.proto.HardwareState.LaserState\x12\x32\n\nwifi_state\x18\x11 \x01(\x0e\x32\x1e.proto.HardwareState.WiFiState\x12\x11\n\twifi_name\x18\x12 \x01(\t\x12\x15\n\rwifi_strength\x18\x13 \x01(\r\x12=\n\x0e\x62reak_sw_state\x18\x14 \x01(\x0e\x32%.proto.HardwareState.BreakSwitchState\x12\x38\n\rjacking_state\x18\x15 \x01(\x0e\x32!.proto.HardwareState.JackingState\x12\x32\n\ngnss_state\x18\x16 \x01(\x0e\x32\x1e.proto.HardwareState.GnssState\x12\x17\n\x0f\x63pu_temperature\x18\x1e \x01(\x05\x12\x16\n\x0e\x62ox_temprature\x18\x1f \x01(\x05\x12\x18\n\x10general_io_input\x18  \x01(\x04\x12\x19\n\x11general_io_output\x18! \x01(\x04\x12\x33\n\x12src_hardware_state\x18\" \x01(\x0b\x32\x17.proto.SRCHardwareState\x12\x33\n\x0ehardware_state\x18$ \x01(\x0e\x32\x1b.proto.HardwareState.HState\x12\x1b\n\x13hardware_error_code\x18% \x01(\r\x12\x0f\n\x07\x63o_info\x18( \x01(\x05\x12\x10\n\x08gas_info\x18) \x01(\x05\"M\n\x0c\x42\x61tteryState\x12\x0e\n\nBATTERY_NA\x10\x00\x12\x14\n\x10\x42\x41TTERY_CHARGING\x10\x01\x12\x17\n\x13\x42\x41TTERY_NO_CHARGING\x10\x02\"L\n\nLaserState\x12\x0c\n\x08LASER_NA\x10\x00\x12\x11\n\rLASER_INITING\x10\x01\x12\x0c\n\x08LASER_OK\x10\x02\x12\x0f\n\x0bLASER_ERROR\x10\x03\"V\n\tWiFiState\x12\x0b\n\x07WIFI_NA\x10\x00\x12\x15\n\x11WIFI_DISCONNECTED\x10\x01\x12\x12\n\x0eWIFI_CONNECTED\x10\x02\x12\x11\n\rWIFI_SCANNING\x10\x03\"F\n\x10\x42reakSwitchState\x12\x0f\n\x0b\x42REAK_SW_NA\x10\x00\x12\x10\n\x0c\x42REAK_SW_OFF\x10\x01\x12\x0f\n\x0b\x42REAK_SW_ON\x10\x02\"a\n\x0cJackingState\x12\x0b\n\x07JACK_NA\x10\x00\x12\x0b\n\x07JACK_UP\x10\x01\x12\r\n\tJACK_DOWN\x10\x02\x12\x12\n\x0eJACK_MOVING_UP\x10\x03\x12\x14\n\x10JACK_MOVING_DOWN\x10\x04\"C\n\tGnssState\x12\x0b\n\x07GNSS_NA\x10\x00\x12\x15\n\x11GNSS_DISCONNECTED\x10\x01\x12\x12\n\x0eGNSS_CONNECTED\x10\x02\"U\n\x06HState\x12\x10\n\x0cH_STATE_ZERO\x10\x00\x12\x16\n\x12H_STATE_INITIALING\x10\x01\x12\x0e\n\nH_STATE_OK\x10\x02\x12\x11\n\rH_STATE_ERROR\x10\x03\"\xc0\x01\n\x10SRCHardwareState\x12\x16\n\x0em1_status_code\x18\x01 \x01(\r\x12\x16\n\x0em2_status_code\x18\x02 \x01(\r\x12\x16\n\x0em3_status_code\x18\x03 \x01(\r\x12\x16\n\x0em4_status_code\x18\x04 \x01(\r\x12\x19\n\x11total_power_cycle\x18\x05 \x01(\r\x12\x1a\n\x12total_poweron_time\x18\x06 \x01(\r\x12\x15\n\rtotal_mileage\x18\x07 \x01(\r\"N\n\x0bLaserPoints\x12\x0f\n\x07req_seq\x18\x01 \x01(\r\x12\x0e\n\x02xs\x18\x02 \x03(\x05\x42\x02\x10\x01\x12\x0e\n\x02ys\x18\x03 \x03(\x05\x42\x02\x10\x01\x12\x0e\n\x02zs\x18\x04 \x03(\x05\x42\x02\x10\x01\",\n\tMapHeader\x12\r\n\x05stamp\x18\x01 \x01(\x01\x12\x10\n\x08\x66rame_id\x18\x02 \x01(\t\"t\n\x0bMapMetaData\x12\x15\n\rmap_load_time\x18\x02 \x01(\x01\x12\x12\n\nresolution\x18\x03 \x01(\x02\x12\r\n\x05width\x18\x04 \x01(\r\x12\x0e\n\x06height\x18\x05 \x01(\r\x12\x1b\n\x06origin\x18\x06 \x01(\x0b\x32\x0b.proto.Pose\"\x81\x01\n\x10OccupancyGridMap\x12 \n\x06header\x18\x01 \x01(\x0b\x32\x10.proto.MapHeader\x12 \n\x04info\x18\x02 \x01(\x0b\x32\x12.proto.MapMetaData\x12\x1b\n\x13\x63ompressed_char_len\x18\x03 \x01(\r\x12\x0c\n\x04\x64\x61ta\x18\x04 \x03(\r\"p\n\x08\x43loudMap\x12\x0f\n\x07req_seq\x18\x01 \x01(\r\x12\x12\n\ncloud_size\x18\x02 \x01(\r\x12\x14\n\x0corg_char_len\x18\x03 \x01(\r\x12\x1b\n\x13\x63ompressed_char_len\x18\x04 \x01(\r\x12\x0c\n\x04\x64\x61ta\x18\x05 \x03(\r\"\xdb\x04\n\x06Result\x12/\n\x0cresult_state\x18\x01 \x01(\x0e\x32\x19.proto.Result.ResultState\x12-\n\x0bresult_code\x18\x02 \x01(\x0e\x32\x18.proto.Result.ResultCode\"_\n\x0bResultState\x12\x11\n\rRESPONSE_NONE\x10\x00\x12\x17\n\x13RESPONSE_PROCESSING\x10\x01\x12\x0f\n\x0bRESPONSE_OK\x10\x02\x12\x13\n\x0fRESPONSE_FAILED\x10\x03\"\x8f\x03\n\nResultCode\x12\x14\n\x10RESULT_CODE_NONE\x10\x00\x12\x14\n\x10RESULT_UNDEFINED\x10\x01\x12\x1c\n\x18RESULT_INVALID_CMD_PARAM\x10\x02\x12#\n\x1fRESULT_USER_OR_PASSWROD_INVALID\x10\x03\x12\x1d\n\x19RESULT_SYSTEM_STATE_ERROR\x10\x04\x12\x1f\n\x1bRESULT_OPERATION_MODE_ERROR\x10\x05\x12\x1e\n\x1aRESULT_LOAD_MAP_FILE_ERROR\x10\x06\x12\x1b\n\x17RESULT_PRE_TASK_RUNNING\x10\x07\x12\x15\n\x11RESULT_NAV_NO_WAY\x10\x08\x12\x19\n\x15RESULT_NAV_NO_STATION\x10\t\x12\x1e\n\x1aRESULT_NAV_PATH_PLAN_ERROR\x10\n\x12\x1e\n\x1aRESULT_NAV_PLAN_STEP_ERROR\x10\x0b\x12#\n\x1fRESULT_NAV_MOTION_CONTROL_ERROR\x10\x0c\"\xd3\x02\n\x07UiState\x12\x30\n\x0cparent_state\x18\x01 \x01(\x0e\x32\x1a.proto.UiState.ParentState\x12.\n\x0b\x63hild_state\x18\x02 \x01(\x0e\x32\x19.proto.UiState.ChildState\"\x93\x01\n\x0bParentState\x12\x15\n\x11PARENT_STATE_NONE\x10\x00\x12\x19\n\x15PARENT_STATE_RUNNABLE\x10\x01\x12\x18\n\x14PARENT_STATE_MAPPING\x10\x02\x12\x1d\n\x19PARENT_STATE_LOCALIZATION\x10\x03\x12\x19\n\x15PARENT_STATE_HW_FAULT\x10\x04\"P\n\nChildState\x12\x14\n\x10\x43HILD_STATE_NONE\x10\x00\x12\x16\n\x12\x43HILD_STATE_MANUAL\x10\x01\x12\x14\n\x10\x43HILD_STATE_AUTO\x10\x02*W\n\x0bMessageType\x12\x0f\n\x0bMSG_REQUEST\x10\x00\x12\x10\n\x0cMSG_RESPONSE\x10\x01\x12\x0f\n\x0bMSG_COMMAND\x10\x02\x12\x14\n\x10MSG_NOTIFICATION\x10\x03*\xdf\x06\n\x0b\x43ommandType\x12\x0c\n\x08\x43MD_ZERO\x10\x00\x12\x19\n\x15\x43MD_HEARTBEATS_UPDATE\x10\x01\x12\x1d\n\x19\x43MD_ENABLE_MANUAL_CONTROL\x10\x02\x12\x1e\n\x1a\x43MD_DISABLE_MANUAL_CONTROL\x10\x03\x12\x18\n\x14\x43MD_SET_MANUAL_SPEED\x10\x04\x12\x17\n\x13\x43MD_SET_SPEED_LEVEL\x10\x05\x12\x17\n\x13\x43MD_SPEED_SLOW_DOWN\x10\x06\x12\x16\n\x12\x43MD_PAUSE_MOVEMENT\x10\x08\x12\x19\n\x15\x43MD_CONTINUE_MOVEMENT\x10\t\x12\x15\n\x11\x43MD_STOP_MOVEMENT\x10\n\x12\x1d\n\x19\x43MD_ENABLE_OBSTACLE_AVOID\x10\x0b\x12\x1e\n\x1a\x43MD_DISABLE_OBSTACLE_AVOID\x10\x0c\x12\x19\n\x15\x43MD_NEW_MOVEMENT_TASK\x10\x10\x12\x1c\n\x18\x43MD_CANCEL_MOVEMENT_TASK\x10\x11\x12\x17\n\x13\x43MD_NEW_ACTION_TASK\x10\x12\x12\x1a\n\x16\x43MD_CANCEL_ACTION_TASK\x10\x13\x12\x18\n\x14\x43MD_CANCEL_EMERGENCY\x10\x15\x12\x19\n\x15\x43MD_TRIGGER_EMERGENCY\x10\x16\x12\x1a\n\x16\x43MD_ENABLE_AUTO_CHARGE\x10\x17\x12\x13\n\x0f\x43MD_STOP_CHARGE\x10\x18\x12\x16\n\x12\x43MD_START_LOCATION\x10 \x12\x15\n\x11\x43MD_STOP_LOCATION\x10!\x12\x19\n\x15\x43MD_SET_LOCATION_MODE\x10\"\x12!\n\x1d\x43MD_SET_LOCATION_INITIAL_POSE\x10#\x12\x15\n\x11\x43MD_NEW_MAP_START\x10$\x12\x14\n\x10\x43MD_NEW_MAP_STOP\x10%\x12\x16\n\x12\x43MD_NEW_MAP_CANCEL\x10&\x12\x11\n\rCMD_SYNC_TIME\x10\x30\x12\x13\n\x0f\x43MD_CALIBRATION\x10\x31\x12\x13\n\x0f\x43MD_REBOOT_PLNS\x10\x32\x12\x1c\n\x18\x43MD_SET_GENERAL_IO_OUPUT\x10\x35\x12\x1a\n\x16\x43MD_SET_SPEAKER_VOLUME\x10\x36\x12\x16\n\x12\x43MD_EXTEND_CONTROL\x10\x37*\x86\x01\n\x0cLocationMode\x12\x16\n\x12LOCATION_MODE_NONE\x10\x00\x12\x17\n\x13LOCATION_MODE_LIDAR\x10\x01\x12\x16\n\x12LOCATION_MODE_GNSS\x10\x02\x12\x16\n\x12LOCATION_MODE_MIX2\x10\x03\x12\x15\n\x11LOCATION_MODE_INS\x10\x04\x42\x02H\x03\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_MESSAGETYPE = _descriptor.EnumDescriptor(
  name='MessageType',
  full_name='proto.MessageType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MSG_REQUEST', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MSG_RESPONSE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MSG_COMMAND', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MSG_NOTIFICATION', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=11227,
  serialized_end=11314,
)
_sym_db.RegisterEnumDescriptor(_MESSAGETYPE)

MessageType = enum_type_wrapper.EnumTypeWrapper(_MESSAGETYPE)
_COMMANDTYPE = _descriptor.EnumDescriptor(
  name='CommandType',
  full_name='proto.CommandType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CMD_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_HEARTBEATS_UPDATE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_ENABLE_MANUAL_CONTROL', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_DISABLE_MANUAL_CONTROL', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_SET_MANUAL_SPEED', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_SET_SPEED_LEVEL', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_SPEED_SLOW_DOWN', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_PAUSE_MOVEMENT', index=7, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_CONTINUE_MOVEMENT', index=8, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_STOP_MOVEMENT', index=9, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_ENABLE_OBSTACLE_AVOID', index=10, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_DISABLE_OBSTACLE_AVOID', index=11, number=12,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_NEW_MOVEMENT_TASK', index=12, number=16,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_CANCEL_MOVEMENT_TASK', index=13, number=17,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_NEW_ACTION_TASK', index=14, number=18,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_CANCEL_ACTION_TASK', index=15, number=19,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_CANCEL_EMERGENCY', index=16, number=21,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_TRIGGER_EMERGENCY', index=17, number=22,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_ENABLE_AUTO_CHARGE', index=18, number=23,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_STOP_CHARGE', index=19, number=24,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_START_LOCATION', index=20, number=32,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_STOP_LOCATION', index=21, number=33,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_SET_LOCATION_MODE', index=22, number=34,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_SET_LOCATION_INITIAL_POSE', index=23, number=35,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_NEW_MAP_START', index=24, number=36,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_NEW_MAP_STOP', index=25, number=37,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_NEW_MAP_CANCEL', index=26, number=38,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_SYNC_TIME', index=27, number=48,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_CALIBRATION', index=28, number=49,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_REBOOT_PLNS', index=29, number=50,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_SET_GENERAL_IO_OUPUT', index=30, number=53,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_SET_SPEAKER_VOLUME', index=31, number=54,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CMD_EXTEND_CONTROL', index=32, number=55,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=11317,
  serialized_end=12180,
)
_sym_db.RegisterEnumDescriptor(_COMMANDTYPE)

CommandType = enum_type_wrapper.EnumTypeWrapper(_COMMANDTYPE)
_LOCATIONMODE = _descriptor.EnumDescriptor(
  name='LocationMode',
  full_name='proto.LocationMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LOCATION_MODE_NONE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCATION_MODE_LIDAR', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCATION_MODE_GNSS', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCATION_MODE_MIX2', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCATION_MODE_INS', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=12183,
  serialized_end=12317,
)
_sym_db.RegisterEnumDescriptor(_LOCATIONMODE)

LocationMode = enum_type_wrapper.EnumTypeWrapper(_LOCATIONMODE)
MSG_REQUEST = 0
MSG_RESPONSE = 1
MSG_COMMAND = 2
MSG_NOTIFICATION = 3
CMD_ZERO = 0
CMD_HEARTBEATS_UPDATE = 1
CMD_ENABLE_MANUAL_CONTROL = 2
CMD_DISABLE_MANUAL_CONTROL = 3
CMD_SET_MANUAL_SPEED = 4
CMD_SET_SPEED_LEVEL = 5
CMD_SPEED_SLOW_DOWN = 6
CMD_PAUSE_MOVEMENT = 8
CMD_CONTINUE_MOVEMENT = 9
CMD_STOP_MOVEMENT = 10
CMD_ENABLE_OBSTACLE_AVOID = 11
CMD_DISABLE_OBSTACLE_AVOID = 12
CMD_NEW_MOVEMENT_TASK = 16
CMD_CANCEL_MOVEMENT_TASK = 17
CMD_NEW_ACTION_TASK = 18
CMD_CANCEL_ACTION_TASK = 19
CMD_CANCEL_EMERGENCY = 21
CMD_TRIGGER_EMERGENCY = 22
CMD_ENABLE_AUTO_CHARGE = 23
CMD_STOP_CHARGE = 24
CMD_START_LOCATION = 32
CMD_STOP_LOCATION = 33
CMD_SET_LOCATION_MODE = 34
CMD_SET_LOCATION_INITIAL_POSE = 35
CMD_NEW_MAP_START = 36
CMD_NEW_MAP_STOP = 37
CMD_NEW_MAP_CANCEL = 38
CMD_SYNC_TIME = 48
CMD_CALIBRATION = 49
CMD_REBOOT_PLNS = 50
CMD_SET_GENERAL_IO_OUPUT = 53
CMD_SET_SPEAKER_VOLUME = 54
CMD_EXTEND_CONTROL = 55
LOCATION_MODE_NONE = 0
LOCATION_MODE_LIDAR = 1
LOCATION_MODE_GNSS = 2
LOCATION_MODE_MIX2 = 3
LOCATION_MODE_INS = 4


_REQUEST_REQUESTTYPE = _descriptor.EnumDescriptor(
  name='RequestType',
  full_name='proto.Request.RequestType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='REQUEST_LOGIN', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_INFO', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_STATE', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_SYSTEM_STATE', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_HARDWARE_STATE', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_MAP_LIST', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_LASER_POINTS', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_TASK_STATE', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_FILE_OPERATE', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_LOGOUT', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_CHANGE_PW', index=10, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_TIMESTAMP', index=11, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_GET_CUR_MAP', index=12, number=12,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_GET_MAPPING_DATA', index=13, number=13,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_GET_MAPPING_TRACK', index=14, number=14,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=459,
  serialized_end=832,
)
_sym_db.RegisterEnumDescriptor(_REQUEST_REQUESTTYPE)

_REQUEST_FILEOPERATETYPE = _descriptor.EnumDescriptor(
  name='FileOperateType',
  full_name='proto.Request.FileOperateType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FILE_OPERATE_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IMPORT_MAP_FILE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IMPORT_CONFIG_FILE', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IMPORT_SYSTEM_UPDATE_FILE', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESTORE_FACTORY_SETTINGS', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=835,
  serialized_end=981,
)
_sym_db.RegisterEnumDescriptor(_REQUEST_FILEOPERATETYPE)

_RESPONSE_RESPONSETYPE = _descriptor.EnumDescriptor(
  name='ResponseType',
  full_name='proto.Response.ResponseType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_LOGIN', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_INFO', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_STATE', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_SYSTEM_STATE', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_HARDWARE_STATE', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_MAP_LIST', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_LASER_POINTS', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_TASK_STATE', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_FILE_OPERATE', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_LOGOUT', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_CHANGE_PW', index=10, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_TIMESTAMP', index=11, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_GET_CUR_MAP', index=12, number=12,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_GET_MAPPING_DATA', index=13, number=13,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_GET_MAPPING_TRACK', index=14, number=14,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_COMMAND', index=15, number=20,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_NOTIFY', index=16, number=21,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1650,
  serialized_end=2082,
)
_sym_db.RegisterEnumDescriptor(_RESPONSE_RESPONSETYPE)

_NOTIFICATION_NOTIFYTYPE = _descriptor.EnumDescriptor(
  name='NotifyType',
  full_name='proto.Notification.NotifyType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_MOVE_TASK_FINISHED', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_ACTION_TASK_FINISHED', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_CALIBRATION_FINISHED', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_UPDATE_FINISHED', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_MOVE_PATH_SENT', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_PASS_THROUGH_STATION', index=5, number=5,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=2819,
  serialized_end=3016,
)
_sym_db.RegisterEnumDescriptor(_NOTIFICATION_NOTIFYTYPE)

_NOTIFYRESPONSE_NOTIFYTYPE = _descriptor.EnumDescriptor(
  name='NotifyType',
  full_name='proto.NotifyResponse.NotifyType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_MOVE_TASK_FINISHED', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_ACTION_TASK_FINISHED', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_CALIBRATION_FINISHED', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_UPDATE_FINISHED', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_MOVE_PATH_SENT', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOTIFY_PASS_THROUGH_STATION', index=5, number=5,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=2819,
  serialized_end=3016,
)
_sym_db.RegisterEnumDescriptor(_NOTIFYRESPONSE_NOTIFYTYPE)

_UPDATERESULT_RESULTCODE = _descriptor.EnumDescriptor(
  name='ResultCode',
  full_name='proto.UpdateResult.ResultCode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UPDATE_SUCCESS', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UPDATE_FAILED', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=3441,
  serialized_end=3492,
)
_sym_db.RegisterEnumDescriptor(_UPDATERESULT_RESULTCODE)

_PATH_PATHTYPE = _descriptor.EnumDescriptor(
  name='PathType',
  full_name='proto.Path.PathType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PATH_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PATH_LINE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PATH_BEZIER', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PATH_ROTATE', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=4351,
  serialized_end=4425,
)
_sym_db.RegisterEnumDescriptor(_PATH_PATHTYPE)

_MOTIONCONTROLSTATE_MCSTATE = _descriptor.EnumDescriptor(
  name='MCState',
  full_name='proto.MotionControlState.MCState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MC_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MC_INITIALING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MC_IDLE', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MC_PATH_RUNNING', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MC_VELOCITY_RUNNING', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=4554,
  serialized_end=4654,
)
_sym_db.RegisterEnumDescriptor(_MOTIONCONTROLSTATE_MCSTATE)

_MOVEMENTTASK_TASKSTATE = _descriptor.EnumDescriptor(
  name='TaskState',
  full_name='proto.MovementTask.TaskState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MT_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MT_WAIT_FOR_START', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MT_RUNNING', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MT_PAUSED', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MT_FINISHED', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=5024,
  serialized_end=5117,
)
_sym_db.RegisterEnumDescriptor(_MOVEMENTTASK_TASKSTATE)

_MOVEMENTTASK_DSTSTATIONTYPE = _descriptor.EnumDescriptor(
  name='DstStationType',
  full_name='proto.MovementTask.DstStationType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DS_DEFAULT', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DS_NO_ROTATE', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=5119,
  serialized_end=5169,
)
_sym_db.RegisterEnumDescriptor(_MOVEMENTTASK_DSTSTATIONTYPE)

_MOVEMENTTASK_TASKTYPE = _descriptor.EnumDescriptor(
  name='TaskType',
  full_name='proto.MovementTask.TaskType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MT_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MT_MOVE_FOLLOW_PATH', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MT_MOVE_TO_STATION', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MT_MOVE_TO_POSE', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MT_MOVE_MIX2', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=5171,
  serialized_end=5282,
)
_sym_db.RegisterEnumDescriptor(_MOVEMENTTASK_TASKTYPE)

_MOVEMENTTASK_AVOIDPOLICY = _descriptor.EnumDescriptor(
  name='AvoidPolicy',
  full_name='proto.MovementTask.AvoidPolicy',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AT_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBSTACLE_AVOID_WAIT', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBSTACLE_AVOID_REPLAN', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBSTACLE_AVOID_NONE', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=5284,
  serialized_end=5387,
)
_sym_db.RegisterEnumDescriptor(_MOVEMENTTASK_AVOIDPOLICY)

_ACTIONTASK_TASKSTATE = _descriptor.EnumDescriptor(
  name='TaskState',
  full_name='proto.ActionTask.TaskState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AT_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AT_WAIT_FOR_START', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AT_RUNNING', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AT_PAUSED', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AT_FINISHED', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=5519,
  serialized_end=5612,
)
_sym_db.RegisterEnumDescriptor(_ACTIONTASK_TASKSTATE)

_ACTIONTASK_TASKTYPE = _descriptor.EnumDescriptor(
  name='TaskType',
  full_name='proto.ActionTask.TaskType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AT_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AT_WAITING_TIMEOUT', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AT_JACK_UP', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AT_JACK_DOWN', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AT_TAKE_PHOTOS', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=5614,
  serialized_end=5715,
)
_sym_db.RegisterEnumDescriptor(_ACTIONTASK_TASKTYPE)

_SYSTEMSTATE_SYSSTATE = _descriptor.EnumDescriptor(
  name='SysState',
  full_name='proto.SystemState.SysState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_INITIALING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_IDLE', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_ERROR', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_HARDWARE_ERROR', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_START_LOCATING', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NEWMAP_DRAWING', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NEWMAP_SAVING', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_MANUAL_PAUSED', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NAV_INITIALING', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NAV_FINDING_PATH', index=10, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NAV_WAITING_FINISH', index=11, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NAV_WAITING_FINISH_SLOW', index=12, number=12,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NAV_REFINDING_PATH', index=13, number=13,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NAV_PAUSED', index=14, number=14,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NAV_NO_WAY', index=15, number=15,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NAV_NO_STATION', index=16, number=16,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_NAV_PATH_ERROR', index=17, number=17,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_PATH_NAV_INITIALING', index=18, number=18,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_PATH_WAITING_FINISH', index=19, number=19,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_PATH_WAITING_FINISH_SLOW', index=20, number=20,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_PATH_PAUSED', index=21, number=21,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYS_STATE_TASK_PATH_ERROR', index=22, number=22,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=6623,
  serialized_end=7397,
)
_sym_db.RegisterEnumDescriptor(_SYSTEMSTATE_SYSSTATE)

_SYSTEMSTATE_LOCATIONSTATE = _descriptor.EnumDescriptor(
  name='LocationState',
  full_name='proto.SystemState.LocationState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LOCATION_STATE_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCATION_STATE_NONE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCATION_STATE_INITIALING', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCATION_STATE_RUNNING', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCATION_STATE_RELOCATING', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCATION_STATE_ERROR', index=5, number=5,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=7400,
  serialized_end=7581,
)
_sym_db.RegisterEnumDescriptor(_SYSTEMSTATE_LOCATIONSTATE)

_SYSTEMSTATE_EMERGENCYSTATE = _descriptor.EnumDescriptor(
  name='EmergencyState',
  full_name='proto.SystemState.EmergencyState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='STATE_EMERGENCY_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STATE_EMERGENCY_NONE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STATE_EMERGENCY_TRIGER', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STATE_EMERGENCY_RECOVERABLE', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=7583,
  serialized_end=7710,
)
_sym_db.RegisterEnumDescriptor(_SYSTEMSTATE_EMERGENCYSTATE)

_SYSTEMSTATE_OPERATIONSTATE = _descriptor.EnumDescriptor(
  name='OperationState',
  full_name='proto.SystemState.OperationState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='OPERATION_NONE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPERATION_AUTO', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPERATION_MANUAL', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=7712,
  serialized_end=7790,
)
_sym_db.RegisterEnumDescriptor(_SYSTEMSTATE_OPERATIONSTATE)

_SYSTEMSTATE_OBAENABLESTATE = _descriptor.EnumDescriptor(
  name='ObaEnableState',
  full_name='proto.SystemState.ObaEnableState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='OBA_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBA_ENABLED', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBA_DISABLED', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=7792,
  serialized_end=7855,
)
_sym_db.RegisterEnumDescriptor(_SYSTEMSTATE_OBAENABLESTATE)

_SYSTEMSTATE_RUNSTATE = _descriptor.EnumDescriptor(
  name='RunState',
  full_name='proto.SystemState.RunState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RUN_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RUN_IDLE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RUN_RUNNING', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RUN_BLOCK_SLOWDOWN', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RUN_BLOCK_STOP', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=7857,
  serialized_end=7954,
)
_sym_db.RegisterEnumDescriptor(_SYSTEMSTATE_RUNSTATE)

_SYSTEMSTATE_LOADSTATE = _descriptor.EnumDescriptor(
  name='LoadState',
  full_name='proto.SystemState.LoadState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LOAD_NONE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOAD_FREE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOAD_FULL', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=7956,
  serialized_end=8012,
)
_sym_db.RegisterEnumDescriptor(_SYSTEMSTATE_LOADSTATE)

_SYSTEMSTATE_GNSSMSGSTATE = _descriptor.EnumDescriptor(
  name='GnssMsgState',
  full_name='proto.SystemState.GnssMsgState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='GNSS_MSG_INVALID', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GNSS_MSG_SINGLE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GNSS_MSG_DIFF', index=2, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GNSS_MSG_FLOAT', index=3, number=5,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=8014,
  serialized_end=8110,
)
_sym_db.RegisterEnumDescriptor(_SYSTEMSTATE_GNSSMSGSTATE)

_HARDWARESTATE_BATTERYSTATE = _descriptor.EnumDescriptor(
  name='BatteryState',
  full_name='proto.HardwareState.BatteryState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BATTERY_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BATTERY_CHARGING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BATTERY_NO_CHARGING', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=9022,
  serialized_end=9099,
)
_sym_db.RegisterEnumDescriptor(_HARDWARESTATE_BATTERYSTATE)

_HARDWARESTATE_LASERSTATE = _descriptor.EnumDescriptor(
  name='LaserState',
  full_name='proto.HardwareState.LaserState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LASER_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LASER_INITING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LASER_OK', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LASER_ERROR', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=9101,
  serialized_end=9177,
)
_sym_db.RegisterEnumDescriptor(_HARDWARESTATE_LASERSTATE)

_HARDWARESTATE_WIFISTATE = _descriptor.EnumDescriptor(
  name='WiFiState',
  full_name='proto.HardwareState.WiFiState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='WIFI_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WIFI_DISCONNECTED', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WIFI_CONNECTED', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WIFI_SCANNING', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=9179,
  serialized_end=9265,
)
_sym_db.RegisterEnumDescriptor(_HARDWARESTATE_WIFISTATE)

_HARDWARESTATE_BREAKSWITCHSTATE = _descriptor.EnumDescriptor(
  name='BreakSwitchState',
  full_name='proto.HardwareState.BreakSwitchState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BREAK_SW_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BREAK_SW_OFF', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BREAK_SW_ON', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=9267,
  serialized_end=9337,
)
_sym_db.RegisterEnumDescriptor(_HARDWARESTATE_BREAKSWITCHSTATE)

_HARDWARESTATE_JACKINGSTATE = _descriptor.EnumDescriptor(
  name='JackingState',
  full_name='proto.HardwareState.JackingState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='JACK_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JACK_UP', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JACK_DOWN', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JACK_MOVING_UP', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JACK_MOVING_DOWN', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=9339,
  serialized_end=9436,
)
_sym_db.RegisterEnumDescriptor(_HARDWARESTATE_JACKINGSTATE)

_HARDWARESTATE_GNSSSTATE = _descriptor.EnumDescriptor(
  name='GnssState',
  full_name='proto.HardwareState.GnssState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='GNSS_NA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GNSS_DISCONNECTED', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GNSS_CONNECTED', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=9438,
  serialized_end=9505,
)
_sym_db.RegisterEnumDescriptor(_HARDWARESTATE_GNSSSTATE)

_HARDWARESTATE_HSTATE = _descriptor.EnumDescriptor(
  name='HState',
  full_name='proto.HardwareState.HState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='H_STATE_ZERO', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='H_STATE_INITIALING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='H_STATE_OK', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='H_STATE_ERROR', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=9507,
  serialized_end=9592,
)
_sym_db.RegisterEnumDescriptor(_HARDWARESTATE_HSTATE)

_RESULT_RESULTSTATE = _descriptor.EnumDescriptor(
  name='ResultState',
  full_name='proto.Result.ResultState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_NONE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_PROCESSING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_OK', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESPONSE_FAILED', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=10386,
  serialized_end=10481,
)
_sym_db.RegisterEnumDescriptor(_RESULT_RESULTSTATE)

_RESULT_RESULTCODE = _descriptor.EnumDescriptor(
  name='ResultCode',
  full_name='proto.Result.ResultCode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RESULT_CODE_NONE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_UNDEFINED', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_INVALID_CMD_PARAM', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_USER_OR_PASSWROD_INVALID', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_SYSTEM_STATE_ERROR', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_OPERATION_MODE_ERROR', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_LOAD_MAP_FILE_ERROR', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_PRE_TASK_RUNNING', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_NAV_NO_WAY', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_NAV_NO_STATION', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_NAV_PATH_PLAN_ERROR', index=10, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_NAV_PLAN_STEP_ERROR', index=11, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESULT_NAV_MOTION_CONTROL_ERROR', index=12, number=12,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=10484,
  serialized_end=10883,
)
_sym_db.RegisterEnumDescriptor(_RESULT_RESULTCODE)

_UISTATE_PARENTSTATE = _descriptor.EnumDescriptor(
  name='ParentState',
  full_name='proto.UiState.ParentState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PARENT_STATE_NONE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PARENT_STATE_RUNNABLE', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PARENT_STATE_MAPPING', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PARENT_STATE_LOCALIZATION', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PARENT_STATE_HW_FAULT', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=10996,
  serialized_end=11143,
)
_sym_db.RegisterEnumDescriptor(_UISTATE_PARENTSTATE)

_UISTATE_CHILDSTATE = _descriptor.EnumDescriptor(
  name='ChildState',
  full_name='proto.UiState.ChildState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CHILD_STATE_NONE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CHILD_STATE_MANUAL', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CHILD_STATE_AUTO', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=11145,
  serialized_end=11225,
)
_sym_db.RegisterEnumDescriptor(_UISTATE_CHILDSTATE)


_MESSAGE = _descriptor.Descriptor(
  name='Message',
  full_name='proto.Message',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='proto.Message.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='seq', full_name='proto.Message.seq', index=1,
      number=2, type=7, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='session_id', full_name='proto.Message.session_id', index=2,
      number=3, type=6, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='request', full_name='proto.Message.request', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='response', full_name='proto.Message.response', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='command', full_name='proto.Message.command', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='notification', full_name='proto.Message.notification', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=242,
)


_REQUEST = _descriptor.Descriptor(
  name='Request',
  full_name='proto.Request',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_type', full_name='proto.Request.request_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='login_request', full_name='proto.Request.login_request', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='file_op', full_name='proto.Request.file_op', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='param_str', full_name='proto.Request.param_str', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='param_str1', full_name='proto.Request.param_str1', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='param_str2', full_name='proto.Request.param_str2', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _REQUEST_REQUESTTYPE,
    _REQUEST_FILEOPERATETYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=245,
  serialized_end=981,
)


_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='proto.Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='response_type', full_name='proto.Response.response_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='login_info', full_name='proto.Response.login_info', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='info', full_name='proto.Response.info', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='system_state', full_name='proto.Response.system_state', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hardware_state', full_name='proto.Response.hardware_state', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='map_list', full_name='proto.Response.map_list', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='laser_points', full_name='proto.Response.laser_points', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='movement_task', full_name='proto.Response.movement_task', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='action_task', full_name='proto.Response.action_task', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='occupancy_grid_map', full_name='proto.Response.occupancy_grid_map', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mapping_cloud', full_name='proto.Response.mapping_cloud', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mapping_trajectory', full_name='proto.Response.mapping_trajectory', index=11,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='result', full_name='proto.Response.result', index=12,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='notify_response', full_name='proto.Response.notify_response', index=13,
      number=14, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='list', full_name='proto.Response.list', index=14,
      number=20, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='proto.Response.timestamp', index=15,
      number=21, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='retInfo', full_name='proto.Response.retInfo', index=16,
      number=22, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _RESPONSE_RESPONSETYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=984,
  serialized_end=2082,
)


_COMMAND = _descriptor.Descriptor(
  name='Command',
  full_name='proto.Command',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='command', full_name='proto.Command.command', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='param_int', full_name='proto.Command.param_int', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='param_int1', full_name='proto.Command.param_int1', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='param_int2', full_name='proto.Command.param_int2', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pose', full_name='proto.Command.pose', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='param_str', full_name='proto.Command.param_str', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='param_str1', full_name='proto.Command.param_str1', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='location_mode', full_name='proto.Command.location_mode', index=7,
      number=8, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='movement_task', full_name='proto.Command.movement_task', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='action_task', full_name='proto.Command.action_task', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='general_io_ouput', full_name='proto.Command.general_io_ouput', index=10,
      number=11, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='extend_cmd', full_name='proto.Command.extend_cmd', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2085,
  serialized_end=2430,
)


_NOTIFICATION = _descriptor.Descriptor(
  name='Notification',
  full_name='proto.Notification',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='notify_type', full_name='proto.Notification.notify_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='reserved_field', full_name='proto.Notification.reserved_field', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cur_station_no', full_name='proto.Notification.cur_station_no', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cur_pose', full_name='proto.Notification.cur_pose', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='task_result', full_name='proto.Notification.task_result', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='movement_task', full_name='proto.Notification.movement_task', index=5,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='action_task', full_name='proto.Notification.action_task', index=6,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cal_result', full_name='proto.Notification.cal_result', index=7,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='update_result', full_name='proto.Notification.update_result', index=8,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pass_station_type', full_name='proto.Notification.pass_station_type', index=9,
      number=14, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _NOTIFICATION_NOTIFYTYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2433,
  serialized_end=3016,
)


_NOTIFYRESPONSE = _descriptor.Descriptor(
  name='NotifyResponse',
  full_name='proto.NotifyResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ack', full_name='proto.NotifyResponse.ack', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='proto.NotifyResponse.type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _NOTIFYRESPONSE_NOTIFYTYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3019,
  serialized_end=3296,
)


_CALIBRATIONRESULT = _descriptor.Descriptor(
  name='CalibrationResult',
  full_name='proto.CalibrationResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='proto.CalibrationResult.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y', full_name='proto.CalibrationResult.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='theta', full_name='proto.CalibrationResult.theta', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='status', full_name='proto.CalibrationResult.status', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3298,
  serialized_end=3370,
)


_UPDATERESULT = _descriptor.Descriptor(
  name='UpdateResult',
  full_name='proto.UpdateResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='result_code', full_name='proto.UpdateResult.result_code', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _UPDATERESULT_RESULTCODE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3372,
  serialized_end=3492,
)


_LOGINREQUEST = _descriptor.Descriptor(
  name='LoginRequest',
  full_name='proto.LoginRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='username', full_name='proto.LoginRequest.username', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='password', full_name='proto.LoginRequest.password', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3494,
  serialized_end=3544,
)


_LOGININFO = _descriptor.Descriptor(
  name='LoginInfo',
  full_name='proto.LoginInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='session_id', full_name='proto.LoginInfo.session_id', index=0,
      number=1, type=6, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='result_str', full_name='proto.LoginInfo.result_str', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3546,
  serialized_end=3597,
)


_INFO = _descriptor.Descriptor(
  name='Info',
  full_name='proto.Info',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='licence_valid', full_name='proto.Info.licence_valid', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='plns_serial_no', full_name='proto.Info.plns_serial_no', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hardware_version', full_name='proto.Info.hardware_version', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tc_version', full_name='proto.Info.tc_version', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='base_version', full_name='proto.Info.base_version', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='vehicle_serial_no', full_name='proto.Info.vehicle_serial_no', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3600,
  serialized_end=3748,
)


_FILEINFO = _descriptor.Descriptor(
  name='FileInfo',
  full_name='proto.FileInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='proto.FileInfo.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='size', full_name='proto.FileInfo.size', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='last_write_time', full_name='proto.FileInfo.last_write_time', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='file_compressed', full_name='proto.FileInfo.file_compressed', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='md5sum', full_name='proto.FileInfo.md5sum', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3750,
  serialized_end=3854,
)


_MAPLIST = _descriptor.Descriptor(
  name='MapList',
  full_name='proto.MapList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='req_seq', full_name='proto.MapList.req_seq', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='list', full_name='proto.MapList.list', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3856,
  serialized_end=3913,
)


_FILELIST = _descriptor.Descriptor(
  name='FileList',
  full_name='proto.FileList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='req_seq', full_name='proto.FileList.req_seq', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='list', full_name='proto.FileList.list', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3915,
  serialized_end=3973,
)


_POSE = _descriptor.Descriptor(
  name='Pose',
  full_name='proto.Pose',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='proto.Pose.x', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y', full_name='proto.Pose.y', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='z', full_name='proto.Pose.z', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='roll', full_name='proto.Pose.roll', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pitch', full_name='proto.Pose.pitch', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='yaw', full_name='proto.Pose.yaw', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='proto.Pose.confidence', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3975,
  serialized_end=4076,
)


_PATH = _descriptor.Descriptor(
  name='Path',
  full_name='proto.Path',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='proto.Path.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sx', full_name='proto.Path.sx', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sy', full_name='proto.Path.sy', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ex', full_name='proto.Path.ex', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ey', full_name='proto.Path.ey', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cx', full_name='proto.Path.cx', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cy', full_name='proto.Path.cy', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dx', full_name='proto.Path.dx', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dy', full_name='proto.Path.dy', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rotate_angle', full_name='proto.Path.rotate_angle', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='direction', full_name='proto.Path.direction', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='obst_stop', full_name='proto.Path.obst_stop', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='limit_v', full_name='proto.Path.limit_v', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='limit_w', full_name='proto.Path.limit_w', index=13,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='limit_acc', full_name='proto.Path.limit_acc', index=14,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='limit_dec', full_name='proto.Path.limit_dec', index=15,
      number=16, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _PATH_PATHTYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=4079,
  serialized_end=4425,
)


_MOTIONCONTROLSTATE = _descriptor.Descriptor(
  name='MotionControlState',
  full_name='proto.MotionControlState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='proto.MotionControlState.state', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='v_x', full_name='proto.MotionControlState.v_x', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='v_y', full_name='proto.MotionControlState.v_y', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='w', full_name='proto.MotionControlState.w', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='path_no', full_name='proto.MotionControlState.path_no', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _MOTIONCONTROLSTATE_MCSTATE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=4428,
  serialized_end=4654,
)


_MOVEMENTTASK = _descriptor.Descriptor(
  name='MovementTask',
  full_name='proto.MovementTask',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='proto.MovementTask.state', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dst_station_type', full_name='proto.MovementTask.dst_station_type', index=1,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='proto.MovementTask.type', index=2,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='stations', full_name='proto.MovementTask.stations', index=3,
      number=5, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='poses', full_name='proto.MovementTask.poses', index=4,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='paths', full_name='proto.MovementTask.paths', index=5,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='avoid_policy', full_name='proto.MovementTask.avoid_policy', index=6,
      number=8, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='remain_time', full_name='proto.MovementTask.remain_time', index=7,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='remain_distance', full_name='proto.MovementTask.remain_distance', index=8,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='total_distance', full_name='proto.MovementTask.total_distance', index=9,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _MOVEMENTTASK_TASKSTATE,
    _MOVEMENTTASK_DSTSTATIONTYPE,
    _MOVEMENTTASK_TASKTYPE,
    _MOVEMENTTASK_AVOIDPOLICY,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=4657,
  serialized_end=5387,
)


_ACTIONTASK = _descriptor.Descriptor(
  name='ActionTask',
  full_name='proto.ActionTask',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='proto.ActionTask.state', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='proto.ActionTask.type', index=1,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='timeout', full_name='proto.ActionTask.timeout', index=2,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='id', full_name='proto.ActionTask.id', index=3,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ACTIONTASK_TASKSTATE,
    _ACTIONTASK_TASKTYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=5390,
  serialized_end=5715,
)


_SYSTEMSTATE = _descriptor.Descriptor(
  name='SystemState',
  full_name='proto.SystemState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='req_seq', full_name='proto.SystemState.req_seq', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sys_state', full_name='proto.SystemState.sys_state', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='location_state', full_name='proto.SystemState.location_state', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='location_mode', full_name='proto.SystemState.location_mode', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='location_pose', full_name='proto.SystemState.location_pose', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='emergency_state', full_name='proto.SystemState.emergency_state', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mc_state', full_name='proto.SystemState.mc_state', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='movement_state', full_name='proto.SystemState.movement_state', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='action_state', full_name='proto.SystemState.action_state', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='station_no', full_name='proto.SystemState.station_no', index=9,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='map_name', full_name='proto.SystemState.map_name', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='map_saving_progress', full_name='proto.SystemState.map_saving_progress', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='operation_state', full_name='proto.SystemState.operation_state', index=12,
      number=13, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='oba_enable_state', full_name='proto.SystemState.oba_enable_state', index=13,
      number=14, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='run_state', full_name='proto.SystemState.run_state', index=14,
      number=15, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='load_state', full_name='proto.SystemState.load_state', index=15,
      number=16, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gnss_msg_state', full_name='proto.SystemState.gnss_msg_state', index=16,
      number=18, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gnss_pose', full_name='proto.SystemState.gnss_pose', index=17,
      number=19, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='satellite_num', full_name='proto.SystemState.satellite_num', index=18,
      number=20, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='speed_level', full_name='proto.SystemState.speed_level', index=19,
      number=31, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='fault_code', full_name='proto.SystemState.fault_code', index=20,
      number=32, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='total_mileage', full_name='proto.SystemState.total_mileage', index=21,
      number=33, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='uiState', full_name='proto.SystemState.uiState', index=22,
      number=34, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SYSTEMSTATE_SYSSTATE,
    _SYSTEMSTATE_LOCATIONSTATE,
    _SYSTEMSTATE_EMERGENCYSTATE,
    _SYSTEMSTATE_OPERATIONSTATE,
    _SYSTEMSTATE_OBAENABLESTATE,
    _SYSTEMSTATE_RUNSTATE,
    _SYSTEMSTATE_LOADSTATE,
    _SYSTEMSTATE_GNSSMSGSTATE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=5718,
  serialized_end=8110,
)


_HARDWARESTATE = _descriptor.Descriptor(
  name='HardwareState',
  full_name='proto.HardwareState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='req_seq', full_name='proto.HardwareState.req_seq', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cpu_usage', full_name='proto.HardwareState.cpu_usage', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='memory_usage', full_name='proto.HardwareState.memory_usage', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='disk_usage', full_name='proto.HardwareState.disk_usage', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='remain_disk_space', full_name='proto.HardwareState.remain_disk_space', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='battery_state', full_name='proto.HardwareState.battery_state', index=5,
      number=9, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='battery_percentage', full_name='proto.HardwareState.battery_percentage', index=6,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='battery_voltage', full_name='proto.HardwareState.battery_voltage', index=7,
      number=11, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='battery_current', full_name='proto.HardwareState.battery_current', index=8,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='battery_tempreature', full_name='proto.HardwareState.battery_tempreature', index=9,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='battery_remain_time', full_name='proto.HardwareState.battery_remain_time', index=10,
      number=14, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='laser_state', full_name='proto.HardwareState.laser_state', index=11,
      number=16, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='wifi_state', full_name='proto.HardwareState.wifi_state', index=12,
      number=17, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='wifi_name', full_name='proto.HardwareState.wifi_name', index=13,
      number=18, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='wifi_strength', full_name='proto.HardwareState.wifi_strength', index=14,
      number=19, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='break_sw_state', full_name='proto.HardwareState.break_sw_state', index=15,
      number=20, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='jacking_state', full_name='proto.HardwareState.jacking_state', index=16,
      number=21, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gnss_state', full_name='proto.HardwareState.gnss_state', index=17,
      number=22, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cpu_temperature', full_name='proto.HardwareState.cpu_temperature', index=18,
      number=30, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='box_temprature', full_name='proto.HardwareState.box_temprature', index=19,
      number=31, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='general_io_input', full_name='proto.HardwareState.general_io_input', index=20,
      number=32, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='general_io_output', full_name='proto.HardwareState.general_io_output', index=21,
      number=33, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='src_hardware_state', full_name='proto.HardwareState.src_hardware_state', index=22,
      number=34, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hardware_state', full_name='proto.HardwareState.hardware_state', index=23,
      number=36, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hardware_error_code', full_name='proto.HardwareState.hardware_error_code', index=24,
      number=37, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='co_info', full_name='proto.HardwareState.co_info', index=25,
      number=40, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gas_info', full_name='proto.HardwareState.gas_info', index=26,
      number=41, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _HARDWARESTATE_BATTERYSTATE,
    _HARDWARESTATE_LASERSTATE,
    _HARDWARESTATE_WIFISTATE,
    _HARDWARESTATE_BREAKSWITCHSTATE,
    _HARDWARESTATE_JACKINGSTATE,
    _HARDWARESTATE_GNSSSTATE,
    _HARDWARESTATE_HSTATE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=8113,
  serialized_end=9592,
)


_SRCHARDWARESTATE = _descriptor.Descriptor(
  name='SRCHardwareState',
  full_name='proto.SRCHardwareState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='m1_status_code', full_name='proto.SRCHardwareState.m1_status_code', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='m2_status_code', full_name='proto.SRCHardwareState.m2_status_code', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='m3_status_code', full_name='proto.SRCHardwareState.m3_status_code', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='m4_status_code', full_name='proto.SRCHardwareState.m4_status_code', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='total_power_cycle', full_name='proto.SRCHardwareState.total_power_cycle', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='total_poweron_time', full_name='proto.SRCHardwareState.total_poweron_time', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='total_mileage', full_name='proto.SRCHardwareState.total_mileage', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9595,
  serialized_end=9787,
)


_LASERPOINTS = _descriptor.Descriptor(
  name='LaserPoints',
  full_name='proto.LaserPoints',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='req_seq', full_name='proto.LaserPoints.req_seq', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='xs', full_name='proto.LaserPoints.xs', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='ys', full_name='proto.LaserPoints.ys', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='zs', full_name='proto.LaserPoints.zs', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9789,
  serialized_end=9867,
)


_MAPHEADER = _descriptor.Descriptor(
  name='MapHeader',
  full_name='proto.MapHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='stamp', full_name='proto.MapHeader.stamp', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='frame_id', full_name='proto.MapHeader.frame_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9869,
  serialized_end=9913,
)


_MAPMETADATA = _descriptor.Descriptor(
  name='MapMetaData',
  full_name='proto.MapMetaData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='map_load_time', full_name='proto.MapMetaData.map_load_time', index=0,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='resolution', full_name='proto.MapMetaData.resolution', index=1,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width', full_name='proto.MapMetaData.width', index=2,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height', full_name='proto.MapMetaData.height', index=3,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='origin', full_name='proto.MapMetaData.origin', index=4,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9915,
  serialized_end=10031,
)


_OCCUPANCYGRIDMAP = _descriptor.Descriptor(
  name='OccupancyGridMap',
  full_name='proto.OccupancyGridMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='proto.OccupancyGridMap.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='info', full_name='proto.OccupancyGridMap.info', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='compressed_char_len', full_name='proto.OccupancyGridMap.compressed_char_len', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='proto.OccupancyGridMap.data', index=3,
      number=4, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10034,
  serialized_end=10163,
)


_CLOUDMAP = _descriptor.Descriptor(
  name='CloudMap',
  full_name='proto.CloudMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='req_seq', full_name='proto.CloudMap.req_seq', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cloud_size', full_name='proto.CloudMap.cloud_size', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='org_char_len', full_name='proto.CloudMap.org_char_len', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='compressed_char_len', full_name='proto.CloudMap.compressed_char_len', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='proto.CloudMap.data', index=4,
      number=5, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10165,
  serialized_end=10277,
)


_RESULT = _descriptor.Descriptor(
  name='Result',
  full_name='proto.Result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='result_state', full_name='proto.Result.result_state', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='result_code', full_name='proto.Result.result_code', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _RESULT_RESULTSTATE,
    _RESULT_RESULTCODE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10280,
  serialized_end=10883,
)


_UISTATE = _descriptor.Descriptor(
  name='UiState',
  full_name='proto.UiState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='parent_state', full_name='proto.UiState.parent_state', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='child_state', full_name='proto.UiState.child_state', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _UISTATE_PARENTSTATE,
    _UISTATE_CHILDSTATE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10886,
  serialized_end=11225,
)

_MESSAGE.fields_by_name['type'].enum_type = _MESSAGETYPE
_MESSAGE.fields_by_name['request'].message_type = _REQUEST
_MESSAGE.fields_by_name['response'].message_type = _RESPONSE
_MESSAGE.fields_by_name['command'].message_type = _COMMAND
_MESSAGE.fields_by_name['notification'].message_type = _NOTIFICATION
_REQUEST.fields_by_name['request_type'].enum_type = _REQUEST_REQUESTTYPE
_REQUEST.fields_by_name['login_request'].message_type = _LOGINREQUEST
_REQUEST.fields_by_name['file_op'].enum_type = _REQUEST_FILEOPERATETYPE
_REQUEST_REQUESTTYPE.containing_type = _REQUEST
_REQUEST_FILEOPERATETYPE.containing_type = _REQUEST
_RESPONSE.fields_by_name['response_type'].enum_type = _RESPONSE_RESPONSETYPE
_RESPONSE.fields_by_name['login_info'].message_type = _LOGININFO
_RESPONSE.fields_by_name['info'].message_type = _INFO
_RESPONSE.fields_by_name['system_state'].message_type = _SYSTEMSTATE
_RESPONSE.fields_by_name['hardware_state'].message_type = _HARDWARESTATE
_RESPONSE.fields_by_name['map_list'].message_type = _MAPLIST
_RESPONSE.fields_by_name['laser_points'].message_type = _LASERPOINTS
_RESPONSE.fields_by_name['movement_task'].message_type = _MOVEMENTTASK
_RESPONSE.fields_by_name['action_task'].message_type = _ACTIONTASK
_RESPONSE.fields_by_name['occupancy_grid_map'].message_type = _OCCUPANCYGRIDMAP
_RESPONSE.fields_by_name['mapping_cloud'].message_type = _CLOUDMAP
_RESPONSE.fields_by_name['mapping_trajectory'].message_type = _LASERPOINTS
_RESPONSE.fields_by_name['result'].message_type = _RESULT
_RESPONSE.fields_by_name['notify_response'].message_type = _NOTIFYRESPONSE
_RESPONSE.fields_by_name['list'].message_type = _FILELIST
_RESPONSE_RESPONSETYPE.containing_type = _RESPONSE
_COMMAND.fields_by_name['command'].enum_type = _COMMANDTYPE
_COMMAND.fields_by_name['pose'].message_type = _POSE
_COMMAND.fields_by_name['location_mode'].enum_type = _LOCATIONMODE
_COMMAND.fields_by_name['movement_task'].message_type = _MOVEMENTTASK
_COMMAND.fields_by_name['action_task'].message_type = _ACTIONTASK
_NOTIFICATION.fields_by_name['notify_type'].enum_type = _NOTIFICATION_NOTIFYTYPE
_NOTIFICATION.fields_by_name['cur_pose'].message_type = _POSE
_NOTIFICATION.fields_by_name['task_result'].message_type = _RESULT
_NOTIFICATION.fields_by_name['movement_task'].message_type = _MOVEMENTTASK
_NOTIFICATION.fields_by_name['action_task'].message_type = _ACTIONTASK
_NOTIFICATION.fields_by_name['cal_result'].message_type = _CALIBRATIONRESULT
_NOTIFICATION.fields_by_name['update_result'].message_type = _UPDATERESULT
_NOTIFICATION_NOTIFYTYPE.containing_type = _NOTIFICATION
_NOTIFYRESPONSE.fields_by_name['type'].enum_type = _NOTIFYRESPONSE_NOTIFYTYPE
_NOTIFYRESPONSE_NOTIFYTYPE.containing_type = _NOTIFYRESPONSE
_UPDATERESULT.fields_by_name['result_code'].enum_type = _UPDATERESULT_RESULTCODE
_UPDATERESULT_RESULTCODE.containing_type = _UPDATERESULT
_MAPLIST.fields_by_name['list'].message_type = _FILEINFO
_FILELIST.fields_by_name['list'].message_type = _FILEINFO
_PATH.fields_by_name['type'].enum_type = _PATH_PATHTYPE
_PATH_PATHTYPE.containing_type = _PATH
_MOTIONCONTROLSTATE.fields_by_name['state'].enum_type = _MOTIONCONTROLSTATE_MCSTATE
_MOTIONCONTROLSTATE_MCSTATE.containing_type = _MOTIONCONTROLSTATE
_MOVEMENTTASK.fields_by_name['state'].enum_type = _MOVEMENTTASK_TASKSTATE
_MOVEMENTTASK.fields_by_name['dst_station_type'].enum_type = _MOVEMENTTASK_DSTSTATIONTYPE
_MOVEMENTTASK.fields_by_name['type'].enum_type = _MOVEMENTTASK_TASKTYPE
_MOVEMENTTASK.fields_by_name['poses'].message_type = _POSE
_MOVEMENTTASK.fields_by_name['paths'].message_type = _PATH
_MOVEMENTTASK.fields_by_name['avoid_policy'].enum_type = _MOVEMENTTASK_AVOIDPOLICY
_MOVEMENTTASK_TASKSTATE.containing_type = _MOVEMENTTASK
_MOVEMENTTASK_DSTSTATIONTYPE.containing_type = _MOVEMENTTASK
_MOVEMENTTASK_TASKTYPE.containing_type = _MOVEMENTTASK
_MOVEMENTTASK_AVOIDPOLICY.containing_type = _MOVEMENTTASK
_ACTIONTASK.fields_by_name['state'].enum_type = _ACTIONTASK_TASKSTATE
_ACTIONTASK.fields_by_name['type'].enum_type = _ACTIONTASK_TASKTYPE
_ACTIONTASK_TASKSTATE.containing_type = _ACTIONTASK
_ACTIONTASK_TASKTYPE.containing_type = _ACTIONTASK
_SYSTEMSTATE.fields_by_name['sys_state'].enum_type = _SYSTEMSTATE_SYSSTATE
_SYSTEMSTATE.fields_by_name['location_state'].enum_type = _SYSTEMSTATE_LOCATIONSTATE
_SYSTEMSTATE.fields_by_name['location_mode'].enum_type = _LOCATIONMODE
_SYSTEMSTATE.fields_by_name['location_pose'].message_type = _POSE
_SYSTEMSTATE.fields_by_name['emergency_state'].enum_type = _SYSTEMSTATE_EMERGENCYSTATE
_SYSTEMSTATE.fields_by_name['mc_state'].message_type = _MOTIONCONTROLSTATE
_SYSTEMSTATE.fields_by_name['movement_state'].message_type = _MOVEMENTTASK
_SYSTEMSTATE.fields_by_name['action_state'].message_type = _ACTIONTASK
_SYSTEMSTATE.fields_by_name['operation_state'].enum_type = _SYSTEMSTATE_OPERATIONSTATE
_SYSTEMSTATE.fields_by_name['oba_enable_state'].enum_type = _SYSTEMSTATE_OBAENABLESTATE
_SYSTEMSTATE.fields_by_name['run_state'].enum_type = _SYSTEMSTATE_RUNSTATE
_SYSTEMSTATE.fields_by_name['load_state'].enum_type = _SYSTEMSTATE_LOADSTATE
_SYSTEMSTATE.fields_by_name['gnss_msg_state'].enum_type = _SYSTEMSTATE_GNSSMSGSTATE
_SYSTEMSTATE.fields_by_name['gnss_pose'].message_type = _POSE
_SYSTEMSTATE.fields_by_name['uiState'].message_type = _UISTATE
_SYSTEMSTATE_SYSSTATE.containing_type = _SYSTEMSTATE
_SYSTEMSTATE_LOCATIONSTATE.containing_type = _SYSTEMSTATE
_SYSTEMSTATE_EMERGENCYSTATE.containing_type = _SYSTEMSTATE
_SYSTEMSTATE_OPERATIONSTATE.containing_type = _SYSTEMSTATE
_SYSTEMSTATE_OBAENABLESTATE.containing_type = _SYSTEMSTATE
_SYSTEMSTATE_RUNSTATE.containing_type = _SYSTEMSTATE
_SYSTEMSTATE_LOADSTATE.containing_type = _SYSTEMSTATE
_SYSTEMSTATE_GNSSMSGSTATE.containing_type = _SYSTEMSTATE
_HARDWARESTATE.fields_by_name['battery_state'].enum_type = _HARDWARESTATE_BATTERYSTATE
_HARDWARESTATE.fields_by_name['laser_state'].enum_type = _HARDWARESTATE_LASERSTATE
_HARDWARESTATE.fields_by_name['wifi_state'].enum_type = _HARDWARESTATE_WIFISTATE
_HARDWARESTATE.fields_by_name['break_sw_state'].enum_type = _HARDWARESTATE_BREAKSWITCHSTATE
_HARDWARESTATE.fields_by_name['jacking_state'].enum_type = _HARDWARESTATE_JACKINGSTATE
_HARDWARESTATE.fields_by_name['gnss_state'].enum_type = _HARDWARESTATE_GNSSSTATE
_HARDWARESTATE.fields_by_name['src_hardware_state'].message_type = _SRCHARDWARESTATE
_HARDWARESTATE.fields_by_name['hardware_state'].enum_type = _HARDWARESTATE_HSTATE
_HARDWARESTATE_BATTERYSTATE.containing_type = _HARDWARESTATE
_HARDWARESTATE_LASERSTATE.containing_type = _HARDWARESTATE
_HARDWARESTATE_WIFISTATE.containing_type = _HARDWARESTATE
_HARDWARESTATE_BREAKSWITCHSTATE.containing_type = _HARDWARESTATE
_HARDWARESTATE_JACKINGSTATE.containing_type = _HARDWARESTATE
_HARDWARESTATE_GNSSSTATE.containing_type = _HARDWARESTATE
_HARDWARESTATE_HSTATE.containing_type = _HARDWARESTATE
_MAPMETADATA.fields_by_name['origin'].message_type = _POSE
_OCCUPANCYGRIDMAP.fields_by_name['header'].message_type = _MAPHEADER
_OCCUPANCYGRIDMAP.fields_by_name['info'].message_type = _MAPMETADATA
_RESULT.fields_by_name['result_state'].enum_type = _RESULT_RESULTSTATE
_RESULT.fields_by_name['result_code'].enum_type = _RESULT_RESULTCODE
_RESULT_RESULTSTATE.containing_type = _RESULT
_RESULT_RESULTCODE.containing_type = _RESULT
_UISTATE.fields_by_name['parent_state'].enum_type = _UISTATE_PARENTSTATE
_UISTATE.fields_by_name['child_state'].enum_type = _UISTATE_CHILDSTATE
_UISTATE_PARENTSTATE.containing_type = _UISTATE
_UISTATE_CHILDSTATE.containing_type = _UISTATE
DESCRIPTOR.message_types_by_name['Message'] = _MESSAGE
DESCRIPTOR.message_types_by_name['Request'] = _REQUEST
DESCRIPTOR.message_types_by_name['Response'] = _RESPONSE
DESCRIPTOR.message_types_by_name['Command'] = _COMMAND
DESCRIPTOR.message_types_by_name['Notification'] = _NOTIFICATION
DESCRIPTOR.message_types_by_name['NotifyResponse'] = _NOTIFYRESPONSE
DESCRIPTOR.message_types_by_name['CalibrationResult'] = _CALIBRATIONRESULT
DESCRIPTOR.message_types_by_name['UpdateResult'] = _UPDATERESULT
DESCRIPTOR.message_types_by_name['LoginRequest'] = _LOGINREQUEST
DESCRIPTOR.message_types_by_name['LoginInfo'] = _LOGININFO
DESCRIPTOR.message_types_by_name['Info'] = _INFO
DESCRIPTOR.message_types_by_name['FileInfo'] = _FILEINFO
DESCRIPTOR.message_types_by_name['MapList'] = _MAPLIST
DESCRIPTOR.message_types_by_name['FileList'] = _FILELIST
DESCRIPTOR.message_types_by_name['Pose'] = _POSE
DESCRIPTOR.message_types_by_name['Path'] = _PATH
DESCRIPTOR.message_types_by_name['MotionControlState'] = _MOTIONCONTROLSTATE
DESCRIPTOR.message_types_by_name['MovementTask'] = _MOVEMENTTASK
DESCRIPTOR.message_types_by_name['ActionTask'] = _ACTIONTASK
DESCRIPTOR.message_types_by_name['SystemState'] = _SYSTEMSTATE
DESCRIPTOR.message_types_by_name['HardwareState'] = _HARDWARESTATE
DESCRIPTOR.message_types_by_name['SRCHardwareState'] = _SRCHARDWARESTATE
DESCRIPTOR.message_types_by_name['LaserPoints'] = _LASERPOINTS
DESCRIPTOR.message_types_by_name['MapHeader'] = _MAPHEADER
DESCRIPTOR.message_types_by_name['MapMetaData'] = _MAPMETADATA
DESCRIPTOR.message_types_by_name['OccupancyGridMap'] = _OCCUPANCYGRIDMAP
DESCRIPTOR.message_types_by_name['CloudMap'] = _CLOUDMAP
DESCRIPTOR.message_types_by_name['Result'] = _RESULT
DESCRIPTOR.message_types_by_name['UiState'] = _UISTATE
DESCRIPTOR.enum_types_by_name['MessageType'] = _MESSAGETYPE
DESCRIPTOR.enum_types_by_name['CommandType'] = _COMMANDTYPE
DESCRIPTOR.enum_types_by_name['LocationMode'] = _LOCATIONMODE

Message = _reflection.GeneratedProtocolMessageType('Message', (_message.Message,), dict(
  DESCRIPTOR = _MESSAGE,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.Message)
  ))
_sym_db.RegisterMessage(Message)

Request = _reflection.GeneratedProtocolMessageType('Request', (_message.Message,), dict(
  DESCRIPTOR = _REQUEST,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.Request)
  ))
_sym_db.RegisterMessage(Request)

Response = _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), dict(
  DESCRIPTOR = _RESPONSE,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.Response)
  ))
_sym_db.RegisterMessage(Response)

Command = _reflection.GeneratedProtocolMessageType('Command', (_message.Message,), dict(
  DESCRIPTOR = _COMMAND,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.Command)
  ))
_sym_db.RegisterMessage(Command)

Notification = _reflection.GeneratedProtocolMessageType('Notification', (_message.Message,), dict(
  DESCRIPTOR = _NOTIFICATION,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.Notification)
  ))
_sym_db.RegisterMessage(Notification)

NotifyResponse = _reflection.GeneratedProtocolMessageType('NotifyResponse', (_message.Message,), dict(
  DESCRIPTOR = _NOTIFYRESPONSE,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.NotifyResponse)
  ))
_sym_db.RegisterMessage(NotifyResponse)

CalibrationResult = _reflection.GeneratedProtocolMessageType('CalibrationResult', (_message.Message,), dict(
  DESCRIPTOR = _CALIBRATIONRESULT,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.CalibrationResult)
  ))
_sym_db.RegisterMessage(CalibrationResult)

UpdateResult = _reflection.GeneratedProtocolMessageType('UpdateResult', (_message.Message,), dict(
  DESCRIPTOR = _UPDATERESULT,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.UpdateResult)
  ))
_sym_db.RegisterMessage(UpdateResult)

LoginRequest = _reflection.GeneratedProtocolMessageType('LoginRequest', (_message.Message,), dict(
  DESCRIPTOR = _LOGINREQUEST,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.LoginRequest)
  ))
_sym_db.RegisterMessage(LoginRequest)

LoginInfo = _reflection.GeneratedProtocolMessageType('LoginInfo', (_message.Message,), dict(
  DESCRIPTOR = _LOGININFO,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.LoginInfo)
  ))
_sym_db.RegisterMessage(LoginInfo)

Info = _reflection.GeneratedProtocolMessageType('Info', (_message.Message,), dict(
  DESCRIPTOR = _INFO,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.Info)
  ))
_sym_db.RegisterMessage(Info)

FileInfo = _reflection.GeneratedProtocolMessageType('FileInfo', (_message.Message,), dict(
  DESCRIPTOR = _FILEINFO,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.FileInfo)
  ))
_sym_db.RegisterMessage(FileInfo)

MapList = _reflection.GeneratedProtocolMessageType('MapList', (_message.Message,), dict(
  DESCRIPTOR = _MAPLIST,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.MapList)
  ))
_sym_db.RegisterMessage(MapList)

FileList = _reflection.GeneratedProtocolMessageType('FileList', (_message.Message,), dict(
  DESCRIPTOR = _FILELIST,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.FileList)
  ))
_sym_db.RegisterMessage(FileList)

Pose = _reflection.GeneratedProtocolMessageType('Pose', (_message.Message,), dict(
  DESCRIPTOR = _POSE,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.Pose)
  ))
_sym_db.RegisterMessage(Pose)

Path = _reflection.GeneratedProtocolMessageType('Path', (_message.Message,), dict(
  DESCRIPTOR = _PATH,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.Path)
  ))
_sym_db.RegisterMessage(Path)

MotionControlState = _reflection.GeneratedProtocolMessageType('MotionControlState', (_message.Message,), dict(
  DESCRIPTOR = _MOTIONCONTROLSTATE,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.MotionControlState)
  ))
_sym_db.RegisterMessage(MotionControlState)

MovementTask = _reflection.GeneratedProtocolMessageType('MovementTask', (_message.Message,), dict(
  DESCRIPTOR = _MOVEMENTTASK,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.MovementTask)
  ))
_sym_db.RegisterMessage(MovementTask)

ActionTask = _reflection.GeneratedProtocolMessageType('ActionTask', (_message.Message,), dict(
  DESCRIPTOR = _ACTIONTASK,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.ActionTask)
  ))
_sym_db.RegisterMessage(ActionTask)

SystemState = _reflection.GeneratedProtocolMessageType('SystemState', (_message.Message,), dict(
  DESCRIPTOR = _SYSTEMSTATE,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.SystemState)
  ))
_sym_db.RegisterMessage(SystemState)

HardwareState = _reflection.GeneratedProtocolMessageType('HardwareState', (_message.Message,), dict(
  DESCRIPTOR = _HARDWARESTATE,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.HardwareState)
  ))
_sym_db.RegisterMessage(HardwareState)

SRCHardwareState = _reflection.GeneratedProtocolMessageType('SRCHardwareState', (_message.Message,), dict(
  DESCRIPTOR = _SRCHARDWARESTATE,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.SRCHardwareState)
  ))
_sym_db.RegisterMessage(SRCHardwareState)

LaserPoints = _reflection.GeneratedProtocolMessageType('LaserPoints', (_message.Message,), dict(
  DESCRIPTOR = _LASERPOINTS,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.LaserPoints)
  ))
_sym_db.RegisterMessage(LaserPoints)

MapHeader = _reflection.GeneratedProtocolMessageType('MapHeader', (_message.Message,), dict(
  DESCRIPTOR = _MAPHEADER,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.MapHeader)
  ))
_sym_db.RegisterMessage(MapHeader)

MapMetaData = _reflection.GeneratedProtocolMessageType('MapMetaData', (_message.Message,), dict(
  DESCRIPTOR = _MAPMETADATA,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.MapMetaData)
  ))
_sym_db.RegisterMessage(MapMetaData)

OccupancyGridMap = _reflection.GeneratedProtocolMessageType('OccupancyGridMap', (_message.Message,), dict(
  DESCRIPTOR = _OCCUPANCYGRIDMAP,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.OccupancyGridMap)
  ))
_sym_db.RegisterMessage(OccupancyGridMap)

CloudMap = _reflection.GeneratedProtocolMessageType('CloudMap', (_message.Message,), dict(
  DESCRIPTOR = _CLOUDMAP,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.CloudMap)
  ))
_sym_db.RegisterMessage(CloudMap)

Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), dict(
  DESCRIPTOR = _RESULT,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.Result)
  ))
_sym_db.RegisterMessage(Result)

UiState = _reflection.GeneratedProtocolMessageType('UiState', (_message.Message,), dict(
  DESCRIPTOR = _UISTATE,
  __module__ = 'plns_pb2'
  # @@protoc_insertion_point(class_scope:proto.UiState)
  ))
_sym_db.RegisterMessage(UiState)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('H\003'))
_LASERPOINTS.fields_by_name['xs'].has_options = True
_LASERPOINTS.fields_by_name['xs']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_LASERPOINTS.fields_by_name['ys'].has_options = True
_LASERPOINTS.fields_by_name['ys']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_LASERPOINTS.fields_by_name['zs'].has_options = True
_LASERPOINTS.fields_by_name['zs']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
