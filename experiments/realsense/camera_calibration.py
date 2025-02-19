# Eye-in-hand calibration
# 0. Get the camera intrinsic matrix K (assume the camera is calibrated)
# 1. Get the transformation matrix from the camera to the chessboard
# 2. Get the transformation matrix from the end-effector to the robot base
# 3. Use the cv2.calibrateHandEye function to get the transformation matrix from the camera to the end-effector
# 4. Verify the result

import cv2
import numpy as np
import glob
from math import *
import pandas as pd
import os

# TODO:根据实际情况修改
K=np.array([[614.9035034179688, 0, 323.47271728515625]
            [0, 614.9575805664062, 237.75799560546875]
            [0, 0, 1]],dtype=np.float64) # 内参 
chess_board_x_num=8 #棋盘格x方向格子数
chess_board_y_num=7 #棋盘格y方向格子数
chess_board_len=22 #单位棋盘格长度,mm


#用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

#用于根据位姿计算变换矩阵
def pose_robot(x, y, z, Tx, Ty, Tz):
    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
    # RT1=np.linalg.inv(RT1)
    return RT1

#用来从棋盘格图片得到相机外参
def get_RT_from_chessboard(img_path,chess_board_x_num,chess_board_y_num,K,chess_board_len):
    '''
    :param img_path: 读取图片路径
    :param chess_board_x_num: 棋盘格x方向格子数
    :param chess_board_y_num: 棋盘格y方向格子数
    :param K: 相机内参
    :param chess_board_len: 单位棋盘格长度,mm
    :return: 相机外参
    '''
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), None)
    print(corners)
    corner_points=np.zeros((2,corners.shape[0]),dtype=np.float64)
    for i in range(corners.shape[0]):
        corner_points[:,i]=corners[i,0,:]
    print(corner_points)
    object_points=np.zeros((3,chess_board_x_num*chess_board_y_num),dtype=np.float64)
    flag=0
    for i in range(chess_board_y_num):
        for j in range(chess_board_x_num):
            object_points[:2,flag]=np.array([(11-j-1)*chess_board_len,(8-i-1)*chess_board_len])
            flag+=1
    print(object_points)

    retval,rvec,tvec  = cv2.solvePnP(object_points.T,corner_points.T, K, distCoeffs=None)
    # print(rvec.reshape((1,3)))
    # RT=np.column_stack((rvec,tvec))
    RT=np.column_stack(((cv2.Rodrigues(rvec))[0],tvec))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    # RT=pose(rvec[0,0],rvec[1,0],rvec[2,0],tvec[0,0],tvec[1,0],tvec[2,0])
    # print(RT)

    # print(retval, rvec, tvec)
    # print(RT)
    # print('')
    return RT

folder = r"calib"#棋盘格图片存放文件夹 TODO:根据实际情况修改
# files = os.listdir(folder)
# file_num=len(files)
# RT_all=np.zeros((4,4,file_num))

# print(get_RT_from_chessboard('calib/2.bmp', chess_board_x_num, chess_board_y_num, K, chess_board_len))
'''
这个地方很奇怪的特点, 有些棋盘格点检测得出来, 有些检测不了, 可以通过函数get_RT_from_chessboard的运行时间来判断
'''
good_picture=[1,3,5,6,7,8,10,11,12,13,14,15]#存放可以检测出棋盘格角点的图片
# good_picture=[1,3,10,11,12]
file_num=len(good_picture)

#计算board to cam 变换矩阵
R_all_chess_to_cam_1=[]
T_all_chess_to_cam_1=[]
for i in good_picture:
    # print(i)
    image_path=folder+'/'+str(i)+'.bmp'
    RT=get_RT_from_chessboard(image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)

    # RT=np.linalg.inv(RT)

    R_all_chess_to_cam_1.append(RT[:3,:3])
    T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3,1)))
# print(T_all_chess_to_cam.shape)

#计算end to base变换矩阵
file_address='calib/机器人基坐标位姿.xlsx'#从记录文件读取机器人六个位姿 TODO:根据实际情况修改
sheet_1 = pd.read_excel(file_address)
R_all_end_to_base_1=[]
T_all_end_to_base_1=[]
# print(sheet_1.iloc[0]['ax'])
for i in good_picture:
    # print(sheet_1.iloc[i-1]['ax'],sheet_1.iloc[i-1]['ay'],sheet_1.iloc[i-1]['az'],sheet_1.iloc[i-1]['dx'],
    #                                   sheet_1.iloc[i-1]['dy'],sheet_1.iloc[i-1]['dz'])
    RT=pose_robot(sheet_1.iloc[i-1]['ax'],sheet_1.iloc[i-1]['ay'],sheet_1.iloc[i-1]['az'],sheet_1.iloc[i-1]['dx'],
                                      sheet_1.iloc[i-1]['dy'],sheet_1.iloc[i-1]['dz'])
    # RT=np.column_stack(((cv2.Rodrigues(np.array([[sheet_1.iloc[i-1]['ax']],[sheet_1.iloc[i-1]['ay']],[sheet_1.iloc[i-1]['az']]])))[0],
    #                    np.array([[sheet_1.iloc[i-1]['dx']],
    #                                   [sheet_1.iloc[i-1]['dy']],[sheet_1.iloc[i-1]['dz']]])))
    # RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    # RT = np.linalg.inv(RT)

    R_all_end_to_base_1.append(RT[:3, :3])
    T_all_end_to_base_1.append(RT[:3, 3].reshape((3, 1)))

# print(R_all_end_to_base_1)
R,T=cv2.calibrateHandEye(R_all_end_to_base_1,T_all_end_to_base_1,R_all_chess_to_cam_1,T_all_chess_to_cam_1)#手眼标定
RT=np.column_stack((R,T))
RT = np.row_stack((RT, np.array([0, 0, 0, 1])))#即为camera to end-effector 变换矩阵
print('相机相对于末端的变换矩阵为：')
print(RT)

#结果验证，原则上来说，每次结果相差较小
for i in range(len(good_picture)):

    RT_end_to_base=np.column_stack((R_all_end_to_base_1[i],T_all_end_to_base_1[i]))
    RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
    # print(RT_end_to_base)

    RT_chess_to_cam=np.column_stack((R_all_chess_to_cam_1[i],T_all_chess_to_cam_1[i]))
    RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
    # print(RT_chess_to_cam)

    RT_cam_to_end=np.column_stack((R,T))
    RT_cam_to_end=np.row_stack((RT_cam_to_end,np.array([0,0,0,1])))
    # print(RT_cam_to_end)

    RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam#即为固定的棋盘格相对于机器人基坐标系位姿
    RT_chess_to_base=np.linalg.inv(RT_chess_to_base)
    print('第',i,'次')
    print(RT_chess_to_base[:3,:])
    print('')

