import numpy as np
from PIL import ImageGrab
import cv2
from directkeys import PressKey, ReleaseKey
import time
import pyautogui
from numpy import ones, vstack
from numpy.linalg import lstsq
from statistics import mean
from getkeys import key_check
import os

np.load_old = np.load
np.load = lambda *a, **k: np.load_old(*a, allow_pickle=True, **k)   # 改变npload参数，否则报错

W = 0x11; A = 0x1E; S = 0x1F; D = 0x20

def keys_to_output(keys):
    # [A, W, D]
    output = [0, 0, 0] # 0 is a boolean
    keys_dict = {'A':0, 'W':1, 'D':2}
    for key in ['A', 'W', 'D']:
        if key in keys:
            output[keys_dict[key]] = 1
    return output


def roi(img, vert):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vert, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):
    
    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker 
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]] # 取得y坐标
        min_y = min(ys) # 得到线段中最小的y坐标
        max_y = 600
        new_lines = []
        line_dict = {}

        # 获得xy坐标
        for idx,i in enumerate(lines):
            for xyxy in i:
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T    # 堆叠
                k, b = lstsq(A, y_coords)[0]    # 通过最小二乘法解超定方程组

                # 通过最小而成法得到的斜率k和b计算新的x坐标
                x1 = (min_y-b) / k
                x2 = (max_y-b) / k

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
    except Exception as e:
        pass


def process_img(origin_img):
    # 将图片转为灰色的单一值更方便训练
    process_img = cv2.Canny(origin_img, threshold1=300, threshold2=400)

    # (a,b)为高斯模糊的卷积内核大小，必须为奇数,屏蔽过密的区域
    process_img = cv2.GaussianBlur(process_img, (5, 5), 0)   
    vert = np.array([[10, 500], [10, 280], [300, 180], [500, 180], [800, 280], [800, 500]], np.int32)
    process_img = roi(process_img, [vert])

    lines = cv2.HoughLinesP(process_img, 1, np.pi/180, 180, 20, 15)   # 检测图中满足要求的直线并返回坐标
    m1 = 0; m2 = 0  # 预设路径末点均值
    try:
        l1, l2, m1, m2 = draw_lanes(process_img, lines)   # 根据坐标画线
        cv2.line(orgin_img, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 20)  #画道路线，分别为两个xy坐标，RGB颜色和宽度
        cv2.line(orgin_img, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 20)
    except Exception as e:
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(process_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                                
            except Exception as e:
                pass
    except Exception as e:
        pass
    return process_img, origin_img, m1, m2

# 控制行动
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('data loading')
    training_data = list(np.load(file_name))
else:
    print('file not exist')
    training_data = []

for i in range(4):
    print(4-i)
    time.sleep(1)

while 1:
    # grab参数bbox(x0,y0, x1, y1)，位置从屏幕左上角算起
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (80, 60))
    
    keys = key_check()
    output = keys_to_output(keys)
    training_data.append([screen, output])

    # stored data
    if len(training_data) % 500 == 0:
        print(len(training_data))
        np.save(file_name, training_data)