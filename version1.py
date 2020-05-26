import numpy as np
from PIL import ImageGrab
import cv2
from directkeys import PressKey, ReleaseKey
import time

W = 0x11; A = 0x1E; S = 0x1F; D = 0x20

for i in range(4):
    print(4-i)
    time.sleep(1)


def process_img(orgin_img):
    # 将图片转为灰色的单一值更方便训练
    process_img = cv2.cvtColor(orgin_img, cv2.COLOR_BGR2GRAY)  
    process_img = cv2.Canny(process_img, threshold1=200, threshold2=300)
    return process_img

start_time = time.time()
press = False
while 1:
    # grab参数bbox(x0,y0, x1, y1)，位置从屏幕左上角算起
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    gray_screen = process_img(screen)
    # 转为numpy以便imshow，并且由于color出错需要改变颜色
    cv2.imshow('window', cv2.cvtColor(gray_screen, cv2.COLOR_BGR2RGB))
    # 按位取与0xFF使输入保留低8位，25为等待案件的delaytime=25ms

    # 此处不能用time.sleep因为会影响画面刷新
    if not press and int(time.time()-start_time) % 3 < 1.5:
        PressKey(W)     # 相当于按住W
        press = True
    elif press and int(time.time()-start_time) % 3 > 1.5:
        ReleaseKey(W)   # 松开W
        press = False

    if cv2.waitKey(25) and 0xFF == ord('q'): 
        cv2.destroyAllWindows()  # 输入q时关闭窗口
        break
