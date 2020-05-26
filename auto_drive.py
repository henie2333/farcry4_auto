import numpy as np
from PIL import ImageGrab
import cv2
from directkeys import PressKey, ReleaseKey
import time
import pyautogui

import torch
import torch.autograd.variable as Variable

W = 0x11; A = 0x1E; S = 0x1F; D = 0x20

from getkeys import key_check
import random

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 10


t_time = 0.06

def straight():
    if random.randrange(4) == 2:
        ReleaseKey(W)
    else:
        PressKey(W)
        ReleaseKey(A)
        ReleaseKey(D)

def left():
    ReleaseKey(D)
    PressKey(W)
    PressKey(A)
    #ReleaseKey(W)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    ReleaseKey(A)
    PressKey(W)
    PressKey(D)
    
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)


def autodrive(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80,60))
            screen = Variable(torch.from_numpy(screen.reshape(1,1, 80, 60)))
            screen = torch.tensor(screen, dtype=torch.float32).to(device)
            prediction = model(screen)[0]
            
            left_thresh = 0.9
            right_thresh = 0.8
            fwd_thresh = 0.7


            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[2] > right_thresh:
                right()
            elif prediction[0] > left_thresh:
                left()
            else:
                straight()
            

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

model = torch.load('model_v0.pkl')
autodrive(model)    