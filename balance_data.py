import numpy as np
import pandas as  pd
from collections import Counter
from random import shuffle
import cv2
import time

np.load_old = np.load
np.load = lambda *a, **k: np.load_old(*a, allow_pickle=True, **k) 
train_data = np.load('training_data.npy')

df = pd.DataFrame(train_data)

print(df.head())
print(Counter(df[1].apply(str)))    # counter计数数组中各元素种类和数量

lefts = []
rights = []
forwards = []

shuffle(train_data) # 避免过拟合以及数据输入顺序对网络的影响

# [A, W, D], [left, forward, right, slow_yoll]
for data in train_data:
    img = data[0]
    action = data[1]
    
    if action == [1, 0, 0]:
        lefts.append([img, action])
    elif action == [0, 1, 0]:
        forwards.append([img, action])
    elif action == [0, 0, 1]:
        rights.append([img, action])
    

forwards = forwards[:min(len(lefts), len(rights))]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]


final_data = forwards + rights + lefts
shuffle(final_data)
print(len(final_data))

np.save('training_data_v2.npy', final_data)