import numpy as np
import pandas as pd
from collections import Counter

np.load_old = np.load
np.load = lambda *a, **k: np.load_old(*a, allow_pickle=True, **k) 
train_data = np.load('training_data.npy')
for i in range(len(train_data)):    # [A, W, D]
    if train_data[i][1] == [1, 1, 1]:
        train_data[i][1] = [0, 1, 0]
    elif train_data[i][1][0] == 1:
        train_data[i][1] = [1, 0, 0]
    elif train_data[i][1][2] == 1:
        train_data[i][1] = [0, 0, 1]


df = pd.DataFrame(train_data)

print(df.head())
print(Counter(df[1].apply(str)))  

np.save('training_data.npy', train_data)
a = 1
