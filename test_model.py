import numpy as np
import alexnet
import torch
import torch.autograd.variable as Variable

np.load_old = np.load
np.load = lambda *a, **k: np.load_old(*a, allow_pickle=True, **k) 
train_data = np.load('training_data.npy')

WIDTH = 80
HEIGHT = 60
LR = 1E-3
EPOCH = 8
MODEL_NAME = 'PY-CAR-LR_{}-EPOCHS_{}.model'.format(LR, EPOCH)


train_data = np.load('training_data_v2.npy')

test = train_data[-500:]

# [A, W, D], [left, forward, right, slow_yoll]

train = train_data[:-500]

train_img = np.array([each[0] for each in train]).reshape((len(train), WIDTH, HEIGHT))
# [A, W, D], [left, forward, right, slow_yoll]
train_action = np.array([i[1] for i in train])

test_img = np.array([each[0] for each in test]).reshape((len(test), WIDTH, HEIGHT))
test_action = np.array([i[1] for i in test])



def test_accuracy(test_img = train_img, test_action = train_action):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load('model_v1.pkl')
    accuracy = []
    for data, target in zip(test_img, test_action):

        data = Variable(torch.from_numpy(data.reshape(1,1, 80, 60)))
        data = torch.tensor(data, dtype=torch.float32)
        output = model(data)
        output = output.argmax(dim=1)
        output = int(output.data[0])
        accuracy.append(1 if output==target[0] else 0)
    print(sum(accuracy)/len(accuracy))

test_accuracy()