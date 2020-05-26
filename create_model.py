import numpy as np
from alexnet import AlexNet
import torch.nn as nn
import torch.autograd.variable as Variable
import torch
from random import shuffle

np.load_old = np.load
np.load = lambda *a, **k: np.load_old(*a, allow_pickle=True, **k) 
train_data = np.load('training_data.npy')

WIDTH = 80
HEIGHT = 60
LR = 0.00005
EPOCH = 8
MODEL_NAME = 'PY-CAR-LR_{}-EPOCHS_{}.model'.format(LR, EPOCH)


train_data = np.load('training_data_v2.npy')

def test_accuracy(model, test_img, test_action):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.cuda()
    accuracy = []
    for data, target in zip(test_img, test_action):

        data = Variable(torch.from_numpy(data.reshape(1,1, 80, 60)))
        data = torch.tensor(data, dtype=torch.float32).to(device=device)
        output = model(data)
        output = int(output.argmax(dim=1))
        accuracy.append(1 if output==target.argmax() else 0)
    return (sum(accuracy)/len(accuracy))


def train(epoch, LR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model_v0.pkl')
    model = model.cuda()
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))

    test = train_data[-200:]
    test_img = np.array([each[0] for each in test]).reshape((-1, WIDTH, HEIGHT,1))
    test_action = np.array([i[1] for i in test])

    model.train()
    old_accuracy = test_accuracy(model, test_img, test_action)
    print('old accuracy is {}'.format(old_accuracy))
    
    for epoch in range(EPOCH):
        t = 0
        shuffle(train_data)
        train = train_data[:-300]


        train_img = np.array([each[0] for each in train]).reshape((-1, WIDTH, HEIGHT,1))
        # [A, W, D], [left, forward, right, slow_yoll]
        train_action = np.array([[i[1]] for i in train])
        test_img = np.array([each[0] for each in test]).reshape((-1, WIDTH, HEIGHT,1))
        test_action = np.array([i[1] for i in test])

        for data, target in zip(train_img, train_action):
            t += 1
            if t%500 == 0: print('epoch {} {}% has been down'.format(epoch, t/len(train)*100))
            data = Variable(torch.from_numpy(data.reshape(1, 1, 80, 60)))
            target = Variable(torch.from_numpy(target)).type(torch.FloatTensor)
            data = torch.tensor(data, dtype=torch.float32)

            target =target.to(device)
            data = data.to(device=device)

            optimizer.zero_grad()
            output = model(data)
            #target = target.reshape(1,4)
            #output = output.view(output.size(0), -1)
            #output.squeeze_()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        accuracy = test_accuracy(model, test_img, test_action)
        print('epoch {} accuracy is {}'.format(epoch, accuracy))
        if accuracy >= old_accuracy:
            torch.save(model, 'model_v{}.pkl'.format(epoch))
            old_accuracy = accuracy


train(EPOCH, LR)