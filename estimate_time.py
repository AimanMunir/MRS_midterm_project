import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

dev = "cpu"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Data(object):
    def __init__(self):
        self.path = './data'
        self.initial_pose = None
        self.goal_position = None
        self.count_steps = None

    def load(self):
        for i in range(450):
            tmp_initial_pose = np.load(self.path + f'/initial_pose/initial_pose_{i}.npy')
            tmp_goal_position = np.load(self.path + f'/goal_position/goal_position_{i}.npy')
            tmp_count_step = np.load(self.path + f'/iteration/iteration_{i}.npy')
            if self.initial_pose is None:
                self.initial_pose = tmp_initial_pose
                self.goal_position = tmp_goal_position
                self.count_steps = tmp_count_step
            else:
                self.initial_pose = np.hstack((self.initial_pose, tmp_initial_pose))
                self.goal_position = np.hstack((self.goal_position, tmp_goal_position))
                self.count_steps = np.hstack((self.count_steps, tmp_count_step))

        return np.concatenate((self.initial_pose.T, self.goal_position.T), axis=1), self.count_steps[:, None] / 400


if __name__ == '__main__':
    net = Net().to(dev)
    data = Data()

    x, y = data.load()

    tensor_x = torch.Tensor(x).to(dev)
    tensor_y = torch.Tensor(y).to(dev)

    my_dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)

    for epoch in range(5000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 10 == 0:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 10))
            #     running_loss = 0.0
        print('epoch: %d, loss: %.3f' % (epoch, running_loss / i))

    print('Finished Training')
