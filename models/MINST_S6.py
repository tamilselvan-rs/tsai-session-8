import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2,padding=1),
            nn.Dropout(0.1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU()
        )

        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x) # 28 > 26 > 13 | 1>3>4
        x = self.conv2(x) # 13 > 11 > 6 | 4 > 8 > 9
        x = self.conv3(x) # 6 > 3
        x = self.conv4(x)
        #print(x.shape)
        x = x.view(-1, 64) # 4*4*256 = 4096
        x = self.fc1(x)
        # x = self.fc2(x)
        return F.log_softmax(x, dim=1)
