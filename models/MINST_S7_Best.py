import torch.nn as nn
import torch.nn.functional as F

'''
Model Params
------------
Best Train Accuracy: 98.75%
Best Test Accuracy: 99.45%
# of Parameters: 7,992
RF Out: 28
Batch Size: 128
LR: 0.01

Target
------
- 99.4% accuracy before 15th Epoch

Insights
--------
- Random Shift and Random Rotation added to data has improved the training accuracy & the model achieved max accuracy than its previous version
- Model isn't overfitting
- Achieved 99.4 in 16th Epoch onwards

Referrence
----------
https://pytorch.org/vision/stable/auto_examples/plot_transforms.html

'''

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()

        '''
        Input Block
        In => 28 * 28 * 1
        Out => 26 * 26 * 10
        RF => 3
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        '''
        Convolution Block 1
        In => 26 * 26 * 10
        Out => 24 * 24 * 10
        Stride In => 1
        Jin => 1
        Jout => 1
        RF => 5
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1))

        '''
        Pooling Layer => Max Pool
        In => 24 * 24 * 10
        Out => 12 * 12 * 10
        Stride In => 2
        Jin => 1
        Jout => 2
        RF => 6
        '''
        self.transitionBlock1 = nn.Sequential(
            nn.MaxPool2d(2,2)
        )

        '''
        Convolution Block 3
        In => 12 * 12 * 10
        Out => 10 * 10 * 10
        Stride In => 1
        Jin => 2
        Jout => 2
        RF => 10
        '''
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1))

        '''
        Convolution Block 4
        In => 10 * 10 * 10
        Out => 8 * 8 * 10
        Stride In => 1
        Jin => 2
        Jout => 2
        RF => 14
        '''
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        '''
        Convolution Block 5
        In => 8 * 8 * 10
        Out => 6 * 6 * 10
        Stride In => 1
        Jin => 2
        Jout => 2
        RF => 18
        '''
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        '''
        Convolution Block 6
        In => 6 * 6 * 10
        Out => 4 * 4 * 10
        Stride In => 1
        Jin => 2
        Jout => 2
        RF => 22
        '''
        self.conv6 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3)
        )

        '''
        Pooling Layer => Global Average Pooling
        In => 4 * 4 * 10
        Out => 1 * 1 * 10
        Stride In => 1
        Jin => 2
        Jout => 2
        RF => 28
        '''
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )

        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transitionBlock1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
