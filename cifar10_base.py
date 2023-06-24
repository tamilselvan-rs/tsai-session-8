import torch.nn as nn
import torch.nn.functional as F
from modelhelper import block 

DROP_OUT = 0.1
NORMALISATION = 'bn'
GROUP_SIZE = 0
ACTIVATION = 'relu'

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()

        '''
        in: 32x32x3
        out: 32x32x32
        RF: 3
        '''
        self.conv1 = block(
            in_channels=3,
            num_filters=32,
            filter_size=3,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
            padding=1
        )
        
        '''
        in: 32x32x32
        out: 32x32x32
        RF: 5
        '''
        self.conv2 = block(
            in_channels=32,
            num_filters=32,
            filter_size=3,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
            padding=1
        )

        '''
        in: 32x32x32
        out: 32x32x32
        RF: 5
        '''
        self.ant1 = block(
            in_channels=32,
            num_filters=32,
            filter_size=1,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
        )

        '''
        in: 32x32x32
        out: 16x16x32
        RF: 6
        '''
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

        '''
        in: 16x16x32
        out: 16x16x32
        RF: 8
        '''
        self.conv3 = block(
            in_channels=32,
            num_filters=32,
            filter_size=3,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
            padding=1
        )

        '''
        in: 16x16x32
        out: 16x16x32
        RF: 10
        '''
        self.conv4 = block(
            in_channels=32,
            num_filters=32,
            filter_size=3,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
            padding=1
        )

        '''
        in: 16x16x32
        out: 16x16x32
        RF: 12
        '''
        self.conv5 = block(
            in_channels=32,
            num_filters=32,
            filter_size=3,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
            padding=1
        )

        '''
        in: 16x16x32
        out: 16x16x32
        RF: 12
        '''
        self.ant2 = block(
            in_channels=32,
            num_filters=32,
            filter_size=1,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
        )

        '''
        in: 16x16x32
        out: 8x8x32
        RF: 14
        '''
        self.transition2 = self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

        '''
        in: 8x8x32
        out: 8x8x32
        RF: 18
        '''
        self.conv6 = block(
            in_channels=32,
            num_filters=16,
            filter_size=3,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
            padding=1
        )

        '''
        in: 8x8x32
        out: 6x6x32
        RF: 22
        '''
        self.conv7 = block(
            in_channels=16,
            num_filters=16,
            filter_size=3,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT
        )

        '''
        in: 6x6x32
        out: 4x4x32
        RF: 26
        '''
        self.conv8 = block(
            in_channels=16,
            num_filters=16,
            filter_size=3,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
        )
        
        '''
        in: 4x4x16
        out: 1*1*16
        RF: 34
        '''
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )

        '''
        in: 1x1x32
        out: 1*1*10
        RF: 34
        '''
        self.conv9 = block(
            in_channels=16,
            num_filters=10,
            filter_size=1,
            norm=NORMALISATION,
            group_size=GROUP_SIZE,
            activation=ACTIVATION,
            drop_out=DROP_OUT,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.ant1(x)
        x = self.transition1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ant2(x)
        x = self.transition2(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.gap(x)
        x = self.conv9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
