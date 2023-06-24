import torch.nn as nn


def get_normalization_module(type='bn', num_filters=0, group_size=0):
    if type == 'bn':
        return nn.BatchNorm2d(num_features=num_filters)
    
    if type == 'gn':
        return nn.GroupNorm(num_groups= group_size, num_channels= num_filters)

    if type == 'ln':
        return nn.GroupNorm(num_groups=1, num_channels=num_filters)
    return None

def get_activation_module(type='relu'):
    if type == 'relu':
        return nn.ReLU()

def block(in_channels, num_filters, filter_size=3, padding=0, norm='skip', activation='skip', group_size=0, drop_out=0):

    sequence = nn.Sequential()

    '''
    Add default Convolution step
    #no of filters => output
    filter size => kernel
    '''
    sequence.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            padding=padding
        )
    )

    '''
    Add normalization
    Allowed Values:
        `bn` = Batch Normalization
        `gn` = Group Normalization
        `ln` = Layer Normalization
        `skip` = Skip Normalization Step
    '''
    if norm != 'skip':
        sequence.append(
            get_normalization_module(norm, num_filters, group_size=group_size)
        )


    '''
    Add Activation
    Allowed Values:
        `relu` = ReLU Activation Function
        `skip` = Skip Activation Step
    '''
    if activation != 'skip':
        sequence.append(
           get_activation_module(type=activation)
        )

    '''
    Add Dropout
    '''
    if drop_out > 0:
        sequence.append(
            nn.Dropout2d(drop_out)
        )

    return sequence