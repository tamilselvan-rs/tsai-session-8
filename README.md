# tsai-session-8

## Folder Structure

```
.  
|__ cifar10_base.py # Base Model with configurable Dropout/Normalization/Activation  
|__ modelhelper.py  # Utils related to Model Training and Testing + Building blocks of an architecture  
|__ dataloader.py   # Utils related to Data Loading (batchsize and data source selection)  
|__ plots.py        # Plot Helper  
|__ common.py       # Runner configuration (Transformations and Device)
|__ *.ipynb         # Executions
|__ models/         # Older MNIST Models of S7 and S8
```

## How to configure Normalization for Model?

Batch Normalization  
1. Open [cifar10_base.py](./cifar10_base.py)
2. Edit `NORMALISATION` to `bn`
3. `GROUP_SIZE` will be ignored

Group Normalization
1. Open [cifar10_base.py](./cifar10_base.py)
2. Edit `NORMALISATION` to `gn`
3. Set `GROUP_SIZE` to `DESIRED_NUMBER`

Layer Normalization
1. Open [cifar10_base.py](./cifar10_base.py)
2. Edit `NORMALISATION` to `ln`
3. `GROUP_SIZE` will be auto detected

## How to maintain resolution while increasing the RF?

1. Use Padding

## Skip Connections
1. Model with skip connection achieved best accuracy of 71.07 at 20th Epoch [Cifar-Model-No-Skip](./cifar_10_batch_norm-no-skip.ipynb)
2. Model with skip connection achieved best accuracy of 71.97 at 20th Epoch [Cifar-Model](./cifar-10-batch-norm.ipynb)

[Ref](https://theaisummer.com/skip-connections/)


## Older Models of MNIST
[S6](./models/MINST_S6.py)  
[S7 Best](./models/MINST_S7_Best.py)
