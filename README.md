# EasyTorch

## Introduction
This is a template code to reduce the implementation burden in PyTorch coding.
You can examine training or evaluation easily by simply preparing model and training data.

## Requirement
* PyTorch 1.0 or later
 
## Installation
You can install source code by using following command.

```bash
git clone git@github.com:ak-yoshi/EasyTorch.git
```

## Usage
### 1. data setting to data manager
You can setup __DataManager__ class by introducing user-defined train/eval/test dataset class.
And you can also set properties (batch_size & num_workers) related to __Dataloader__ class in PyTorch.

```python
data_manager = DataManager.DataManager(train_data=ud_train_data, test_data=ud_test_data,
                                       batch_size=batch_size, num_workers=num_workers)
```

Here, it is needed to inhelit __Data__ class at user-defined dataset class implementation.
And user-defined dataset class must have specified variables(data size, data contents and target contents).

```python
from framework import Data

class UserDefinedDataSet(Data.Data):

  # initializer
  def __init__(self, filepath):
    self._len = 0
    self._data = []
    self._target = []
    ...
```

### 2. declaration of network
Before declaration of network, you must define user-defined model, criterion, evaluator(optional) and optimizer.
Here, it is needed to inhelit __nn.Module__ class and define __forward()__ method in user-defined module class.

```python
import torch.nn as nn

class UserDefinedModel(nn.Module):

  # initializer
  def __init__(self, **kwargs):
    ...
  
  # forward function
  def forward(self, x):
    ...
    return x
```

After that, all components needed for deep learning are taken into __Network__ class.

```python
net = Network.Network(model=model, criterion=criterion, optimizer=optimizer, evaluator=evaluator, 
                      data_manager=data_manager, device=device, device_ids=device_ids, non_blocking=True)
```

### 3. training
You can examine training by calling __train()__ method.
The number of learning epochs can be controlled by the number of loop iterations

```python
for _ in range(num_epoch):
  net.train()
```

### 4. evaluation/testing
You can examine evaluation/testing by calling __eval()__ and __test()__ method.

```python
net.eval()
net.test()
```

## Reference
[1] [PyTorchのテンプレコードを用意してどんなデータセットにも楽々ディープラーニング](https://deoxy.hatenablog.com/entry/2020/12/05/235908)