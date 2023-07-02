import torch
from torchvision import transforms as transforms

from framework import Data

#
# class to store dataset
class Dataset():

  # initializer
  def __init__(self, data:Data.Data, transform:transforms=None)->None:
    # initialize member function
    self._len = data.get_len()
    self._data = data.get_data()
    self._target = data.get_target()
    self._transform = transform

  # length getter
  def __len__(self)->int:
    return self._len

  # item getter
  def __getitem__(self, index)->dict:
    # create dict from data & target
    index = torch.tensor(index, dtype = int)
    data = self._transform(self._data[index]) if self._transform != None else torch.tensor(self._data[index])
    target = torch.tensor(self._target[index])
    return { "index" : index, "data" : data, "target" : target }