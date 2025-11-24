import sys
import torch
from torchvision import transforms

#
# base class to hold data
class Data():

  # initializer
  def __init__(self)->None:
    self._len = 0
    self._data = []
    self._target = []

  # length getter
  def get_len(self):
    if hasattr(self, "_len"):
      return self._len
    else:
      print("error : _len is not defined.")
      sys.exit()

  # data getter
  def get_data(self):
    if hasattr(self, "_data"):
      return self._data
    else:
      print("error : _data is not defined.")
      sys.exit()

  # target getter
  def get_target(self):
    if hasattr(self, "_target"):
      return self._target
    else:
      print("error : _target is not defined.")
      sys.exit()

#
# class to hold dataset
class Dataset():

  # initializer
  def __init__(self, data:Data, transform:transforms=None)->None:
    # initialize member function
    self._len = data.get_len()
    self._data = data.get_data()
    self._target = data.get_target()
    self._transform = transform

  # length getter
  def __len__(self)->int:
    return self._len

  # item getter
  def __getitem__(self, index:int)->dict:
    # create dict from data & target
    index = torch.tensor(index, dtype = int)
    data = self._transform(self._data[index]) if self._transform != None else torch.tensor(self._data[index])
    target = torch.tensor(self._target[index])
    return {"index":index, "data":data, "target":target}

#
# convert dictionary to user-defined torch tensor
# tensor type : { "index", "data", "target", "size" }
def custom_collate_fn(dict:dict)->dict:
  index = torch.stack([dict[i]["index"] for i in range(len(dict))])
  data = torch.stack([dict[i]["data"] for i in range(len(dict))])
  target = torch.stack([dict[i]["target"] for i in range(len(dict))])
  return {"index":index, "data":data, "target":target, "size":len(dict)}

#
# class to hold train, evaluate and test loader
class DataLoader():

  # initializer
  def __init__(self, **kwargs:dict)->None:
    train_data = kwargs.get("train_data", None)
    eval_data  = kwargs.get("eval_data", None)
    test_data  = kwargs.get("test_data", None)
    transform = kwargs.get("transform", None)
    self._train_dataset = Dataset(train_data, transform) if (train_data != None) else None
    self._eval_dataset = Dataset(eval_data, transform) if (eval_data != None) else None
    self._test_dataset = Dataset(test_data, transform) if (test_data != None) else None
    self._batch_size = kwargs.get("batch_size", 100)
    self._num_workers = kwargs.get("num_workers", 1)
    self._shuffle = kwargs.get("shuffle", False)
    self._drop_last = kwargs.get("drop_last", False)

  # train data loader getter
  def get_train_data_loader(self)->torch.utils.data.DataLoader:
    if (self._train_dataset == None):
      print("error : failed to load train dataset.")
      sys.exit()
    return torch.utils.data.DataLoader(self._train_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                                       collate_fn=custom_collate_fn, pin_memory=True, shuffle=self._shuffle, drop_last=self._drop_last)

  # eval data loader getter
  def get_eval_data_loader(self)->torch.utils.data.DataLoader:
    if (self._eval_dataset == None):
      print("error : failed to load eval dataset.")
      sys.exit()
    return torch.utils.data.DataLoader(self._eval_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                                       collate_fn=custom_collate_fn, pin_memory=True, shuffle=self._shuffle, drop_last=self._drop_last)

  # test data loader getter
  def get_test_data_loader(self)->torch.utils.data.DataLoader:
    if (self._test_dataset == None):
      print("error : failed to load test dataset.")
      sys.exit()
    return torch.utils.data.DataLoader(self._test_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                                       collate_fn=custom_collate_fn, pin_memory=True, shuffle=self._shuffle, drop_last=self._drop_last)