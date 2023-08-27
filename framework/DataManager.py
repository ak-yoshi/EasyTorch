import sys
import torch
from torch.utils.data import DataLoader

from framework import Dataset

#
# convert dictionary to user-defined torch tensor
# tensor type : { "index", "data", "target", "size" }
def custom_collate_fn(dict)->dict:
  index = torch.stack([dict[i]["index"] for i in range(len(dict))])
  data = torch.stack([dict[i]["data"] for i in range(len(dict))])
  target = torch.stack([dict[i]["target"] for i in range(len(dict))])
  return {"index":index, "data":data, "target":target, "size":len(dict)}

#
# class to manage model train, evaluate and test loader
class DataManager():

  # initializer
  def __init__(self, **kwargs)->None:
    train_data = kwargs.get("train_data", None)
    eval_data  = kwargs.get("eval_data", None)
    test_data  = kwargs.get("test_data", None)
    transform = kwargs.get("transform", None)
    self._train_dataset = Dataset.Dataset(train_data, transform) if (train_data != None) else None
    self._eval_dataset = Dataset.Dataset(eval_data, transform) if (eval_data != None) else None
    self._test_dataset = Dataset.Dataset(test_data, transform) if (test_data != None) else None
    self._batch_size = kwargs.get("batch_size", 100)
    self._num_workers = kwargs.get("num_workers", 1)
    self._shuffle = kwargs.get("shuffle", False)
    self._drop_last = kwargs.get("drop_last", False)

  # train data loader getter
  def get_train_data_loader(self)->DataLoader:
    if (self._train_dataset == None):
      print("error : failed to load train dataset.")
      sys.exit()
    return DataLoader(self._train_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                      collate_fn=custom_collate_fn, pin_memory=True, shuffle=self._shuffle, drop_last=self._drop_last)

  # eval data loader getter
  def get_eval_data_loader(self)->DataLoader:
    if (self._eval_dataset == None):
      print("error : failed to load eval dataset.")
      sys.exit()
    return DataLoader(self._eval_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                      collate_fn=custom_collate_fn, pin_memory=True, shuffle=self._shuffle, drop_last=self._drop_last)

  # test data loader getter
  def get_test_data_loader(self)->DataLoader:
    if (self._test_dataset == None):
      print("error : failed to load test dataset.")
      sys.exit()
    return DataLoader(self._test_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                      collate_fn=custom_collate_fn, pin_memory=True, shuffle=self._shuffle, drop_last=self._drop_last)