import sys
import inspect
import torch
import torch.nn as nn

#
# class to manage model
class Model(nn.Module):

  # initializer
  def __init__(self, model:nn.Module)->None:
    super().__init__()
    # initialize member function
    self._model = model
    self._device = "cpu"
    self._device_id = [0]
    self._non_blocking = False
    # check if forward function is defined
    if not hasattr(self._model, "forward"):
      print("error : forward function is not defined in _model.")
      sys.exit()

  # run type setter
  def set_run_type(self, device:str, device_ids:list, non_blocking:bool)->None:
    # set run type
    self._device = device
    self._non_blocking = non_blocking
    # send model to target device
    if (self._device != "cpu"):
      self._model = nn.DataParallel(self._model.to(device), device_ids=device_ids)

  # forward function calculator
  def forward(self, **kwargs)->dict:
    # calc forward function using input data
    data = kwargs["data"].to(self._device, non_blocking=self._non_blocking)
    out = self._model(data)
    kwargs["out"] = out
    return kwargs

  # model function loader
  def load(self, filepath:str)->None:
    self._model = torch.load(filepath)

  # model function saver
  def save(self, filepath:str)->None:
    torch.save(self._model, filepath)
