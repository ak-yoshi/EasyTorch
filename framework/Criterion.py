import sys
import torch.nn as nn

#
# class to manage criterion
# usage : calculate loss function by forward()
class Criterion(nn.Module):

  # initializer
  def __init__(self, criterion:nn.Module)->None:
    super().__init__()
    # initialize member function
    self._criterion = criterion
    self._device = "cpu"
    self._non_blocking = False
    # check if forward function is defined
    if not hasattr(self._criterion, "forward"):
      print("error : forward function is not defined in _criterion.")
      sys.exit()

  # run type setter
  def set_run_type(self, device:str, non_blocking:bool)->None:
    # set run type
    self._device = device
    self._non_blocking = non_blocking
    # send model to target device
    if (self._device == "cpu"):
      self._criterion.cpu()
    elif (self._device == "cuda"):
      self._criterion.cuda()
    else:
      print("error : invalid device type is specified.")
      sys.exit()

  # forward function calculator
  def forward(self, **kwargs)->dict:
    # calc forward function using out & target data
    out = kwargs["out"].to(self._device, non_blocking=self._non_blocking)
    target = kwargs["target"].to(self._device, non_blocking=self._non_blocking)
    return self._criterion(out, target)