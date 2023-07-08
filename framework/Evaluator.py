import sys
import torch.nn as nn

#
# class to manage evaluator
# usage : calculate evaluate function by forward()
class Evaluator(nn.Module):

  # initializer
  def __init__(self, evaluator:nn.Module)->None:
    super().__init__()
    # initialize member function
    self._evaluator = evaluator
    self._device = "cpu"
    self._non_blocking = False
    # check if forward function is defined
    if not hasattr(self._evaluator, "forward"):
      print("error : forward function is not defined in _evaluator.")
      sys.exit()

  # run type setter
  def set_run_type(self, device:str, non_blocking:bool)->None:
    # set run type
    self._device = device
    self._non_blocking = non_blocking
    # send model to target device
    if (self._device == "cpu"):
      self._evaluator.cpu()
    elif (self._device == "cuda"):
      self._evaluator.cuda()
    else:
      print("error : invalid device type is specified.")
      sys.exit()

  # forward function calculator
  def forward(self, **kwargs)->dict:
    # calc forward function using out & target data
    out = kwargs["out"].to(self._device, non_blocking=self._non_blocking)
    target = kwargs["target"].to(self._device, non_blocking=self._non_blocking)
    return self._evaluator(out, target)