import sys
import torch.nn as nn

#
# class to manage optimizer
# usage : initialize gradient by zero_grad()
# usage : update weight by step()
class Optimizer():

  # initializer
  def __init__(self, optimizer:nn.Module)->None:
    super().__init__()
    # initialize member function
    self._optimizer = optimizer
    # check if function is defined
    if self._optimizer and not hasattr(self._optimizer, "zero_grad"):
      print("error : zero_grad() is not defined in _optimizer.")
      sys.exit()
    if self._optimizer and not hasattr(self._optimizer, "step"):
      print("error : step() is not defined in _optimizer.")
      sys.exit()

  # available checker
  def check_available(self)->bool:
    return self._optimizer

  # gradient initializer
  def zero_grad(self)->None:
    self._optimizer.zero_grad()

  # param updater
  def step(self)->None:
    self._optimizer.step()