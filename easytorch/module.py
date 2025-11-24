import sys
import torch

#
# class to hold model
# usage : calculate network function by forward()
class Model(torch.nn.Module):

  # initializer
  def __init__(self, model:torch.nn.Module)->None:
    super().__init__()
    # initialize member function
    self._model = model
    self._device = "cpu"
    self._device_id = [0]
    self._non_blocking = False
    # check if forward is defined
    if self._model and not hasattr(self._model, "forward"):
      print("error : forward() is not defined in _model.")
      sys.exit()

  # available checker
  def check_available(self)->bool:
    return self._model

  # run type setter
  def set_run_type(self, device:str, device_ids:list, non_blocking:bool)->None:
    # set run type
    self._device = device
    self._non_blocking = non_blocking
    # send model to target device
    if (self._model):
      if (self._device == "cpu"):
        self._model.cpu()
      elif (self._device == "cuda"):
        self._model = torch.nn.DataParallel(self._model.cuda(), device_ids=device_ids)
      else:
        print("error : invalid device type is specified.")
        sys.exit()

  # forward function calculator
  def forward(self, **kwargs:dict)->dict:
    # calc forward function using input data
    data = kwargs["data"].to(self._device, non_blocking=self._non_blocking)
    out = self._model(data)
    kwargs["out"] = out
    return kwargs

  # model state loader
  def load(self, filepath:str)->None:
    state_dict = torch.load(filepath)
    self._model.load_state_dict(state_dict)

  # model state saver
  def save(self, filepath:str)->None:
    torch.save(self._model.state_dict(), filepath)

#
# class to hold criterion
# usage : calculate loss function by forward()
class Criterion(torch.nn.Module):

  # initializer
  def __init__(self, criterion:torch.nn.Module)->None:
    super().__init__()
    # initialize member function
    self._criterion = criterion
    self._device = "cpu"
    self._non_blocking = False
    # check if forward is defined
    if self._criterion and not hasattr(self._criterion, "forward"):
      print("error : forward() is not defined in _criterion.")
      sys.exit()

  # available checker
  def check_available(self)->bool:
    return self._criterion

  # run type setter
  def set_run_type(self, device:str, non_blocking:bool)->None:
    # set run type
    self._device = device
    self._non_blocking = non_blocking
    # send model to target device
    if (self._criterion):
      if (self._device == "cpu"):
        self._criterion.cpu()
      elif (self._device == "cuda"):
        self._criterion.cuda()
      else:
        print("error : invalid device type is specified.")
        sys.exit()

  # forward function calculator
  def forward(self, **kwargs:dict)->dict:
    # calc forward function using out & target data
    out = kwargs["out"].to(self._device, non_blocking=self._non_blocking)
    target = kwargs["target"].to(self._device, non_blocking=self._non_blocking)
    return self._criterion(out, target)

#
# class to hold optimizer
# usage : initialize gradient by zero_grad()
# usage : update weight by step()
class Optimizer():

  # initializer
  def __init__(self, optimizer:torch.nn.Module)->None:
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
#
# class to hold evaluator
# usage : calculate evaluate function by forward()
class Evaluator(torch.nn.Module):

  # initializer
  def __init__(self, evaluator:torch.nn.Module)->None:
    super().__init__()
    # initialize member function
    self._evaluator = evaluator
    self._device = "cpu"
    self._non_blocking = False
    # check if forward is defined
    if self._evaluator and not hasattr(self._evaluator, "forward"):
      print("error : forward() is not defined in _evaluator.")
      sys.exit()

  # available checker
  def check_available(self)->bool:
    return self._evaluator

  # run type setter
  def set_run_type(self, device:str, non_blocking:bool)->None:
    # set run type
    self._device = device
    self._non_blocking = non_blocking
    # send model to target device
    if (self._evaluator):
      if (self._device == "cpu"):
        self._evaluator.cpu()
      elif (self._device == "cuda"):
        self._evaluator.cuda()
      else:
        print("error : invalid device type is specified.")
        sys.exit()

  # forward function calculator
  def forward(self, **kwargs:dict)->dict:
    # calc forward function using out & target data
    out = kwargs["out"].to(self._device, non_blocking=self._non_blocking)
    target = kwargs["target"].to(self._device, non_blocking=self._non_blocking)
    return self._evaluator(out, target)