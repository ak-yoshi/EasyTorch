import sys
import torch

from framework import Model
from framework import Criterion
from framework import Evaluator
from framework import Optimizer

#
# class to manage model training/evaluation
# usage : execute training by net.train()
# usage : execute evaluation by net.eval()
class Network():

  # initializer
  def __init__(self, **kwargs):
    # initialize member function
    self._model = Model.Model(kwargs.get("model", None))
    self._criterion = Criterion.Criterion(kwargs.get("criterion", None))
    self._evaluator = Evaluator.Evaluator(kwargs.get("evaluator", None))
    self._optimizer = Optimizer.Optimizer(kwargs.get("optimizer", None))
    self._data_manager = kwargs.get("data_manager", None)
    self._save_checkpoint = kwargs.get("save_checkpoint", False)
    self._epoch = 0
    self._loss_min = 1e10
    self._train_loss_hist = []
    self._eval_loss_hist = []
    # set device type & non blocking flag
    device = kwargs.get("device", "cpu")
    device_ids = kwargs.get("device_ids", [0])
    non_blocking = kwargs.get("non_blocking", False)
    self._model.set_run_type(device, device_ids, non_blocking)
    self._criterion.set_run_type(device, non_blocking)
    self._evaluator.set_run_type(device, non_blocking)

  # trained model loader
  def load(self):
    self._model.load("model/model.pth")
    print("trained model was loaded.")

  # trained model saver
  def save(self):
    self._model.save("model/model.pth")
    print("trained model at current step was saved.")

  # available checker
  def check_available(self):
    if (not self._model.check_available()):
      print("error : model is not loaded.")
      sys.exit()
    if (not self._criterion.check_available()):
      print("error : criterion is not loaded.")
      sys.exit()
    if (not self._optimizer.check_available()):
      print("error : optimizer is not loaded.")
      sys.exit()
    if (not self._data_manager):
      print("error : data manager is not loaded.")
      sys.exit()

  # training function
  def train(self):
    self.check_available()
    total_loss = 0.0
    total_size = 0
    self._model.train()
    for batch in self._data_manager.get_train_data_loader():
      self._optimizer.zero_grad()
      out = self._model(**batch)
      loss = self._criterion(**out)
      total_loss += loss
      loss.backward()
      self._optimizer.step()
      total_size += batch["size"]
    self._epoch += 1
    self._train_loss_hist.append(total_loss.item()/total_size)
    print(f"train mode : epoch = {self._epoch:>5}, loss = {total_loss.item()/total_size:.3e}")
    if (total_loss.item()/total_size < self._loss_min and self._save_checkpoint == True):
      self._loss_min = total_loss.item()/total_size
      self.save()

  # eval function
  def eval(self):
    self.check_available()
    total_loss = 0.0
    total_eval = 0.0
    total_size = 0
    self._model.eval()
    with torch.no_grad():
      for batch in self._data_manager.get_eval_data_loader():
        out = self._model(**batch)
        loss = self._criterion(**out)
        eval = self._evaluator(**out) if self._evaluator.check_available() else 0.0
        total_loss += loss
        total_eval += eval
        total_size += batch["size"]
      self._eval_loss_hist.append(total_loss.item()/total_size)
    if (self._evaluator):
      print(f"eval mode : loss = {total_loss.item()/total_size:.3e}, eval = {total_eval/total_size:.3e}")
    else:
      print(f"eval mode : loss = {total_loss.item()/total_size:.3e}")

  # test function
  def test(self):
    self.check_available()
    total_loss = 0.0
    total_eval = 0.0
    total_size = 0
    self._model.eval()
    with torch.no_grad():
      for batch in self._data_manager.get_test_data_loader():
        out = self._model(**batch)
        loss = self._criterion(**out)
        eval = self._evaluator(**out) if self._evaluator.check_available() else 0.0
        total_loss += loss
        total_eval += eval
        total_size += batch["size"]
    if (self._evaluator):
      print(f"test mode : loss = {total_loss.item()/total_size:.3e}, eval = {total_eval/total_size:.3e}")
    else:
      print(f"test mode : loss = {total_loss.item()/total_size:.3e}")