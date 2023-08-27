import sys

#
# class to manage data
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