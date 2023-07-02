import sys
import glob
import pickle
import numpy as np
import pandas as pd

from framework import Data

#
# class to load CIFAR10
class CIFAR10(Data.Data):

  # initializer
  def __init__(self, filepath):
    # initialize member function
    self._len = 0
    self._data = np.empty((0, 3 * 32 * 32))
    self._target = []
    # load list of file path
    filepath_list = glob.glob(filepath + "*")
    if (not filepath_list):
      print("error : failed to load data file.")
      sys.exit(0)
    # load data
    for _, filepath in enumerate(filepath_list):
      dict = pickle.load(open(filepath, "rb"), encoding="bytes")
      self._len += len(dict[b"labels"])
      self._data = np.append(self._data, dict[b"data"], axis=0)
      self._target.extend(dict[b"labels"])
    self._data = self._data.astype("float32").reshape(-1, 3, 32, 32)/255.0