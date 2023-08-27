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
    super().__init__()
    # load list of file path
    filepath_list = glob.glob(filepath + "/*")
    if (not filepath_list):
      print("error : failed to load data file.")
      sys.exit(0)
    # load data
    for _, filepath in enumerate(filepath_list):
      dict = pickle.load(open(filepath, "rb"), encoding="bytes")
      for data, target in zip(dict[b"data"], dict[b"labels"]):
        self._len += 1
        self._data.append(data)
        self._target.append(target)
    # convert to numpy 
    self._data = np.array(self._data)
    self._target = np.array(self._target)
    # normalize
    self._data = self._data.astype("float32").reshape(-1, 3, 32, 32)/255.0