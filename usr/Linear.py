import torch
import torch.nn as nn
import torch.nn.functional as F

#
## class to calculate linear model
class Linear(nn.Module):

  # initializer
  def __init__(self, **kwargs):
    super().__init__()
    in_units  = kwargs.get("in_units")
    mid_units = kwargs.get("mid_units")
    out_units = kwargs.get("out_units")
    self._input_size = in_units
    # append first layer
    self._linear_list = nn.ModuleList([nn.Linear(in_features=in_units, out_features=mid_units[0])])
    # append after second layer
    for i in range(1, len(mid_units)):
      self._linear_list.append(nn.Linear(in_features=mid_units[i-1], out_features=mid_units[i]))
    # append final layer
    self._final_layer = nn.Linear(in_features=mid_units[-1], out_features=out_units)

  # forward function calculator
  def forward(self, x):
    # calc forward function using input data
    x = x.view(-1, self._input_size)
    for _, linear in enumerate(self._linear_list):
      x = torch.relu(linear(x))
    x = self._final_layer(x)
    x = F.log_softmax(x, dim=1)
    return x