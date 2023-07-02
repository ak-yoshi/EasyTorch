import torch
import torch.nn as nn

#
## class to count correct rate
class CorrectRate(nn.Module):

  # initializer
  def __init__(self):
    super().__init__()

  # forward function calculator
  def forward(self, out, target):
    # calc forward function using output data
    out = out.argmax(1)
    return torch.eq(out, target).sum()