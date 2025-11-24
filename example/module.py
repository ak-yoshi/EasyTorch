import torch

#
## class to infer using neural network
class NeuralNetwork(torch.nn.Module):

  # initializer
  def __init__(self, **kwargs):
    super().__init__()
    in_units  = kwargs.get("in_units")
    mid_units = kwargs.get("mid_units")
    out_units = kwargs.get("out_units")
    self._input_size = in_units
    # append first layer
    self._linear_list = torch.nn.ModuleList([torch.nn.Linear(in_features=in_units, out_features=mid_units[0])])
    # append after second layer
    for i in range(1, len(mid_units)):
      self._linear_list.append(torch.nn.Linear(in_features=mid_units[i-1], out_features=mid_units[i]))
    # append final layer
    self._final_layer = torch.nn.Linear(in_features=mid_units[-1], out_features=out_units)

  # forward function calculator
  def forward(self, x):
    # calc forward function using input data
    x = x.view(-1, self._input_size)
    for _, linear in enumerate(self._linear_list):
      x = torch.relu(linear(x))
    x = self._final_layer(x)
    x = torch.nn.functional.log_softmax(x, dim=1)
    return x

#
## class to count correct rate
class CorrectRate(torch.nn.Module):

  # initializer
  def __init__(self):
    super().__init__()

  # forward function calculator
  def forward(self, out, target):
    # calc forward function using output data
    out = out.argmax(1)
    return torch.eq(out, target).sum()