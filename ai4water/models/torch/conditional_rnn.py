
from ai4water.backend import torch

nn = torch.nn if torch is not None else None

class Conditionalize(nn.Module):

    def __init__(self, input_size, output_size, num_max_conditions=10, **kwargs):

        super(Conditionalize, self).__init__()

        self.units = output_size

        self.cond_to_init_state_dense_1 = nn.Linear(in_features=input_size, out_features=output_size)


    def forward(self, inputs):
        cond = inputs
        return self.cond_to_init_state_dense_1(cond)


i = nn.LSTM(3)
