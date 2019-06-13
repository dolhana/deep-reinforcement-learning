import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, hidden_units):
        super(Model, self).__init__()
        