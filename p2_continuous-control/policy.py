import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyContinuous:
    """Policy defines the behavior of the agent.
    """
    def __init__(self, policy_network, device='cpu'):
        self.device = device
        self.policy_network = policy_network

    def action(self, state):
        logits = self.policy_network(torch.as_tensor(state, dtype=torch.float))

    def pd(self, state):
        """Returns the action probability distribution at `state`

        Args:
          state: represents the state

        Returns:
          probability distribution of actions
        """
        pass

    def __call__(self, state, action):
        """Returns the probability of `action` to be taken at `state`

        Args:
          state: represents the state
          action: the action to be taken

        Returns:
          the probability of the actions to be takedn
        """
        pass

    def log_gradient(self, state, action):
        """Returns the gradient of log of policy at `state` and `action`
        """
        pass


class PolicyNetwork(nn.Module):
    """Policy network with linear layers
    """
    def __init__(self, n_state_dims, n_action_dims, hidden_units):
        super(PolicyNetwork, self).__init__()
        self.n_state_dims = n_state_dims
        self.n_action_dims = n_action_dims
        self.hidden_units = hidden_units

        self.hidden_layers = nn.ModuleList()
        input_units = self.n_state_dims
        for units in self.hidden_units:
            self.hidden_layers.append(nn.Linear(input_units, units))
        self.output_layer = nn.Linear(input_units, self.n_action_dims)

    def forward(self, state):
        x = state
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)
