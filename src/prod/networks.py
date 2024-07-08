"""Create Policy, Value and Discriminator Notebooks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Policy network inspired by PyTorch implementation of GAIL.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        trajectories (bool): Whether to use trajectories.
    """
    
    def __init__(self, state_dim, action_dim, trajectories):
        super(PolicyNetwork, self).__init__()
        self.trajectories = trajectories
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 10, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 10, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(10 * ((state_dim // 8) ** 2), action_dim)
        )
    
    def forward(self, state):
        """Forward pass of the policy network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output action distribution or mean and covariance matrix.
        """
        if self.trajectories:
            prob = F.softmax(self.model(state), dim=1)
            distribution = torch.distributions.Categorical(prob)
        else:
            mean = self.model(state)
            log_std = nn.Parameter(torch.zeros_like(mean))
            std = torch.exp(log_std)
            distribution = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))
            
        return distribution


class ValueNetwork(nn.Module):
    """Value network.

    Args:
        state_dim (int): Dimension of the state space.
    """
    
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 10, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 10, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(10 * ((state_dim // 8) ** 2), 1)
        )
    
    def forward(self, state):
        """Forward pass of the value network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output value tensor.
        """
        return self.model(state)


class Discriminator(nn.Module):
    """Discriminator network.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        trajectories (bool): Whether to use trajectories.
    """
    
    def __init__(self, state_dim, action_dim, trajectories):
        super(Discriminator, self).__init__()
        self.trajectories = trajectories
        self.state_dim = state_dim
        self.action_dim = action_dim

        if self.trajectories:
            self.emb_actions = nn.Embedding(action_dim, state_dim)
            self.model_dim = 2 * state_dim
        else:
            self.model_dim = state_dim + action_dim
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 10, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 10, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(10 * ((self.model_dim // 8) ** 2), 1)
        )
    
    def forward(self, state, action):
        """Forward pass of the discriminator.

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.trajectories:
            action_emb = self.emb_actions(action)
            x = torch.cat([state, action_emb], dim=1)
        else:
            x = torch.cat([state, action], dim=1)
        
        logits = self.model(x.unsqueeze(1))
        return torch.sigmoid(logits)


class Expert(nn.Module):
    """Expert model.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        trajectories (bool): Whether to use trajectories.
        train_config (dict, optional): Training configuration. Default is None.
    """
    
    def __init__(self, state_dim, action_dim, trajectories, train_config=None):
        super(Expert, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trajectories = trajectories

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.trajectories)
    
    def get_networks(self):
        """Get the networks.

        Returns:
            list: List of networks.
        """
        return [self.pi]
    
    def action(self, state):
        """Get action from the policy network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output action tensor.
        """
        self.pi.eval()
        with torch.no_grad():
            distribution = self.pi(state)
            action = distribution.sample()
        
        return action


