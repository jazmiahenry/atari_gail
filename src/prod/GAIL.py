"""GAIL Model."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GAIL(nn.Module):
    """Generative Adversarial Imitation Learning (GAIL) model.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        trajectories (bool): Whether to use trajectories.
        train_config (dict, optional): Training configuration. Default is None.
    """
    def __init__(self, state_dim, action_dim, trajectories, train_config=None):
        super(GAIL, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trajectories = trajectories
        self.train_config = train_config
        
        self.pi = PolicyNetwork(state_dim, action_dim, trajectories)
        self.value = ValueNetwork(state_dim)
        self.d = Discriminator(state_dim, action_dim, trajectories)
    
    def networks(self):
        """Get the networks.

        Returns:
            list: List of networks.
        """
        return [self.pi, self.value, self.d]
    
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
        
        return action.cpu().numpy()
    
    def model_train(self, env, expert, render=False):
        """Train the model.

        Args:
            env: The environment to train in.
            expert: The expert policy.
            render (bool, optional): Whether to render the environment. Default is False.
        """
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config.get("horizon", None)
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]
        
        optimizer_d = optim.Adam(self.d.parameters(), lr=0.1)
        
        exp_rwd = []
        exp_obs = []
        exp_act = []
        
        steps = 0
        
        while steps < num_steps_per_iter:
            obs = []
            rwds = []
            
            t = 0
            done = False
            
            ob = env.reset()
        
            while not done and steps < num_steps_per_iter:
                act = expert.action(ob)

                obs.append(ob)
                exp_obs.append(ob)
                exp_act.append(act)

                if render:
                    env.render()
                ob, rwd, done, _ = env.step(act)

                rwds.append(rwd)

                t += 1
                steps += 1

                if horizon is not None and t >= horizon:
                    done = True
                    break

            if done:
                exp_rwd.append(np.sum(rwds))
        
        exp_rwd_mean = np.mean(exp_rwd)
        print(f"Mean of Expert Reward: {exp_rwd_mean}")
    
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []
            lmbs = []

            steps = 0

            while steps < num_steps_per_iter:
                obs = []
                acts = []
                rwds = []
                costs = []
                disc_costs = []
                gms = []
                lmbs = []

                t = 0
                done = False

                ob = env.reset()

                while not done and steps < num_steps_per_iter:
                    act = self.action(ob)

                    obs.append(ob)
                    acts.append(act)

                    if render:
                        env.render()
                    ob, rwd, done, _ = env.step(act)

                    rwds.append(rwd)
                    gms.append(gae_gamma ** t)
                    lmbs.append(gae_lambda ** t)

                    t += 1
                    steps += 1

                    if horizon is not None and t >= horizon:
                        done = True
                        break

                if done:
                    rwd_iter.append(np.sum(rwds))
                    
            obs = torch.tensor(obs, dtype=torch.float32)
            acts = torch.tensor(acts, dtype=torch.float32)
            rwds = torch.tensor(rwds, dtype=torch.float32)
            gms = torch.tensor(gms, dtype=torch.float32)
            lmbs = torch.tensor(lmbs, dtype=torch.float32)
            
            costs = -torch.log(self.d(obs, acts)).squeeze().detach()
            disc_costs = gms * costs
            
            disc_rets = torch.tensor([sum(disc_costs[i:]) for i in range(t)], dtype=torch.float32)
            rets = disc_rets / gms

            self.value.eval()
            vals = self.value(obs).detach()
            next_vals = torch.cat((self.value(obs)[1:], torch.tensor([[0.]], dtype=torch.float32)))
            deltas = costs.unsqueeze(-1) + gae_gamma * next_vals - vals

            advs = torch.tensor([((gms * lmbs)[:t - j].unsqueeze(-1) * deltas[j:]).sum()
                for j in range(t)], dtype=torch.float32)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            optimizer_d.zero_grad()
            exp_scores = self.d(torch.tensor(exp_obs, dtype=torch.float32), torch.tensor(exp_act, dtype=torch.float32))
            nov_scores = self.d(obs, acts)
            bce_loss = nn.BCEWithLogitsLoss()
            loss_d = bce_loss(exp_scores, nov_scores)
            loss_d.backward()
            optimizer_d.step()

            # Further training steps for the policy network and value network can be added here

            rwd_iter_mean = np.mean(rwd_iter)
            print(f"Iteration {i + 1}, Reward Mean: {rwd_iter_mean}")

        return exp_rwd_mean, rwd_iter_mean

    def L(self, obs, acts, advs, old_distb):
        """Calculate the loss function.

        Args:
            obs (torch.Tensor): Observations.
            acts (torch.Tensor): Actions.
            advs (torch.Tensor): Advantages.
            old_distb (torch.distributions.Distribution): Old distribution.

        Returns:
            torch.Tensor: Loss value.
        """
        distb = self.pi(obs)
        return (advs * torch.exp(distb.log_prob(acts) - old_distb.log_prob(acts))).mean()
    
    def kld(self, old_distb, obs):
        """Calculate the KL divergence.

        Args:
            old_distb (torch.distributions.Distribution): Old distribution.
            obs (torch.Tensor): Observations.

        Returns:
            torch.Tensor: KL divergence value.
        """
        distb = self.pi(obs)

        if self.trajectories:
            old_p = old_distb.probs
            p = distb.probs
            return (old_p * (torch.log(old_p) - torch.log(p))).sum(-1).mean()
        else:
            old_mean = old_distb.mean
            old_cov = old_distb.covariance_matrix.sum(-1)
            mean = distb.mean
            cov = distb.covariance_matrix.sum(-1)
            return (0.5) * (
                (old_cov / cov).sum(-1)
                + (((old_mean - mean) ** 2) / cov).sum(-1)
                - self.action_dim
                + torch.log(cov).sum(-1)
                - torch.log(old_cov).sum(-1)
            ).mean()
