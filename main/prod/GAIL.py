#!/usr/bin/env python
# coding: utf-8




import numpy as np
import tensorflow as tf
from tensorflow import keras

from networks import PolicyNetwork, ValueNetwork, Discriminator
from functions import *



class GAIL(tf.keras.Model):
    def __init__(self, state_fim, action_dim, trajectories, train_config = None):
        super(GAIL, self).__init__()
        
        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.trajectories)
        self.value = ValueNetwork(self.state_dim)
        
        self.d = Discriminator(self.state_dim, self.action_dim, self.trajectories)
    
    def networks(self):
        return [self.pi, self.value]
    
    def action(self, state):
        self.pi.evaluate()
        
        state = keras.cast(state, "int32")
        distribution = self.pi(state)
        
        action = distribution.numpy()
        
        return action
    
    def model_train(self, env, expert):
        
        
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]
        
        
        optimizer = tf.keras.optimizers.Adam(self.d, learning_rate = .1)
        
        #model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        
        exp_rwd = []
        exp_obs = []
        exp_act = []
        
        steps = 0
        
        while steps < nums_steps_per_iter:
            obs = []
            rwds = []
            
            t = 0
            
            done = False
            
            ob = env.reset()
        
            while not done and steps < num_steps_per_iter:
            
                act = expert.act(ob)

                obs.append(ob)
                exp_obs.append(ob)
                exp_act.append(act)

                if render:
                    env.render()
                ob, rwd, done, info = env.step(act)

                rwds.append(rwd)

                t += 1
                steps += 1

                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break
                        
        if done:
                exp_rwd_iter.append(np.sum(rwd))
        
        
        obs = keras.cast(obs, "int32")
        rwds = keras.cast(rwd, "int32")
        
        rwd_mean = np.mean(rwd_iter)
        print ("Mean of Expert Reward: {}".format(rwd_mean))
    
        obs = keras.cast(obs, "int32")
        rwd = keras.cast(rwd, "int32")
    
        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
        while steps < num_steps_per_iter:
            obs = []
            acts = []
            rwd = []
            costs = []
            disc_costs = []
            gms = []
            lmbs = []

            t = 0
            done = False

            ob = env.reset()

            while not done and steps < num_steps_per_iter:
                act = self.act(ob)

                obs.append(ob)
                obs.append(ob)

                ep_acts.append(act)
                acts.append(act)

                if render:
                    env.render()
                ob, rwd, done, info = env.step(act)

                ep_rwds.append(rwd)
                ep_gms.append(gae_gamma ** t)
                ep_lmbs.append(gae_lambda ** t)

                t += 1
                steps += 1

                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break

            if done:
                rwd_iter.append(np.sum(rwd))
                    
                
                
            obs = keras.cast(obs, "int32").numpy()
            acts = keras.cast(acts, "int32").numpy()
            rwds = keras.cast(rwds, "int32")
            gms = keras.cast(ms, "int32")
            lmbs = keras.cast(lmbs, "int32")
                
            costs = (-1) * keras.log(self.d(obs, acts)).tf.squeeze().detach()
                
            disc_costs = gms * costs
                
            disc_rets = keras.cast([sum(disc_costs[i:]) for i in range(t)])
            rets = disc_rets/gms
                
            rets.append(rets)
                
            self.value.evalute()
            vals = self.value(obs).detach()
            next_vals = tfp.distributions.Categorical((self.value(obs)[1:], keras.cast([[0.], "int32"]))).detach()
            deltas = costs.unsqueeze(-1) + gae_gamma * next_vals - vals
                
            advs = keras.cast([((gms * lmbs)[:t - j].unsqueeze(-1) * deltas[j:]).sum()
                for j in range(t)], "int32")
                
            advs.append(advs)

            gms.append(gms)

            rwd_iter_means.append(np.mean(rwd_iter))
            print(
                "Iterations: {},   Reward Mean: {}"
                .format(i + 1, np.mean(rwd_iter))
            )

            obs = keras.cast(obs, "int32").numpy()
            acts = keras.cast(acts, "int32").numpy()
            rets = tfp.distributions.Categorical(rets)
            advs = tfp.distributions.Categorical(advs)
            gms = tfp.distributions.Categorical(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()
                
        
            self.d.model_train()
            
            exp_scores = self.d.find_logits(exp_obs, exp_acts)
            nov_scores = self.d.find_logits(obs, acts)

            #opt_d.gradients(stop_gradient(tf.constant(0.))
            bce = keras.losses.BinaryCrossentropy(from_logits = True)
            loss = bce(exp_scores, nov_scores).numpy()

            loss.GradientTape()
            
            opt_d.minimize()

            self.value.model_train()
            old_params = flat_params(self.value).output
            old_v = self.value(obs).output
            
            
        def constraint():
            return ((old_v - self.value(obs)) **2).mean()
        
            flat_grad = keras.flatten(constraint(), self.value)

        #def kl_divergence_regularizer(inputs):
        
            #kullback_leibler_divergence = keras.losses.kullback_leibler_divergence
            #K = keras.backend
            
            #means = K.mean(inputs, axis=0)
            #return 0.01 * (kullback_leibler_divergence(0.05, means)
                     #+ kullback_leibler_divergence(1 - 0.05, 1 - means))
        
        def kld():
            distb = self.pi(obs)
            
            if self.discrete:
                    old_p = old_distb.probs
                    p = distb.probs

                    return (old_p * (tf.log(old_p) - tf.log(p))).sum(-1).mean()

            else:
                old_mean = old_distb.mean
                old_cov = old_distb.covariance_matrix.sum(-1)
                mean = distb.mean()
                cov = distb.covariance_matrix.sum(-1)

                return (0.5) * (
                        (old_cov / cov).sum(-1)
                        + (((old_mean - mean) ** 2) / cov).sum(-1)
                        - self.action_dim
                        + tf.log(cov).sum(-1)
                        - tf.log(old_cov).sum(-1)).mean()

            grad_kld_old_param = keras.flatten(kld(), self.pi)
            
        def get_hessian():
            loss = tf.reduce_sum(self.pi(obs))
            return tf.hessians(loss, acts)
        
        def L():
            distb = self.pi(obs)

            return (advs * tf.exp(distb.log_prob(acts) - old_distb.log_prob(acts))).mean()
        
        def hv(v):
            hessian = get_hessian()
            
            g = flat_params(L(), self.pi).stop_gradients()

            s = conjugate_gradient(hv, g).stop_gradients()
            hs = hv(s).stop_gradients()

            new_params = rescale_and_linesearch(
                g, s, hs, max_kl, L, kld, old_params, self.pi
            )

            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts)).mean()
            
            grad_disc_causal_entropy = flat_params(disc_causal_entropy, self.pi)
            
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)

        return exp_rwd_mean, rwd_iter_means
