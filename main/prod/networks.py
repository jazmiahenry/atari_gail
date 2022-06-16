#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import tensorflow.keras as keras

from SharpCosSim2D import SharpCosSim2D


class PolicyNetwork(tf.keras.Sequential):
    #inspired by pytorch application of gail: https://github.com/hcnoh/gail-pytorch/
    
    def __init__(self, state_dim, action_dim, trajectories):
        super(PolicyNetwork, self).__init__()

        
        self.model = keras.Sequential(
        [
            layers.InputLayer(input_shape = state_dim),
            SharpCosSim2D(5, 10, 1), #incorporation of sliding window sharpened cosign similarity class
            layers.MaxPool2D((2, 2)),
            SharpCosSim2D(3, 10, 1),
            layers.MaxPool2D((2, 2)),
            SharpCosSim2D(1, 10, 1),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, input_shape = action_dim, activation = None) #to designate a linear transformation
        ])
        
    def forward(self, state):
        if self.trajectories:
            prob = tf.keras.activations.softmax(self.model(state), axis = 1)
            distribution = tf.keras.utils.to_categorical(prob, num_classes = None, dtype='float32')
        else:
            mean = self.model(state)
            
            std = tf.keras.activations.exponential(self.log_std)
            cov_matrix = tf.eye(self.action_dim) * (std ** 2)
            
            distribution = tfp.distributions.MultivariateNormalDiag(mean, cov_matrix)
            
        return distribution


class ValueNetwork(tf.keras.Sequential):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        
        
        self.model = keras.Sequential(
            [
                layers.InputLayer(input_shape = state_dim),
                SharpCosSim2D(5, 10, 1), #incorporation of sliding window sharpened cosign similarity class
                layers.MaxPool2D((2, 2)),
                SharpCosSim2D(3, 10, 1),
                layers.MaxPool2D((2, 2)),
                SharpCosSim2D(1, 10, 1),
                layers.MaxPool2D((2, 2)),
                layers.Flatten(),
                layers.Dense(10, input_shape = 1, activation = None) #to designate a linear transformation
            ])
    
    def forward(self, state):
        return self.model(state)



class Discriminator(tf.keras.Sequential):
    def __init__(self, state_dim, action_dim, trajectories):
        super(Discriminator, self).__init__()
        
        if self.trajectories:
            self.emb_actions = tf.keras.layers.Embedding(action_dim, state_dim)
            self.model_dim = 2 * state_dim
        else:
            self.model_dim = state_dim + action_dim
        
        self.model = keras.Sequential(
            [
                layers.InputLayer(input_shape = self.model_dim),
                SharpCosSim2D(5, 10, 1), #incorporation of sliding window sharpened cosign similarity class
                layers.MaxPool2D((2, 2)),
                SharpCosSim2D(3, 10, 1),
                layers.MaxPool2D((2, 2)),
                SharpCosSim2D(1, 10, 1),
                layers.MaxPool2D((2, 2)),
                layers.Flatten(),
                layers.Dense(10, input_shape = 1, activation = None) #to designate a linear transformation
            ])
           
    def find_logits(state, action):
        return keras.sparse_categorical_crossentropy(state, action, from_logits=True)
                
    def forward(self, state, action):
        return tf.nn.sigmoid(self.find_logits)



class Expert(tf.keras.Model):
    def __init__(
        self,
        state_dim,
        action_dim,
        trajectories,
        train_config=None
    ):
        super(Expert, self).__init__()
        
        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.trajectories)
        
    def get_networks(self):
        return [self.pi]
    
    def action(self, state):
        self.pi.evaluate()
        
        state = keras.cast(state, "int32")
        distribution = self.pi(state)
        
        action = distribution.numpy()
        
        return action

