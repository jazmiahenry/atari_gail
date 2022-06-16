#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


class SharpCosSim2D(tf.keras.layers.Layer):
    """Model built based on sharpened cosine similarity logic."""
    
    def __init__(self, kernel_size = 1, units = 32):
        super(SharpCosSim2D, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
    
    def build(self, input_shape = (28, 28, 1)):
        self.input_shape = input_shape
        
        self.output_y = math.ceil(self.input_shape[1]/ 1)
        self.output_x = math.ceil(self.input_shape[2]/ 1)
        self.flat_size = self.output_x * self.output_y
        self.channels = self.input_shape[3]
        
        self.w = self.add_weight(
            shape=(1, self.channels * tf.square(self.kernel_size), self.units),
            initializer = tf.keras.initializers.GlorotNormal(),
            trainable=True)
        
        self.b = self.add_weight(
            shape=(self.units,), initializer= "zeros", trainable=True)

        self.p = self.add_weight(
            shape=(self.units,), initializer= "ones", trainable=True)

        self.q = self.add_weight(
            shape=(1,), initializer= "zeros", trainable=True)
        
    def l2_normal(self, x, axis=None, epsilon=1e-12):
        square_sum = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        x_inv_norm = tf.sqrt(tf.maximum(square_sum, epsilon))
        return x_inv_norm

    def forward(self, x):
        return tf.nn.sigmoid(x) * tf.nn.softplus(x)
    
    def call(self, inputs, training=None):
        self.stack = lambda x: x
        x = self.stack(inputs)
        x = tf.reshape(x, (-1, self.flat_size, self.channels * tf.square(self.kernel_size)))
        x_norm = (self.l2_normal(x, axis=2))
        w_norm = (self.l2_normal(self.w, axis=1))
        x = tf.matmul(x / x_norm, self.w / w_norm)
        sign = tf.sign(x)
        x = tf.abs(x) + 1e-12
        x = tf.pow(x  + tf.square(self.b), self.forward(self.p))
        x = sign * x
        x = tf.reshape(x, (-1, self.output_y, self.output_x, self.units))
        return x

