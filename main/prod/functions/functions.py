#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[ ]:


def flat_params(parameters):
    params = []
        
    for param in parameters:
        params.append(param.numpy().reshape(-1))
    return np.concatenate(params)

def set_flat_params(model, flat_params):
    idx = 0
    
    if trainable_only:
        variables = model.trainable_variables
    else:
        variables = model.variables

    for param in variables:
    # This will be 1 if param.shape is empty, corresponding to a single value.
        flat_size = int(np.prod(list(param.shape)))
        flat_param_to_assign = flat_params[idx:idx + flat_size]
    # Explicit check here because of: b/112443506
        if len(param.shape):  # pylint: disable=g-explicit-length-test
            flat_param_to_assign = flat_param_to_assign.reshape(*param.shape)
        else:
            flat_param_to_assign = flat_param_to_assign[0]
        param.assign(flat_param_to_assign)
        idx += flat_size
    return model

def conjugate_gradient(b, max_iter=10, residual_tol=1e-10):
    x = tf.zeros_like(b)
    r = b - tf.keras.metrics.Mean(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = tf.keras.metrics.Mean(p)
        alpha = rsold / tf.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if tf.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

def rescale_and_linesearch(
    g, s, hs, max_kl, L, kld, old_params, pi, max_iter= 10,
    success_ratio = 0.1):
    
    
    set_params(self.pi, old_params)
    L_old = L().stop_gradients()

    beta = tf.sqrt((2 * max_kl) / tf.dot(s, hs))

    for i in range(max_iter):
        new_params = old_params + beta * s

        set_params(self.pi, new_params)
        kld_new = kld().stop_gradients()

        L_new = L().stop_gradients()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
            return new_params

        beta *= 0.5

    return old_params


# In[ ]:


if __name__ == '__main__':
    pass

