import tensorflow as tf
import numpy as np

def gaussian_nll_loss(input, target, var, reduction='mean'):
    '''
    Gaussian loss with variance as input (not sigma)
    It returns without the constant term in the loss
    '''
    ax = list(range(1, len(target.shape)))
    
    #check validity of reduction mode
    if reduction !='none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + "is not valid" )

    #Entries of scale must be non negative
    #tf.debugging.assert_non_negative(
    #    var, message="variance has negative numbers", summarize="have you missed to make variance positive with softplus", name=None  
    #)

    per_pixel_loss = 0.5*tf.math.log(var) + ((input - target)**2)/var
    loss = tf.reduce_mean(per_pixel_loss, axis=ax)

    #Apply reduction
    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)
    else:
        return loss
    

def laplace_nll_loss(input, target, scale, eps=1e-06, reduction='mean'):
    '''
    laplace loss nll
    '''
    ax = list(range(1, len(target.shape)))
    
    #check validity of reduction mode
    if reduction !='none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + "is not valid" )

    #Entries of scale must be non negative
    #tf.debugging.assert_non_negative(
    #    scale, message="variance has negative numbers", summarize="have you missed to make variance positive with softplus", name=None  
    #)

    per_pixel_loss = (tf.math.log(2*scale) + tf.abs(input - target)/scale)
    loss = tf.reduce_mean(per_pixel_loss, axis=ax)

    #Apply reduction
    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)
    else:
        return loss
    


def generalized_nll_loss(input, target, alpha, beta, eps=1e-6, reduction='none'):
  """Generalized Gaussian Negative Log-Likelihood Loss.

  Args:
    input: Tensor of shape (batch_size, num_features).
    target: Tensor of shape (batch_size, num_features).
    alpha: Tensor of shape (batch_size, num_features) or (batch_size, 1).
    beta: Tensor of shape (batch_size, num_features) or (batch_size, 1).
    eps: Small value to prevent numerical instability.
    reduction: Reduction method ('none', 'mean', or 'sum').

  Returns:
    Tensor of shape (batch_size,) if reduction is 'none',
    scalar if reduction is 'mean' or 'sum'.
  """

  # Ensure input and target have the same shape
  input = tf.reshape(input, [-1, tf.shape(input)[-1]])
  target = tf.reshape(target, [-1, tf.shape(target)[-1]])
  if tf.shape(input) != tf.shape(target):
    raise ValueError("input and target must have the same size")

  # Ensure alpha and beta have the correct shape
  alpha = tf.reshape(alpha, [-1, tf.shape(input)[-1]])
  beta = tf.reshape(beta, [-1, tf.shape(input)[-1]])
  if tf.shape(alpha)[-1] != tf.shape(input)[-1] and tf.shape(alpha)[-1] != 1:
    raise ValueError("alpha is of incorrect size")
  if tf.shape(beta)[-1] != tf.shape(input)[-1] and tf.shape(beta)[-1] != 1:
    raise ValueError("beta is of incorrect size")

  # Check validity of reduction mode
  if reduction not in ('none', 'mean', 'sum'):
    raise ValueError(f"{reduction} is not valid")

  # Ensure alpha and beta are non-negative
  if tf.reduce_any(alpha < 0):
    raise ValueError("alpha has negative entry/entries")
  if tf.reduce_any(beta < 0):
    raise ValueError("beta has negative entry/entries")

  # Clamp for stability
  alpha = tf.maximum(alpha, eps)
  beta = tf.maximum(beta, eps)

  # Calculate loss (without constant)
  loss = tf.pow(tf.abs(input - target) / alpha, beta) - tf.math.log(beta) + tf.math.log(2 * alpha) + tf.math.lgamma(1 / beta)

  # Apply reduction
  if reduction == 'mean':
    return tf.reduce_mean(loss)
  elif reduction == 'sum':
    return tf.reduce_sum(loss)
  else:
    return loss
'''
def laplace_nll_loss(input, target, scale, eps=1e-06, reduction='mean'):
    
    # Inputs and target should be of same shape
    input = tf.reshape(input, [tf.shape(input).numpy()[0], -1])
    target = tf.reshape(target, [tf.shape(target).numpy()[0], -1])
    if input.shape != target.shape:
        raise ValueError("input and target must have same size")

    # Second dim of scale must match the input size
    scale = tf.reshape(scale, [tf.shape(scale).numpy()[0], -1])
    if scale.shape != target.shape:
        raise ValueError("scale must have same size as input")

    #check validity of reduction mode
    if reduction !='none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + "is not valid" )

    #Entries of scale must be non negative
    tf.debugging.assert_non_negative(
        scale, message="scale has negative numbers", summarize="have you missed to make scale positive", name=None  
    )

    # Clamp for stability
    scale = tf.identity(scale)
    scale = tf.stop_gradient(scale)
    scale = tf.clip_by_value(scale, clip_value_min=eps, clip_value_max=100)

    #Calculate loss
    loss = (tf.math.log(2*scale) + tf.abs(input - target)/scale)
    loss = tf.reshape(loss, [tf.shape(input).numpy()[0], -1])
    loss = tf.reduce_sum(loss, axis=1)

    #Apply reduction
    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)
    else:
        return loss

'''
