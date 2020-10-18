import tensorflow as tf
import numpy as np


# FUCK YOU TENSORFLOW
if float('.'.join(tf.__version__.split('.')[:-1])) > 1.12:
  from tensorflow.compat.v1 import AUTO_REUSE
else:
  from tensorflow import AUTO_REUSE


def discrete_logistics_likelihood(loc, scale, obs, side_censored=True):
    from tensorflow_probability import distributions as tfd
    x_lo = obs - 1/255/2
    x_hi = obs + 1/255/2
    if side_censored:
        x_lo = tf.where(x_lo>=0, x_lo, tf.to_float(-1000)*tf.ones_like(x_lo))
        x_hi = tf.where(x_hi<=1, x_hi, tf.to_float(+1000)*tf.ones_like(x_hi))
    dist = tfd.Logistic(tf.to_double(loc), tf.to_double(scale))
    # tf.where doesn't backpropagate correctly when there's NaN in the unused branch. So instead of
    # ret = tf.where(
    #     tf.math.abs(obs - loc) <= 2*scale,
    #     tf.math.log(dist.cdf(tf.to_double(x_hi)) - dist.cdf(tf.to_double(x_lo))),
    #     dist.log_prob(tf.to_double(obs)) + tf.to_double(tf.math.log(1/255)))
    # we need
    p_exact = dist.cdf(tf.to_double(x_hi)) - dist.cdf(tf.to_double(x_lo))
    p_exact = tf.where(
        tf.math.abs(obs - loc) <= 2*scale, p_exact, tf.ones_like(p_exact))
    ret = tf.where(
        tf.math.abs(obs - loc) <= 2*scale, 
        tf.math.log(p_exact),
        dist.log_prob(tf.to_double(obs)) + tf.to_double(tf.math.log(1/255)))
    return tf.to_float(ret)


class RunningAverage(object):
    def __init__(self, beta=0.9):
        self.minima = (1e100, None)
        self.beta = beta
        self.avg  = None
        self.c_iter = 0

    def update(self, val, itr=None):
        if itr is None:
            self.c_iter += 1
            itr = self.c_iter
        if self.avg is None:
            self.avg = val
        else:
            self.avg = self.avg * self.beta + val * (1-self.beta)
        self.minima = min(self.minima, (self.avg, itr))

    def __float__(self):
        return float(self.avg) if self.avg is not None else 1e100


class EarlyStopper(object):

    def __init__(self, tol):
        self._tol = tol
        self._lv = []
        self._min = (np.inf, np.inf)

    def add_check(self, lv):
        self._lv.append(lv)
        cp = len(self._lv)
        self._min = min(self._min, (lv, cp))
        return (cp > self._min[1] + self._tol)


def add_bool_flag(parser, name, dest=None):
  dest = dest or name
  parser.add_argument('-'+name, action='store_true', dest=dest)
  parser.add_argument('-no_'+name, action='store_false', dest=dest)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable(
        'u', [1, w_shape[-1]],
        initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def sn_dense(inp, units, activation, name):
    last_dim = int(inp.shape[-1])
    with tf.name_scope(name):
        w = tf.get_variable(name='kernel', shape=[last_dim, num_units],
                            initializer='glorot_uniform')
        b = tf.get_variable(name='bias', shape=[num_units],
                            initializer=tf.zeros_initializer())
        w1 = spectral_norm(w)
    return inp @ w1 + b


def tile_images(imgs):
    z = int(imgs.shape[0] ** 0.5)
    assert z*z == imgs.shape[0]
    imgs = imgs.reshape([z,z]+list(imgs.shape[1:]))
    return np.concatenate(np.concatenate(imgs, axis=1), axis=1)


def save_image(path, imgs):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    if len(imgs.shape) == 3 and imgs.shape[-1] == 1:
        imgs = imgs[..., 0]
    if len(imgs.shape) == 2:
        plt.imsave(path, imgs, cmap='gray')
    else:
        plt.imsave(path, imgs)

