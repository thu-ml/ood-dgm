import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils import *
from tsvaenet.util import *


ENCODER = 'encoder'
DECODER = 'decoder'
COND_ENERGY = 'cond_energy'
MARGINAL_ENERGY = 'mar_energy'


class DaibAE(object):

  def __init__(self, x, h_dim, z_dim, eps_dim, latent_space,
               observation=None, wae_det_enc=False,
               is_training_ph=None, skip_last_fc=False, cap_dupl=1, name=None):
    assert latent_space == 'euc'
    self.z_dim, self.h_dim, self.name = z_dim, h_dim, name
    self.is_training = is_training_ph
    self.wae_det_enc = wae_det_enc
    self.skip_last_fc = skip_last_fc
    self.cap_dupl = cap_dupl
    self.x = x
    self.z, self.mean_z, self.sd_z = self._encoder(x)
    self._logits = self._decoder(self.z)
    self.qxz_mean = tf.nn.sigmoid(self._logits)

  def _encoder(self, x):
    assert x.shape.ndims == 4  # NHWC, [0,1]
    final_side_length = self.x.get_shape().as_list()[1]
    y = x
    with tf.variable_scope(self.name+'/'+ENCODER, reuse=AUTO_REUSE) as scp:
      self._encoder_scope = scp.name
      if final_side_length > 32:
        y = lrelu(conv2d(y, 32*self.cap_dupl, 4, 4, 2, 2, name='conv0', use_sn=True))
        print('conv0', y.get_shape().as_list())
      y = lrelu(conv2d(y, 64*self.cap_dupl, 4, 4, 2, 2, name='conv1', use_sn=True))
      print('conv1', y.get_shape().as_list())
      y = conv2d(y, 128*self.cap_dupl, 4, 4, 2, 2, name='conv2', use_sn=True)
      y = batch_norm(y, is_training=self.is_training, scope='bn2')
      y = lrelu(y)
      print('bn2', y.get_shape().as_list())
      if not self.skip_last_fc:
        y = tf.reshape(y, [tf.shape(x)[0], np.prod(y.shape.as_list()[1:])])
        print('prefc3', y.get_shape().as_list())
        y = linear(y, 1024, scope="fc3", use_sn=True)
        y = batch_norm(y, is_training=self.is_training, scope='bn3')
        y = lrelu(y)
        gaussian_params = linear(y, 2 * self.z_dim, scope="en4", use_sn=True)
      else:
        y = conv2d(y, 256, 4, 4, 2, 2, name='conv3', use_sn=True)
        y = batch_norm(y, is_training=self.is_training, scope='bn3')
        print('bn3', y.get_shape().as_list())
        y = lrelu(y)
        y = tf.reshape(y, [tf.shape(x)[0], np.prod(y.shape.as_list()[1:])])
        gaussian_params = linear(y, 2 * self.z_dim, scope="en4", use_sn=True)
        # y = conv2d(y, self.z_dim * 2, 1, 1, 1, 1, name='conv4')  # [?,4,4,2*z_dims]
        # print('pregap', y.get_shape().as_list())
        # gaussian_params = tf.reduce_mean(y, axis=[1, 2])
      z = mean_z = gaussian_params[:, :self.z_dim]
      sd_z = None
      if not self.wae_det_enc:
        # Dai used softplus here
        sd_z = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])
        z = z + tf.random_normal(shape=tf.shape(z)) * sd_z
    return z, mean_z, sd_z

  def _decoder(self, z):
    batch_size = tf.shape(z)[0]
    final_side_length = self.x.get_shape().as_list()[1]
    data_depth = self.x.get_shape().as_list()[-1]
    with tf.variable_scope(self.name+'/'+DECODER, reuse=AUTO_REUSE) as scp:
      self._decoder_scope = scp.name
      if not self.skip_last_fc:
        assert final_side_length <= 32, "use skip_last_fc"
        y = linear(z, 1024, 'fc1')
        y = tf.nn.relu(batch_norm(y, is_training=self.is_training, scope='bn1'))
        y = linear(
          y, 128 * (final_side_length // 4) * (final_side_length // 4), scope='fc2')
        y = tf.nn.relu(batch_norm(y, is_training=self.is_training, scope='bn2'))
        y = tf.reshape(
          y, [batch_size, final_side_length // 4, final_side_length // 4, 128])
      else:
        if final_side_length <= 32:
          y = linear(z, 4*4*256, 'fc1')
          y = tf.nn.relu(batch_norm(y, is_training=self.is_training, scope='bn1'))
          y = tf.reshape(y, [batch_size, 4, 4, 256])
        else:
          y = linear(z, 4*4*512, 'fc1')
          y = tf.nn.relu(batch_norm(y, is_training=self.is_training, scope='bn1'))
          y = tf.reshape(y, [batch_size, 4, 4, 512])
          y = deconv2d(
            y, [batch_size, final_side_length//8, final_side_length//8, 256*self.cap_dupl], 
            4, 4, 2, 2, name='conv1h')
        print('decoder-before-conv2', y.get_shape().as_list())
        y = deconv2d(
          y, [batch_size, final_side_length//4, final_side_length//4, 128*self.cap_dupl], 
          4, 4, 2, 2, name='conv2')
        y = tf.nn.relu(batch_norm(y, is_training=self.is_training, scope='bn2'))
      print('decoder-before-conv3', y.get_shape().as_list())
      y = deconv2d(
        y,
        [batch_size, final_side_length // 2, final_side_length // 2, 64*self.cap_dupl],
        4, 4, 2, 2, name='conv3')
      y = tf.nn.relu(batch_norm(y, is_training=self.is_training, scope='bn3'))
      print('decoder-before-conv4', y.get_shape().as_list())
      y = deconv2d(
        y,
        [batch_size, final_side_length, final_side_length, data_depth],
        4, 4, 2, 2, name='conv4')
      xhat = y
    print('recon shape', xhat.shape.as_list())
    # if self.x.shape.ndims == 2:
    #   assert int(xhat.shape[-1]) == 1
    #   xhat = tf.reshape(xhat, [batch_size, 784])
    return xhat


class DaibVAE(DaibAE):

  def __init__(self, x, h_dim, z_dim, observation, is_training_ph,
               sd_parameterization='softplus',
               skip_last_fc=False,
               cap_dupl=1, name='first'):
    assert observation in ['normal', 'dlog', 'dlogs']
    with tf.variable_scope(name):
      # Do not move this into decoder - will break VAEWarmup
      self._pxz_gamma_raw = tf.get_variable(
        'pxz_gamma', [], tf.float32, tf.zeros_initializer())
    if sd_parameterization == 'softplus':
      self.pxz_gamma = tf.nn.softplus(self._pxz_gamma_raw*5) + 1e-6
    else:
      self.pxz_gamma = tf.exp(self._pxz_gamma_raw) + 1e-6
    #
    super(DaibVAE, self).__init__(
      x, h_dim, z_dim, 0, 'euc', observation=observation,
      is_training_ph=is_training_ph,
      skip_last_fc=skip_last_fc, cap_dupl=cap_dupl, name=name)
    # DaibAE created self.{x,z}
    self.q_z = tfd.Normal(self.mean_z, self.sd_z)
    if observation == 'normal':
      self.recon_nll = tf.reduce_sum(
        -tfd.Normal(self.qxz_mean, self.pxz_gamma).log_prob(x),
        axis=[-1,-2,-3])
    else:
      rclh = discrete_logistics_likelihood(
        self.qxz_mean, self.pxz_gamma, x, side_censored=(observation=='dlogs'))
      self.recon_nll = -tf.reduce_sum(rclh, [-1,-2,-3])
    assert self.recon_nll.shape.ndims == 1

  def _encoder(self, x):
    z, mean_z, sd_z = super(DaibVAE, self)._encoder(x)
    return z, mean_z, sd_z


class SmallVAE(object):

  def __init__(self, x, h_dim, z_dim, depth=3, activation=tf.nn.relu, clip_sd_by=None,
               inp_mean=None, inp_sd=None,
               name='second', sd_parameterization='softplus'):
    self.x, self.h_dim, self.z_dim = x, h_dim, z_dim
    self.depth, self.activation = depth, activation
    self.sd_parameterization = sd_parameterization
    self.inp_mean, self.inp_sd = inp_mean, inp_sd
    self.name = name
    #
    self.z_mean, self.z_sd = self._encoder(self.x)
    self.q_z = tfd.Normal(self.z_mean, self.z_sd)
    self.z = self.q_z.sample()
    self.qxz_mean = self._logits = self._decoder(self.z)
    #
    with tf.variable_scope(self.name):
      self._pxz_gamma_raw = tf.get_variable(
        'pxz_gamma', [], tf.float32, tf.zeros_initializer())
    if sd_parameterization == 'softplus':
      self.pxz_gamma = tf.nn.softplus(self._pxz_gamma_raw*5) + 1e-4
    else:
      self.pxz_gamma = tf.exp(self._pxz_gamma_raw)
    if clip_sd_by is not None:
      if sd_parameterization != 'exp':
        print('WARNING: pxz_gamma might not be updated')
      self.pxz_gamma = tf.maximum(self.pxz_gamma, clip_sd_by)
    self.recon_nll = tf.reduce_sum(
      -tfd.Normal(self._logits, self.pxz_gamma).log_prob(x), axis=-1)

  def _encoder(self, x):
    with tf.variable_scope(self.name+'/'+ENCODER, reuse=AUTO_REUSE):
      h = x
      for j in range(self.depth):
        h = tf.layers.dense(h, units=self.h_dim, activation=self.activation,
                            kernel_initializer=tf.initializers.he_normal())
      z_mean = tf.layers.dense(h, units=self.z_dim, activation=None) + \
        tf.layers.dense(x, units=self.z_dim, activation=None,
                        kernel_initializer=tf.initializers.orthogonal())
      z_sd = tf.layers.dense(
        tf.concat([h,x], axis=-1), units=self.z_dim, activation=tf.nn.softplus)
      # if self.sd_parameterization == 'softplus':
      # else:
      #   z_sd = tf.layers.dense(h, units=self.z_dim, activation=tf.exp)
    return z_mean, z_sd

  def _decoder(self, z):
    with tf.variable_scope(self.name+'/'+DECODER, reuse=AUTO_REUSE) as scp:
      self._decoder_scope = scp.name
      h = z
      for _ in range(self.depth):
        h = tf.layers.dense(
          h, units=self.h_dim, activation=self.activation,
          kernel_initializer=tf.initializers.he_normal())
      logits = tf.layers.dense(h, units=self.x.shape[-1], activation=None)
      logits += tf.layers.dense(z, units=self.x.shape[-1], activation=None,
                                kernel_initializer=tf.initializers.orthogonal())
    return logits * self.inp_sd + self.inp_mean
