import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from db_model_wrapper import ENCODER, DECODER, COND_ENERGY, MARGINAL_ENERGY
import db_model_wrapper


def get_n_observations(dset):
    return {
        'mnist': 50000,
        'fashion': 50000,
        'cifar10': 50000
    }[dset]


def optimize(loss, scope, args, lr, clip_norm=-1, return_grad_norm=False):
    optm = {
        'adam': lambda: tf.train.AdamOptimizer(learning_rate=lr),
        'rmsprop': lambda: tf.train.RMSPropOptimizer(learning_rate=lr)
    }[args.optimizer]()
    if scope is not None:
        tvars = []
        for s in scope:
            tv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=s)
            assert len(tv)>0
            tvars += tv
        grad_and_vars = optm.compute_gradients(loss, var_list=tvars)
    else:
        grad_and_vars = optm.compute_gradients(loss, var_list=None)
    if clip_norm > 0:
        grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var)
                          for grad, var in grad_and_vars]
    opt_op = optm.apply_gradients(grad_and_vars)
    if return_grad_norm:
        gn = [tf.norm(g) for g, _ in grad_and_vars]
        rt = {
            'grad_norm/avg': tf.add_n(gn) / tf.to_float(len(gn)),
            'grad_norm/max': tf.reduce_max(tf.convert_to_tensor(gn))
        }
        return opt_op, rt
    return opt_op


class VAEWarmup(object):

    def __init__(self, model, args, lr=None, clip_norm=-1):
        # to be compatible with previous spec (pixelwise MSE)
        self.rce = tf.reduce_mean(
          model.recon_nll / tf.to_float(tf.reduce_prod(model.x.shape[1:])) * 2)
        self.train_step = optimize(
            self.rce, [model._decoder_scope], args, lr, clip_norm)
        self.print = {
            f'{model.name}_warmup/rce': self.rce,
            '__loss': self.rce
        }

    def step(self, sess, fd, c_iter):
        sess.run(self.train_step, fd)


class ExplicitVAE(object):

    def __init__(self, model, args, var_scope=None, lr=None, clip_norm=-1):
        self.model, self.args = model, args
        self.reconstruction_loss = tf.reduce_mean(model.recon_nll)
        self.p_z = tfd.Normal(tf.zeros_like(model.z), tf.ones_like(model.z))
        kl = model.q_z.kl_divergence(self.p_z)
        self.kl = tf.reduce_mean(tf.reduce_sum(kl, axis=-1))
        self.ELBO = - self.reconstruction_loss - self.kl
        self.train_step, gn = optimize(
            -self.ELBO, var_scope, args, lr, clip_norm, return_grad_norm=True)
        self.print = {
            'loss/recon_nll': self.reconstruction_loss, 
            'loss/recon_pixwise': tf.reduce_mean((model.qxz_mean-model.x)**2),
            'loss/total': -self.ELBO, 
            'loss/KL': self.kl,
            '__loss': -self.ELBO,
        }
        self.print.update(gn)
        if hasattr(model, 'pxz_gamma'):
            self.print['pxz_sd'] = tf.reduce_mean(model.pxz_gamma)
            
    def step(self, sess, fd, c_iter):
        sess.run(self.train_step, fd)

