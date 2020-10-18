import os
import argparse
import numpy as np
import tensorflow as tf
if float('.'.join(tf.__version__.split('.')[:-1])) > 1.12:
  import tensorflow.compat.v1 as tfv1
else:
  tfv1 = tf
tf.logging.set_verbosity(tfv1.logging.ERROR)
from tensorflow_probability import distributions as tfd
from tqdm import trange
import experiments.utils
from experiments.slave import LogContext

import optimizers, db_model_wrapper
from utils import tile_images, save_image, add_bool_flag, EarlyStopper
from datasets import load_data


parser = experiments.utils.parser('svae')
parser.add_argument('-seed', type=int, default=1234)
# ==== do not change ====
parser.add_argument('--net_depth_s2', '-ds2', type=int, default=3)
parser.add_argument('-h_dims', type=int, default=2048)
parser.add_argument('-deq', action='store_true')
# ====
# observation model: discretized logistics [side censored] / normal
parser.add_argument('-observation', type=str, default='dlog',
                    choices=['dlog', 'dlogs', 'normal'])  
parser.add_argument('-z_dims_s1', type=int, default=64)
parser.add_argument('-z_dims_s2', type=int, default=64)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-n_iter_s0', type=int, default=20000)
parser.add_argument('-n_iter_s1', type=int, default=200000)
parser.add_argument('-n_iter_s2', type=int, default=1000)
parser.add_argument('-lr_halve_every_s1', type=int, default=50000)
parser.add_argument('-lr_halve_every_s2', type=int, default=150000)
parser.add_argument('-clip_grad_s1', type=float, default=-1)
parser.add_argument('-n_particles', type=int, default=1)
parser.add_argument('-sd_param', type=str, default='exp',
                    choices=['softplus', 'exp'])
parser.add_argument('-optimizer', type=str, default='adam',
                    choices=['adam', 'rmsprop'])
parser.add_argument('-lr', type=float, default=1e-4)
# ====
parser.add_argument('-save_every', type=int, default=2000)
# when True, use test set for early stop, and dset.validation becomes undefined
parser.add_argument('-test', action='store_true')
parser.add_argument('--compute_fid_only', '-cf', action='store_true',
                    default=False)
parser.add_argument('-do_fid', action='store_true', default=True)
# Don't load SVHN for training
parser.add_argument('-dataset', type=str, default='cifar10')
#                   choices=['mnist', 'fashion', 'cifar10', 'cifar100', 'cel64', 'cel32', 'imagenet'])
parser.add_argument('-restore_s1_from', type=str, default='')
# ==== more VAE stuff ====
add_bool_flag(parser, 'decoder_warmup')
add_bool_flag(parser, 'decoder_warmup_s2')
add_bool_flag(parser, 's1_early_stop')
add_bool_flag(parser, 'normalize_s2')
parser.add_argument('--capacity_duplex', '-cap_dupl', type=int, default=1)
# positive value for axis-dependent sd
parser.add_argument('-s2_clip_sd', type=float, default=0.)
add_bool_flag(parser, 'skip_last_fc')  # whether to use a DCGAN-like arch
parser.set_defaults(test=True, deq=True, decoder_warmup=True, decoder_warmup_s2=False,
                    s1_early_stop=True, normalize_s2=True, skip_last_fc=True)


INCEPTION_PATH = os.path.expanduser('~/inception')


def load_fid(dtest, args):
  import fid

  def transform_for_fid(im):
    assert len(im.shape) == 4 and im.dtype == np.float32
    if im.shape[-1] == 1:
      assert im.shape[-2] == 28
      im = np.tile(im, [1, 1, 1, 3])
    if not (im.std()<1. and im.min()>-1.):
      print('WARNING: abnormal image range', im.std(), im.min())
    return (im+1)*128

  inception_path = fid.check_or_download_inception(INCEPTION_PATH)
  inception_graph = tf.Graph()
  with inception_graph.as_default():
    fid.create_inception_graph(str(inception_path))

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  inception_sess = tf.Session(config=config, graph=inception_graph)
  
  stats_path = os.path.join(INCEPTION_PATH, f'{args.dataset}-stats.npz')
  if not os.path.exists(stats_path):
    mu0, sig0 = fid.calculate_activation_statistics(
      transform_for_fid(dtest), inception_sess, args.batch_size, verbose=True)
    np.savez(stats_path, mu0=mu0, sig0=sig0)
  else:
    sdict = np.load(stats_path)
    mu0, sig0 = sdict['mu0'], sdict['sig0']

  def compute(images):
    m, s = fid.calculate_activation_statistics(
      transform_for_fid(images), inception_sess, args.batch_size, verbose=True)
    return fid.calculate_frechet_distance(m, s, mu0, sig0)

  return compute, locals()


FIRST_STAGE = 'first_stage'
SECOND_STAGE = 'second_stage'


class Graph(object):

  def __init__(self, args, HW, C):
    self.HW, self.C, self.args = HW, C, args
    x_ph = tf.placeholder(tf.float32, shape=(None, HW, HW, C), name='x')
    lr_ph = tf.placeholder(tf.float32, shape=(), name='lr')
    is_training_ph = tf.placeholder(tf.bool, shape=(), name='is_training')
    # FIRST STAGE
    model_s1 = db_model_wrapper.DaibVAE(
      x=x_ph, h_dim=args.h_dims//2, z_dim=args.z_dims_s1,
      observation=args.observation, is_training_ph=is_training_ph,
      sd_parameterization=args.sd_param, skip_last_fc=args.skip_last_fc,
      cap_dupl=args.capacity_duplex,
      name=FIRST_STAGE,
    )
    optimizer_s1 = optimizers.ExplicitVAE(
      model_s1, args, [FIRST_STAGE], lr_ph, args.clip_grad_s1)
    assert model_s1.z.shape.ndims == 2
    if args.decoder_warmup:
      optimizer_s0 = optimizers.VAEWarmup(model_s1, args, lr_ph)

    # SECOND STAGE
    # - normalize input
    if args.normalize_s2:
      s2_inp_mean = s2_inp_mean_raw = tf.get_variable(
        's2_inp_mean', [args.z_dims_s1], tf.float32, tf.zeros_initializer(), trainable=False)
      s2_inp_sd   = s2_inp_sd_raw   = tf.get_variable(
        's2_inp_sd', [args.z_dims_s1], tf.float32, tf.zeros_initializer(), trainable=False)
      s2im_ph = tf.placeholder(tf.float32, shape=[args.z_dims_s1], name='s2im')
      s2is_ph = tf.placeholder(tf.float32, shape=[args.z_dims_s1], name='s2is')
      set_s2_stats_op = tf.group([
        tf.assign(s2_inp_mean, s2im_ph), tf.assign(s2_inp_sd, s2is_ph)])
    else:
      s2_inp_mean = tf.zeros([args.z_dims_s1])
      s2_inp_sd   = tf.ones([args.z_dims_s1])
    # - calculate clip_sd
    if args.s2_clip_sd > 0:
      qzx_sd_med = tf.get_variable(
        'qzx_sd_med', [args.z_dims_s1], tf.float32, tf.zeros_initializer(), trainable=False)
      qsm_ph = tf.placeholder(tf.float32, shape=[args.z_dims_s1], name='qsm')
      set_qsm_op = tf.assign(qzx_sd_med, qsm_ph)
      s2_clip = args.s2_clip_sd * qzx_sd_med
    else:
      s2_clip = None
    s2_inp = model_s1.z
    # -
    model_s2 = db_model_wrapper.SmallVAE(
      x=s2_inp, h_dim=args.h_dims, z_dim=args.z_dims_s2,
      depth=args.net_depth_s2, sd_parameterization=args.sd_param,
      clip_sd_by=s2_clip, inp_mean=s2_inp_mean, inp_sd=s2_inp_sd,
      name=SECOND_STAGE)
    if args.decoder_warmup_s2:
      optimizer_s02 = optimizers.VAEWarmup(model_s2, args, lr_ph)
    optimizer_s2 = optimizers.ExplicitVAE(model_s2, args, [SECOND_STAGE], lr_ph)

    # samples - recon
    #   - s1
    x_recon_s1 = tf.nn.sigmoid(model_s1._decoder(model_s1.z))
    #   - s2
    z_recon = model_s2._decoder(model_s2.z)
    x_recon_s2 = tf.nn.sigmoid(model_s1._decoder(z_recon))
    # samples - from prior. actually mean of p(x|z) (for normal/dlog[s])
    #   - s1
    dist_pz = tfd.Normal(tf.zeros_like(model_s1.z), tf.ones_like(model_s1.z))
    pz_sample = dist_pz.sample()
    samples_s1 = tf.nn.sigmoid(model_s1._decoder(pz_sample))
    #   - s2
    dist_pz = tfd.Normal(tf.zeros_like(model_s2.z), tf.ones_like(model_s2.z))
    pz_sample = model_s2._decoder(dist_pz.sample())
    pz_sample += tf.random_normal(shape=tf.shape(pz_sample)) * model_s2.pxz_gamma
    samples_s2 = tf.nn.sigmoid(model_s1._decoder(pz_sample))
    # logging
    # model_s1.sd_z
    x_recon_err_s1 = tf.reduce_mean((x_recon_s1 - model_s1.x)**2)
    x_recon_err = tf.reduce_mean((x_recon_s2 - model_s1.x)**2)
    toprint_s2 = dict(('s2_'+k, v) for k, v in optimizer_s2.print.items())
    vars(self).update(locals())

  def process_batch(self, dat):
    if self.args.observation == 'sigmoid':
      dat = (dat > np.random.random(size=dat.shape)).astype(np.float32)
    if self.args.deq:
      dat = dat + np.random.uniform(size=dat.shape) / 256
    return dat

  def train_step_0(self, sess, raw_batch, lr, ci):
    self.optimizer_s0.step(sess, {
      self.x_ph: self.process_batch(raw_batch),
      self.is_training_ph: True,
      self.lr_ph: lr
    }, ci)

  def train_step_1(self, sess, raw_batch, lr, ci):
    self.optimizer_s1.step(sess, {
      self.x_ph: self.process_batch(raw_batch),
      self.is_training_ph: True,
      self.lr_ph: lr
    }, ci)

  def train_step_02(self, sess, z_batch, lr, ci):
    self.optimizer_s02.step(sess, {
      self.model_s1.z: z_batch,
      self.is_training_ph: False,
      self.lr_ph: lr
    }, ci)

  def train_step_2(self, sess, z_batch, lr, ci):
    self.optimizer_s2.step(sess, {
      self.model_s1.z: z_batch,
      self.is_training_ph: False,
      self.lr_ph: lr
    }, ci)

  def gen_qzx(self, sess, train_images):
    print('Generating q(z|x) from S1')
    N = train_images.shape[0]
    self.s1_mu_z = np.zeros((N, self.args.z_dims_s1))
    self.s1_sig_z = np.zeros((N, self.args.z_dims_s1))
    B = self.args.batch_size
    for j in trange(0, N, B):
      im_raw = train_images[j:j+B]
      m, s = sess.run([self.model_s1.mean_z, self.model_s1.sd_z], {
        self.x_ph: self.process_batch(im_raw),
        self.is_training_ph: False
      })
      self.s1_mu_z[j:j+B] = m
      self.s1_sig_z[j:j+B] = s
    s2im = self.s1_mu_z.mean(axis=0)
    s2is = (self.s1_mu_z**2 + self.s1_sig_z**2).mean(axis=0) ** 0.5 + 1e-6
    self.s2im, self.s2is = s2im, s2is
    print(f'Mean mean={np.abs(self.s1_mu_z).mean()}, sd={s2is.mean()}+-{s2is.std()}')
    if self.args.normalize_s2:
      sess.run(self.set_s2_stats_op, {
        self.s2im_ph: s2im, self.s2is_ph: s2is})
    qzx_sd_med = np.median(self.s1_sig_z, axis=0)
    if self.args.s2_clip_sd > 0:
      sess.run(self.set_qsm_op, {self.qsm_ph: qzx_sd_med})

  def get_fids(self, stg, sess, val_images, compute_fid):
    if val_images.shape[0] > 10000:
      idcs = np.arange(val_images.shape[0])
      rng = np.random.RandomState(1234)
      rng.shuffle(idcs)
      val_images = val_images[idcs[:10000]]

    BS = self.args.batch_size
    samples = []
    for j in range(0, val_images.shape[0], BS*2):
      ss = sess.run(getattr(self, f'samples_{stg}'), {
          self.x_ph: np.zeros((BS*2, self.HW, self.HW, self.C)),
          self.is_training_ph: False
      })
      samples.append(ss)
    recons = []
    for j in range(0, val_images.shape[0], BS*2):
      rc = sess.run(getattr(self, f'x_recon_{stg}'), {
          self.x_ph: val_images[j:j+BS*2],
          self.is_training_ph: False
      })
      recons.append(rc)
    samples = np.concatenate(samples, axis=0)
    fids = {
      'fid/sample': compute_fid(samples),
      'fid/recon': compute_fid(np.concatenate(recons, axis=0))
    }
    print(fids)
    return fids, samples


def compute_lr(init, halve_every, cur_iter):
  return init * (0.5 ** (cur_iter // halve_every))


def compute_fid_inception(dset, sess, G, stg=2, bsz=1000):
  """following Dai who used inception for all datasets including MNIST"""
  import fid_daib
  ss = []
  for _ in range(0, 10000, bsz):
    ss_ = sess.run(getattr(G, 'samples_s'+str(stg)), {
      G.x_ph: np.zeros_like(dset.test.images[:bsz]).astype('f'),
      G.is_training_ph: False
    })
    ss.append(ss_)
  generated = np.concatenate(ss, axis=0)
  test_images = dset.test.images
  # might clear the trained weights
  tf.reset_default_graph()
  score = fid_daib.evaluate_fid_score(generated*255, test_images*255, None)
  return score


def main(args):
  dset = load_data(args.dataset, args.test)
  if not args.test:
    val_images = test_images = dset.validation.images
  else:
    val_images = dset.test.images

  def get_val_batch():
    idcs = np.arange(val_images.shape[0])
    np.random.shuffle(idcs)
    return val_images[idcs[:args.batch_size*2]]

  compute_fid, _fid_lc = load_fid(dset.test.images, args)

  use_restored_s1 = len(args.restore_s1_from) > 0
  if use_restored_s1:
    args_json = json.load(open(os.path.join(args.restore_s1_from, 'hps.txt')))
    assert args_json['dataset'] == args.dataset and \
      args_json['sd_param'] == args.sd_param
    keys_to_restore = ['sd_param', 'z_dims_s1', 'skip_last_fc']
    vars(args).update(dict((k, args_json[k]) for k in keys_to_restore))
    
  tf.set_random_seed(args.seed)
  np.random.seed(args.seed)
  G = Graph(args, val_images.shape[1], val_images.shape[3])
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  if use_restored_s1:
    s1_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=FIRST_STAGE)
    saver = tfv1.train.Saver(var_list=s1_vars)
    ckpt_dir = tf.train.get_checkpoint_state(args.restore_s1_from).model_checkpoint_path
    print('RESTORING S1 FROM', ckpt_dir)
    saver.restore(sess, ckpt_dir)

  saver = tfv1.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=1)
  print(args)
  B = 100

  if args.decoder_warmup and not use_restored_s1:
    print('Warmup stage:')
    es = EarlyStopper(args.n_iter_s0//(B*4))
    with LogContext(args.n_iter_s0//B, logdir=args.dir, tfsummary=True) as ctx:
      for i in ctx:
        for j in range(B): # training
          x_mb, _ = dset.train.next_batch(args.batch_size)
          G.train_step_0(sess, x_mb, args.lr*2, i)

        x_mb = G.process_batch(get_val_batch())
        to_log = sess.run(
          {**G.optimizer_s0.print}, {G.x_ph: x_mb, G.is_training_ph: False})
        ctx.log_scalars(to_log, list(to_log))
        if es.add_check(to_log['__loss']):
          print('Early stopping')
          break

  if not use_restored_s1:
    print('First stage:')
    es = EarlyStopper(args.n_iter_s1//(B*8))
    with LogContext(args.n_iter_s1//B, logdir=args.dir, tfsummary=True) as ctx:
      for i in ctx:
        for j in range(B): # training
          lr = compute_lr(args.lr, args.lr_halve_every_s1, i*B+j)
          x_mb, _ = dset.train.next_batch(args.batch_size)
          G.train_step_1(sess, x_mb, lr, i)
        # plot validation
        x_mb = G.process_batch(get_val_batch())
        to_log = sess.run(
          {**G.optimizer_s1.print}, {G.x_ph: x_mb, G.is_training_ph: False})
        # if i % 80 == 0:
        #   fids, _ = G.get_fids('s1', sess, val_images, compute_fid)
        #   to_log.update(fids)
        ctx.log_scalars(to_log, list(to_log))
        if args.s1_early_stop and es.add_check(to_log['__loss']):
          print('Early stop')
          break

  e_start = args.n_iter_s1//B
  N = dset.train.images.shape[0]
  B = (N + args.batch_size - 1) // args.batch_size
  G.gen_qzx(sess, dset.train.images)

  if args.production:
    if use_restored_s1:
      fids, _ = G.get_fids('s1', sess, val_images, compute_fid)
      print('S1 FID:', fids)
    else:
      saver.save(sess, os.path.join(args.dir, 'model'), global_step=e_start)
      print('Model saved')

  if args.decoder_warmup_s2:
    print('S2 warmup stage:')
    es = EarlyStopper(args.n_iter_s0//(B*4))
    with LogContext(args.n_iter_s0//B, logdir=args.dir, tfsummary=True) as ctx:
      for i in ctx:
        idcs = np.arange(N); np.random.shuffle(idcs)
        z_samples = np.random.normal(G.s1_mu_z, G.s1_sig_z)[idcs]
        for j in range(B): # training
          G.train_step_02(sess, z_samples[j*B:(j+1)*B], args.lr, i)
        to_log = sess.run(
          G.optimizer_s02.print, {G.model_s1.z: z_samples, G.is_training_ph: False})
        ctx.log_scalars(to_log, list(to_log))
        if es.add_check(to_log['__loss']):
          print('Early stopping')
          break

  with LogContext(args.n_iter_s2//B+1, logdir=args.dir, tfsummary=True,
          n_ep_start=e_start) as ctx:
    for i in ctx:
      idcs = np.arange(N); np.random.shuffle(idcs)
      z_samples = np.random.normal(G.s1_mu_z, G.s1_sig_z)[idcs]
      for j in range(B):  # training
        lr = compute_lr(args.lr, args.lr_halve_every_s2, i*B+j)
        z_mb = z_samples[j*B: (j+1)*B]
        G.train_step_2(sess, z_mb, lr, i)
      # plot validation
      x_mb = G.process_batch(get_val_batch())
      to_log = sess.run(
        G.toprint_s2, {G.x_ph: x_mb, G.is_training_ph: False})
      if i % 80 == 0:
        fids, _ = G.get_fids('s2', sess, val_images, compute_fid)
        to_log.update(fids)
      ctx.log_scalars(to_log, list(to_log))
      if (i * 100) % args.save_every == 0 and args.save_every > 0:
        saver.save(sess, os.path.join(args.dir, 'model'), global_step=i+e_start)
        print('Model saved')

  print('Test/validation:')
  test_images = val_images
  _, sample_images = G.get_fids('s2', sess, val_images, compute_fid)
  im_tiled = tile_images(sample_images[:100].reshape((100, G.HW, G.HW, G.C))) 
  save_image(os.path.join(args.dir, 'sampled.png'), im_tiled)
  score = compute_fid_inception(dset, sess, G)
  print('FID SCORE (Bin Dai\'s) =', score)


def do_fid_inception(args):
  args_json = json.load(open(os.path.join(args.dir, 'hps.txt')))
  vars(args).update(args_json)
  dset = load_data(args.dataset, True)
  ckpt_dir = tf.train.get_checkpoint_state(args.dir).model_checkpoint_path
  val_images = dset.test.images
  G = Graph(args, val_images.shape[1], val_images.shape[3])
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())
  saver = tfv1.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=1)
  print('RESTORING WEIGHTS FROM', ckpt_dir)
  saver.restore(sess, ckpt_dir)
  score = compute_fid_inception(dset, sess, G)
  print('FID SCORE = {}'.format(score))

 
if __name__ == '__main__':
  args = parser.parse_args()
  assert args.dataset != 'svhn', "SVHN training set not supported"
  print(os.environ['LD_LIBRARY_PATH'])
  print('================')
  if args.compute_fid_only:
    do_fid_inception(args)
  else:
    experiments.utils.preflight(args)
    main(args)
