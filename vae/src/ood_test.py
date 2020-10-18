import tqdm
import numpy as np
import tensorflow as tf
if float('.'.join(tf.__version__.split('.')[:-1])) > 1.12:
    import tensorflow.compat.v1 as tfv1
else:
    tfv1 = tf
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt

import optimizers, db_model_wrapper
from utils import *
from datasets import load_data
from lsvae import parser, INCEPTION_PATH, load_fid, compute_fid_inception, Graph
import datasets, oodutils

import os, json, sys
import argparse
import experiments.utils


# Test args
t_parser = experiments.utils.parser('ood')
t_parser.add_argument('-max_lag', type=int, default=500)  # will be x3 for all-channel test
t_parser.add_argument('-model_dir', type=str, default='/home/ziyu/s-vae-tf/run/ood_dump')
t_parser.add_argument('-comparewith', type=str, default='test', choices=['train', 'test'])
t_parser.add_argument('-test_type', default='bp', choices=['ljb', 'bp'])
t_parser.add_argument('-more_cifar_od', action='store_true')
t_parser.set_defaults(more_cifar_od=False)
                      
test_args = t_parser.parse_args()
DIR = test_args.model_dir
L = test_args.max_lag

if DIR.endswith('/'):
    DIR = DIR[:-1]
DUMP_DIR = test_args.dir
os.makedirs(DUMP_DIR, exist_ok=True)
def SAVE(name):
    plt.savefig(os.path.join(DUMP_DIR, name+'.png'))
    sys.stderr.write(f'{name}.png saved\n')
print(test_args)

# load VAE args
args_json = json.loads(open(os.path.join(DIR, 'hps.txt')).readline())
args = parser.parse_args('')
del args_json['production']
vars(args).update(args_json)

# load inlier dset, build graph
dset = load_data(args.dataset, args.test)
val_images = dset.test.images

def get_val_batch():
    idcs = np.arange(val_images.shape[0])
    np.random.shuffle(idcs)
    return val_images[idcs[:args.batch_size*2]]

if args.production:
    compute_fid, _fid_lc = load_fid(dset.test.images, args)
G = Graph(args, val_images.shape[1], val_images.shape[3])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
ckpt_dir = tf.train.get_checkpoint_state(DIR).model_checkpoint_path
print(ckpt_dir)
saver = tfv1.train.Saver()
saver.restore(sess, ckpt_dir)
if args.production:
    G.gen_qzx(sess, dset.test.images[:10000])
    _ = G.get_fids('s1', sess, val_images, compute_fid)


def get_recon(inp, bsz=200):
    recon = []
    for j in tqdm.trange(0, inp.shape[0], bsz):
        recon.append(
            sess.run(G.x_recon_s1, {G.x_ph: inp[j:j+bsz], G.is_training_ph: False}))
    recon = np.concatenate(recon, axis=0)
    return inp-recon, np.mean(np.reshape(inp-recon, [recon.shape[0],-1])**2, axis=-1)

def get_proc_fn(inl_train_ims):
    """ inl_train_ims: [Ntrain, ...]; float tensor """
    inl_train_mean = inl_train_ims.mean(axis=0)
    inl_train_centered = (inl_train_ims - inl_train_mean).reshape((inl_train_ims.shape[0], -1))
    inl_train_mean = inl_train_mean.reshape((-1,))
    inl_cov = inl_train_centered.T @ inl_train_centered / inl_train_centered.shape[0]
    inl_cov_chol = np.linalg.cholesky(inl_cov)
    def proc(inp):
        inp = inp.reshape((inp.shape[0], -1))
        inp_resi = np.linalg.solve(inl_cov_chol, (inp-inl_train_mean).T).T
        return inp_resi
    return proc

# compute inlier statistics
inl_train_residual, inl_train_rce = get_recon(dset.train.images)
inl_test_residual, inl_test_rce = get_recon(dset.test.images)
_ = plt.hist(inl_train_rce, bins=150, alpha=0.3, label='inl_train', density=True)
_ = plt.hist(inl_test_rce, bins=150, alpha=0.3, label='inl_test', density=True)
plt.legend()
SAVE('inl-rce')
# mtg - all channels
proc = get_proc_fn(inl_train_residual)
inl_train_mtg = proc(inl_train_residual)
inl_test_mtg = proc(inl_test_residual)
# mtg - single channels
def extract_single_chan(inp):
    inp = inp.reshape((inp.shape[0], -1))
    return inp[:, ::3]
proc_ss = get_proc_fn(extract_single_chan(inl_train_residual))
inl_train_mtg_ss = proc_ss(extract_single_chan(inl_train_residual))
inl_test_mtg_ss = proc_ss(extract_single_chan(inl_test_residual))

# MTG TEST

def test_on(test_ims, chan, L=L, comparewith='test', normalize=True):
    assert test_ims.shape[1] == dset.train.images.shape[1] == 32
    resi, rce = get_recon(test_ims)
    mtg = proc(resi) if chan == 'all' else proc_ss(extract_single_chan(resi))
    inl_base_mtg = inl_train_mtg if chan == 'all' else inl_train_mtg_ss
    if comparewith == 'train':
        inl_ref_mtg = inl_base_mtg
        inl_ref_rce = inl_train_rce
    else:
        inl_ref_mtg = inl_test_mtg if chan == 'all' else inl_test_mtg_ss
        inl_ref_rce = inl_test_rce
    plt.figure(facecolor='w')
    skip_corr = 32 if chan != 'all' else 96  # XXX
    auroc = oodutils.time_series_test(
        inl_base_mtg, inl_ref_mtg, mtg, test_args.test_type, 
        B=None, L=L, SK=skip_corr, normalize_fea=normalize)
    print('WN', auroc)
    # Empirically equiv to DGM LH
    inl_score, od_score = oodutils.proc_scores(inl_ref_rce, rce, 'meddist')        
    print('DGM-RCE-2S', oodutils.get_roc(inl_score, od_score)[2])
    print('DGM-RCE-1S', oodutils.get_roc(-inl_ref_rce, -rce)[2])    
    # Gaussian LH for the residuals
    inl_ref_resi_lh = -(inl_ref_mtg**2).sum(axis=1)
    resi_lh = -(mtg**2).sum(axis=1)
    print('LN-LH-2S', oodutils.get_roc(
        *oodutils.proc_scores(inl_ref_resi_lh, resi_lh, 'meddist'))[2])
    print('LN-LH-1S', oodutils.get_roc(inl_ref_resi_lh, resi_lh)[2])

# CMP TEST

from tensorflow_probability import distributions as tfd
import cv2, scipy
if args.observation == 'normal':
    # Discrete log likelihood
    p_xz = tfd.Normal(tf.to_double(G.model_s1.qxz_mean), tf.to_double(G.model_s1.pxz_gamma))
    x_lo = G.x_ph - 1/255/2; x_lo = tf.where(x_lo>=0, x_lo, tf.to_float(-1000)*tf.ones_like(x_lo))
    x_hi = G.x_ph + 1/255/2; x_hi = tf.where(x_hi<=1, x_hi, tf.to_float(+1000)*tf.ones_like(x_hi))
    recon_loglh = tf.where(
        tf.math.abs(G.x_ph-G.model_s1.qxz_mean) <= 2*G.model_s1.pxz_gamma,
        tf.to_float(tf.math.log(p_xz.cdf(tf.to_double(x_hi)) - p_xz.cdf(tf.to_double(x_lo)))),
        tf.to_float(p_xz.log_prob(tf.to_double(G.x_ph))) + tf.math.log(1/255))
    recon_loglh = tf.reduce_sum(recon_loglh, [1,2,3])
else:
    recon_loglh = -G.model_s1.recon_nll

p_z = tfd.Normal(tf.zeros_like(G.model_s1.z), tf.ones_like(G.model_s1.z))
kl = tf.reduce_sum(G.model_s1.q_z.kl_divergence(p_z), [1])
elbo = recon_loglh - kl
assert elbo.shape.ndims == 1
def get_logp_ub(inp, bsz=500, n_p=10, continuous=False, rce=False):
    elbos = []
    to_run = c_elbo if continuous else elbo
    if rce:
        to_run = to_run + tf.reduce_sum(kl, [1])
    for j in tqdm.trange(n_p):
        ee = []
        for k in range(0, inp.shape[0], bsz):
            ee.append(sess.run(to_run, {G.x_ph:inp[k:k+bsz], G.is_training_ph:False}))
        elbos.append(np.concatenate(ee, axis=0))
    logp = scipy.special.logsumexp(np.array(elbos), axis=0) - np.log(n_p)
    return logp / (inp.shape[1]**2 * inp.shape[3])  # logp (nats) per dim
def compression(logp, orig_ims):
    stats = []
    LW = orig_ims.shape[1]
    for k in tqdm.trange(orig_ims.shape[0]):
        rval, buf = cv2.imencode('.png', orig_ims[k], [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        assert rval
        bpd_png = 8 * buf.shape[0] / (LW*LW*3)
        stats.append(-logp[k].mean() / 3 - bpd_png * np.log(2))
    return stats
# cmp - inl stats
if test_args.comparewith == 'train':
    val_images = dset.train.images
else:
    val_images = dset.test.images
inl_logp_pdim = get_logp_ub(val_images)  # these are actually logprobs
inl_cmp_stats = compression(inl_logp_pdim*3, np.cast[np.uint8](val_images*255))

if args.dataset.startswith('synth'):
    pref, suff = args.dataset.split('_')
    od_dsets = [pref + '_' + {'sep':'mixed', 'mixed':'sep'}[suff]]
else:
    od_dsets = {
        'cifar10': ['cel32', 'svhn', 'cifar100c', 'cifar10'],
        'cel32': ['svhn', 'cifar10', 'cel32'],
        'imagenet': ['svhn', 'cifar10', 'cel32'],
    }[args.dataset]
    if test_args.more_cifar_od and args.dataset == 'cifar10':
        od_dsets += ['imagenet', 'facescrub', 'mnist', 'fashion', 'omniglot', 'trafficsign', 'random', 'const']

for dname in od_dsets:
    od_dset = load_data(dname, True)
    od_test_ims = od_dset.test.images
    if dname == args.dataset and test_args.comparewith != 'train':
        od_test_ims = od_dset.train.images
    print('==============' + dname + '-all ===============')
    test_on(od_test_ims, chan='all', L=L*3, comparewith=test_args.comparewith)
    SAVE(dname + '-all')
    # print('==============' + dname + '-single ===============')
    # test_on(od_test_ims, chan='single', L=L, comparewith=test_args.comparewith)
    # SAVE(dname + '-single')
    print('==============' + dname + '-compr ===============')    
    od_logp_pdim = get_logp_ub(od_test_ims)
    od_stats = compression(od_logp_pdim*3, np.cast[np.uint8](od_test_ims*255))
    print('DGM-LH-1S AUC =', oodutils.get_roc(
        np.array(inl_logp_pdim), np.array(od_logp_pdim))[2])
    print('DGM-LH-2S AUC =', oodutils.get_roc(
        *oodutils.proc_scores(np.array(inl_logp_pdim), np.array(od_logp_pdim), 'meddist'))[2])
    print('Compression AUC =', oodutils.get_roc(
        -np.array(inl_cmp_stats), -np.array(od_stats))[2])

