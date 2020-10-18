"""
Generate test sequences from trained PixelCNN++ model. 
Usage: 
    python gen_repr.py --save_dir path/to/model.cpkt -dp path/to/save.pkl -d inlier_dataset_name -od od_dset_1,od_dset_2
"""
import os
import os.path as osp
import sys
import json
import argparse
import time
import tqdm

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting
from data import alldata
from utl import discretized_mix_logistic_loss_

# -----------------------------------------------------------------------------
root_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default=osp.join(root_dir, 'data'), help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default=osp.join(root_dir, 'save'), help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet|mnist|fashion|celeba')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-k', '--nr_resblocks', type=int, default=3, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
parser.add_argument('-od', '--outlier_data', type=str, default='', help='outlier datasets to dump')
parser.add_argument('-dp', '--dump_path', type=str, default='/tmp/last.pkl')
parser.add_argument('-clp', '--clip', action='store_true', help='clip AR mean estimate to [0,1]')
parser.set_defaults(clip=True)

args = parser.parse_args()
OUTLIER_DATA = list(args.outlier_data.strip().split(','))
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

loss_fun = nn.discretized_mix_logistic_loss

# initialize data loaders for train/test splits
train_data, test_data, fake_3channels = alldata.get_dataloaders(
    args.data_set, args.data_dir, args.batch_size*args.nr_gpu, rng, return_labels=args.class_conditional)
obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'grayscale data not supported'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]

if args.class_conditional:
    num_labels = train_data.get_num_labels()
    y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
    h_init = tf.one_hot(y_init, num_labels)
    y_sample = np.split(np.mod(np.arange(args.batch_size*args.nr_gpu), num_labels), args.nr_gpu)
    h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
    ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(args.nr_gpu)]
    hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
else:
    h_init = None
    h_sample = [None] * args.nr_gpu
    hs = h_sample

# create the model
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': False, 'nr_resblocks': args.nr_resblocks }
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))
ema_params = [ema.average(p) for p in all_params]

test_outs = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # test
        out = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        test_outs.append(out)
output_reprs = tf.concat(test_outs, axis=0)

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    if init:
        feed_dict = {x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if y is not None:
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=config)

ckpt_file = args.save_dir# + '/params_' + args.data_set + '.ckpt'
print('restoring parameters from', ckpt_file)
saver.restore(sess, ckpt_file)


import tqdm


logp_all, ar_all, _ = discretized_mix_logistic_loss_(tf.concat(xs, axis=0), output_reprs, clip=args.clip)


def gen_repr(images):
    SS = 50000
    if images.shape[0] > SS:
        print('Down-sample 50000 images')
        indices = np.arange(images.shape[0])
        np.random.RandomState(23).shuffle(indices)
        images = images[indices[:SS]]
    
    BSZ = args.batch_size * args.nr_gpu
    images = images[:images.shape[0]//BSZ*BSZ]
    lpas, aas = [], []
    for sta in tqdm.trange(0, images.shape[0], BSZ):
        im = images[sta: sta+BSZ]
        feed_dict = make_feed_dict(im)
        lpa, aa = sess.run((logp_all, ar_all), feed_dict)
        lpas.append(lpa)
        aas.append(aa)
        
    return np.concatenate(lpas, axis=0), np.concatenate(aas, axis=0), images


todump = {
    'inl_train': gen_repr(train_data.data),
    'inl_test': gen_repr(test_data.data)
}

for od_name in OUTLIER_DATA:
    od_test, od_f3ch = alldata.get_test_images(od_name, args.data_dir)
    # assert od_f3ch == fake_3channels, od_name
    ooo = gen_repr(od_test)
    todump['od_'+od_name] = ooo

import pickle
with open(args.dump_path, 'wb') as fout:
    pickle.dump(todump, fout, protocol=4)
print('REPR DUMPED TO', args.dump_path)
