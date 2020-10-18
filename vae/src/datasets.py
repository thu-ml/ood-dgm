import os
import numpy as np
import tensorflow as tf
from scipy import io as sio
import cv2
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base


def load_synth(dataset, test):
  dname = dataset.split('-')[1] + '.npz'
  data_dir = os.path.join('/data/ziyu/ooddata', dname)
  ddict = np.load(data_dir, allow_pickle=True)
  print("Loading custom dataset", data_dir, ddict['args'][None][0])
  images = ddict['samples']
  assert len(images.shape) == 4 and images.shape[1] == 32 and images.shape[3] == 3
  assert images.dtype == np.uint8
  # synthetic dataset, no need to shuffle
  spl = images.shape[0] * 9 // 10
  train_images, test_images = images[:spl], images[spl:]
  if not test:
    validation_size = 5000
    validation_images = train_images[-validation_size:]
    train_images = train_images[:-validation_size]
  else:
    validation_images = train_images[:5000]
  train = DataSet(train_images)
  validation = DataSet(validation_images)
  test = DataSet(test_images)
  return base.Datasets(train=train, validation=validation, test=test)


def load_cifar100(test, exclude_classes=[]):
  data_dir = os.path.join(os.path.dirname(__file__), 'data')
  import cifar100_data
  train_dl = cifar100_data.DataLoader(
    data_dir, 'train', -1, exclude_classes=exclude_classes)
  test_dl = cifar100_data.DataLoader(
    data_dir, 'test', -1, exclude_classes=exclude_classes)
  assert train_dl.data.dtype == np.uint8
  train_images = train_dl.data
  test_images = test_dl.data
  if not test:
    validation_size = 5000
    np.random.seed(23)
    idcs = np.arange(train_images.shape[0])
    np.random.shuffle(idcs)
    train_images = train_images[idcs]
    validation_images = train_images[-validation_size:]
    train_images = train_images[:-validation_size]
  else:
    validation_images = train_images[:5000]
  train = DataSet(train_images)
  validation = DataSet(validation_images)
  test = DataSet(test_images)
  return base.Datasets(train=train, validation=validation, test=test)


def load_cifar10(test):
  source_url = 'http://ml.cs.tsinghua.edu.cn/~ziyu/static/cifar10/'
  train_dir = 'data/cifar10'
  TRAIN_IMAGES = 'train.npy'
  TEST_IMAGES = 'test.npy'
  lc = base.maybe_download(
      TRAIN_IMAGES, train_dir, source_url + TRAIN_IMAGES)
  train_images = np.load(lc)
  lc = base.maybe_download(
      TEST_IMAGES, train_dir, source_url + TEST_IMAGES)
  test_images = np.load(lc)
  if not test:
    validation_size = 5000
    np.random.seed(23)
    idcs = np.arange(train_images.shape[0])
    np.random.shuffle(idcs)
    train_images = train_images[idcs]
    validation_images = train_images[-validation_size:]
    train_images = train_images[:-validation_size]
  else:
    validation_images = train_images[:5000]
  train = DataSet(train_images)
  validation = DataSet(validation_images)
  test = DataSet(test_images)
  return base.Datasets(train=train, validation=validation, test=test)


def load_svhn(test):
  DFMT = '/data/LargeData/svhn/{}_32x32.mat'
  # train_images = np.transpose(sio.loadmat(DFMT.format('train'))['X'], [3,0,1,2])
  test_images = np.transpose(sio.loadmat(DFMT.format('test'))['X'], [3,0,1,2])
  # NOTE Train and validation are placeholders
  train = DataSet(test_images)
  validation = DataSet(test_images[:5000])
  test = DataSet(test_images)
  return base.Datasets(train=train, validation=validation, test=test)


def load_celeba(test):  # deprecated
  assert test
  ddct = np.load(os.path.expanduser('~/celebA/64.npz'))
  train_images = ddct['train']
  test_images = ddct['test']
  train = DataSet(train_images, scale=False)
  validation = DataSet(train_images[:5000], scale=False)
  test = DataSet(test_images, scale=False)
  return base.Datasets(train=train, validation=validation, test=test)


def load_cel32(test):
  ddct = np.load(os.path.expanduser('~/celebA/celeba-32.npz'))
  train_images = ddct['train']
  test_images = ddct['test']
  assert train_images.dtype == np.uint8
  if test:
    train = DataSet(train_images, scale=True)
    validation = DataSet(train_images[:5000], scale=True)
  else:
    validation_size = 5000
    np.random.seed(23)
    idcs = np.arange(train_images.shape[0])
    np.random.shuffle(idcs)
    train_images = train_images[idcs]
    train      = DataSet(train_images[:-validation_size], scale=True)
    validation = DataSet(train_images[-validation_size:], scale=True)
  test = DataSet(test_images, scale=True)
  return base.Datasets(train=train, validation=validation, test=test)


def load_imagenet32(test):
  ddct = np.load(os.path.expanduser('/data/ziyu/imgnet_32x32.npz'))
  train_images = ddct['trainx']
  test_images = ddct['testx']
  assert train_images.dtype == np.uint8
  if test:
    train = DataSet(train_images, scale=True)
    validation = DataSet(train_images[:5000], scale=True)
  else:
    validation_size = 10000
    np.random.seed(23)
    idcs = np.arange(train_images.shape[0])
    np.random.shuffle(idcs)
    train_images = train_images[idcs]
    train      = DataSet(train_images[:-validation_size], scale=True)
    validation = DataSet(train_images[-validation_size:], scale=True)
  test = DataSet(test_images, scale=True)
  return base.Datasets(train=train, validation=validation, test=test)


def gen_npz_loader(file_name):
  def load(test):
    assert test, NotImplementedError(file_name)
    ddct = np.load(os.path.expanduser('/data/ziyu/ooddata/'+file_name+'.npz'))
    test_images = ddct['test']
    assert test_images.dtype == np.uint8
    # placeholders; won't use
    train = DataSet(test_images[:1000], scale=True)
    validation = DataSet(test_images[:1000], scale=True)
    test = DataSet(test_images, scale=True)
    return base.Datasets(train=train, validation=validation, test=test)
  return load


def convert_mnist_data(inp_images):
  inp_images = inp_images.reshape((-1, 28, 28, 1))
  out = []
  for _ in range(inp_images.shape[0]):
    out.append(cv2.resize(np.tile(inp_images[_], (1,1,3)), (32,32)))
  return np.array(out)


def load_data(dname, test):
  dfn = {
    'cifar10': load_cifar10,
    'svhn': load_svhn,
    'cel64': load_celeba,
    'cel32': load_cel32,
    'imagenet': load_imagenet32,
  }
  if dname in dfn:
    return dfn[dname](test)

  if dname in ['facescrub', 'random', 'const', 'trafficsign', 'omniglot']:
    return gen_npz_loader(dname)(test)

  if dname.startswith('synth'):
    return load_synth(dname, test)

  if dname == 'cifar100c':
    classes_to_keep = [
        2, 3, 4, 5, 6, 7, 9, 17, 10
    ]
    exclude_classes = [i for i in range(20) if not (i in classes_to_keep)]
    return load_cifar100(test, exclude_classes)

  from tensorflow.examples.tutorials.mnist import input_data
  url = {
    'mnist': 'http://yann.lecun.com/exdb/mnist/',
    'fashion': 'http://ml.cs.tsinghua.edu.cn/~ziyu/static/fashion/',
  }[dname]
  dsets = input_data.read_data_sets(
    'data/'+dname, one_hot=False, source_url=url,
    validation_size=(0 if test else 5000))
  # dsets.train._images = dsets.train.images.reshape((-1, 28, 28, 1))
  # dsets.validation._images = dsets.validation.images.reshape((-1, 28, 28, 1))
  # dsets.test._images = dsets.test.images.reshape((-1, 28, 28, 1))
  dsets.train._images = convert_mnist_data(dsets.train.images)
  dsets.validation._images = convert_mnist_data(dsets.validation.images)
  dsets.test._images = convert_mnist_data(dsets.test.images)
  return dsets


class DataSet(object):
  """Container class for a dataset. From tf.examples.tutorials.mnist
  """

  def __init__(self,
               images,
               one_hot=False,
               dtype=dtypes.float32,
               scale=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if dtype == dtypes.float32 and scale:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      np.random.shuffle(self._images)
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        np.random.shuffle(self._images)
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      return np.concatenate(
          (images_rest_part, images_new_part), axis=0), None
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], None

