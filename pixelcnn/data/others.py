import os
import numpy as np
import tensorflow as tf
from scipy import io as sio
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base


def load_svhn(test):
  DFMT = '/data/LargeData/svhn/{}_32x32.mat'
  train_images = np.transpose(sio.loadmat(DFMT.format('train'))['X'], [3,0,1,2])
  test_images = np.transpose(sio.loadmat(DFMT.format('test'))['X'], [3,0,1,2])
  # XXX Train and validation are placeholders
  train = DataSet(train_images)
  validation = DataSet(train_images[:5000])
  test = DataSet(test_images)
  return base.Datasets(train=train, validation=validation, test=test)


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

