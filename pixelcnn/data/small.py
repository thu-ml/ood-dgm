import os
import sys
import tarfile
from six.moves import urllib

import numpy as np
from imageio import imread


def load_data(dname):
    from tensorflow.examples.tutorials.mnist import input_data
    url = {
        'mnist': 'http://yann.lecun.com/exdb/mnist/',
        'fashion': 'http://ml.cs.tsinghua.edu.cn/~ziyu/static/fashion/',
    }[dname]
    dsets = input_data.read_data_sets(
        'data/'+dname, one_hot=False, source_url=url,
        validation_size=0)
    def MP(i):
        assert i.min()>=-1e-9 and i.max()<=1+1e-9
        return np.cast[np.uint8](np.clip(i * 255, 0, 255))
    dsets.train._images = MP(dsets.train.images.reshape((-1, 28, 28, 1)))
    dsets.test._images = MP(dsets.test.images.reshape((-1, 28, 28, 1)))
    return dsets


class DataLoader(object):
    """ convert TF dataloader """

    def __init__(self, tf_dset, subset, batch_size, rng=None, shuffle=False, **kwargs):
        """ 
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data = getattr(tf_dset, subset).images
        assert self.data.dtype == np.uint8
        self.data = np.tile(self.data, [1,1,1,3]) # fake 3 channels
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        self.p += self.batch_size

        return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


def get_dataloaders(dname, batch_size, rng=None):
    tf_dset = load_data(dname)
    dl_train = DataLoader(tf_dset, 'train', batch_size, rng=rng, shuffle=True)
    dl_test = DataLoader(tf_dset, 'test', batch_size, rng=None, shuffle=False)
    return dl_train, dl_test

