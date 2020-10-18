"""
The synthetic CIFAR-10 dataset.
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def load(dname, subset='train'):
    dname = dname.split('-')[1] + '.npz'
    data_dir = os.path.join('/data/ziyu/ooddata', dname)
    ddict = np.load(data_dir, allow_pickle=True)
    if subset == 'train':
        print("Loading custom dataset", data_dir, ddict['args'][None][0])
    images = ddict['samples']
    labels = ddict['labels']
    assert len(images.shape) == 4 and images.shape[1] == 32 and images.shape[3] == 3
    assert images.dtype == np.uint8
    # synthetic dataset, no need to shuffle
    spl = images.shape[0] * 9 // 10
    if subset == 'train':
        return images[:spl], labels[:spl]
    elif subset == 'test':
        return images[spl:], labels[spl:]
    else:
        raise NotImplementedError(subset)

def get_dataloader(dname):
    def init_dl(*args, **kw):
        return DataLoader(dname, *args, **kw)
    return init_dl

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, dname, data_dir, subset, batch_size, rng=None, shuffle=False,
                 return_labels=False, exclude_classes=[]):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels
        self.exclude_classes = exclude_classes

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load CIFAR-10 training data to RAM
        self.data, self.labels = load(dname, subset=subset)
        assert len(self.labels.shape) == 1
        mask = np.logical_not(np.isin(self.labels, self.exclude_classes))
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        if len(self.exclude_classes) > 0:
            print(self.data.shape[0], 'samples left after masking')

        assert self.data.shape[-1] == 3
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

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
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


