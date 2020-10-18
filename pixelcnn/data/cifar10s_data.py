"""
The synthetic dataset generated from CIFAR-10, used in Section 4.
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def maybe_download_and_extract(
    data_dir,
    url='http://ml.cs.tsinghua.edu.cn/~ziyu/static/ood/cifar10-synth-vae.npz'):
    filename = url.split('/')[-1]
    if not os.path.exists(os.path.join(data_dir, filename)):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def load(data_dir, subset='train'):
    def gen_label(inp):  # placeholders
        return np.random.randint(0, 10, size=(inp.shape[0],))

    if subset == 'od_test':
        url = 'http://ml.cs.tsinghua.edu.cn/~ziyu/static/ood/cifar10-knockoff-vae.npz'
        maybe_download_and_extract(data_dir, url)
        dat = np.load(os.path.join(data_dir, 'cifar10-knockoff-vae.npz'))['knockoff_samples']
        assert dat.dtype == np.uint8 and len(dat.shape) == 4
        return dat, gen_label(dat)

    maybe_download_and_extract(data_dir)
    dat = np.load(os.path.join(data_dir, 'cifar10-synth-vae.npz'))['samples']
    assert dat.dtype == np.uint8 and len(dat.shape) == 4
    split = dat.shape[0] * 4 // 5
    if subset=='train':
        ret = dat[:split]
    elif subset=='test':
        ret = dat[split:]
    else:
        raise NotImplementedError('subset should be either train or test')
    return ret, gen_label(ret)


class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False,
                 exclude_classes=[]):
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
        self.data, self.labels = load(os.path.join(data_dir,'cifar-10-synth'), subset=subset)
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


