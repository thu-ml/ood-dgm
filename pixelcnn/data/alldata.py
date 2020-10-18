import numpy as np
from . import cifar10_data, cifar100_data, imagenet_data, celeba32_data, small, \
    others, cifar10s_data, other_synth_data
import cv2


def get_dataloader_class(dset, mut_rate=None):
    if dset.startswith('synth'):
        DataLoader = other_synth_data.get_dataloader(dset)
    elif dset.startswith('mut'):
        assert dset == 'mut_cifar', NotImplementedError(dset)
        DataLoader = cifar10_data.get_mutated_dataloader(mut_rate)
    else:
        DataLoader = {
            'cifar': cifar10_data.DataLoader,
            'cifar10s': cifar10s_data.DataLoader,
            'cifar100': cifar100_data.DataLoader,
            'imagenet': imagenet_data.DataLoader,
            'celeba32': celeba32_data.DataLoader
        }[dset]
    return DataLoader


def get_dataloaders(dset, data_dir, batch_size, rng, return_labels=False):
    if dset in ['fashion', 'mnist']:
        fake_3channels = True
        train_data, test_data = small.get_dataloaders(dset, batch_size, rng=rng)
    elif dset == 'cifar100c':
        fake_3channels = False
        classes_to_keep = [
            2, 3, 4, 5, 6, 7, 9, 10, 17
        ]
        exclude_classes = [i for i in range(20) if not (i in classes_to_keep)]
        DataLoader = cifar100_data.DataLoader
        train_data = DataLoader(
            data_dir, 'train', batch_size, rng=rng, shuffle=True, return_labels=return_labels,
            exclude_classes=exclude_classes)
        test_data = DataLoader(
            data_dir, 'test', batch_size, shuffle=False, return_labels=return_labels,
            exclude_classes=exclude_classes)
    else:
        fake_3channels = False
        DataLoader = get_dataloader_class(dset)
        train_data = DataLoader(
            data_dir, 'train', batch_size, rng=rng, shuffle=True, return_labels=return_labels)
        test_data = DataLoader(
            data_dir, 'test', batch_size, shuffle=False, return_labels=return_labels)
    return train_data, test_data, fake_3channels


def convert_small_data(ims):
    assert ims.shape[3] == 3 and ims.dtype == np.uint8
    ret = [cv2.resize(im, (32,32), interpolation=cv2.INTER_AREA) for im in ims]
    return np.array(ret)


def get_test_images(dset, data_dir):
    if dset == 'svhn':
        dat_sv = others.load_svhn(test=True)
        ret = dat_sv.test.images * 255
        assert ret.min() > -1e-9 and ret.max() < 255+1e-9
        return np.cast[np.uint8](np.clip(ret, 0, 255)), False
    if dset in ['random', 'const', 'omniglot', 'facescrub', 'trafficsign']:
        ret = np.load('/data/ziyu/ooddata/' + dset + '.npz')['test']
        assert ret.shape[-1] == 3 and ret.dtype == np.uint8
        return ret, False  # Not shuffled
    if dset == 'cifar_gray':
        fake_3ch = False
        ret = cifar10_data.DataLoader(
            data_dir, 'test', 32, shuffle=False, return_labels=False).data
        ret = np.tile(np.mean(ret, axis=-1)[...,None], [1,1,1,3])
        return ret, False
    if dset == 'cifar10s_knockout':
        fake_3ch = False
        test_data = cifar10s_data.DataLoader(
            data_dir, 'od_test', 32, shuffle=False, return_labels=False)
    else: 
        _, test_data, fake_3ch = get_dataloaders(dset, data_dir, 32, None)
    if dset in ['mnist', 'fashion']:
        return convert_small_data(test_data.data), fake_3ch
    else:
        return test_data.data, fake_3ch

