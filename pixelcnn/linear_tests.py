from data import alldata
import os, sys
import numpy as np
import tqdm
import argparse
from sklearn.metrics import roc_curve, roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument('--max_lags', '-L', default=1200, type=int)
parser.add_argument('-sc', action='store_true', dest='single_channel')
parser.add_argument('-fc', action='store_false', dest='single_channel')
parser.add_argument('-synth', action='store_true', dest='synth_only')
parser.set_defaults(single_channel=True, synth_only=False)


TEST_DSETS = {
    'cifar': ['celeba32', 'svhn', 'cifar', 'cifar100c', 'imagenet'],
    'celeba32': ['celeba32', 'svhn', 'cifar'],
    'imagenet': ['svhn', 'cifar', 'celeba32', 'imagenet'],
    'synth-snakeray_mixed': ['synth-snakeray_sep'],
    'synth-elephant_mixed': ['synth-elephant_sep'],
    'synth-bus_mixed': ['synth-bus_sep'],
}


def autocorr5(x,lags):
    '''
    adapted from https://stackoverflow.com/a/51168178/7509266
    Fixed the incorrect denominator: np.correlate(x,x,'full')[d-1+k] returns
        sum_{i=k}^{d-1} x[i]x[i-k]
    so we should divide it by (d-k) instead of d
    '''
    mean=x.mean()
    var=np.var(x)
    xp=x-mean
    ruler = len(x) - np.arange(len(x))
    corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/ruler
    return corr[:(lags)]

def get_autocorr_(fea, B=500, L=200, test_typ='bp', skip_corr=1):
    idcs = np.arange(fea.shape[0])
    if B is not None:
        np.random.shuffle(idcs)
    else:
        B = idcs.shape[0]
    corrs = []
    N = fea[0].shape[0]
    for j in tqdm.trange(B):
        ac_j = autocorr5(fea[idcs[j]], L)
        corrs.append(ac_j)
    corrs = np.array(corrs)[:,::skip_corr]
    if test_typ == 'ljb':
        ruler = (N - np.arange(1, L+1))[None, ::skip_corr].astype('f')
        stats = N * (N+2) * (corrs[:,1:]**2 / ruler[1:]).mean(axis=-1)
    elif test_typ == 'bp':
        stats = N * (corrs[:,1:]**2).astype('f').mean(axis=-1)
    else:
        raise NotImplementedError()
    return stats, corrs

def get_wn_stats(fea, B=500, L=200, test_typ='bp', skip_corr=1):
    """
    :param fea: a ndarray of shape [n_samples, seq_len]. Each row corresponds to a test sequence
                (R or W) of an input
    :param B: sub-sample B sequences from fea. Use None to disable sub-sampling.
    :param L: the maximum lag to include in test
    :param skip_corr: only include lags that are multiples of skip_corr in the test statistics
    :return: a ndarray of shape [B] or [n_samples], corresponding to the WN test statistics
    """
    stats, corrs = get_autocorr_(fea, B, L, test_typ, skip_corr)
    return stats

def get_roc(fea_inl, fea_ood):
    assert len(fea_inl.shape) == 1
    y = np.concatenate([np.ones_like(fea_inl), np.zeros_like(fea_ood)], axis=0)
    scores = np.concatenate([(fea_inl), (fea_ood)], axis=0)
    fpr, tpr, thrs = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    return fpr, tpr, auc

def proc_scores(inl, oul):
    inl_mean = np.median(inl)
    return -np.abs(inl-inl_mean), -np.abs(oul-inl_mean)


def run(TRAIN_DSET, args):

    # =========== load data ===========
    inl_train_loader, inl_test_loader, _ = alldata.get_dataloaders(TRAIN_DSET, './data', 256, None)
    test_datasets = TEST_DSETS[TRAIN_DSET]
    od_test_ims = []
    for dname in test_datasets:
        od_test_ims.append(alldata.get_test_images(dname, './data')[0])

    assert all(i.dtype == np.uint8 for i in od_test_ims)
    od_test_ims = [i.astype('f') / 255 for i in od_test_ims]

    # =========== process inlier data, get whitening transformation =========== 
    inl_train_ims = np.cast[np.float32](inl_train_loader.data) / 255
    inl_test_ims = np.cast[np.float32](inl_test_loader.data) / 255
    inl_train_mean = inl_train_ims.mean(axis=0)
    inl_train_centered = (inl_train_ims - inl_train_mean).reshape((inl_train_ims.shape[0], -1))
    inl_train_mean = inl_train_mean.reshape((-1,))
    inl_cov = inl_train_centered.T @ inl_train_centered / inl_train_centered.shape[0]
    inl_cov_chol = np.linalg.cholesky(inl_cov)
    #inl_cov_chol_d = np.linalg.cholesky(inl_cov.astype('d'))
    def proc(inp, mean, cov_chol):
        inp = inp.reshape((inp.shape[0], -1))
        inp_resi = np.linalg.solve(cov_chol, (inp-mean).T).T
        return inp_resi
    inl_proc = proc(inl_test_ims, inl_train_mean, inl_cov_chol)
    LW = inl_test_ims.shape[2]

    print('============= MVN LH TESTS =============')
    for od_ims, od_name in zip(od_test_ims, test_datasets):
        od_fea = proc(od_ims, inl_train_mean, inl_cov_chol)
        if od_name != TRAIN_DSET:  # compares to inlier test samples by default
            inl_stats = np.mean(inl_proc ** 2, axis=-1)
        else:  # for inlier, compare training set with test set
            inl_proc_train = proc(inl_train_ims, inl_train_mean, inl_cov_chol)
            inl_stats = np.mean(inl_proc_train ** 2, axis=-1)
        od_stats = np.mean(od_fea ** 2, axis=-1)
        _, _, roc = get_roc(*proc_scores(inl_stats, od_stats))
        _, _, s1roc = get_roc(-inl_stats, -od_stats)
        print(od_name, '2s', roc, '1s', s1roc)

    print('============= WN TEST =============')
    ch_sk = 3 if args.single_channel else 1
    SK = LW if args.single_channel else LW*3
    for od_ims, od_name in zip(od_test_ims, test_datasets):
        od_fea = proc(od_ims, inl_train_mean, inl_cov_chol)
        if od_name != TRAIN_DSET:  # compares to inlier test samples by default
            inl_stats = get_wn_stats(inl_proc[:, ::ch_sk], B=None, L=args.max_lags, skip_corr=SK)
        else:  # for inlier, compare training set with test set
            inl_proc_train = proc(inl_train_ims, inl_train_mean, inl_cov_chol)
            inl_stats = get_wn_stats(inl_proc_train[:, ::ch_sk], B=None, L=args.max_lags, skip_corr=SK)
        od_stats = get_wn_stats(od_fea[:, ::ch_sk], B=None, L=args.max_lags, skip_corr=SK)
        auroc_1s = get_roc(-inl_stats, -od_stats)[2]
        print(od_name, auroc_1s)
        sys.stdout.flush()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    for tds in TEST_DSETS:
        if args.synth_only and not tds.startswith('synth'):
            continue
        print('#########################', tds)
        run(tds, args)
