import numpy as np
from matplotlib import pyplot as plt
import tqdm
from sklearn.metrics import roc_curve, roc_auc_score


def normalize_feature(inl_train, ood, batch_dims=1):
    inl_train = inl_train.reshape([np.prod(inl_train.shape[:batch_dims]), -1])
    ood = ood.reshape([np.prod(ood.shape[:batch_dims]), -1])
    _mean = inl_train.mean(axis=0)[None]
    _sd = inl_train.std(axis=0)[None]
    return (ood - _mean) / _sd

def autocorr5(x,lags,normalize=True):
    '''numpy.correlate, non partial'''
    mean=x.mean()
    var=np.var(x)
    xp=x-mean
    n_samples = len(x) - np.arange(len(x))
    if normalize:
        corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/n_samples
    else:
        corr=np.correlate(xp,xp,'full')[len(x)-1:]/n_samples
    return corr[:(lags)]

def get_autocorr(fea, B=500, L=200, test_typ='bp', skip_corr=1, normalize=True):
    idcs = np.arange(fea.shape[0])
    if B is not None:
        np.random.shuffle(idcs)
    else:
        B = idcs.shape[0]
    corrs = []
    N = fea[0].shape[0]
    for j in tqdm.trange(B):
        ac_j = autocorr5(fea[idcs[j]], L, normalize=normalize)
        corrs.append(ac_j)
    corrs = np.array(corrs)[:,::skip_corr]
    if test_typ == 'ljb':
        ruler = (N - np.arange(1, L+1))[None, ::skip_corr].astype('f')
        stats = N * (N+2) * (corrs[:,1:]**2 / ruler[:,1:]).mean(axis=-1)  # normalized; *L would follow ChiSq[L]
    else:
        stats = N * (corrs[:,1:]**2).astype('f').mean(axis=-1)
    return stats, corrs

def plot_autocorrelations(corrs):
    plt.plot(np.arange(corrs.shape[1]), np.mean(corrs, axis=0))
    plt.fill_between(np.arange(corrs.shape[1]),
                     np.mean(corrs, axis=0)-np.std(corrs, axis=0), np.mean(corrs, axis=0)+np.std(corrs, axis=0), alpha=0.1)

def autocorr_plot(fea, B=500, L=200, test_typ='bp', skip_corr=1, normalize=True):
    stats, corrs = get_autocorr(fea, B, L, test_typ, skip_corr, normalize)
    if not normalize:
        stats = np.clip(stats, -1e9, 1e9)
    plot_autocorrelations(corrs)
    return stats

def get_roc(fea_inl, fea_ood):
    assert len(fea_inl.shape) == 1
    y = np.concatenate([np.ones_like(fea_inl), np.zeros_like(fea_ood)], axis=0)
    scores = np.concatenate([(fea_inl), (fea_ood)], axis=0)
    fpr, tpr, thrs = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    return fpr, tpr, auc

def plot_roc(fea_inl, fea_ood):
    fpr, tpr, auc = get_roc(fea_inl, fea_ood)
    plt.plot(fpr, tpr)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve, auc = {:.3f}'.format(auc))
    return auc

def proc_scores(inl, oul, score_type):
    if score_type == 'meddist':
        inl_mean = np.median(inl)
        return -np.abs(inl-inl_mean), -np.abs(oul-inl_mean)
    
def time_series_test(inl_train_fea, inl_test_fea, oul_fea, test_typ, B=500, L=100, SK=1, normalize=True, statsonly=False,
                     score_type='meddist', normalize_fea=True):
    if normalize_fea:
        oul_fea = normalize_feature(inl_train_fea, oul_fea)
        inl_fea = normalize_feature(inl_train_fea, inl_test_fea)
    else:
        inl_fea = inl_test_fea
    if not statsonly:
        plt.figure(figsize=(12,2.5), facecolor='w'); plt.subplot(141)
        plt.title('outlier ACF')
        oul_stats = autocorr_plot(oul_fea, L=L, B=B, test_typ=test_typ, skip_corr=SK, normalize=normalize)
        plt.subplot(142)
        plt.title('inlier ACF')
        inl_stats = autocorr_plot(inl_fea, L=L, B=B, test_typ=test_typ, skip_corr=SK, normalize=normalize)
        plt.subplot(143)
    else:
        oul_stats, _ = get_autocorr(oul_fea, L=L, B=B, test_typ=test_typ, skip_corr=SK, normalize=normalize)
        inl_stats, _ = get_autocorr(inl_fea, L=L, B=B, test_typ=test_typ, skip_corr=SK, normalize=normalize)
    plt.title('Histogram: WN test stat')
    _ = plt.hist(inl_stats, label='inlier', bins=50, alpha=0.5, density=True)
    _ = plt.hist(oul_stats, label='outlier', bins=50, alpha=0.5, density=True)
    plt.legend()
    ret_single_side = get_roc(-inl_stats, -oul_stats)[2]
    return ret_single_side
        

def tile_images(imgs):
    z = int(imgs.shape[0] ** 0.5)
    if z*z < imgs.shape[0]:
        imgs = np.concatenate([imgs, np.zeros_like(imgs[:(z+1)*(z+1) - imgs.shape[0]])], axis=0)
        z = z+1
    imgs = imgs.reshape([z,z]+list(imgs.shape[1:]))
    return np.concatenate(np.concatenate(imgs, axis=1), axis=1)
