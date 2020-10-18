import os
import numpy as np
import pickle
from sklearn.metrics import roc_curve, roc_auc_score
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r1', type=str, default='/data/ziyu/ood-cifar-inl-withclip-od-gray.pkl',
                    help='test sequences generated from OOD datasets')
parser.add_argument('-r2', type=str, default='/data/ziyu/ood-cifar-mut0.10.pkl',
                    help='test sequences from the noise-corrupted inlier dataset')
parser.add_argument('-cw', type=str, default='test', choices=['train', 'test'],
                    help='whether to compare with inlier training or test set')

args = parser.parse_args()


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
    plt.title('ROC curve, auc = {:.2f}'.format(auc))

def proc_scores(inl, oul, score_type):
    if score_type == 'meddist':
        inl_mean = np.median(inl)
        return -np.abs(inl-inl_mean), -np.abs(oul-inl_mean)
    elif score_type == 'percentile':
        inl_pct = -np.abs(np.mean(inl[:,None] < inl[None,:], axis=-1) - 0.5)
        oul_pct = -np.abs(np.mean(oul[:,None] < inl[None,:], axis=-1) - 0.5)
        return inl_pct, oul_pct
    
def proc_repr(reprs):
    if 'inlier_train' in reprs:
        reprs['inl_train'] = reprs['inlier_train']
        reprs['inl_test'] = reprs['inlier_test']
        del reprs['inlier_train']
        del reprs['inlier_test']
        

reprs1 = pickle.load(open(args.r1, 'rb'))
reprs2 = pickle.load(open(args.r2, 'rb'))
proc_repr(reprs1)
proc_repr(reprs2)

def get_lr(key):
    df = np.abs(reprs1[key][2] - reprs2[key][2]).max()
    # the inlier sequences should follow the same order in the two files
    assert df<1e-4
    return (reprs1[key][0] - reprs2[key][0]).mean((1,2))

base = 'inl_' + args.cw
for k in reprs1:
    if k.startswith('inl') and k.split('_')[1] == args.cw:
        continue
    print(k, get_roc(get_lr(base), get_lr(k))[2])
