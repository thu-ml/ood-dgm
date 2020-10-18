# Further Analysis of Outlier Detection with Deep Generative Models

## Datasets

Most datasets will be downloaded automatically. For the GAN-generated dataset, 
see the instructions in `gan_dset/`.

For the other datasets, you will need to download them manually and change some 
hard-coded paths. The files below can be downloaded from 
`http://ml.cs.tsinghua.edu.cn/~ziyu/static/ood/${file_name}`:

* CelebA: `celeba-32.npz`
* SVHN: `test_32x32.mat`
* TinyImageNet: `imgnet_32x32.npz`
* For the experiment in Appendix B: 
`{const,random,facescrub,omniglot,trafficsign}.npz`

## Using the Code

To run the experiments in paper, see the instructions in `vae/` and `pixelcnn/`
(for PixelCNN++ and the linear model). 
All code is tested under Python 3.6 and TensorFlow 1.

The proposed test is implemented in `pixelcnn/linear_tests.py` which is mostly 
self-contained.

## Acknowledgement

This repository contains code adapted from multiple sources. See the README in 
each directory.
