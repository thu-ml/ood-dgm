# Dependencies

`pip install -r requirements.txt`.

You also need to download and install (`pip install -e .`) `https://github.com/meta-inf/experiments`

Download [this file](http://ml.cs.tsinghua.edu.cn/~ziyu/static/ood/classify_image_graph_def.pb) and place it to `~/inception`. Alternatively you can remove the FID logic from `examples/lsvae.py`.

# Running the Code

```
python train_vae.py -dataset cifar10 -dir dir/to/model
python ood_test.py -test_type bp -model_dir dir/to/model
```

I have removed a lot of unused code, so let me know if something breaks.

# Acknowledgement

This repository is forked from [thu-ml/wmvl](http://github.com/thu-ml/wmvl) and contains code adapted from [nicola-decao/s-vae-tf](https://github.com/nicola-decao/s-vae-tf) and [bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).

