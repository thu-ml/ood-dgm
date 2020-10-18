The code is adapted from [huggingface/pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN).

Please follow the instruction [here](
https://github.com/huggingface/pytorch-pretrained-BigGAN#download-and-conversion-scripts)
to download the pretrained BigGAN model. You need to place the converted model 
to `~/biggan-128`. After this, you can generate the datasets by

```
pip install -r requirements.txt
python gen.py -dir data/bus-vendingmachine/inlier.npz -l bus \
    -l2 vending_machine -trunc 0.7 -crop 0.125 -n 200000
python gen.py -dir data/bus-vendingmachine/ood.npz -l bus,vending_machine \
    -trunc 0.7 -crop 0.125 -n 200000
```

You can generate the other datasets using the parameters provided in Table 6,
Appendix C.2.

