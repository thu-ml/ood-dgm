import numpy as np
from PIL import Image
import torch
import cv2
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
from tqdm import trange
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', '-ns', type=int, default=1000)
parser.add_argument('--dump_dir', '-dir', type=str, default='/tmp/last.npz')
# https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
parser.add_argument('--labels', '-l', type=str, default='')
parser.add_argument('--batch_size', '-bsz', type=int, default=64)
parser.add_argument('--crop_ratio', '-crop', type=float, default=0.25)
parser.add_argument('--labels_alt', '-l2', type=str, default='')
parser.add_argument('--mixture_prop', '-mp', type=float, default=0.5)
parser.add_argument('--truncation', '-trunc', type=float, default=0.8)
parser.add_argument('--model_dir', type=str,
                    default=os.path.expanduser('~/biggan-128/pt'))


def tile_images(images):
    batch_size = images.shape[0]
    SL = images.shape[1]
    B = int(batch_size ** 0.5)
    assert(B*B == batch_size)
    return images.reshape((B, B, SL, SL, 3)).transpose((0,2,1,3,4)).reshape((B*SL, B*SL, 3)) 


def gen_image(noise_vector, class_vector, crop, truncation):
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)
    noise_vector = noise_vector.to('cuda')
    class_vector = class_vector.to('cuda')
    # Generate a batch of images
    with torch.no_grad():
        out = model(noise_vector, class_vector, truncation).to('cpu').numpy()
    out = (out + 1) / 2
    assert out.min() >= 0 and out.max() <= 1 and out.shape[1] == 3
    out = out.transpose((0,2,3,1))
    out = np.cast[np.uint8](np.clip(out*255, 0, 255))
    crop_base = int(128 * crop)
    def transform_image(im):
        return cv2.resize(
            im[crop_base:128-crop_base, crop_base:128-crop_base],
            (32,32), interpolation=cv2.INTER_AREA)
    out = np.array([transform_image(out[i]) for i in range(out.shape[0])])
    return out


def main():
    args = parser.parse_args()
    import logging
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    global model
    model = BigGAN.from_pretrained(args.model_dir).to('cuda')

    label_str = args.labels.strip()
    labels = [l.replace('_', ' ') for l in label_str.split(',') if len(l)>0]
    class_base_vecs = one_hot_from_names(labels)
    label_alt_str = args.labels_alt.strip()
    labels_alt = [l.replace('_', ' ') for l in label_alt_str.split(',') if len(l)>0]
    print(labels, labels_alt)
    if len(labels_alt) > 0:
        assert len(labels_alt) == len(labels)
        c1 = one_hot_from_names(labels_alt)
        class_base_vecs = args.mixture_prop * class_base_vecs + (1-args.mixture_prop) * c1

    outs = []
    labels = []
    for _ in trange(0, args.n_samples // args.batch_size):
        # Prepare a input
        cls = np.random.randint(0, class_base_vecs.shape[0], size=(args.batch_size,))
        class_vector = class_base_vecs[cls]
        noise_vector = truncated_noise_sample(
            truncation=args.truncation, batch_size=args.batch_size)
        outs.append(gen_image(
            noise_vector, class_vector, args.crop_ratio, args.truncation))
        labels.append(cls)

    outs = np.concatenate(outs)
    labels = np.concatenate(labels)
    np.savez(args.dump_dir+'.npz', args=vars(args), samples=outs, labels=labels)

    Image.fromarray(tile_images(outs[:81])).save(args.dump_dir+'.samples-81.png')
    Image.fromarray(tile_images(outs[:16])).save(args.dump_dir+'.samples-16.png')
    

if __name__ == '__main__':
    main()
