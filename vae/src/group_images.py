from utils import tile_images, save_image
import os, sys
import numpy as np
from matplotlib import pyplot as plt

dir_ = sys.argv[1]
ims = []
for j in range(100):
  im = plt.imread(os.path.join(dir_, '{}.png'.format(j)))
  ims.append(im)
ims = np.array(ims)
save_image(os.path.join(dir_, 'all.png'), tile_images(ims))

