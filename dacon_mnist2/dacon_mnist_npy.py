import warnings
warnings.filterwarnings('ignore')
import numpy as np
import PIL
import pandas as pd

from numpy import asarray
from PIL import Image

im=list()
for i in range(25000):
    filepath='C:/data/dacon_mnist2/dirty_mnist/%05d.png'%i
    image=Image.open(filepath)
    image_data=asarray(image)
    im.append(image_data)

# im1=list()
# for i in range(25000, 50000):
#     filepath1='../data/dacon/data/dirty_mnist/%05d.png'%i
#     image1=Image.open(filepath1)
#     image_data1=asarray(image1)
#     im1.append(image_data1)

np.save('C:/data/dacon_mnist2/npy_data/a.npy', arr=im)
# np.save('../data/dacon/data/b.npy', arr=im1)

# im_npy=np.load('C:/data/dacon_mnist2/npy_data/a.npy')
# im_npy1=np.load('../data/dacon/data/b.npy')

# image_set=np.concatenate((im_npy, im_npy1))

# np.save('../data/dacon/data/image_set.npy', arr=image_set)