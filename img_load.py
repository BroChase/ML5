import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from skimage.color import rgb2lab
from skimage.transform import resize


# Load Images from File.
def get_images(path, n_labels, resize_img):
    images = []
    labels = []
    Dataset = namedtuple('Dataset', ['X', 'y'])
    for filename in glob.iglob(path + '/**/color*.png', recursive=True):
        img = plt.imread(filename).astype(np.float32)
        # convert image from rbg to lab color space L*a*b* designed to approximate human vision.
        # lightness.
        img = rgb2lab(img / 255.0)[:, :, 0]
        if resize_img:
            img = resize(img, resize_img, mode='reflect')

        images.append(img.astype(np.float32))
        # Get the image Label
        lhs, rhs = filename.split('_', 1)
        lhs, rhs = rhs.split('_', 1)
        label = np.zeros((n_labels, ), dtype=np.float32)
        label[int(lhs)] = 1.0
        labels.append(label)

    return Dataset(X=to_tensorflow_format(images).astype(np.float32),
                   y=np.matrix(labels).astype(np.float32))


# joins a sequence of arrays along a new axis
# Axis=0; axis in the result array along which the input arrays are stacked.
# cast as float34
def to_tensorflow_format(imgs):
    return np.stack([img[:, :, np.newaxis] for img in imgs], axis=0).astype(np.float32)
