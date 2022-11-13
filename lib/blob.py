"""Blob helper functions."""

import numpy as np
# from scipy.misc import imread, imresize
import cv2

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, target_height, target_width):
    """Mean correction is performed inside the detectron2 model
    """

    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return im