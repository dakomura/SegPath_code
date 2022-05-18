from skimage.color import rgb2hed
import numpy as np


def extract_H(img_he, split=False):
    if split:  # rgb2hed for large image
        he_hed = rgb2hed_split(img_he)[:, :, 0]
    else:
        he_rgb = np.array(img_he)[:, :, :3]
        he_hed = rgb2hed(he_rgb)[:, :, 0]
    region_he = he_hed - np.min(he_hed)
    region_he = region_he * 255 / np.max(region_he)
    region_he = 255 - np.asarray(region_he, dtype='uint8')

    return region_he


def rgb2hed_split(im, split_width=100):
    # segfault when the image is large
    imgwidth, imgheight = im.size
    img = None
    for leftmost in range(0, imgwidth, split_width):
        box = (leftmost, 0, leftmost + split_width, imgheight)
        cropped = im.crop(box)
        cropped_hed = rgb2hed(np.array(cropped)[:, :, :3])
        if leftmost == 0:
            img = cropped_hed
        else:
            img = get_concat_w(img, cropped_hed)
    return img[:imgheight, :imgwidth, :]


def get_concat_w(im1, im2):
    dst = np.empty((im1.shape[0], im1.shape[1] + im2.shape[1], 3), dtype=float)
    dst[:, :im1.shape[1], :] = im1
    dst[:, im1.shape[1]:, :] = im2
    return dst
