import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
import copy

import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler())
logger.info('Started')

def get_mask(ihc_th, min_size):
    ihc_th = ihc_th.astype(np.uint8)

    if min_size is None:
        mask = np.zeros(ihc_th.shape)
        mask[ihc_th > 0] = 1

    else:
        mask = np.zeros(ihc_th.shape, dtype=np.uint8)

        contours, hierarchy = cv2.findContours(ihc_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > min_size:
                cv2.drawContours(mask, [cnt], 0, 1, -1)

    return mask

def get_overlay(img, orig_mask):
    _, mask = cv2.threshold(orig_mask, 128, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.merge([mask, mask, mask]).astype(np.uint8)
    mm = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    return mm

parser = argparse.ArgumentParser()

parser.add_argument("input",
                    help="input WSI file(HE)")

parser.add_argument("--msize_opal",
                    help="minimum size of IF positive region",
                    default=150,
                    type=int)

parser.add_argument("--th_opal",
                    help="IF intensity cutoff",
                    default=50,
                    type=int)

args = parser.parse_args()

th_opal = args.th_opal

msize_opal = args.msize_opal

fn_im = args.input
fn_ihc = fn_im.replace("_HE", "_IHC_nonrigid")
ihc = imread(fn_ihc)[:, :, ::-1]

fn_out = fn_im.replace("_HE", "_IHC_nonrigid_mask2")

logger.info(f"target HE: {fn_im}")

he = imread(fn_im)[:,:,::-1]

opal = ihc[:, :, 0] > th_opal

opal[he[:,:,0] < 100] = 0
opal[he[:,:,1] > 130] = 0
opal[he[:,:,0] - he[:,:,2] < 0] = 0
mask_opal = get_mask(opal, msize_opal)

cv2.imwrite(fn_out, mask_opal)

logger.info(f"out mask: {fn_out}")

logger.info("finished")
