import sys
import os
import glob

import cv2

import math

import joblib

import numpy as np

import pandas as pd
from skimage.color import rgb2hed, rgb2gray

from PIL import Image


## blur level
def blur(img):
    gray = rgb2gray(img)
    b = variance_of_laplacian(gray)
    return b


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

## correlation between DAPI and Hematoxylin signal
def corr(hefile, ihcfile):
    he = np.asarray(Image.open(hefile))[:, :, :3]
    h = rgb2hed(he)[:, :, 0]
    dapi = np.asarray(Image.open(ihcfile))[:, :, 2]

    dapi_i = []
    h_i = []
    for i in range(30, 180):
        m = np.mean(h[dapi == i])
        dn = np.count_nonzero(dapi == i)
        if not math.isnan(m) and dn > 1000:
            dapi_i.append(i)
            h_i.append(m)

    c = np.corrcoef(dapi_i, h_i)[0, 1]
    b = blur(he)

    return c, b


indir = sys.argv[1]

infiles = glob.glob(os.path.join(indir,"*_HE.png"))

outfile = indir.rstrip("/")+"_QC.csv"

if not os.path.exists(outfile):
    cs=[]
    bs=[]
    infiles_final = []
    for i in range(len(infiles)):
        hefile = infiles[i]
        ihcfile = hefile.replace("_HE", "_IHC_nonrigid")

        c, b = corr(hefile, ihcfile)
        cs.append(c)
        bs.append(b)
        infiles_final.append(hefile)

    df=pd.DataFrame({'imgfile':infiles_final,
                    'corr':cs,
                    'blur':bs})

    df.to_csv(outfile, index=False)

