import numpy as np
import cv2
import glob
import os, sys
from PIL import Image
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("indir",
                    help="input directory",
                    type=str)

args = parser.parse_args()

indir = args.indir

#####################

files = glob.glob(os.path.join(indir, "*","*_IHC_cellpose_nonrigid.png"))

for file in tqdm(files):
    outfile = file.replace("_IHC_cellpose_nonrigid.png","_IHC_cellpose_mask.png")
    assert file != outfile
    img = np.array(Image.open(file))[:,:,0]
    mask_save = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)

    mask_save[img > 0] = 1

    cv2.imwrite(outfile, mask_save)