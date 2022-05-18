import os
import glob
import pprint
import argparse
import collections
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import joblib
import sys
import pandas as pd
import matplotlib
import seaborn as sns
import tqdm
from os.path import basename as bn
from os.path import dirname as dn


from sklearn.linear_model import Ridge

def get_input_files(path: str, tvt: str) -> str:
    """
    get train/val/test image files

    :type tvt: str
    """
    return glob.glob(path + "/" + tvt + "/*_HE.png")


def get_mask_file(infile, ab):
    ab2 = ab.replace("_cellpose","").replace("SMA","αSMA")
    ihcfile = f"/patch/{ab2}_HR/"+os.path.basename(infile).replace("_HE","_IHC_nonrigid").replace(f"-{ab2}_",f"-{ab2}/")
    ihcfile = ihcfile.replace("_a0","/a0")
    return ihcfile

def get_cellpose(infile, ab):
    ab2 = ab.replace("_cellpose","")
    cellfile = f"/patch/{ab2}_HR/"+os.path.basename(infile).replace("_HE.png","_IHC_nonrigid_mask.pkl").replace(f"-{ab2}_",f"-{ab2}/")
    cellfile = cellfile.replace("_a0","/a0")
    return joblib.load(cellfile)

def get_rbc(infile, ab):
    rbcfile = f"./segmentation_results/{ab}/{os.path.basename(infile)}.RBC.npy"
    return np.load(rbcfile)

def get_nearest(df, wsi, x, y, nn=9):
    df_wsi = df.loc[df['wsi']==wsi].reset_index()
    dist = [np.abs(x-df_wsi['x'][i]) + np.abs(y-df_wsi['y'][i])  for i in range(df_wsi.shape[0])]
    close = np.argpartition(dist, nn)[:nn]
    
    return df_wsi, list(close)


def calc_coef_each(infile, ab):
    ridge = Ridge(alpha=0.0, random_state=0, fit_intercept=False)
    segfile = f"./segmentation_results/{ab}/{os.path.basename(infile)}.npy"
    mfile = get_mask_file(infile, ab)

    ks = 11
    sk = 4

    im2 = cv2.imread(mfile)[:,:,2]
    im3 = np.load(segfile)
    im3 = (im3*255).astype(np.uint8)

    im2 = cv2.blur(im2, (ks,ks))[::sk,::sk]
    im3 = cv2.blur(im3, (ks,ks))[::sk,::sk]
    if 'cellpose' in ab:
        cells = get_cellpose(infile, ab)[::sk, ::sk]
        xd = im2[cells>0]
        yd = im3[cells>0]
    else:
        rbc = get_rbc(infile, ab)[::sk, ::sk]
        xd = im2[rbc==False]
        yd = im3[rbc==False]

    if len(xd) <= 2: return 0
        
    s = (xd//20+1) * (yd//20+1) # sample weight for ridge regression
    ridge.fit(xd[:,np.newaxis], yd, s)
    
    return ridge.coef_[0]
    

def calc_coef(df, ab):
    coefs = joblib.Parallel(n_jobs=20)(joblib.delayed(calc_coef_each)(infile, ab) for infile in df['file'])
    return coefs

    
ab = sys.argv[1]
rnd = "_2nd"

datadir = f"/dataset_clean{rnd}/{ab}_HR"

x_files = []
for tvt in ['train', 'val', 'test']:
    x_files_tmp = sorted(get_input_files(datadir, tvt))
    x_files.extend(x_files_tmp)

wsis = [bn(x).split("_")[0] for x in x_files]
xs = [bn(x).split("_")[-3] for x in x_files]
ys = [bn(x).split("_")[-2] for x in x_files]

df = pd.DataFrame({'file':x_files, 'wsi':wsis, 'x':xs, 'y':ys})

df.x = [int(x) for x in df.x]
df.y = [-int(y) for y in df.y]

pred_px = []
outdir = f"/segmentation_results/{ab}"
for n in tqdm.tqdm(range(len(x_files))):
    resfile = os.path.join(outdir, bn(x_files[n])+rnd+".npy")
    pr_mask = np.load(resfile)
    psum = np.count_nonzero(pr_mask > 0.5)
    pred_px.append(np.log10(psum+1))

df['log_p1'] = pred_px
df['coef'] = calc_coef(df, ab)

joblib.dump(df, f"./results/{ab}_HR_df_coef{rnd}.pkl")