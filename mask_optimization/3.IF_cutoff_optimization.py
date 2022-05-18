import os
import glob
import pprint
import argparse
import tqdm
import collections
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
import sys
from scipy import stats
from PIL import Image

from os.path import basename as bn
from os.path import dirname as dn

def out_nextround(df, ab, limit=None, outdir="/dataset_clean_2nd"):
    def create_file(infile, otsu, ab):
        tvt = bn(dn(infile))
        filename = bn(infile)
        maskfilename = filename.replace("_HE","_IHC_nonrigid_mask2")
            
        outfile = os.path.join(outdir,ab+"_HR",tvt,filename)        
        outmaskfile = os.path.join(outdir,ab+"_HR",tvt,maskfilename)
    
        mfile = get_mask_file(infile, ab)
        im = cv2.imread(mfile)[:,:,2]
        im = (im>=otsu).astype(np.uint8)
        
        os.symlink(infile, outfile)
        Image.fromarray(im).save(outmaskfile)
        
        return None
    
    for tvt in ['train','val','test']:
        os.makedirs(os.path.join(outdir,ab+"_HR",tvt), exist_ok=True)
    
    if limit == None:
        limit = df.shape[0]
    _ = joblib.Parallel(n_jobs=-1)(joblib.delayed(create_file)(df['file'][k], df['final_final_otsu'][k], ab) for k in range(limit))
    

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
    ab2 = ab
    rbcfile = f"./segmentation_results/{ab}/{os.path.basename(infile)}.RBC.npy"
    return np.load(rbcfile)

def get_nearest2(df_wsi, x, y, nn=9):
    nn = min(nn, df_wsi.shape[0])
    dist = [np.abs(x-df_wsi['x'][i]) + np.abs(y-df_wsi['y'][i])  for i in range(df_wsi.shape[0])]
    close = np.argpartition(dist, nn)[:nn]
    
    return close

def get_nearest2_dist(df_wsi, x, y, nn=9):
    dist = np.array([np.abs(x-df_wsi['x'][i]) + np.abs(y-df_wsi['y'][i])  for i in range(df_wsi.shape[0])])
    index = np.array(range(len(dist)))[df_wsi['success'] == True]
    dist = dist[df_wsi['success'] == True]

    nn = min(nn, len(dist)-1)
    close = np.argpartition(dist, nn)[:nn]
    
    return index[close], dist[np.array(close)]


def is_success(df_wsi, k, ab, coef_th=1):
    if df_wsi['coef'][k] <= coef_th:
        return False
    
    infile = df_wsi['file'][k]
    mfile = get_mask_file(infile, ab)    

    im2 = cv2.imread(mfile)[:,:,2]
    if 'cellpose' in ab:
        cells = get_cellpose(infile, ab)
        xd = im2[cells>0]
    else:
        rbc = get_rbc(infile, ab)
        xd = im2[rbc==False]
    return np.max(xd) >= 10

def get_otsu_mat(df_wsi, nearest, ab, rnd=""):
    xds = np.array([])
    xd_blurs = np.array([])
    yd_blurs = np.array([])
    for n in nearest:
        if not df_wsi['success'][n]:
            continue
            
        infile = df_wsi['file'][n]
        segfile = f"./segmentation_results/{ab}/{os.path.basename(infile)}{rnd}.npy"
        mfile = get_mask_file(infile, ab)

        ks = 11

        im2 = cv2.imread(mfile)[:,:,2]
        im3 = np.load(segfile)
        im3 = (im3*255).astype(np.uint8)

        im2_blur = cv2.blur(im2, (ks,ks))
        im3_blur = cv2.blur(im3, (ks,ks))
        if 'cellpose' in ab:
            cells = get_cellpose(infile, ab)
            xd = im2[cells>0]
            xd_blur = im2_blur[cells>0]
            yd_blur = im3_blur[cells>0]
        else:
            rbc = get_rbc(infile, ab)
            xd = im2[rbc==False]
            xd_blur = im2_blur[rbc==False]
            yd_blur = im3_blur[rbc==False]


        xds = np.concatenate([xds, xd])
        xd_blurs = np.concatenate([xd_blurs, xd_blur])
        yd_blurs = np.concatenate([yd_blurs, yd_blur])
    
    otsu, _ = cv2.threshold(xd_blurs.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    mat = 0
    if ab == 'AE13':
        otsu = int(otsu*0.8)
    return min(50, max(otsu, 10)), mat


def get_success_sub(df_wsi, k, rnd=""):
    if df_wsi['success'][k]:
        nearest = get_nearest2(df_wsi, df_wsi['x'][k], df_wsi['y'][k], 9)
        otsu, mat = get_otsu_mat(df_wsi, nearest, ab, rnd)
        return otsu
    else:
        return None

def get_success(df_wsi, ab, rnd=""):
    df_wsi['success'] = joblib.Parallel(n_jobs=-1)(joblib.delayed(is_success)(df_wsi, k, ab) for k in range(df_wsi.shape[0]))
    cutoff_o = joblib.Parallel(n_jobs=-1)(joblib.delayed(get_success_sub)(df_wsi, k, rnd) for k in range(df_wsi.shape[0]))
           
    return cutoff_o

def otsu_smooth_sub(df_wsi, k, fotsu, norm):
    nearest, dist = get_nearest2_dist(df_wsi, df_wsi['x'][k], df_wsi['y'][k], 16)
    dist = dist / 3000
    nearest_otsu = fotsu[nearest]
    w = norm.pdf(dist)
    if len(nearest_otsu) == 0:
        return 20
    else:
        return np.average(nearest_otsu, weights=w)

def otsu_smooth(df_wsi):
    norm = stats.norm(0, 1)
    fotsu = df_wsi['final_otsu']
    ffotsu = joblib.Parallel(n_jobs=-1)(joblib.delayed(otsu_smooth_sub)(df_wsi, k, fotsu, norm) for k in range(df_wsi.shape[0]))
    
    return ffotsu

ab = sys.argv[1]
rnd = "_2nd"
assert ab in ["MNDA_cellpose","CD3_CD20_cellpose","CD45_cellpose","MIST1_cellpose","ERG_cellpose","AE13","SMA"]

result_file = f"./results/{ab}_HR_df_coef{rnd}.pkl"
df = joblib.load(result_file)
wsis = np.unique(df['wsi'])

if rnd == "_2nd":
    outdir = "/wsi/analysis/CellType/dataset_clean_3rd"
else:
    outdir = "/wsi/analysis/CellType/dataset_clean_2nd"

for wsi in tqdm.tqdm(wsis):
    df_wsi = df.loc[df['wsi']==wsi].reset_index()
    df_wsi['final_otsu'] = get_success(df_wsi, ab, rnd)
    df_wsi['final_final_otsu'] = otsu_smooth(df_wsi)
    joblib.dump(df_wsi,f"./final_otsu/{os.path.basename(wsi)}_otsu.pkl")
