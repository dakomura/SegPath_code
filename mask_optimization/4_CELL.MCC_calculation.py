import os
import tqdm
import numpy as np
import cv2
import joblib
import sys
import random
from sklearn.metrics import matthews_corrcoef as mcc
import copy

from os.path import basename as bn
from os.path import dirname as dn

def get_cellpose(infile, ab):
    ab2 = ab.replace("_cellpose","")
    cellfile = f"/patch/{ab2}_HR/"+os.path.basename(infile).replace("_HE.png","_IHC_nonrigid_mask.pkl").replace(f"-{ab2}_",f"-{ab2}/")
    cellfile = cellfile.replace("_a0","/a0")
    return joblib.load(cellfile)

def max_mat(mask, y_pred, xmin=0.1, xmax=0.8):
    mcc_max = 0
    mccs = []
    mmax = 0
    for xx in np.arange(xmin, xmax, 0.01):
        y_true = mask >= xx
        if np.sum(y_true) > 3 and np.sum(y_pred) > 3 and np.sum(y_true & y_pred) > 1:
            mcc_val = mcc(y_true, y_pred)
        else:
            mcc_val = 0
        if mcc_val > mcc_max:
            mcc_max = mcc_val
            mmax = xx
        mccs.append(mcc_val)
    return mmax, mccs

def get_ovm(cellpose_m, cid, area):
    ov_m = np.count_nonzero((cellpose_m==cid)) / area    
    return ov_m

def get_ovp(cellpose_p, cid, area):
    ov_p = np.count_nonzero((cellpose_p==cid)) / area
    return ov_p > 0.5

ab = sys.argv[1]
ab += "_cellpose"
rnd1 = "_2nd"
rnd2 = "_3rd"
assert ab in ["MNDA_cellpose","CD3_CD20_cellpose","CD45_cellpose","MIST1_cellpose","ERG_cellpose"]

result_file = f"./results/{ab}_HR_df_coef{rnd1}.pkl"
df = joblib.load(result_file)

ov_ms_all = []
ov_ps_all = []


infiles_rand = random.sample(list(df['file']), 5000)
for infile in tqdm.tqdm(infiles_rand):
    cellpose = get_cellpose(infile, ab)
    segfile = f"./segmentation_results/{ab}/{os.path.basename(infile)}{rnd1}.npy"
    seg = np.load(segfile)
    mask = infile.replace("dataset_clean"+rnd1,"dataset_clean"+rnd2).replace("_HE","_IHC_nonrigid_mask2")
    mask = cv2.imread(mask,0)
    
    cellpose_m = copy.deepcopy(cellpose)
    cellpose_m[mask == 0] = 0
    cellpose_p = copy.deepcopy(cellpose)
    cellpose_p[seg <= 0.5] = 0
    
    area_count = joblib.Parallel(n_jobs=-1)(joblib.delayed(np.count_nonzero)(cellpose==cid) for cid in range(1, np.max(cellpose)+1))
    ov_ms = joblib.Parallel(n_jobs=-1)(joblib.delayed(get_ovm)(cellpose_m, cid, area_count[cid-1]) for cid in range(1, np.max(cellpose)+1))
    ov_ps = joblib.Parallel(n_jobs=-1)(joblib.delayed(get_ovp)(cellpose_p, cid, area_count[cid-1]) for cid in range(1, np.max(cellpose)+1))

    ov_ms_all.extend(ov_ms)
    ov_ps_all.extend(ov_ps)


ov_ms_all = np.asarray(ov_ms_all)
ov_ps_all = np.asarray(ov_ps_all)

mmax, mccs = max_mat(ov_ms_all, ov_ps_all)

joblib.dump({'m':ov_ms_all, 'p':ov_ps_all, 'mmax':mmax, 'mccs':mccs}, f"./overlap/{ab}_HR_overlap{rnd1}.pkl")