import os
import numpy as np
import cv2
import joblib
import sys
from PIL import Image
from skimage.color import rgb2hed

def get_cellpose(infile, ab):
    ab2 = ab.replace("_cellpose","")
    cellfile = f"/patch/{ab2}_HR/"+os.path.basename(infile).replace("_HE.png","_IHC_nonrigid_mask.pkl").replace(f"-{ab2}_",f"-{ab2}/")
    cellfile = cellfile.replace("_a0","/a0")
    return joblib.load(cellfile)


def write_ovm(infile, ab, th, rnd1, rnd2):
    cellpose = get_cellpose(infile, ab)
    
    mask_final = np.zeros(cellpose.shape, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    for cid in range(1, np.max(cellpose)+1):
        if cid == 1:
            #RBC mask out
            rbc = f"/segmentation_results/{ab}/" + os.path.basename(infile) + ".RBC.npy"
            rbc = np.load(rbc)

            mask = infile.replace("dataset_clean"+rnd1,"dataset_clean"+rnd2).replace("_HE","_IHC_nonrigid_mask2")
            mask = cv2.imread(mask,0)
            mask[rbc == True] = 0
            
            #Hematoxylin
            ihc_hed = rgb2hed(cv2.imread(infile)[:,:,::-1])[:,:,0]
        area = np.count_nonzero(cellpose==cid)
        ov_m = np.count_nonzero((cellpose==cid) & (mask>0)) / area
        h = np.count_nonzero((cellpose==cid) & (ihc_hed>0.05))
        
        if ov_m >= th and h > 10:
            mask2 = np.zeros(mask.shape, dtype=np.uint8)
            mask2[cellpose == cid] = 1
            mask2 = cv2.erode(mask2, kernel, iterations=1)
            mask_final[mask2 > 0] = 1
        
    outmaskfile = infile.replace("dataset_clean"+rnd1,"dataset_clean"+rnd2).replace("_HE","_IHC_cellpose_mask")
    Image.fromarray(mask_final).save(outmaskfile)
    return 0

ab = sys.argv[1]
ab += "_cellpose"
rnd1 = "_2nd"
rnd2 = "_3rd"
assert ab in ["MNDA_cellpose","CD3_CD20_cellpose","CD45_cellpose","MIST1_cellpose","ERG_cellpose"]
    
result_file = f"./results/{ab}_HR_df_coef{rnd1}.pkl"
assert os.path.exists(result_file)
df = joblib.load(result_file)

overlap_file = f"./overlap/{ab}_HR_overlap{rnd1}.pkl"
assert os.path.exists(overlap_file)
results = joblib.load(overlap_file)
mmax = results['mmax']    
_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(write_ovm)(infile, ab, mmax, rnd1, rnd2) for infile in df['file'])