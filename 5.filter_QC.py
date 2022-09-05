#!/usr/bin/env python
# coding: utf-8


# remove low quality images

import sys
import os
import glob

import joblib

import numpy as np

import pandas as pd

from tqdm import tqdm
import shutil

from os.path import basename as bn
from os.path import dirname as dn
from os.path import join

def to_mask(f):
    if "cellpose" in f:
        out = f.replace("_HE.png","_IHC_cellpose_mask.png")
    else:
        out = f.replace("_HE.png","_IHC_nonrigid_mask2.png")
        
    assert os.path.exists(out)
    
    return out

def to_out(f):
    return f.replace("/CellType/dataset/","/CellType/dataset_clean/")

def to_csv(f):
    f2 = f.replace("/dataset/","/patch/").replace("_cellpose","").replace("/test","").replace("/train","").replace("/val","")
    b = bn(f)
    subd = b.split("_a")[0]
    d = dn(f2)
    
    csvfile = join(d,subd+"_QC.csv")
    assert os.path.exists(csvfile)
    return csvfile

def to_img(f):
    f2 = f.replace("/dataset/","/patch/").replace("_cellpose","").replace("/test","").replace("/train","").replace("/val","")
    b = bn(f)
    subd = b.split("_a")[0]
    afterd = "a"+"_".join(b.split("_a")[1:])
    d = dn(f2)
    
    imgfile = join(d, subd, afterd)
    
    assert os.path.exists(imgfile)
    return imgfile

def get_cb(f, cth = 0.5, bth = 0.0005):
    csvfile = to_csv(f)
    imgfile = to_img(f)
    
    df = pd.read_csv(csvfile).fillna(0)
    data = df.loc[df.imgfile == imgfile, :]
    
    corr = float(data["corr"])
    blur = float(data["blur"])
    
    return (corr > cth and blur > bth, corr, blur)

abss = []
imgfiles = []
lowQ = []
cs = []
bs = []

indir=sys.argv[1]
ab=sys.argv[2] #"MIST1_HR"
#indir=join("/dataset/",ab)
infiles = glob.glob(join(indir,"*/*_HE.png"))
print(f"antibody {ab} : {len(infiles)} imgfiles")

count=0
for f in tqdm(infiles):
    try:
        ok, c, b = get_cb(f)
        if not ok:
            abss.append(ab)
            imgfiles.append(f)
            lowQ.append(f)
            cs.append(c)
            bs.append(b)
            count+=1

        else:
            mask = to_mask(f)
            out_f = to_out(f)
            out_mask = to_out(mask)

            os.makedirs(dn(out_f), exist_ok=True)

            if not os.path.exists(out_f):
                shutil.copy(f, out_f, follow_symlinks=False)

            if not os.path.exists(out_mask):
                shutil.copy(mask, out_mask, follow_symlinks=False)
    except:
        print(f)
        
print(f"{count} imgs removed")

df_stat = pd.DataFrame({'antibody':abss,
                        'imgfile':imgfiles,
                        'maskfile':[to_mask(f) for f in imgfiles],
                        'corr':cs,
                        'blur':bs})

df_stat.to_csv("./QC_stat.csv")

