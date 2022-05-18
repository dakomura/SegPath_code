import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
import copy
import glob
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from torchvision import transforms
import albumentations as albu

## MLFlow
import mlflow
mlflow.set_tracking_uri("http://192.168.0.1:5000")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.1:4000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio-secret-key"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device name", torch.cuda.get_device_name(0))

import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler())
logger.info('Started')

def load_model(ab: str, run_uuid: str):
    tracking = mlflow.tracking.MlflowClient()

    model_tmp_path = tracking.download_artifacts(run_uuid, 'model/data/model.pth')
    best_model = torch.load(model_tmp_path).to(DEVICE).half()

    return best_model

def get_pred(image, best_model, norm, crop):
    image_norm = norm(image=image)['image']
    x_tensor = torch.from_numpy(image_norm).to(DEVICE).unsqueeze(0)
    x_tensor = x_tensor.permute(0,3,1,2).half()
    pr_mask = best_model.predict(x_tensor)
    pr_mask = torch.sigmoid(pr_mask)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round()).astype(np.uint8)
    pr_mask = crop(image=pr_mask)['image']
    
    return pr_mask

def get_norm():
    """Add paddings to make image shape divisible by 32"""
    transform = [
        albu.augmentations.transforms.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32),
        albu.augmentations.transforms.Normalize()
    ]
    return albu.Compose(transform)

def get_crop():
    #back to original image size
    transform = [
        albu.CenterCrop(984, 984),
    ]
    return albu.Compose(transform)

def get_mask(ihc_th, min_size = 200):
    ihc_th = ihc_th.astype(np.uint8)

    if min_size is None:
        mask = np.zeros(ihc_th.shape)
        mask[ihc_th > 0] = 200

    else:
        mask = np.zeros(ihc_th.shape, dtype=np.uint8)

        contours, hierarchy = cv2.findContours(ihc_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > min_size:
                cv2.drawContours(mask, [cnt], 0, 200, -1)

    return mask

parser = argparse.ArgumentParser()

parser.add_argument("input",
                    help="input dir")

parser.add_argument("--th_opal",
                    help="IF intensity cutoff",
                    default=50,
                    type=int)

## load model for RBC segmentation stored in MLFlow server
ab, uuid, is_sigmoid = "GLPA_HR", "ca378a666ab5440dbc3ec4d602bfa041", True
best_model = load_model(ab, uuid)
norm = get_norm()
crop = get_crop()


args = parser.parse_args()

th_opal = args.th_opal

indir = args.input
he_files = glob.glob(os.path.join(indir,"*_HE.png"))
for fn_im in he_files:

    fn_ihc = fn_im.replace("_HE", "_IHC_nonrigid")
    ihc = imread(fn_ihc)[:, :, ::-1]

    fn_out = fn_im.replace("_HE", "_IHC_nonrigid_mask2")

    logger.info(f"target HE: {fn_im}")

    img = imread(fn_im)


    ## get IF-positive regions
    opal = ihc[:, :, 0] > th_opal
    mask_opal = get_mask(opal, None)
    mask_area_before = np.count_nonzero(mask_opal)

    ## RBC segmentation
    out_seg = get_pred(img[:,:,::-1], best_model, norm, crop)
    ## RBC mask
    mask_opal[out_seg > 0] = 0
    mask_area_after = np.count_nonzero(mask_opal)
    mask_change = mask_area_after - mask_area_before
    logger.info(f"remove {mask_change} px mask for RBC from {fn_im}")

    mask_both = copy.deepcopy(mask_opal)
    mask_both_save = np.zeros([mask_both.shape[0], mask_both.shape[1]], dtype = np.uint8)

    mask_both_save[mask_both == 200] = 1

    cv2.imwrite(fn_out, mask_both_save)

    logger.info(f"out mask: {fn_out}")

logger.info("finished")
