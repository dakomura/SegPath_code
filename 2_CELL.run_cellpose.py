## apply Cellpose for leucocytes, myeloid cells, lymphocytes, plasma cells and endothelial cells.

from genericpath import exists
import numpy as np
from cellpose import utils, io, models
import joblib
import copy
import cv2
import glob
import os, sys
from PIL import Image
import argparse
import logging
from tqdm import tqdm

sys.path.append('./utils')
from TqdmLoggingHandler import TqdmLoggingHandler

parser = argparse.ArgumentParser()

parser.add_argument("indir",
                    help="input file directory",
                    type=str)

parser.add_argument("-p", "--pos_th",
                    help="IF intensity cutoff for mask generation(0-255)",
                    default=25,
                    type=int)

parser.add_argument("-d", "--diameter",
                    help="expected nucleus diameter(px)",
                    default=30,
                    type=int)

parser.add_argument("-b", "--bs",
                    help="batch size for cellpose",
                    default=64,
                    type=int)

parser.add_argument("-o", "--overlap",
                    help="overlap rate for positive cell",
                    default=0.4,
                    type=float)

parser.add_argument("-c", "--cpu",
                    help="CPU mode",
                    action='store_false')

parser.add_argument("--reuse",
                    help="reuse cellpose results",
                    action='store_true')

parser.add_argument("--skip",
                    help="skip if the output file exists",
                    action='store_true')

parser.add_argument("--cellpose_th",
                    help="Cell probability threshold",
                    default=0.1,
                    type=float)


args = parser.parse_args()

reuse = args.reuse

indir = args.indir
pos_th = args.pos_th
cpu = args.cpu
bs = args.bs
diameter = args.diameter if args.diameter > 0 else None
overlap_th = args.overlap
skip = args.skip

cth = args.cellpose_th


logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter_tqdm = logging.Formatter("%(asctime)s %(levelname)8s %(message)s")
tqdm_handler = TqdmLoggingHandler(level=logging.INFO)
tqdm_handler.setFormatter(formatter_tqdm)
logger.addHandler(tqdm_handler)

handler2 = logging.FileHandler(
    filename=os.path.join(indir, "cellpose_{}.log".format(os.path.basename(indir))))
handler2.setLevel(logging.INFO)
handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
logger.addHandler(handler2)

logger.info('Started')

logger.info(f'input dir : {indir}')
logger.info(f'positive threshold : {pos_th}')
logger.info(f'expected diameter : {diameter}')
logger.info(f'overlap ratio : {overlap_th}')
logger.info(f'cellpose prob th : {cth}')


fname_pre = "_IHC_nonrigid"
fname_post = "_IHC_cellpose_nonrigid"

#####################

def overlap_mask(img, mask, pos_th, overlap_th):
    mask_final = np.zeros(mask.shape, dtype=int)
    mask_final_each = np.zeros(mask.shape, dtype=int)

    kernel = np.ones((3, 3), np.uint8)
    for j in range(1, np.max(mask) + 1):

        area = len(mask[mask == j])
        if area == 0:
            continue

        mask2 = np.zeros(mask.shape, dtype=np.uint8)
        mask2[mask == j] = 1

        # morphological erosion

        mask3 = cv2.erode(mask2, kernel, iterations=1)

        signal = np.zeros(mask.shape, dtype=int)
        signal[img[:, :, 0] > pos_th] = 1

        img_and = mask2 * signal
        overlap = float(len(img_and[img_and > 0])) / float(area)
        if overlap > overlap_th:
            mask_final[mask3 == 1] = 255
            mask_final_each[mask3 == 1] = j

    return mask_final, mask_final_each


def mask_overlay(img, masks, colors=None):
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    img = utils.normalize99(img)
    img -= img.min()
    img /= img.max()
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(img*1.5, 0, 1.0)
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = np.random.rand()
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def image_to_rgb(img0, channels=[0,0]):
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim<3:
        img = img[:,:,np.newaxis]
    if img.shape[0]<5:
        img = np.transpose(img, (1,2,0))
    if channels[0]==0:
        img = img.mean(axis=-1)[:,:,np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:,:,i])>0:
            img[:,:,i] = utils.normalize99(img[:,:,i])
            img[:,:,i] = np.clip(img[:,:,i], 0, 1)
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1]==1:
        RGB = np.tile(img,(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    return RGB

def get_mask_img(img, maski, channels=[3, 0]):
    img0 = img.copy()
    if img0.shape[0] < 4:
        img0 = np.transpose(img0, (1, 2, 0))
    if img0.shape[-1] < 3 or img0.ndim < 3:
        img0 = image_to_rgb(img0, channels=channels)
    else:
        if img0.max() <= 50.0:
            img0 = np.uint8(np.clip(img0 * 255, 0, 1))

    overlay = mask_overlay(img0, maski)

    return overlay

def split_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]

def imread2(infile, i):
    return io.imread(infile), i

#####################

if not reuse:
    model = models.Cellpose(gpu=cpu, model_type='nuclei')

infiles = glob.glob(os.path.join(indir,"*{}.png".format(fname_pre)))

if skip:
    infiles = [x for x in infiles if not os.path.exists(x.replace(fname_pre, fname_post))]

logging.info(f"{len(infiles)} files for processing")

infiles_spl = split_list(infiles, bs)

for k, infiles_sub in enumerate(infiles_spl):

    imgs = [io.imread(infile) for infile in tqdm(infiles_sub)]

    if reuse:
        masks = []
        for infile in infiles_sub:
            mask_file = infile.replace(".png", "_mask.pkl")
            mask = joblib.load(mask_file)
            masks.append(mask)

    else:
        masks, _, _, _ = model.eval(imgs,
                                    diameter=diameter,
                                    channels=[3,0],
                                    batch_size=bs,
                                    cellprob_threshold=cth)

    for infile, img, mask in zip(infiles_sub, imgs, masks):

        if not reuse:
            mask_file = infile.replace(".png", "_mask.pkl")
            assert infile != mask_file
            joblib.dump(mask, mask_file)
        logging.info(f"processing {infile}...")

        mask_final, mask_final_each = overlap_mask(img, mask, pos_th, overlap_th)
        img[:, :, 0] = mask_final

        outfile = infile.replace(fname_pre, fname_post)
        Image.fromarray(np.uint8(img)).save(outfile)
