#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

from utils import log, stain, statistics, region, registration, parse_ndpa


parser = argparse.ArgumentParser()

parser.add_argument("targetdir",
                    help="input directory")

parser.add_argument("outdir",
                    help="output directory")

parser.add_argument("--init_scale",
                    help="scale used for rough registration",
                    default=6,
                    type=int)

parser.add_argument("--regist_scale",
                    help="scale for fine-grained registration",
                    default=1,
                    type=int)

parser.add_argument("-i", "--img_size",
                    help="output image size",
                    default=500,
                    type=int)

parser.add_argument("-p", "--pos_th",
                    help="cutoff IHC intensity(0-255) (including in file name only) ",
                    default=50,
                    type=int)

parser.add_argument("-m", "--mask_th",
                    help="cutoff IHC intensity for mask generation (0-255)",
                    default=10,
                    type=int)

parser.add_argument("-o", "--overwrite",
                    help="overwrite output image files",
                    action='store_true')

args = parser.parse_args()

init_scale = args.init_scale
regist_scale = args.regist_scale
out_scale = 1


imgsize = args.img_size  # output image size

pos_th = args.pos_th

mask_th = args.mask_th

overwrite = args.overwrite

is_large = False

outdir = args.outdir

targetdir = args.targetdir
dirs = glob.glob(os.path.join(targetdir, "*"))

assert os.path.exists(targetdir)

os.makedirs(outdir, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


handler2 = logging.FileHandler(
    filename=os.path.join(outdir, "registration_{}.log".format(os.path.basename(targetdir))))
handler2.setLevel(logging.INFO)
handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
logger.addHandler(handler2)

logger.info('Started')

logger.info(f'init_scale : {init_scale}')
logger.info(f'regist_scale : {regist_scale}')
if is_out_original_scale:
    logger.info('out_scale : 0')
else:
    logger.info(f'out_scale : {out_scale}')
logger.info(f'image size : {imgsize}')
logger.info(f'target dir : {targetdir}')
logger.info(f'output dir : {outdir}')



def get_thumbnail(op_ihc, op_he, scale=6, is_show=False):
    # get whole slide thumbnail (image sizes are different)
    region_ihc = op_ihc.read_region([0, 0], scale,
                                    [int(float(dim) / (2 ** scale)) for dim in op_ihc.dimensions])
    region_he = op_he.read_region([0, 0], scale,
                                  [int(float(dim) / (2 ** scale)) for dim in op_he.dimensions])

    region_ihc = 255 - np.array(region_ihc)[:, :, 0]

    region_he = stain.extract_H(region_he, split=True)

    if is_show:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(region_ihc, cmap="binary")
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(region_he, cmap="binary")
        ax2.axis('off')

    return region_ihc, region_he


def get_file(indir, outputdir, is_overwrite):
    sample = os.path.basename(indir)
    outdir_sample = os.path.join(outputdir, sample)
    if not is_overwrite:
        if os.path.isdir(outdir_sample):
            print("{} exists. skipped.".format(outdir_sample))
            return None, None, None, None
    files = glob.glob(os.path.join(indir, "*.ndpi"))

    os.makedirs(outdir_sample, exist_ok=True)

    ihcfile = targetfile = hefile = ""
    for f in files:
        if 'DAPI' in f:
            ihcfile = f
        elif 'Opal' in f:
            targetfile = f
        else:
            hefile = f

    return sample, outdir_sample, files, [ihcfile, hefile, targetfile]


def get_openslide(wsifiles):
    ihcfile, hefile, targetfile = wsifiles

    op_IHC = openslide.OpenSlide(ihcfile)
    op_HE = openslide.OpenSlide(hefile)
    op_target = openslide.OpenSlide(targetfile)

    return op_IHC, op_HE, op_target


# Main code


for ddir in dirs:
    merge_files = []
    sample, outdir_sample, files, wsifiles = get_file(ddir, outdir, overwrite)
    if sample is None:
        continue

    logger.info(f"sample: {sample}")
    logger.info(f"HE file: {wsifiles[1]}")
    logger.info(f"DAPI file: {wsifiles[0]}")
    logger.info(f"IF file: {wsifiles[2]}")

    op_IHC, op_HE, op_target = get_openslide(wsifiles)
    width, height = op_HE.dimensions

    hefile = wsifiles[1]

    ## read regions marked by pathologists
    logger.info("parse ndpa file")
    ndpafile = parse_ndpa.search_ndpa(hefile)
    if ndpafile is not None:
        excluded_contour = parse_ndpa.create_contour_from_ndpa(op_HE, ndpafile)
        logger.info("{} contours found".format(len(excluded_contour)))
    else:
        logger.info("No ndpa file found")
        excluded_contour = None

    ## extract tissue region candidates
    he_contour, he_contour_img, t = region.get_contour2(op_HE, large=is_large)
    logger.info("Otsu threshold : {}".format(t))
    logger.info("{} tissue regions were identified".format(len(he_contour)))
    if len(he_contour) == 0:
        logger.info("no tissue region. exit.")
        exit()


    logdata = log.Log(op_HE, sample, imgsize, out_scale)

    i, h = get_thumbnail(op_IHC, op_HE, is_show=False, scale=init_scale)

    # first registration
    logger.info("1st registration")
    R = registration.Registration()
    R.rigid_registration(i, h, init_scale, is_show=False)

    ## second registration  using kernel density estimation
    R.peak_rigid_registration(op_IHC,
                          op_HE,
                          step_img=20,
                          imgsize=imgsize,
                          scale=regist_scale)

    imgsize_origscale = int(imgsize * (2 ** out_scale))

    logger.info("2nd registration")
    R2 = registration.Registration()
    for i, x in enumerate(tqdm(range(0, width - imgsize_origscale, imgsize_origscale))):
        for j, y in enumerate(range(0, height - imgsize_origscale, imgsize_origscale)):

            ## extract patches if the patch is within the tissue region
            isin, dist = statistics.is_inside((x, y), he_contour, out_scale, imgsize)
            if not isin:
                continue

            ## remove patches if the patch is within the regions marked by pathologists  (annotated in ndpa)
            isin, annot = region.is_overlap([x, y], imgsize_origscale, excluded_contour)
            if isin:
                continue

            region_dapi, region_H = region.get_region(op_IHC, op_HE, (x, y), (imgsize, imgsize), scale=out_scale,
                                                      shift=R.get_shift())
            ## third registration
            R2.rigid_registration(region_dapi, region_H, out_scale)

            final_shift = (R.get_shift()[0] + R2.get_init_shift()[0],
                           R.get_shift()[1] + R2.get_init_shift()[1])

            region_target2, region_HE2 = region.get_region_he(op_target, op_HE, (x, y), (imgsize, imgsize),
                                                              scale=out_scale,
                                                              shift=final_shift)
            region_dapi, region_H = region.get_region(op_IHC, op_HE, (x, y), (imgsize, imgsize), scale=out_scale,
                                               shift=final_shift)

            ## fourth registration
            R2.rigid_registration(region_dapi, region_H, out_scale)

            final_shift += (R2.get_init_shift()[0], R2.get_init_shift()[1])

            for x2 in [x, x+imgsize]:
                for y2 in [y, y+imgsize]:
                    region_target2, region_HE2 = region.get_region_he(op_target, op_HE, (x2, y2), (imgsize, imgsize),
                                                                      scale=0,
                                                                      shift=final_shift)
                    region_dapi, _ = region.get_region(op_IHC, op_HE, (x2, y2), (imgsize, imgsize), scale=0,
                                                              shift=final_shift)

                    ## nonrigid registration
                    t, d = R.nonrigid_registration(region_HE2, region_dapi, region_target2)

                    merged = np.zeros([region_target2.shape[0], region_target2.shape[1], 3])
                    merged[:, :, 0] = 255 - t
                    merged[:, :, 2] = 255 - 4 * d
                    merged[merged < 0] = 0

                    pr = statistics.positive_rate(255 - t, pos_th)
                    max_blur = statistics.blur(merged)

                    out_merged = os.path.join(outdir_sample,
                                              "a{}_b{}_d{}_{:06d}_{:06d}_IHC_nonrigid.png".format(pr, max_blur, dist, x2, y2))
                    merge_files.append(out_merged)

                    ## remove 20-pixel margins from the edges
                    merged = merged[20:-20, 20:-20, :]
                    region_HE2 = region_HE2[20:-20, 20:-20, :]
                    out_he = os.path.join(outdir_sample, "a{}_b{}_d{}_{:06d}_{:06d}_HE.png".format(pr, max_blur, dist, x2, y2))

                    merged = np.uint8(merged)
                    region_HE2 = np.uint8(region_HE2)


                    logger.info(f"write IHC images : {out_merged}")
                    Image.fromarray(merged).save(out_merged)

                    logger.info(f"write HE images : {out_he}")
                    Image.fromarray(region_HE2).save(out_he)

                    logdata.add(x2, y2, pr, max_blur)

    logdata.plot(he_contour_img, os.path.join(outdir, "stat"))
    logfile = os.path.join(outdir_sample, "log.txt")
    logdata.write_log(logfile)

logger.info('Finished')
