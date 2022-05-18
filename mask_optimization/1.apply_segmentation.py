from genericpath import exists
import os
import glob
import numpy as np
import tqdm
import joblib

from typing import List, Union, Optional, Any
from dataclasses import asdict

import argparse

import pprint

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import kornia.augmentation as K

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.fn as fn
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.types import DALIImageType
import nvidia.dali.types as types

import mlflow.pytorch
import mlflow



BINARY_MODE = "binary"
# ## parameter definition

round = "_3rd"

parser = argparse.ArgumentParser()

parser.add_argument("antibody",
                    help="antibody for the analysis",
                    type=str)

parser.add_argument("-b","--batch",
                    help="batch size",
                    default=16,
                    type=int)

parser.add_argument("--rbc",
                    help="RBC segmentation",
                    action='store_true')

parser.add_argument("--data_dir",
                    help="input directory",
                    default="/dataset_clean"+round,
                    type=str)

parser.add_argument("-o","--out_dir",
                    help="save directory",
                    type=str)

args = parser.parse_args()

args = parser.parse_args()

data_dir = args.data_dir
is_rbc = args.rbc
ab = args.antibody
bs = args.batch
outdir = os.path.join(args.out_dir, ab)

if 'cellpose' in ab:
    postfix = "_IHC_cellpose_mask"
    print("cellpose mode")
else:
    postfix = "_IHC_nonrigid_mask2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device name", torch.cuda.get_device_name(0))

ACTIVATION = None
CLASSES = ['target']
DEVICE = 'cuda'

EPSILON = 1e-15

device_id = 0

# mlflow
mlflow.set_tracking_uri("http://192.168.0.1:5000")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.1:4000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio-secret-key"

if round == "_2nd" or round== "_3rd":
    ab_dict = {"SMA":"8ec5d3fa9e2b441ba01445f6c8d144b2",
            "AE13":"80634b4e6be24794abda2980f29131ae",
            "CD45_cellpose":"73bb275924eb4e5786a7d024fa0639dd",
            "ERG_cellpose":"fe650ed5518845658770a42d0bf6713a",
            "CD3_CD20_cellpose":"3365f9b286c744a9b7988bac5a1d44ae",
            "MIST1_cellpose":"12890d41049243198a9c260a8442dae6", 
            "MNDA_cellpose":"c13449f15d5a4b34a3704687f55e4d19",        
            "RBC":"ca378a666ab5440dbc3ec4d602bfa041",
    }
else:
    ab_dict = {"SMA":"1ba6014721a2450fa12a4a6364e49542",
            "AE13":"86e6a465fc3d4e5a842da811d3d6a13d",
            "CD45_cellpose":"882c9a5edf75459ea0a18e903a203f69",
            "ERG_cellpose":"f0d15f13f64441929e43a9fa68754ff3",
            "CD3_CD20_cellpose":"7c995b2c298441c49c4c461dc046a4eb",
            "MIST1_cellpose":"d4b4aec68078445b89df34785fc3cb5c", 
            "MNDA_cellpose":"0042227385f744e7b874e87a54d38c96",        
            "RBC":"ca378a666ab5440dbc3ec4d602bfa041",
    }


assert ab in ab_dict.keys()

def get_input_files(path: str, tvt: str) -> str:
    """
    get train/val/test image files

    :type tvt: str
    """
    return glob.glob(path + "/" + tvt + "/*_HE.png")


class DALIPipeline(Pipeline):
    def __init__(
            self,
            image_files: List[str],
            bs: int,
            num_threads: int,
            device_id: str,
    ):
        batch_size = bs
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id)

        self.ab = ab
        self.ids = image_files
        self.images_fps = image_files

        self.class_values = [1]

        # make file list
        self.fileroot = os.path.dirname(self.images_fps[0])
        self.tmpfile_img = f"/tmp/cls_{self.ab}_{self.device_id}.txt"
        if os.path.exists(self.tmpfile_img):
            os.remove(self.tmpfile_img)
        with open(self.tmpfile_img, "w") as fout:
            for imgfile in self.images_fps:
                print(f"{os.path.basename(imgfile)}\t0", file=fout)

    def define_graph(self):
        img, _ = fn.readers.file(file_root=self.fileroot, file_list=self.tmpfile_img,
                                 name='Reader1',
                                 shuffle_after_epoch = False,
                                 shard_id=int(self.device_id),
                                 num_shards=1
                                 )

        image = fn.decoders.image(img, device='mixed')
        return image

    def __len__(self):
        return len(self.ids)


def dali_iter(image_files: List[str], bs=20):
    """
    :rtype: DALIGenericIterator
    """
    image_files = list(image_files)
    pipe = DALIPipeline(image_files, 
                          bs=bs,
                          num_threads=4,
                          device_id=0) 
    pipe.build()

    return DALIGenericIterator(pipe,
                               ['image'],
                               reader_name='Reader1',
                               last_batch_policy = LastBatchPolicy.PARTIAL,
                               last_batch_padded = False,
                               dynamic_shape=False,
                               )

def load_model(run_uuid: str):
    tracking = mlflow.tracking.MlflowClient()
    run = tracking.get_run(run_uuid)
    pprint.pprint(run.data.metrics)
    pprint.pprint(run.data.params)

    model_tmp_path = tracking.download_artifacts(run_uuid, 'model/data/model.pth')
    best_model = torch.load(model_tmp_path).to(DEVICE).half()

    return best_model


def get_pred(x_tensor, best_model, transform=None):
    best_model.eval()
    with torch.no_grad():
        pr_mask = best_model.predict(x_tensor)
    if transform is not None:
        pr_mask = transform(pr_mask)
    pr_sigmoid = (torch.sigmoid(pr_mask).squeeze().cpu().numpy())
    if len(pr_sigmoid.shape) == 2:
        pr_sigmoid = pr_sigmoid[np.newaxis,:,:]
    return pr_sigmoid

def get_pred_cls(x_tensor, best_model, transform=None):
    best_model.eval()
    with torch.no_grad():
        pr_mask = best_model.predict(x_tensor)
    if transform is not None:
        pr_mask = transform(pr_mask)
    pr_logit = (pr_mask.squeeze().cpu().numpy() > 0)
    if len(pr_logit.shape) == 2:
        pr_logit = pr_logit[np.newaxis,:,:]
    return pr_logit


class Data():
    def __init__(
        self,
        x_files,
        bs=20,
    ):
        self.batch = dali_iter(x_files, bs=bs)
        self.aug = transforms.Compose([
            transforms.Pad(padding=4,
                        padding_mode="reflect"),
            transforms.Normalize(mean=[124., 116., 104.],
                                std=[58.6, 57.3, 57.6])
        ])

    def get_input(self):
        features = self.batch.next()[0]['image'].permute(0,3,1,2).half()
        features = self.aug(features)

        return features


def to_np(features, idx):
    return features[idx].permute(1,2,0).cpu().detach().numpy().astype(np.float32)

def main():
    datadir = os.path.join(data_dir, ab+"_HR")

    os.makedirs(outdir, exist_ok=True)

    x_files_all = []
    for m in ['train', 'val', 'test']:
        x_files = get_input_files(datadir, m)
        x_files_all.extend(x_files)
            
        data = Data(x_files, bs=bs)

        if is_rbc:
            model = load_model(ab_dict["RBC"])
        else:
            model = load_model(ab_dict[ab])

        tr = transforms.CenterCrop(984)

        for i in tqdm.tqdm(range(len(x_files)//bs + 1)):
            d = data.get_input()
            bsi = d.shape[0]
            ## run segmentation
            if is_rbc:
                ## return 0/1
                pr = get_pred_cls(d, model, tr).transpose(1, 2, 0)
            else:
                ## return probability
                pr = get_pred(d, model, tr).transpose(1, 2, 0)
            for j in range(bsi):
                of = os.path.basename(x_files[c+j])
                if is_rbc:
                    outfile = os.path.join(outdir, f"{of}{round}.RBC.npy")
                else:
                    outfile = os.path.join(outdir, f"{of}{round}.npy")
                ## save segmentation results for j-th image
                np.save(outfile, np.squeeze(pr[:,:,j]))

if __name__ == '__main__':
    main()