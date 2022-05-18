
import os
import glob
import datetime
import joblib
import numpy as np

from typing import List, Union, Optional, Any
from dataclasses import asdict

import argparse
import random
np.random.seed(seed=1)

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import kornia.augmentation as K

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.fn as fn
from nvidia.dali.types import DALIImageType
import nvidia.dali.types as types

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

import torch_optimizer as optim

from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torchmetrics.utilities.distributed import reduce


import segmentation_models_pytorch as smp
from pytorch_toolbelt.losses import BinaryFocalLoss, DiceLoss
from pytorch_toolbelt.losses.functional import soft_dice_score

import mlflow.pytorch
import mlflow

pl.utilities.seed.seed_everything(seed=0, workers=True)

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"

# ## parameter definition

parser = argparse.ArgumentParser()

parser.add_argument("antibody",
                    help="antibody to analyze",
                    type=str)

parser.add_argument("--user",
                    help="user name for MLFlow",
                    type=str)

parser.add_argument("--data_dir",
                    help="input data directory",
                    type=str)

parser.add_argument("--resume",
                    help="Resume file for Optuna Study",
                    default=None,
                    type=str)

parser.add_argument("-i", "--img_size",
                    help="input image size",
                    default=960,
                    type=int)

parser.add_argument("--normvalue",
                    help="RGB normalize mean std (imagenet or histology)",
                    default="imagenet",
                    type=str)

parser.add_argument("--post",
                    help="postfix for MLflow name",
                    default="",
                    type=str)

parser.add_argument("-l", "--loss",
                    help="loss type(combo/dice/bce/ftv/focal/auto)",
                    default="auto",
                    type=str)

parser.add_argument("--lparam1",
                    help="loss parameter 1",
                    default=None,
                    type=float)

parser.add_argument("--lparam2",
                    help="loss parameter 2",
                    default=None,
                    type=float)

parser.add_argument("-e", "--nepoch",
                    help="Number of epoch",
                    default=10,
                    type=int)

parser.add_argument("-t", "--n_trials",
                    help="Number of optuna trial",
                    default=10,
                    type=int)

parser.add_argument("-b", "--nbatch_tr",
                    help="Training batch size",
                    default=8,
                    type=int)

parser.add_argument("--accum_grad",
                    help="use accumulate gradient",
                    default=None,
                    type=int)

parser.add_argument("--oversampling",
                    help="oversampling for training data",
                    action='store_true')

parser.add_argument("-g", "--num_gpus",
                    help="Number of GPU used for training",
                    default=1,
                    type=int)

parser.add_argument("--debug",
                    help="debug mode (only 5% samples are used for train/val",
                    action='store_true')

parser.add_argument("--save_checkpoint",
                    help="save checkpoint to resume training",
                    action='store_true')

args = parser.parse_args()

data_dir = args.data_dir

ab = args.antibody
ims = args.img_size
user = args.user
nepoch = args.nepoch
nbatch_tr = args.nbatch_tr

acg = args.accum_grad
oversampling = args.oversampling

mlflow_post = args.post

resume = args.resume

assert args.normvalue in ['imagenet']
normvalue = args.normvalue

if normvalue == 'imagenet':
    nmean = [124., 116., 104.]
    nstd = [58.6, 57.3, 57.6]

num_gpus = args.num_gpus

n_trials = args.n_trials
debug = args.debug

save_checkpoint = args.save_checkpoint


if 'cellpose' in ab:
    postfix = "_IHC_cellpose_mask"
    print("cellpose mode")
else:
    postfix = "_IHC_nonrigid_mask2"

model_path = "/model/torch"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device name", torch.cuda.get_device_name(0))

ACTIVATION = None
CLASSES = ['target']
DEVICE = 'cuda'

EPSILON = 1e-15

# mlflow
mlflow.set_tracking_uri("http://192.168.0.1:5000")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.1:4000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio-secret-key"


## function definition

####### Original Metrics ##########
class merged_Intersection(ConfusionMatrix):

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        threshold: float = 0.5,
        reduction: str = "sum",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            normalize=None,
            threshold=threshold,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.absent_score = absent_score

    def compute(self) -> Tensor:
        """Computes intersection over union (Dice)"""
        return _intersection_from_confmat(self.confmat, self.num_classes, self.ignore_index, self.absent_score, self.reduction)

    @property
    def is_differentiable(self) -> bool:
        return False

class merged_Union(ConfusionMatrix):

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        threshold: float = 0.5,
        reduction: str = "sum",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            normalize=None,
            threshold=threshold,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.absent_score = absent_score

    def compute(self) -> Tensor:
        """Computes intersection over union (Dice)"""
        return _union_from_confmat(self.confmat, self.num_classes, self.ignore_index, self.absent_score, self.reduction)

    @property
    def is_differentiable(self) -> bool:
        return False

def _intersection_from_confmat(
    confmat: Tensor,
    reduction: str = "sum",
) -> Tensor:
    intersection = confmat[1,1]

    # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
    scores = intersection.float()

    return reduce(scores, reduction=reduction)

def _union_from_confmat(
    confmat: Tensor,
    reduction: str = "sum",
) -> Tensor:

    union = confmat[1,0] + confmat[0,1]
    scores = union.float()

    return reduce(scores, reduction=reduction)


################ Optuna callback (save study) ##############
class SaveStudyCallback:
    def __init__(self, outfile: str):
        self.outfile = outfile
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        joblib.dump(study, self.outfile)

################ Loss ##############

class LogCoshDiceLoss(DiceLoss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        assert y_true.size(0) == y_pred.size(0)

        y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        dims = (0, 2)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask
            y_true = y_true * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        loss = torch.log(torch.cosh(loss))

        return loss.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self,
                 smooth=1,
                 alpha=0.5,
                 beta=0.5,
                 gamma=1.0,
                 weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky


def label_func(fn: str, postfix: str) -> str:
    return fn.replace("_HE", postfix)


def get_input_files(path: str, tvt: str) -> str:
    return glob.glob(path + "/" + tvt + "/*_HE.png")


def get_mask_files(input_files: List[str],
                   postfix: str) -> List[str]:
    return [label_func(x, postfix) for x in input_files]


class DALIPipeline(Pipeline):
    def __init__(
            self,
            image_files: List[str],
            mask_files: List[str],
            batch_size: int,
            imgsize: int,
            mode: str,
            num_threads: int,
            device_id: str,
            num_gpus: int,
    ):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id)

        self.ab = ab
        self.ids = image_files
        self.images_fps = image_files
        self.masks_fps = mask_files
        self.imgsize = imgsize
        self.oversampling = oversampling

        self.num_gpus = num_gpus

        self.mode = mode  # train or val
        self.max_xy = self.imgsize / 960.0 / 2.0  # random crop region

        self.class_values = [1]

        self.fileroot = os.path.dirname(self.images_fps[0])
        self.tmpfile_img = f"/tmp/cls_{self.mode}_{self.device_id}.txt"
        self.tmpfile_mask = f"/tmp/cls_{self.mode}_{self.device_id}_mask.txt"
        with open(self.tmpfile_img, "w") as fout:
            for imgfile in self.images_fps:
                print(f"{os.path.basename(imgfile)}\t0", file=fout)
        with open(self.tmpfile_mask, "w") as fout:
            for imgfile in self.masks_fps:
                print(f"{os.path.basename(imgfile)}\t0", file=fout)

    def define_graph(self):
        img, _ = fn.readers.file(file_root=self.fileroot, file_list=self.tmpfile_img,
                                 name='Reader1',
                                 initial_fill=128,
                                 shuffle_after_epoch = True if self.mode == 'train' else False,
                                 shard_id=int(self.device_id),
                                 num_shards=self.num_gpus,
                                 seed=1,
                                 )
        mask, _ = fn.readers.file(file_root=self.fileroot, file_list=self.tmpfile_mask,
                                  name='Reader2',
                                  initial_fill=128,
                                  shuffle_after_epoch = True if self.mode == 'train' else False,
                                  shard_id=int(self.device_id),
                                  num_shards=self.num_gpus,
                                  seed=1,
                                  )

        image = fn.decoders.image(img, device='mixed')
        mask = fn.decoders.image(mask, device='mixed', output_type=DALIImageType.GRAY)

        if self.mode == 'train':
            self.rng_hue = fn.random.uniform(range=[-0.1, 0.1])  # -0.2, 0.2 was too strong
            self.rng_sat = fn.random.uniform(range=[0.9, 1.1])
            self.rng_bright = fn.random.uniform(range=[0.9, 1.1])
            self.rng_cont = fn.random.uniform(range=[0.9, 1.1])

            crop_pos_x = random.uniform(self.max_xy, 1.0 - self.max_xy)
            crop_pos_y = random.uniform(self.max_xy, 1.0 - self.max_xy)

            bright = self.rng_bright
            cont = self.rng_cont
            hue = self.rng_hue
            sat = self.rng_sat

            image = fn.color_twist(image, brightness=bright,
                                   contrast=cont,
                                   hue=hue,
                                   saturation=sat)

        else:
            crop_pos_x = crop_pos_y = 0.5

        image = fn.crop_mirror_normalize(image, crop=(self.imgsize, self.imgsize), crop_pos_x=crop_pos_x,
                                         crop_pos_y=crop_pos_y,
                                         dtype=types.FLOAT16,
                                         mean=nmean,
                                         std=nstd,)
        mask = fn.crop_mirror_normalize(mask, crop=(self.imgsize, self.imgsize), crop_pos_x=crop_pos_x,
                                        crop_pos_y=crop_pos_y,
                                        dtype=types.FLOAT16)  # , device='mixed')

        return image, mask

    def __len__(self):
        return len(self.ids)


def dali_iter(ab: str, oversampling: bool, image_files: List[str], mask_files: List[str],
              batch_size: int, imgsize: int, num_gpus: int,
              mode: str):
    """

    :rtype: DALIGenericIterator
    """
    image_files = list(image_files)
    mask_files = list(mask_files)

    if mode == 'train':
        c = list(zip(image_files, mask_files))
        random.shuffle(c)
        image_files, mask_files = zip(*c)

    pipes = [DALIPipeline(image_files, mask_files,
                          batch_size=batch_size, imgsize=imgsize,
                          mode=mode,
                          num_threads=4,
                          num_gpus=num_gpus,
                          device_id=device_id) for device_id in range(num_gpus)]
    for pipe in pipes:
        pipe.build()

    return DALIGenericIterator(pipes,
                               ['image', 'mask'],
                               reader_name='Reader1',
                               auto_reset=True,
                               dynamic_shape=False,
                               )


@dataclass
class Config:
    ab: str
    ims: int
    nbatch_tr: int
    BACKBONE: str
    ENCODER_WEIGHTS: Union[None, str]
    ACTIVATION: Union[None, str]
    lr: float
    model_arch: str
    num_gpus: int
    trial: optuna.Trial
    postfix: str
    loss_type: str
    loss_param1: Union[None, float]
    loss_param2: Union[None, float]
    logfile: str
    image_norm: str
    swa: bool
    accumulate_grad: bool
    oversampling: bool


class SegmentCell(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        if conf["loss_type"] == 'combo':
            if conf["loss_param1"] is None:
                loss_param1 = 0.3
            else:
                loss_param1 = conf["loss_param1"]

            self.losses = [
                ("dice", loss_param1, DiceLoss(mode="binary", from_logits=True)),
                ("focal", 1.0 - loss_param1, BinaryFocalLoss()),
            ]
        elif conf["loss_type"] == 'dice':
            self.losses = [
                ("dice", 1.0, DiceLoss(mode="binary", from_logits=True)),
            ]
        elif conf["loss_type"] == 'bce':
            self.losses = [
                ("bce", 1.0, nn.BCEWithLogitsLoss()),
            ]
        elif conf["loss_type"] == 'ftv':
            if conf["loss_param1"] is None:
                loss_param1 = 10
            else:
                loss_param1 = conf["loss_param1"]

            if conf["loss_param2"] is None:
                loss_param2 = 2.0
            else:
                loss_param2 = conf["loss_param2"]

            self.losses = [
                ("ftversky", 1.0, FocalTverskyLoss(beta=loss_param1, gamma=loss_param2)),
            ]
        elif conf["loss_type"] == 'focal':
            if conf["loss_param1"] is None:
                loss_param1 = 10
            else:
                loss_param1 = conf["loss_param1"]

            if conf["loss_param2"] is None:
                loss_param2 = 2.0
            else:
                loss_param2 = conf["loss_param2"]

            self.losses = [
                ("focal", 1.0, BinaryFocalLoss(alpha=1.0/loss_param1, gamma=loss_param2)),
            ]
        elif conf["loss_type"] == 'logcoshdice':
            self.losses = [
                ("dice", 1.0, LogCoshDiceLoss(mode="binary", from_logits=True)),
            ]

        self.model = eval(f"smp.{self.conf['model_arch']}")(
            encoder_name=self.conf["BACKBONE"],
            encoder_weights=self.conf["ENCODER_WEIGHTS"],
            classes=1,
            in_channels=3,
            activation=self.conf["ACTIVATION"],
        )

        self.save_hyperparameters()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(batch)

    def setup(self, stage=0):
        self.datadir = os.path.join(data_dir, self.conf["ab"])
        print(self.datadir)

        self.x_train_files = get_input_files(self.datadir, 'train')
        self.y_train_files = get_mask_files(self.x_train_files, self.conf["postfix"])

        xy = list(zip(self.x_train_files, self.y_train_files))
        random.shuffle(xy)
        self.x_train_files, self.y_train_files = zip(*xy)

        self.x_val_files = get_input_files(self.datadir, 'val')
        self.y_val_files = get_mask_files(self.x_val_files, self.conf["postfix"])

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.conf["BACKBONE"],
                                                                  self.conf["ENCODER_WEIGHTS"])

        self.aug1 = [
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomAffine(degrees=180,
                           scale=(0.9, 1.1),
                           padding_mode="reflection"),
        ]
        self.aug2 = nn.Sequential(
            K.GaussianBlur((3, 3), (0.2, 0.2), p=.3)
        )

        self.train_intersection = merged_Intersection(
            num_classes=2,
            ignore_index=0,
            dist_sync_on_step=True)
        self.train_union = merged_Union(
            num_classes=2,
            ignore_index=0,
            dist_sync_on_step=True)
        self.val_intersection = merged_Intersection(
            num_classes=2,
            ignore_index=0,
            compute_on_step = False,
            dist_sync_on_step=True)
        self.val_union = merged_Union(
            num_classes=2,
            ignore_index=0,
            compute_on_step = False,
            dist_sync_on_step=True)

        print("Classification for ", self.conf["ab"])
        print("# train files = ", len(self.x_train_files))
        print("# val files = ", len(self.x_val_files))

    def train_dataloader(self):
        train_loader = dali_iter(self.conf["ab"],
                                 self.conf["oversampling"],
                                 self.x_train_files,
                                 self.y_train_files,
                                 self.conf["nbatch_tr"],
                                 self.conf["ims"],
                                 self.conf["num_gpus"],
                                 'train')

        return train_loader

    def val_dataloader(self):
        valid_loader = dali_iter(self.conf["ab"],
                                 self.conf["oversampling"],
                                 self.x_val_files,
                                 self.y_val_files,
                                 2,
                                 960,
                                 self.conf["num_gpus"],
                                 'val')

        return valid_loader

    def configure_optimizers(self):
        optimizer = optim.RAdam([
            dict(params=self.model.parameters(), lr=self.conf["lr"],
                 weight_decay=0.0001),
        ])

        self.optimizers = [optimizer]

        return self.optimizers

    def training_step(self, batch, batch_idx):
        features = batch[0]['image']
        masks = batch[0]['mask']

        if self.aug1 is not None:
            features = self.aug2(features)
            for aug1_each in self.aug1:
                features = aug1_each(features)
                masks = aug1_each(masks, aug1_each._params)

        y = self.forward(features)

        total_loss = 0
        for loss_name, weight, loss in self.losses:
            ls_mask = loss(y, masks)
            total_loss += weight * ls_mask

        y = torch.sigmoid(y)

        return {'y': y, 'masks': masks, 'loss':total_loss}

    def training_step_end(self, outputs):
        self.train_intersection(outputs["y"], outputs["masks"].type(torch.int64))
        self.train_union(outputs["y"], outputs["masks"].type(torch.int64))
        self.log('train_loss', outputs["loss"])
        return {'loss': outputs['loss']}

    def training_epoch_end(self, outs):
        train_intersection_epoch = self.train_intersection.compute()
        train_union_epoch = self.train_union.compute()
        train_dice_epoch = train_intersection_epoch / (train_union_epoch / 2.0 + train_intersection_epoch)
        self.log('train_dice_epoch', train_dice_epoch)


    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()

    def validation_step(self, batch, batch_id):
        features = batch[0]['image']
        masks = batch[0]['mask']

        y = self.forward(features)

        total_loss = 0
        for loss_name, weight, loss in self.losses:
            ls_mask = loss(y, masks)
            total_loss += weight * ls_mask

        y = torch.sigmoid(y)

        return {'y': y, 'masks': masks, 'loss':total_loss}

    def validation_step_end(self, outputs):
        self.val_intersection(outputs["y"], outputs["masks"].type(torch.int64))
        self.val_union(outputs["y"], outputs["masks"].type(torch.int64))
        #self.log('val_dice', self.val_metric)
        self.log('val_loss', outputs["loss"])

    def validation_epoch_end(self, outs):
        val_intersection_epoch = self.val_intersection.compute()
        val_union_epoch = self.val_union.compute()
        val_dice_epoch = val_intersection_epoch / (val_union_epoch / 2.0 + val_intersection_epoch)
        self.log('val_dice_epoch', val_dice_epoch)

        with open(self.conf["logfile"], 'w') as f: #loggerはdelayが生じるのでここでrealtimeにvalidation diceをファイル出力する
            vie = val_dice_epoch.to('cpu').detach().numpy()
            print(vie, file=f)
            print("validation Dice:", vie)

    def get_model(self):
        return self.model


def objective(trial: Union[optuna.Trial, None], args) -> float:
    """
    Optuna trial
    :param trial: trial
    :return: Validation Dice
    """

    # create segmentation model with pretrained encoder
    size_params = {'timm-efficientnet-': ['b1', 'b2', 'b3'],
                   'resnet': ['18', '34', '50']}

    if trial is None:
        lr = 1e-3
        size = 0
        is_swa = False
        model_arch = 'DeepLabV3Plus'
        model_backbone = "timm-efficientnet-"
    else:
        lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        size = trial.suggest_int('model size', 0, 2)
        is_swa = trial.suggest_categorical('swa', [True, False])
        model_arch = trial.suggest_categorical('model arch', ['DeepLabV3Plus', 'UnetPlusPlus', 'Unet'])
        model_backbone = trial.suggest_categorical('model backbone', ["timm-efficientnet-", 'resnet'])

    loss_type = args.loss
    assert loss_type in ['combo', 'dice', 'bce', 'ftv', 'focal', 'logcoshdice', 'auto']

    if loss_type == 'auto':
        loss_type2 = trial.suggest_categorical('loss type', ['combo', 'dice', 'bce', 'ftv', 'focal', 'logcoshdice'])
        loss_param1 = loss_param2 = None
        if loss_type2 in ['ftv', 'focal']:
            loss_param1 = trial.suggest_uniform('ftv_alpha', 3, 8)
            loss_param2 = trial.suggest_uniform('ftv_gamma', 0.5, 3)
        if loss_type2 == 'combo':
            loss_param1 = trial.suggest_uniform('combo weight', 0.1, 0.9)
    else:
        loss_type2 = loss_type
        if loss_type2 in ['ftv', 'focal']:
            if args.lparam1 != None:
                loss_param1 = args.lparam1
            else:
                loss_param1 = trial.suggest_uniform('ftv_alpha', 3, 8)
            if args.lparam2 != None:
                loss_param2 = args.lparam2
            else:
                loss_param2 = trial.suggest_uniform('ftv_gamma', 0.5, 3)
        if loss_type2 == 'combo':
            if args.lparam1 != None:
                loss_param1 = args.lparam1
            else:
                loss_param1 = trial.suggest_uniform('combo weight', 0.1, 0.9)
        else:
            loss_param1 = None
            loss_param2 = None


    acg = args.accum_grad
    if acg == None:
        accumulate_grad = trial.suggest_categorical('accum grad', [1, 5, 10])
    else:
        accumulate_grad = acg 

    ENCODER_WEIGHTS = 'noisy-student' if model_backbone == "timm-efficientnet-" else 'imagenet'

    BACKBONE = f"{model_backbone}{size_params[model_backbone][size]}"

    print(f"img size: {ims}")
    print(f"model arch: {model_arch}")
    print(f"model backbone: {BACKBONE}")
    print(f"encoder weights: {ENCODER_WEIGHTS}")

    expname = "{}_{}_{}".format(ab,
                                random.randint(0, 1000),
                                datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    mlflow_logger = MLFlowLogger(experiment_name=ab+mlflow_post,
                                 tags = {'user': user},
                                 tracking_uri="http://192.168.0.1:5000")
    #tb_logger = TensorBoardLogger("test", name="my_model")

    logdir = f"/tmp/{expname}"
    logfile = os.path.join(logdir, "metrics.csv")
    os.makedirs(logdir, exist_ok=True)
    print ("create log dir:", logdir)

    conf = Config(ab, ims,
                  nbatch_tr,
                  BACKBONE,
                  ENCODER_WEIGHTS,
                  ACTIVATION, lr,
                  model_arch,
                  num_gpus,
                  trial,
                  postfix,
                  loss_type2,
                  loss_param1,
                  loss_param2,
                  logfile,
                  normvalue,
                  is_swa,
                  accumulate_grad,
                  oversampling,
                  )

    print(asdict(conf))

    pipeline = SegmentCell(asdict(conf))

    if debug:
        limit_train_batches = 0.05
        limit_val_batches = 0.05
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0

    trainer = pl.Trainer(logger=mlflow_logger,
                         max_epochs=nepoch,
                         limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         gpus=num_gpus if torch.cuda.is_available() else None,
                         precision=16,
                         accelerator='ddp_spawn',
                         deterministic=True,
                         stochastic_weight_avg=is_swa,
                         auto_select_gpus=True,
                         benchmark=True,
                         sync_batchnorm=True,
                         accumulate_grad_batches=accumulate_grad,
                         callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_dice_epoch")], )
    trainer.fit(pipeline)

    model_file = os.path.join(logdir, "model.ckpt")
    trainer.save_checkpoint(model_file)

    trial.set_user_attr("mlflow_runID", mlflow_logger.run_id)
    trial.set_user_attr("checkpoint_file", model_file)


    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.pytorch.log_model(pipeline.get_model(), "model")

    with open(logfile) as f:
        eval_metric = float(f.readline().rstrip())
    return eval_metric


def main():
    if n_trials == 0:
        objective(None)

    else:
        dd = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        savestudy_cb = SaveStudyCallback(f"/optuna_study/{ab}_{dd}.pkl")

        prev_studies = []
        if resume is not None:
            print (f"resume (load from {resume}")
            study = joblib.load(resume)
            for trial in study.trials:
                runID = trial.user_attrs["mlflow_runID"]
                prev_studies.append(runID)
        else:
            study = optuna.create_study(direction='maximize')

        study.optimize(lambda trial: objective(trial, args), n_trials=n_trials, gc_after_trial=True,
                        callbacks = [savestudy_cb])


if __name__ == '__main__':
    main()
