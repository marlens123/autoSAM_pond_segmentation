"""
Copyright 2023 Xinrong Hu et al. https://github.com/xhu248/AutoSAM

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modified
"""

import os
import argparse
import random
import time
import warnings
import numpy as np

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import JaccardIndex

from loss_functions.dice_loss import SoftDiceLoss

from models.build_autosam_seg_model import sam_seg_model_registry

from torch.utils.data import DataLoader
from .data import Dataset
from .utils import compute_class_weights

import wandb

wandb.login()

parser = argparse.ArgumentParser(description="PyTorch AutoSam Training")

parser.add_argument(
    "--pref", default="default", type=str, help="used for wandb logging"
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=150, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=2,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0005,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument(
    "--model_type", type=str, default="vit_b", help="path to splits file"
)
parser.add_argument(
    "--images_train_dir", type=str, default="data/training/train_images_n_2.npy"
)
parser.add_argument(
    "--masks_train_dir", type=str, default="data/training/train_masks_n_2.npy"
)
parser.add_argument(
    "--images_test_dir", type=str, default="data/training/test_images_n_2.npy"
)
parser.add_argument(
    "--masks_test_dir", type=str, default="data/training/test_masks_n_2.npy"
)
parser.add_argument("--augmentation", default=True, action="store_false")
parser.add_argument(
    "--augment_mode",
    default="2",
    help="1 = flip, crop, rotate, sharpen/blur, 2 = flip, rotate, 3 = sharpen/blur",
)
parser.add_argument(
    "--normalize", default=True, action="store_false", help="z-score normalization"
)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--use_class_weights", default=True, action="store_false")
parser.add_argument(
    "--dropout",
    default=False,
    action="store_true",
    help="if to use dropout in the last layer (prob 0.5)",
)
parser.add_argument(
    '-p', 
    '--print-freq', 
    default=10, 
    type=int,
    metavar='N', 
    help='print frequency (default: 10)'
)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s).")
        else:
            print("CUDA is not available.")

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.model_type == "vit_h":
        model_checkpoint = "segment_anything_checkpoints/sam_vit_h_4b8939.pth"
        model = sam_seg_model_registry[args.model_type](
            num_classes=args.num_classes, checkpoint=model_checkpoint
        )
    elif args.model_type == "vit_l":
        model_checkpoint = "segment_anything_checkpoints/sam_vit_l_0b3195.pth"
        model = sam_seg_model_registry[args.model_type](
            num_classes=args.num_classes, checkpoint=model_checkpoint
        )
    elif args.model_type == "vit_b":
        model_checkpoint = "segment_anything_checkpoints/sam_vit_b_01ec64.pth"
        model = sam_seg_model_registry[args.model_type](
            num_classes=args.num_classes,
            checkpoint=model_checkpoint,
            dropout=args.dropout,
        )

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # freeze weights in the image_encoder
    for name, param in model.named_parameters():
        if param.requires_grad and "image_encoder" in name or "iou" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)

    cudnn.benchmark = True

    # compute class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(args.masks_train_dir)
        # for dice loss component
        class_weights_np = class_weights
        # for cce loss component
        class_weights = torch.from_numpy(class_weights).float().cuda(args.gpu)
    else:
        class_weights = None
        class_weights_np = None

    print("Class weights are...:", class_weights)

    # Data loading code
    train_dataset = Dataset(args, mode="train")
    test_dataset = Dataset(args, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.workers
    )

    args.save_dir = "experiments/" + args.save_dir
    writer = SummaryWriter(os.path.join(args.save_dir, "tensorboard" + str(gpu)))

    best_mp_iou = 0.38

    # wandb setup
    wandb.init(project="sam", group="distribution", name=args.pref, config=args)
    wandb.watch(model, log_freq=2)

    for epoch in range(args.epochs):
        print("EPOCH {}:".format(epoch + 1))

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch)

        # train for one epoch
        train(
            train_loader,
            class_weights,
            model,
            optimizer,
            scheduler,
            epoch,
            args,
            writer,
            class_weights_np=class_weights_np,
        )
        _, mp_iou = validate(test_loader, model, epoch, args, writer)

        # save model when melt pond IoU improved
        if mp_iou > best_mp_iou:
            best_mp_iou = mp_iou
            torch.save(
                model.state_dict(), args.save_dir + "/model_mp_{}.pth".format(epoch)
            )

        if epoch == 145:
            torch.save(
                model.state_dict(), args.save_dir + "/model_mp_{}.pth".format(epoch)
            )


def train(
    train_loader,
    class_weights,
    model,
    optimizer,
    scheduler,
    epoch,
    args,
    writer,
    class_weights_np=None,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    dice_loss = SoftDiceLoss(
        batch_dice=True, do_bg=False, rebalance_weights=class_weights_np
    )
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    # switch to train mode
    model.train()

    end = time.time()
    for i, tup in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            img = tup[0].float().cuda(args.gpu, non_blocking=True)
            label = tup[1].long().cuda(args.gpu, non_blocking=True)
        else:
            img = tup[0].float()
            label = tup[1].long()
        b, _, h, w = img.shape

        # compute output
        mask, iou_pred = model(img)
        mask = mask.view(b, -1, h, w)
        iou_pred = iou_pred.squeeze().view(b, -1)

        pred_softmax = F.softmax(mask, dim=1)
        loss = ce_loss(mask, label.squeeze(1)) + dice_loss(
            pred_softmax, label.squeeze(1)
        )

        jaccard = JaccardIndex(task="multiclass", num_classes=args.num_classes).to(
            args.gpu
        )
        jac = jaccard(torch.argmax(mask, dim=1), label.squeeze(1))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar("train_loss", loss, global_step=i + epoch * len(train_loader))
        writer.add_scalar(
            "train_iou", torch.mean(iou_pred), global_step=i + epoch * len(train_loader)
        )

        if i % args.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\t" "loss {loss:.4f}".format(
                    epoch, i, len(train_loader), loss=loss.item()
                )
            )

    wandb.log({"epoch": epoch, "train_loss": loss})
    wandb.log({"epoch": epoch, "train_jac": jac})

    if epoch >= 10:
        scheduler.step(loss)


def validate(val_loader, model, epoch, args, writer):
    loss_list = []
    jac_list_mp = []
    jac_list_si = []
    jac_list_oc = []
    jac_mean = []
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    model.eval()

    with torch.no_grad():
        for i, tup in enumerate(val_loader):
            if args.gpu is not None:
                img = tup[0].float().cuda(args.gpu, non_blocking=True)
                label = tup[1].long().cuda(args.gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]
            b, _, h, w = img.shape

            # compute output
            mask, iou_pred = model(img)
            mask = mask.view(b, -1, h, w)
            iou_pred = iou_pred.squeeze().view(b, -1)
            iou_pred = torch.mean(iou_pred)

            pred_softmax = F.softmax(mask, dim=1)
            loss = dice_loss(
                pred_softmax, label.squeeze(1)
            )  # self.ce_loss(pred, target.squeeze())
            loss_list.append(loss.item())

            jaccard = JaccardIndex(
                task="multiclass", num_classes=args.num_classes, average=None
            ).to(args.gpu)
            jaccard_mean = JaccardIndex(
                task="multiclass", num_classes=args.num_classes
            ).to(args.gpu)

            jac = jaccard(pred_softmax, label.squeeze(1))
            jac_m = jaccard_mean(pred_softmax, label.squeeze(1))

            jac_list_mp.append(jac[0].item())
            jac_list_si.append(jac[1].item())
            jac_list_oc.append(jac[2].item())
            jac_mean.append(jac_m.item())

            wandb.log({"epoch": epoch, "val_loss_{}".format(i): loss.item()})
            wandb.log({"epoch": epoch, "val_jac_mp_{}".format(i): jac[0].item()})
            wandb.log({"epoch": epoch, "val_jac_si_{}".format(i): jac[1].item()})
            wandb.log({"epoch": epoch, "val_jac_oc_{}".format(i): jac[2].item()})
            wandb.log({"epoch": epoch, "val_jac_{}".format(i): jac_m.item()})

    wandb.log({"epoch": epoch, "val_loss": np.mean(loss_list)})
    wandb.log({"epoch": epoch, "val_jac_mp": np.mean(jac_list_mp)})
    wandb.log({"epoch": epoch, "val_jac_si": np.mean(jac_list_si)})
    wandb.log({"epoch": epoch, "val_jac_oc": np.mean(jac_list_oc)})
    wandb.log({"epoch": epoch, "val_jac": np.mean(jac_mean)})

    print(
        "Validating: Epoch: %2d Loss: %.4f IoU_pred: %.4f"
        % (epoch, np.mean(loss_list), iou_pred.item())
    )
    print("new_iou: " + str(jac))
    writer.add_scalar("val_loss", np.mean(loss_list), epoch)
    writer.add_scalar("val_iou", iou_pred.item(), epoch)
    return np.mean(loss_list), np.mean(jac_list_mp)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

if __name__ == "__main__":
    main()