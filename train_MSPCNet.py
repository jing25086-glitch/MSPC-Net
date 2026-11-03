#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSPC-Net trainer (paper method) — lives under new/, does NOT modify existing code/.

Method components (Sec. 3):
  TAP   : topology-aware deep-global / shallow-local prototype alignment
  DEAC  : dynamic entropy-aware contrastive learning (conf>=0.90)
  SDM-BR: signed-distance-field L2 boundary regularization
  + PatchFusion & train-time geometric consistency (auxiliary)

Run from repo root or from new/:
  python new/train_MSPCNet.py --dataset_name Pancreas --labelnum 6 --gpu 0
  python train_MSPCNet.py --dataset_name ToothFairy --labelnum 12 --gpu 0
"""

import os
import sys
import math
import json
import time
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# ---------- path: prefer new/, then fall back to code/ for dataloaders & test_patch ----------
NEW_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(NEW_DIR), "code")
if NEW_DIR not in sys.path:
    sys.path.insert(0, NEW_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(1, CODE_DIR)
# also allow running with cwd=code
_cwd = os.getcwd()
if os.path.basename(_cwd) == "code" and CODE_DIR not in sys.path:
    sys.path.insert(1, _cwd)

from networks.net_factory import net_factory
from losses import (
    compute_tap_loss,
    global_class_prototypes,
    compute_deac_loss,
    compute_deac_loss_labeled,
    compute_sdm_br_loss,
)
from utils.seg_losses import Binary_dice_loss, gaussian_aux_weight
from utils.mix_utils import generate_patchfusion_mask
from utils.augment_utils import random_tta_augment, get_tta_ensemble_loss
from dataloaders.dataset import *
from utils import test_patch

# ========================= argparse =========================
parser = argparse.ArgumentParser(description="MSPC-Net (paper) trainer")
parser.add_argument("--dataset_name", type=str, default="Pancreas")
parser.add_argument("--root_path", type=str, default="./")
parser.add_argument("--exp", type=str, default="MSPCNet")
parser.add_argument("--max_iteration", type=int, default=15000)
parser.add_argument("--labeled_bs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--labelnum", type=int, default=6)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--deepsup", type=bool, default=True)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--warmup_iter", type=int, default=2000)

# nine auxiliary weights (paper Sec. 3.7): 0.30,0.10,0.10,0.10,0.10,0.05,0.10,0.10,0.05
parser.add_argument("--lamda_teacher", type=float, default=0.30)
parser.add_argument("--lamda_proto", type=float, default=0.10)
parser.add_argument("--lamda_mix", type=float, default=0.10)
parser.add_argument("--lamda_mix_proto", type=float, default=0.10)
parser.add_argument("--lamda_tap", type=float, default=0.10)
parser.add_argument("--lamda_deac", type=float, default=0.05)
parser.add_argument("--lamda_sdm", type=float, default=0.10)
parser.add_argument("--patchfusion_weight", type=float, default=0.10)
parser.add_argument("--lambda_tta", type=float, default=0.05)

parser.add_argument("--num_proto", type=int, default=4)
parser.add_argument("--use_tap", type=int, default=1)
parser.add_argument("--use_deac", type=int, default=1)
parser.add_argument("--use_sdm", type=int, default=1)
parser.add_argument("--use_patchfusion", type=int, default=1)
parser.add_argument("--use_tta_train", type=int, default=1)
parser.add_argument("--tta_consistency", type=int, default=1)
parser.add_argument("--tta_scale_min", type=float, default=0.9)
parser.add_argument("--tta_scale_max", type=float, default=1.1)
parser.add_argument("--tta_mode", type=str, default="kl")
parser.add_argument("--patchfusion_prob", type=float, default=0.7)
parser.add_argument("--patchfusion_size", type=int, default=64)
parser.add_argument("--patchfusion_conf_thresh", type=float, default=0.9)
parser.add_argument("--deac_conf_thresh", type=float, default=0.90)
parser.add_argument("--deac_tau0", type=float, default=0.10)
parser.add_argument("--tap_sigma", type=float, default=0.5)
parser.add_argument("--sdm_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
parser.add_argument("--ema_decay", type=float, default=0.99)

args = parser.parse_args()

alpha = 0.99
num_classes = 2
dynamic_threshold_bg = [1 / num_classes for _ in range(4)]
dynamic_threshold_fg = [1 / num_classes for _ in range(4)]
plt_bg, plt_fg = {}, {}

# resolve data / results paths relative to code/ when root is ./
_root = args.root_path
if _root in ("./", "."):
    _root = CODE_DIR + os.sep

snapshot_path = (
    f"{_root}results/{args.dataset_name}_{args.exp}_{args.labelnum}_labeled_"
    f"{args.base_lr}_{alpha}/"
    f"TAP{args.use_tap}_DEAC{args.use_deac}_SDM{args.use_sdm}_"
    f"PF{args.use_patchfusion}_TTA{args.use_tta_train}_"
    f"{args.num_proto}_{args.lamda_tap}_{args.lamda_deac}_{args.lamda_sdm}"
)

if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = _root + "data/LA"
    args.max_samples = 80
elif args.dataset_name == "Pancreas":
    patch_size = (96, 96, 96)
    args.root_path = _root + "data/Pancreas"
    args.max_samples = 62
elif args.dataset_name == "Nurves":
    patch_size = (160, 128, 112)
    args.root_path = _root + "data/Nurves"
    args.max_samples = 90
elif args.dataset_name == "ToothFairy":
    patch_size = (128, 128, 128)
    args.root_path = _root + "data/ToothFairy_Dataset/alpha"
    _tf_lists = os.path.join(os.path.dirname(args.root_path), "lists", "train.list")
    if os.path.exists(_tf_lists):
        with open(_tf_lists, "r") as _f:
            args.max_samples = len([l for l in _f if l.strip()])
    else:
        args.max_samples = len(
            [
                p
                for p in os.listdir(args.root_path)
                if os.path.isdir(os.path.join(args.root_path, p))
                and os.path.exists(os.path.join(args.root_path, p, "gt_alpha.npy"))
            ]
        )
else:
    raise ValueError(args.dataset_name)

train_data_path = args.root_path
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


def convert_to_soft_label(deep_label):
    foreground_prob = deep_label.unsqueeze(1)
    background_prob = 1 - foreground_prob
    return torch.cat([background_prob, foreground_prob], dim=1)


def custom_cross_entropy(output, target, certain_index=None):
    target = convert_to_soft_label(target)
    log_output = F.log_softmax(output, dim=1)
    if certain_index is None:
        return F.kl_div(log_output, target, reduction="mean")
    certain_index = certain_index.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    return F.kl_div(log_output, target, reduction="none")[certain_index].mean()


def get_loss(outputs, label_batch, deepsup, if_pseudo=False, if_soft=False, uncertain_area_all=None):
    dice_loss = Binary_dice_loss
    if deepsup:
        loss_seg, loss_seg_dice = 0, 0
        for i in range(len(outputs)):
            output = outputs[i]
            if if_pseudo:
                deep_label = label_batch[i].long()
            elif if_soft:
                deep_label = F.interpolate(
                    label_batch.float().unsqueeze(1), size=output.shape[2:], mode="trilinear"
                ).float()[:, 0]
            else:
                deep_label = F.interpolate(
                    label_batch.float().unsqueeze(1), size=output.shape[2:], mode="nearest"
                ).long()[:, 0]
            if if_pseudo:
                if uncertain_area_all is None:
                    loss_seg = loss_seg + F.cross_entropy(output, deep_label)
                else:
                    certain_index = uncertain_area_all[i] == 0
                    loss_seg = loss_seg + F.cross_entropy(output, deep_label, reduction="none")[certain_index].mean()
            elif if_soft:
                if uncertain_area_all is None:
                    loss_seg = loss_seg + custom_cross_entropy(output, deep_label)
                else:
                    certain_index = uncertain_area_all[i] == 0
                    loss_seg = loss_seg + custom_cross_entropy(output, deep_label, certain_index)
            else:
                loss_seg = loss_seg + F.cross_entropy(output, deep_label)

            output_soft = F.softmax(output, dim=1)
            if uncertain_area_all is not None:
                certain_index = uncertain_area_all[i] == 0
                loss_seg_dice = loss_seg_dice + dice_loss(
                    output_soft[:, 1][certain_index], deep_label[certain_index]
                )
            else:
                loss_seg_dice = loss_seg_dice + dice_loss(output_soft[:, 1], deep_label)
        return loss_seg / len(outputs), loss_seg_dice / len(outputs)
    loss_seg = F.cross_entropy(outputs, label_batch)
    outputs_soft = F.softmax(outputs, dim=1)
    return loss_seg, Binary_dice_loss(outputs_soft[:, 1], label_batch == 1)


def get_prototype(features, masks, argmax=False, if_bg=True, if_soft=False):
    prototypes = []
    for i, feature in enumerate(features):
        if argmax:
            tmp_mask = torch.argmax(masks[-(i + 1)], dim=1).unsqueeze(1)
        elif if_soft:
            tmp_mask = F.interpolate(
                masks.float().unsqueeze(1), size=feature.shape[2:], mode="trilinear"
            ).float()
        else:
            tmp_mask = F.interpolate(
                masks.float().unsqueeze(1), size=feature.shape[2:], mode="nearest"
            ).long()
        feature_tgt = feature * tmp_mask
        feature_tgt = feature_tgt.sum(dim=(2, 3, 4)) / tmp_mask.sum(dim=(2, 3, 4)).clamp_min(1e-6)
        proto = torch.mean(feature_tgt, dim=0)
        if if_bg:
            tmp_mask_bg = 1 - tmp_mask
            feature_bg = feature * tmp_mask_bg
            feature_bg = feature_bg.sum(dim=(2, 3, 4)) / tmp_mask_bg.sum(dim=(2, 3, 4)).clamp_min(1e-6)
            feature_bg = torch.mean(feature_bg, dim=0)
            proto = torch.cat([feature_bg.unsqueeze(0), proto.unsqueeze(0)], dim=0)
        prototypes.append(proto)
    return prototypes


def get_prototype_loss(prototypes, prototypes_unlabeled, num_prototypes):
    prototypes = prototypes[::-1]
    prototypes_unlabeled = prototypes_unlabeled[::-1]
    loss_prototype = []
    for i in range(num_prototypes):
        loss_tmp = F.mse_loss(prototypes[i], prototypes_unlabeled[i])
        if not torch.isnan(loss_tmp):
            loss_prototype.append(loss_tmp)
    if len(loss_prototype) > 0:
        return sum(loss_prototype) / len(loss_prototype)
    return torch.tensor(0.0, device=prototypes[0].device)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_param.data * alpha + (1 - alpha) * param.data
    return ema_model


def get_eta_time(start_time, iter_num, total_iter):
    elapsed = time.time() - start_time
    est = (elapsed / max(iter_num, 1)) * (total_iter - iter_num)
    h, rem = divmod(est, 3600)
    m, s = divmod(rem, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(h), int(m), int(s))


def _safe_add(loss, term, weight):
    if isinstance(term, torch.Tensor) and term.requires_grad and (not torch.isnan(term)) and term.item() != 0:
        return loss + weight * term / term.detach()
    return loss


class _InferWrapper(nn.Module):
    """Make MSPCNet look like a plain logits model for code/utils/test_patch.py."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        (logits, _feats), _sdm = self.net(x, deepsup=False)
        return logits


if __name__ == "__main__":
    os.makedirs(snapshot_path, exist_ok=True)
    writer = SummaryWriter(snapshot_path + "/log")
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("MSPC-Net trainer from new/ — existing code/ untouched")

    model = net_factory(net_type="MSPCNet", in_chns=1, class_num=num_classes, mode="train", use_sdm_head=bool(args.use_sdm))
    model_teacher = net_factory(net_type="MSPCNet", in_chns=1, class_num=num_classes, mode="train", use_sdm_head=bool(args.use_sdm))
    for p in model_teacher.parameters():
        p.detach_()

    if args.dataset_name == "LA":
        db_train = LAHeart(
            base_dir=train_data_path,
            split="train",
            transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size), ToTensor()]),
        )
    elif args.dataset_name == "Pancreas":
        db_train = Pancreas(
            base_dir=train_data_path,
            split="train",
            transform=transforms.Compose([RandomCrop(patch_size), ToTensor()]),
        )
    elif args.dataset_name == "Nurves":
        db_train = Nurves(
            base_dir=train_data_path,
            split="train",
            transform=transforms.Compose([RandomCrop(patch_size), ToTensor()]),
        )
    elif args.dataset_name == "ToothFairy":
        db_train = ToothFairy(
            base_dir=train_data_path,
            split="train",
            transform=transforms.Compose([RandomCrop(patch_size), ToTensor()]),
        )

    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs
    )
    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True
    )
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice = 0
    start_time = time.time()
    max_epoch = max_iterations // len(trainloader) + 1
    spacing = tuple(args.sdm_spacing)

    for epoch_num in range(max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_labeled, volume_unlabeled = volume_batch[:labeled_bs], volume_batch[labeled_bs:]
            model.train()

            (outputs_labeled, features_labeled), sdm_l = model(volume_labeled, args.deepsup)
            (outputs_unlabeled, features_unlabeled), sdm_u = model(volume_unlabeled, args.deepsup)
            loss_seg, loss_seg_dice = get_loss(outputs_labeled, label_batch[:labeled_bs], args.deepsup)
            loss_supervised = (loss_seg + loss_seg_dice) / 2

            # ----- train-time geometric TTA -----
            loss_tta_supervised = torch.tensor(0.0, device=volume_labeled.device)
            loss_tta_consistency = torch.tensor(0.0, device=volume_labeled.device)
            if args.use_tta_train:
                vol_tta, lab_tta = random_tta_augment(
                    volume_labeled,
                    label_batch[:labeled_bs],
                    scale_min=args.tta_scale_min,
                    scale_max=args.tta_scale_max,
                )
                (out_tta, _), _ = model(vol_tta, args.deepsup)
                ls_t, ld_t = get_loss(out_tta, lab_tta, args.deepsup)
                loss_tta_supervised = (ls_t + ld_t) / 2
                if args.tta_consistency:
                    loss_tta_consistency = get_tta_ensemble_loss(outputs_labeled, out_tta, mode=args.tta_mode)

            # ----- teacher -----
            with torch.no_grad():
                (outputs_teacher, features_teacher), _ = model_teacher(volume_unlabeled, args.deepsup)
            pseudo_label = [torch.argmax(o, dim=1) for o in outputs_teacher]
            outputs_teacher_soft = [F.softmax(o, dim=1).detach().cpu() for o in outputs_teacher]
            pseudo_label_bg = [
                (s[:, 0] > dynamic_threshold_bg[i]).long() for i, s in enumerate(outputs_teacher_soft)
            ]
            pseudo_label_fg = [
                (s[:, 1] > dynamic_threshold_fg[i]).long() for i, s in enumerate(outputs_teacher_soft)
            ]
            uncertain_area_all = []
            for i in range(len(outputs_teacher_soft)):
                uncertain_area_all.append((pseudo_label_bg[i] + pseudo_label_fg[i]) == 0)
            loss_seg_pseudo, loss_seg_dice_pseudo = get_loss(
                outputs_unlabeled, pseudo_label, args.deepsup, if_pseudo=True, uncertain_area_all=uncertain_area_all
            )
            loss_consistency = (loss_seg_pseudo + loss_seg_dice_pseudo) / 2

            # base prototypes (for mix / logging)
            prototype_labeled = get_prototype(features_labeled, label_batch[:labeled_bs])
            prototype_unlabeled = get_prototype(features_unlabeled, outputs_teacher, argmax=True)
            loss_proto = get_prototype_loss(prototype_labeled, prototype_unlabeled, args.num_proto)

            # ----- PatchFusion -----
            loss_patchfusion = torch.tensor(0.0, device=volume_labeled.device)
            if (
                args.use_patchfusion
                and volume_labeled.size(0) > 0
                and volume_unlabeled.size(0) > 0
                and random.random() < args.patchfusion_prob
            ):
                min_pf = min(volume_labeled.size(0), volume_unlabeled.size(0))
                vl, vu = volume_labeled[:min_pf], volume_unlabeled[:min_pf]
                lab_pf = label_batch[:min_pf]
                t_logits = outputs_teacher[-1][:min_pf].detach()
                t_prob = torch.softmax(t_logits, dim=1)
                conf_pf, pseudo_pf = torch.max(t_prob, dim=1)
                conf_mask = (conf_pf > args.patchfusion_conf_thresh).float().unsqueeze(1)
                patch_mask = generate_patchfusion_mask(vl.shape, args.patchfusion_size, device=vl.device)
                x_fused = patch_mask * vl + (1.0 - patch_mask) * vu
                y_fused = torch.where(patch_mask.squeeze(1).bool(), lab_pf.long(), pseudo_pf.long())
                conf_patch = patch_mask + (1.0 - patch_mask) * conf_mask
                (out_pf, feat_pf), sdm_pf = model(x_fused, args.deepsup)
                logits_pf = out_pf[-1] if args.deepsup else out_pf
                ce = F.cross_entropy(logits_pf, y_fused, reduction="none")
                loss_pf_ce = (ce * conf_patch.squeeze(1)).mean()
                loss_pf_sdm = torch.tensor(0.0, device=vl.device)
                if args.use_sdm and sdm_pf is not None:
                    loss_pf_sdm = compute_sdm_br_loss(sdm_pf, y_fused, num_fg=1, spacing=spacing)
                loss_patchfusion = (loss_pf_ce + 0.5 * loss_pf_sdm) * args.patchfusion_weight

            # ----- intensity mix -----
            fused_volume, fused_label = [], []
            for i in range(labeled_bs):
                w = random.uniform(0.25, 0.75)
                fused_volume.append(w * volume_labeled[i] + (1 - w) * volume_unlabeled[i])
                fused_label.append(w * label_batch[i] + (1 - w) * pseudo_label[-1][i].float())
            fused_volume = torch.stack(fused_volume, dim=0)
            fused_label = torch.stack(fused_label, dim=0)
            (outputs_fused, features_fused), _ = model(fused_volume, args.deepsup)
            loss_mix_seg, loss_mix_seg_dice = get_loss(
                outputs_fused, fused_label, args.deepsup, if_soft=True, uncertain_area_all=uncertain_area_all
            )
            loss_mix = (loss_mix_seg + loss_mix_seg_dice) / 2
            prototype_mix = get_prototype(features_fused, fused_label, if_soft=True)
            loss_mix_proto = get_prototype_loss(prototype_labeled, prototype_mix, args.num_proto)

            # ----- TAP -----
            loss_tap = torch.tensor(0.0, device=volume_labeled.device)
            if args.use_tap:
                try:
                    loss_tap = compute_tap_loss(
                        feats_labeled=features_labeled,
                        feats_unlabeled=features_unlabeled,
                        mask_labeled=label_batch[:labeled_bs],
                        mask_unlabeled=pseudo_label[-1],
                        deep_indices=(2, 3),
                        shallow_indices=(0, 1),
                        sigma=args.tap_sigma,
                        num_classes=num_classes,
                    )
                except Exception as e:
                    logging.warning(f"TAP failed: {e}")
                    loss_tap = torch.tensor(0.0, device=volume_labeled.device)

            # ----- DEAC -----
            loss_deac = torch.tensor(0.0, device=volume_labeled.device)
            if args.use_deac:
                try:
                    proto_l = global_class_prototypes(features_labeled[0], label_batch[:labeled_bs], num_classes)
                    loss_deac_l = compute_deac_loss_labeled(
                        features_labeled[0], label_batch[:labeled_bs], proto_l, tau0=args.deac_tau0, num_classes=num_classes
                    )
                    loss_deac_u = compute_deac_loss(
                        feat=features_unlabeled[0],
                        teacher_logits=outputs_teacher[-1],
                        prototypes=proto_l.detach(),
                        conf_thresh=args.deac_conf_thresh,
                        tau0=args.deac_tau0,
                        num_classes=num_classes,
                    )
                    loss_deac = loss_deac_l + loss_deac_u
                except Exception as e:
                    logging.warning(f"DEAC failed: {e}")
                    loss_deac = torch.tensor(0.0, device=volume_labeled.device)

            # ----- SDM-BR -----
            loss_sdm = torch.tensor(0.0, device=volume_labeled.device)
            if args.use_sdm and sdm_l is not None:
                loss_sdm_l = compute_sdm_br_loss(
                    sdm_l, label_batch[:labeled_bs], num_fg=1, spacing=spacing
                )
                loss_sdm_u = compute_sdm_br_loss(
                    sdm_u, pseudo_label[-1].detach(), num_fg=1, spacing=spacing
                )
                loss_sdm = loss_sdm_l + loss_sdm_u

            # ----- total (normalized terms + Gaussian schedule after warmup) -----
            g = gaussian_aux_weight(iter_num, max_iterations)
            loss = loss_supervised / loss_supervised.detach()

            if args.use_tta_train:
                loss = _safe_add(loss, loss_tta_supervised, 0.5 * g)
                if args.tta_consistency:
                    loss = _safe_add(loss, loss_tta_consistency, args.lambda_tta * g)
            if args.use_tap:
                loss = _safe_add(loss, loss_tap, args.lamda_tap * g)
            if args.use_deac:
                loss = _safe_add(loss, loss_deac, args.lamda_deac * g)
            if args.use_sdm:
                loss = _safe_add(loss, loss_sdm, args.lamda_sdm * g)
            if args.use_patchfusion:
                loss = _safe_add(loss, loss_patchfusion, g)

            if iter_num >= args.warmup_iter:
                loss = _safe_add(loss, loss_consistency, args.lamda_teacher * g)
                loss = _safe_add(loss, loss_mix, args.lamda_mix * g)
                loss = _safe_add(loss, loss_proto, args.lamda_proto * g)
                loss = _safe_add(loss, loss_mix_proto, args.lamda_mix_proto * g)

                # EMA dynamic threshold (same as DuCiSC baseline)
                mean_prob_bg, mean_prob_fg = [], []
                current_unlabeled_soft = [F.softmax(o, dim=1).cpu() for o in outputs_unlabeled]
                for i in range(len(current_unlabeled_soft)):
                    index_gt_fg = (pseudo_label[i] == 1).cpu()
                    index_gt_bg = (pseudo_label[i] == 0).cpu()
                    mean_prob_fg.append(
                        current_unlabeled_soft[i][:, 1][index_gt_fg].mean().item() if index_gt_fg.sum() > 0 else 0
                    )
                    mean_prob_bg.append(
                        current_unlabeled_soft[i][:, 0][index_gt_bg].mean().item() if index_gt_bg.sum() > 0 else 0
                    )
                for i in range(len(dynamic_threshold_bg)):
                    if mean_prob_bg[i] != 0:
                        dynamic_threshold_bg[i] = dynamic_threshold_bg[i] * alpha + (1 - alpha) * mean_prob_bg[i]
                        dynamic_threshold_bg[i] = min(0.95, max(0.5, dynamic_threshold_bg[i]))
                    if mean_prob_fg[i] != 0:
                        dynamic_threshold_fg[i] = dynamic_threshold_fg[i] * alpha + (1 - alpha) * mean_prob_fg[i]
                        dynamic_threshold_fg[i] = min(0.95, max(0.5, dynamic_threshold_fg[i]))
                    plt_bg[iter_num] = list(dynamic_threshold_bg)
                    plt_fg[iter_num] = list(dynamic_threshold_fg)

            loss_all = (
                loss_supervised
                + args.lamda_teacher * loss_consistency
                + args.lamda_mix * loss_mix
                + args.lamda_proto * (loss_proto if isinstance(loss_proto, torch.Tensor) else 0)
                + args.lamda_mix_proto * (loss_mix_proto if isinstance(loss_mix_proto, torch.Tensor) else 0)
                + args.lamda_tap * loss_tap
                + args.lamda_deac * loss_deac
                + args.lamda_sdm * loss_sdm
                + loss_patchfusion
                + 0.5 * loss_tta_supervised
                + args.lambda_tta * loss_tta_consistency
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, model_teacher, args.ema_decay, iter_num)
            iter_num += 1

            writer.add_scalar("train/loss_all", float(loss_all.item()) if isinstance(loss_all, torch.Tensor) else loss_all, iter_num)
            writer.add_scalar("train/loss_supervised", loss_supervised.item(), iter_num)
            writer.add_scalar("train/loss_tap", float(loss_tap), iter_num)
            writer.add_scalar("train/loss_deac", float(loss_deac), iter_num)
            writer.add_scalar("train/loss_sdm", float(loss_sdm), iter_num)
            writer.add_scalar("train/gauss_w", g, iter_num)

            if iter_num % 200 == 0:
                logging.info(
                    f"iter {iter_num}: loss={float(loss_all):.4f} "
                    f"sup={loss_supervised.item():.4f} tap={float(loss_tap):.4f} "
                    f"deac={float(loss_deac):.4f} sdm={float(loss_sdm):.4f} "
                    f"cons={loss_consistency.item():.4f} g={g:.3f} "
                    f"eta={get_eta_time(start_time, iter_num, max_iterations)}"
                )

            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                # test_patch uses relative ./data paths — evaluate from CODE_DIR
                _prev_cwd = os.getcwd()
                try:
                    os.chdir(CODE_DIR)
                    eval_model = _InferWrapper(model)
                    if args.dataset_name == "LA":
                        dice_sample, _, _, _ = test_patch.var_all_case(
                            eval_model, num_classes=num_classes, patch_size=patch_size,
                            stride_xy=18, stride_z=4, dataset_name="LA",
                        )
                    elif args.dataset_name == "Pancreas":
                        dice_sample, _, _, _ = test_patch.var_all_case(
                            eval_model, num_classes=num_classes, patch_size=patch_size,
                            stride_xy=16, stride_z=16, dataset_name="Pancreas",
                        )
                    elif args.dataset_name == "Nurves":
                        dice_sample, _, _, _ = test_patch.var_all_case(
                            eval_model, num_classes=num_classes, patch_size=patch_size,
                            stride_xy=16, stride_z=16, dataset_name="Nurves",
                        )
                    elif args.dataset_name == "ToothFairy":
                        dice_sample, _, _, _ = test_patch.var_all_case(
                            eval_model, num_classes=num_classes, patch_size=patch_size,
                            stride_xy=16, stride_z=16, dataset_name="ToothFairy",
                        )
                    else:
                        dice_sample = 0
                finally:
                    os.chdir(_prev_cwd)
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    torch.save(model.state_dict(), os.path.join(snapshot_path, f"{args.exp}_best_model.pth"))
                    logging.info(f"save best model dice={best_dice*100:.2f}")
                logging.info(f"iter {iter_num}, dice=[{dice_sample*100:.2f}/{best_dice*100:.2f}]")
                model.train()

            if iter_num >= max_iterations:
                torch.save(model.state_dict(), os.path.join(snapshot_path, f"iter_{iter_num}.pth"))
                break
        if iter_num >= max_iterations:
            break

    import pickle
    with open(os.path.join(snapshot_path, "ema_data.pkl"), "wb") as f:
        pickle.dump({"bg": plt_bg, "fg": plt_fg}, f)
    logging.info("Training finished.")
