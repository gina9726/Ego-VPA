# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import time
import shutil
import pdb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from sklearn.metrics import confusion_matrix
import wandb

from lavila.data import datasets
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models import models, loss
from lavila.models.tokenizer import (MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer)
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
from lavila.utils.config import load_cfg
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler
from lavila.utils.evaluation_charades import charades_map
from lavila.utils.evaluation_ek100mir import (calculate_k_counts, calculate_IDCG, calculate_mAP, calculate_nDCG)
from lavila.utils.evaluation import accuracy, get_mean_accuracy

def get_args_parser():
    parser = argparse.ArgumentParser(description='lavila finetune and evaluation', add_help=False)
    parser.add_argument('--config',
                        default='configs/charades_ego/finetune.yml',
                        type=str, help='path to config file')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--save_results', action='store_true', help='save results')
    parser.add_argument('--debug', action='store_true', help='debug')
    # System
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    return parser


def main(args, cfg):
    dist_utils.init_distributed_mode(args)

    random_seed(args.seed, dist_utils.get_rank())

    if cfg['model'].get('pretrain', False):
        ckpt_path = cfg['model']['pretrain']
    else:
        raise Exception('no checkpoint found')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    state_dict = OrderedDict()
    skip_load = cfg['model'].pop('skip_load', [])
    for k, v in ckpt['state_dict'].items():
        if any([(x in k) for x in skip_load]):
            state_dict.pop(k, None)
        else:
            state_dict[k.replace('module.', '')] = v

    old_args = vars(ckpt['args'])
    arch = old_args.get('model', 'CLIP_OPENAI_TIMESFORMER_BASE')
    print("=> creating model: {}".format(arch))
    model = getattr(models, arch)(
        pretrained=old_args.get('load_visual_pretrained', None),
        pretrained2d=old_args.get('load_visual_pretrained', None) is not None,
        text_use_cls_token=old_args.get('use_cls_token', False),
        project_embed_dim=old_args.get('project_embed_dim', 256),
        timesformer_gated_xattn=False,
        num_frames=cfg['model'].get('num_frames', cfg['data']['clip_length']),
        model_cfg=cfg['model']
    )
    model.logit_scale.requires_grad = False
    model.cuda(args.gpu)
    if ('TIMESFORMER' in arch or 'EGOVLP' in arch) and cfg['model'].get('inflat_posemb', True):
        # inflate weight
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=cfg['model'].get('num_frames', cfg['data']['clip_length']),
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded resume checkpoint '{}'".format(ckpt_path))

    train_cfg = cfg['training']
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=train_cfg['find_unused_parameters']
        )

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": train_cfg['wd']},
                    {"params": p_non_wd, "weight_decay": 0}]

    if train_cfg['use_zero']:
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=torch.optim.SGD if train_cfg['use_sgd'] else torch.optim.AdamW,
            lr=train_cfg['lr'], betas=train_cfg['betas'], eps=train_cfg['eps'], weight_decay=train_cfg['wd']
        )
    else:
        if train_cfg['use_sgd']:
            optimizer = torch.optim.SGD(optim_params, lr=train_cfg['lr'], momentum=train_cfg['betas'][0], weight_decay=train_cfg['wd'])
        else:
            optimizer = torch.optim.AdamW(optim_params, lr=train_cfg['lr'], betas=train_cfg['betas'],
                                          eps=train_cfg['eps'], weight_decay=train_cfg['wd'])

    scaler = amp.GradScaler(enabled=not train_cfg['disable_amp'])
    # optionally resume from a checkpoint (takes precedence over autoresume)
    best_score = 0
    latest = os.path.join(cfg['output_dir'], 'checkpoint.pt')
    resume = cfg['model']['resume']
    if os.path.isfile(latest) and not args.evaluate:
        resume = ''
    if resume:
        if os.path.isfile(resume):
            print("=> loading resume checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            train_cfg['start_epoch'] = epoch
            if not args.distributed:
                state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k.replace('module.', '')] = v
                result = model.load_state_dict(state_dict, strict=False)
            else:
                result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_score = checkpoint.get('best_score', 0)
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(cfg['output_dir'], 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            train_cfg['start_epoch'] = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_score = latest_checkpoint.get('best_score', 0)
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True
    
    # Data loading code
    print("=> creating dataset")
    if arch.endswith('DISTILBERT_BASE'):
        tokenizer = MyDistilBertTokenizer('distilbert-base-uncased')
    elif arch.endswith('BERT_BASE'):
        tokenizer = MyBertTokenizer('bert-base-uncased')
    elif arch.endswith('BERT_LARGE'):
        tokenizer = MyBertTokenizer('bert-large-uncased')
    elif arch.endswith('GPT2'):
        tokenizer = MyGPT2Tokenizer('gpt2')
    elif arch.endswith('GPT2_MEDIUM'):
        tokenizer = MyGPT2Tokenizer('gpt2-medium')
    elif arch.endswith('GPT2_LARGE'):
        tokenizer = MyGPT2Tokenizer('gpt2-large')
    elif arch.endswith('GPT2_XL'):
        tokenizer = MyGPT2Tokenizer('gpt2-xl')
    else:
        print("Using SimpleTokenizer because of model '{}'. "
              "Please check if this is what you want".format(arch))
        tokenizer = SimpleTokenizer()

    dataset = cfg['data']['dataset']
    if dataset == 'ek100_mir':
        criterion = loss.MaxMarginRankingLoss(margin=0.2, fix_norm=True).cuda(args.gpu)
    elif dataset == 'charades_ego' or dataset == 'egtea':
        criterion = loss.CLIPLoss(
            use_vissl=True,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size
        )

    crop_size = 224 if '336PX' not in arch else 336
    if dataset == 'egtea':
        transforms_list = [
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        transforms_list = [
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        ]
    if 'OPENAI' in arch:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
    else:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
    train_transform = transforms.Compose(transforms_list)

    if dataset == 'egtea':
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in arch else
             transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
            TemporalCrop(frames_per_clip=cfg['data']['clip_length'], stride=cfg['data']['clip_length']),
            SpatialCrop(crop_size=crop_size, num_crops=cfg['data']['num_crops'])
        ])
    else:
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in arch else
             transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
        ])

    # build dataset
    cfg['model']['arch'] = arch
    cfg['model']['norm_embed'] = old_args.get('norm_embed', True)
    data_cfg = cfg['data']
    if dataset == 'ek100_mir':
        train_dataset = datasets.get_dataset(train_transform, tokenizer, cfg, is_training=True)
        val_dataset = datasets.get_dataset(val_transform, tokenizer, cfg, is_training=False)
    elif dataset == 'charades_ego':
        if data_cfg.get('trimmed', True):
            train_dataset = datasets.VideoCaptionDatasetCLIP(
                'charades_ego_trimmed', data_cfg['root'], data_cfg['metadata'],
                transform=train_transform, is_training=True, tokenizer=tokenizer,
                clip_length=data_cfg['clip_length'], clip_stride=data_cfg['clip_stride']
            )
        else:
            train_dataset = datasets.VideoCaptionDatasetCLIP(
                'charades_ego', data_cfg['root'], data_cfg['metadata'],
                transform=train_transform, is_training=True, tokenizer=tokenizer,
                clip_length=data_cfg['clip_length'], clip_stride=data_cfg['clip_stride']
            )
        labels, mapping_vn2act = generate_label_map(dataset)
        val_dataset = datasets.VideoClassyDataset(
            dataset, data_cfg['root'], data_cfg['metadata_val'],
            transform=val_transform, is_training=False,
            label_mapping=mapping_vn2act, is_trimmed=False,
            num_clips=1, clip_length=data_cfg['clip_length'], clip_stride=data_cfg['clip_stride'],
            sparse_sample=data_cfg['sparse_sample']
        )
    elif dataset == 'egtea':
        train_dataset = datasets.VideoCaptionDatasetCLIP(
                dataset, data_cfg['root'], data_cfg['metadata'],
                transform=train_transform, is_training=True, tokenizer=tokenizer,
                clip_length=data_cfg['clip_length'], clip_stride=data_cfg['clip_stride']
            )
        labels, mapping_vn2act = generate_label_map(dataset)
        val_dataset = datasets.VideoClassyDataset(
            dataset, data_cfg['root'], data_cfg['metadata_val'],
            transform=val_transform, is_training=False,
            label_mapping=mapping_vn2act, is_trimmed=False,
            num_clips=data_cfg['num_clips'], clip_length=data_cfg['clip_length'], clip_stride=data_cfg['clip_stride'],
            sparse_sample=data_cfg['sparse_sample']
        )
    else:
        raise NotImplementedError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)  # disable distributed
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_cfg['batch_size'], shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )
    print('len(train_loader) = {}'.format(len(train_loader)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_cfg['batch_size'], shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))

    if args.evaluate:
        if dataset == 'ek100_mir':
            _ = validate_mir(val_loader, model, criterion, args, cfg)
        elif dataset == 'charades_ego' or dataset == 'egtea':
            _ = validate_cls(val_loader, ['{}'], labels, model, tokenizer, args, cfg)
        return

    if train_cfg['fix_lr']:
        lr_schedule = None
    else:
        lr_schedule = cosine_scheduler(
            train_cfg['lr'], train_cfg['lr_end'], train_cfg['epoch'], len(train_loader) // train_cfg['update_freq'],
            warmup_epochs=train_cfg['warmup_epochs'], start_warmup_value=train_cfg['lr_start'],
        )

    if dist_utils.is_main_process() and args.wandb:
        shutil.copy(args.config, cfg['output_dir'])
        wandb_id = os.path.split(cfg['output_dir'])[-1]
        if dataset == 'ek100_mir':
            wandb.init(project='EK100_mir', id=wandb_id, config=args, resume='allow')
        elif dataset == 'egtea':
            wandb.init(project='EGTEA', id=wandb_id, config=args, resume='allow')
        else:
            wandb.init(project='CharadesEgo', id=wandb_id, config=args, resume='allow')

    print(args)
    print(cfg)

    #print("=> zero-shot testing")
    #if dataset == 'ek100_mir':
    #    _ = validate_mir(val_loader, model, criterion, args, cfg)
    #elif dataset == 'charades_ego' or dataset == 'egtea':
    #    _ = validate_cls(val_loader, ['{}'], labels, model, tokenizer, args, cfg)

    print("=> beginning training")
    for epoch in range(train_cfg['start_epoch'], train_cfg['epoch']):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg)

        is_epoch = ((epoch + 1) % train_cfg['save_freq']) == 0

        print('=> saving checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_score': best_score,
            'args': args,
        }, False, cfg['output_dir'], is_epoch=is_epoch)

        if (epoch + 1) % train_cfg['eval_freq'] != 0:
            continue

        if dataset == 'ek100_mir':
            val_stats = validate_mir(val_loader, model, criterion, args, cfg)
            score = (val_stats['avg_mAP'] + val_stats['avg_nDCG']) / 2
            if score > best_score:
                best_score = score
                print(f"=> saving checkpoint (best mAP: {val_stats['avg_mAP']}, best nDCG: {val_stats['avg_nDCG']})")
                dist_utils.save_on_master({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_score': best_score,
                    'best_mAP': {k: v for k, v in val_stats.items() if 'mAP' in k},
                    'best_nDCG': {k: v for k, v in val_stats.items() if 'nDCG' in k},
                    'args': args,
                }, True, cfg['output_dir'], is_epoch=False)

        elif dataset == 'charades_ego':
            val_stats = validate_cls(val_loader, ['{}'], labels, model, tokenizer, args, cfg)
            if val_stats['mAP'] > best_score:
                best_score = val_stats['mAP']
                print(f'=> saving checkpoint (best mAP: {best_score})')
                dist_utils.save_on_master({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_mAP': best_score,
                    'best_score': best_score,
                    'args': args,
                }, True, cfg['output_dir'], is_epoch=False)

        elif dataset == 'egtea':
            val_stats = validate_cls(val_loader, ['{}'], labels, model, tokenizer, args, cfg)
            score = (val_stats['mAcc'] + val_stats['acc1']) / 2
            if score > best_score:
                best_score = score
                print(f"=> saving checkpoint (best mAcc: {val_stats['mAcc']}, best Top-1 Acc: {val_stats['acc1']})")
                dist_utils.save_on_master({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_mAcc': val_stats['mAcc'],
                    'best_Top1': val_stats['acc1'],
                    'best_score': best_score,
                    'args': args,
                }, True, cfg['output_dir'], is_epoch=False)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if dist_utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(cfg['output_dir'], 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    dataset = cfg['data']['dataset']
    if dataset == 'ek100_mir':
        metric_names = ['loss', 'max_margin_loss']
    elif dataset == 'charades_ego' or dataset == 'egtea':
        metric_names = models.get_metric_names(cfg['model']['arch'])
    update_freq = cfg['training']['update_freq']
    iters_per_epoch = len(train_loader) // update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
        relevancies = inputs.pop()

        # compute output
        with amp.autocast(enabled=not cfg['training']['disable_amp']):
            gamma = 1.0 / (1 + np.exp(float(-epoch + cfg['training']['epoch']//2)))
            outputs = model(
                *inputs,
                use_checkpoint=cfg['training']['use_checkpoint'],
                norm_embed=cfg['model']['norm_embed'],
                istrain=True,
                gamma=gamma
            )
            if dataset == 'ek100_mir':
                loss_dict = criterion(outputs, weight=relevancies)
            elif dataset == 'charades_ego' or dataset == 'egtea':
                loss_dict = criterion(outputs)
            loss = loss_dict['loss'] + outputs['ps_loss'] * cfg['training'].get('lambda', 0)
            loss /= update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()
        if (data_iter + 1) % update_freq != 0:
            continue

        clip_grad_value = cfg['training']['clip_grad_value']
        clip_grad_type = cfg['training']['clip_grad_type']
        if clip_grad_value != "":
            scaler.unscale_(optimizer)
            if clip_grad_type == 'norm':
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_value, norm_type=2.
                )
            elif clip_grad_type == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)
            else:
                assert False, f"Unknown clip mode ({clip_grad_type})."
        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        if hasattr(dist_utils.get_model(model), 'logit_scale'):
            # clamp logit scale to [0, 100]
            dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = dist_utils.get_model(model).logit_scale.exp().item()
        else:
            logit_scale = torch.nan

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), cfg['training']['batch_size'])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if dist_utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                           'scaler': scaler.get_scale(), 'logit': logit_scale})
            progress.display(optim_iter)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate_mir(val_loader, model, criterion, args, cfg):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'max_margin_loss']
    update_freq = cfg['training']['update_freq']
    iters_per_epoch = len(val_loader) // update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Test: "
    )

    # switch to eval mode
    model.eval()

    all_video_embed = []
    all_text_embed = []
    with torch.no_grad():
        end = time.time()
        for i, inputs in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
            relevancies = inputs.pop()

            # compute output
            outputs = model(
                *inputs,
                use_checkpoint=cfg['training']['use_checkpoint'],
                norm_embed=cfg['model']['norm_embed']
            )
            loss_dict = criterion(outputs, weight=relevancies)

            for k in loss_dict:
                metrics[k].update(loss_dict[k].item(), cfg['training']['batch_size'])

            image_features = outputs['image_embed']
            text_features = outputs['text_embed']
            all_video_embed.append(image_features.cpu().numpy())
            all_text_embed.append(text_features.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if i % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({**{k: v.item() for k, v in loss_dict.items()}})
                progress.display(i)
    progress.synchronize()
    all_text_embed = np.vstack(all_text_embed)
    all_video_embed = np.vstack(all_video_embed)
    similarity_matrix = np.matmul(all_video_embed, all_text_embed.T)
    similarity_matrix = (similarity_matrix + 1) / 2
    video_id = pd.read_csv(cfg['data']['metadata'].replace('train', 'test')).values[:, 0]
    text_id = pd.read_csv(cfg['data']['metadata'].replace('train', 'test_sentence')).values[:, 0]
    indexes = [video_id.tolist().index(elem) for elem in text_id]
    similarity_matrix = similarity_matrix[:, indexes]
    print(similarity_matrix.shape)
    rel_matrix = pd.read_pickle(
        cfg['data']['relevancy_path']
    )
    metrics = {k: v.avg for k, v in metrics.items()}
    vis_map = calculate_mAP(similarity_matrix, rel_matrix)
    txt_map = calculate_mAP(similarity_matrix.T, rel_matrix.T)
    avg_map = (vis_map + txt_map) / 2
    metrics.update({'V2T_mAP': vis_map, 'T2V_mAP': txt_map, 'avg_mAP': avg_map})
    print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
    vis_k_counts = calculate_k_counts(rel_matrix)
    txt_k_counts = calculate_k_counts(rel_matrix.T)
    vis_IDCG = calculate_IDCG(rel_matrix, vis_k_counts)
    txt_IDCG = calculate_IDCG(rel_matrix.T, txt_k_counts)
    vis_nDCG = calculate_nDCG(similarity_matrix, rel_matrix, k_counts=vis_k_counts, IDCG=vis_IDCG)
    txt_nDCG = calculate_nDCG(similarity_matrix.T, rel_matrix.T, k_counts=txt_k_counts, IDCG=txt_IDCG)
    avg_nDCG = (vis_nDCG + txt_nDCG) / 2
    print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_nDCG, txt_nDCG, avg_nDCG))
    metrics.update({'V2T_nDCG': vis_nDCG, 'T2V_nDCG': txt_nDCG, 'avg_nDCG': avg_nDCG})
    return metrics


def validate_cls(val_loader, templates, labels, model, tokenizer, args, cfg):
    # switch to eval mode
    model.eval()

    all_outputs = []
    all_targets = []
    dataset = cfg['data']['dataset']
    with torch.no_grad():
        text_features = []
        for label in labels:
            if isinstance(label, list):
                texts = [tmpl.format(lbl) for tmpl in templates for lbl in label]
            else:
                texts = [tmpl.format(label) for tmpl in templates]
            texts = tokenizer(texts)
            if isinstance(texts, tuple):
                # Bert-style tokenizer will output both ids and mask
                texts, masks = texts
                texts = texts.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
            else:
                texts = texts.cuda(non_blocking=True)
                masks = None
            texts = texts.view(-1, 77).contiguous()
            masks = masks.view(-1, 77).contiguous() if masks is not None else None
            if masks is not None:
                class_embeddings, _ = dist_utils.get_model(model).encode_text(texts, attention_mask=masks)
            else:
                class_embeddings, _ = dist_utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        print('=> start forwarding')
        end_time = time.time()
        for i, values in enumerate(val_loader):
            images = values[0]
            target = values[1]
            if i % args.print_freq == 0:
                print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            if isinstance(images, torch.Tensor):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # encode images
                image_features, _ = dist_utils.get_model(model).encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # cosine similarity as logits
                logits_per_image = image_features @ text_features.t()
                logits_per_image = torch.softmax(logits_per_image, dim=1)
            else:
                target = target.cuda(non_blocking=True)
                images_list = images
                logits_all_clips = []
                for images in images_list:
                    images = images.cuda(non_blocking=True)
                    image_features, _ = dist_utils.get_model(model).encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits_per_image = image_features @ text_features.t()
                    logits_all_clips.append(logits_per_image)

                logits_all_clips = torch.stack(logits_all_clips, dim=0)
                logits_per_image = logits_all_clips.mean(0)
                logits_per_image = torch.softmax(logits_per_image, dim=1)

            all_outputs.append(logits_per_image.cpu())
            all_targets.append(target.cpu())
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    preds, targets = all_outputs.numpy(), all_targets.numpy()
    if dataset == 'charades_ego':
        if args.evaluate:
            m_ap, _, m_aps = charades_map(preds, targets)
            if args.save_results:
                np.save(os.path.join(cfg['save_dir'], 'results.npy'), m_aps)
        else:
            m_ap, _, _ = charades_map(preds, targets)
        print('mAP = {:.3f}'.format(m_ap))
        return {'mAP': m_ap}
    else:
        cm = confusion_matrix(targets, preds.argmax(axis=1))
        if args.evaluate and args.save_results:
            np.save(os.path.join(cfg['save_dir'], 'results.npy'), cm)
            
        mean_class_acc, acc = get_mean_accuracy(cm)
        print('Mean Acc. = {:.3f}, Top-1 Acc. = {:.3f}'.format(mean_class_acc, acc))
        return {'mAcc': mean_class_acc, 'acc1': acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila finetune and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    if args.debug:
        cfg['output_dir'] = 'runs/debug'
    if not args.evaluate:
        os.makedirs(cfg['output_dir'], exist_ok=True)
    main(args, cfg)

