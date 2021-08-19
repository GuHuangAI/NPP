# -*- coding: utf-8 -*-
""" Search up cell and edge wgen fix BB"""
import os
import sys
import argparse
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from core.config import config
from core.config import update_config
from core.criterion import Criterion_pose, Criterion_par
from core.function import *

from models.model_augment import Network
from dataset.data_loader import LIPDataset as Dataset
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import count_parameters_in_MB


device = torch.device("cuda")


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description='Train parsing network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--global_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus

    # if args.genotype:
    #    config.TRAIN.GENOTYPE = args.genotype


def get_imlist(dataloader):
    # length = len(dataloader)
    eval_im_name_list = []
    for i, batch in enumerate(dataloader):
        _, _, _, meta = batch
        for name in meta['name']:
            eval_im_name_list.append(name)
    return eval_im_name_list


def main():
    args = parse_args()
    reset_config(config, args)

    # tensorboard
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'augment_64_16_2', 'train')

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = True

    gpus = [int(i) for i in config.GPUS.split(',')]
    distributed = len(gpus) > 1
    if distributed:
        print(args.local_rank, args.global_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
        synchronize()

    if not gpus == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # prepare dataloader
    crop_size = (config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    # Image normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Data transform
    data_transform = transforms.Compose([transforms.ToTensor(), normalize, ])
    train_dataset = Dataset(root=config.DATASET.ROOT, \
                             im_root=config.DATASET.TRAIN_IMROOT, \
                             pose_anno_file=config.TRAIN.TRAIN_SET, \
                             parsing_anno_root=config.DATASET.TRAIN_SEGROOT, \
                             transform=data_transform, \
                             pose_net_stride=4, \
                             parsing_net_stride=1, \
                             crop_size=crop_size, \
                             target_dist=1.171, scale_min=0.5, scale_max=1.5, \
                             max_rotate_degree=40, \
                             max_center_trans=40, \
                             flip_prob=0.5, \
                             pose_aux=True, \
                             is_visualization=False)

    test_size = (config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    valid_dataset = Dataset(root=config.DATASET.ROOT, \
                            im_root=config.DATASET.VAL_IMROOT, \
                            pose_anno_file=config.TRAIN.TEST_SET, \
                            parsing_anno_root=config.DATASET.VAL_SEGROOT, \
                            transform=data_transform, \
                            pose_net_stride=4, \
                            parsing_net_stride=1, \
                            crop_size=test_size, \
                            target_dist=1.171, scale_min=0.5, scale_max=1.5, \
                            max_rotate_degree=0, \
                            max_center_trans=0, \
                            flip_prob=0.5, \
                            pose_aux=True, \
                            is_visualization=False,
                            sample=5000,
                            is_train=False)
    print('get imlist...')

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.TRAIN.BATCH_SIZE,
                                               shuffle=True and train_sampler is None,
                                               num_workers=config.WORKERS,
                                               drop_last=True,
                                               pin_memory=True,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=config.SEARCH.BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=config.WORKERS,
                                               pin_memory=True,
                                               sampler=valid_sampler)

    im_list = get_imlist(valid_loader)
    print(len(train_dataset), len(im_list))
    criterion1 = Criterion_pose(out_len=2, use_target_weight=False).cuda()
    criterion2 = Criterion_par(out_len=2).cuda()

    model = Network(config)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model._init_params()
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if
                    (n.startswith('cells1.') or n.startswith('cells2') or n.startswith('stem')) and p.requires_grad],
            'lr': 0.33*config.TRAIN.LR,},

        {
            "params": [p for n, p in model.named_parameters() if
                       not (n.startswith('cells1.') or n.startswith('cells2') or n.startswith('stem')) and p.requires_grad],
        },
    ]

    model.load_pretrain_backbone(path="/export/home/lg/huang/code/NPP/encoder.pth")

    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    logger.info("param size = %fMB", count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(param_dicts, config.TRAIN.LR)
    optimizer.add_param_group({'params': criterion1.parameters(), 'lr': 0.0001})
    optimizer.add_param_group({'params': criterion2.parameters(), 'lr': 0.0001})
    lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
    logger.info("Logger is set - training start")

    ####
    last_epoch = 0
    best_mIOU = 0.
    best_acc = 0.
    is_best = False
    print(config.TRAIN.RESUME)
    if config.TRAIN.RESUME:
        checkpoint_file = "/export/home/lg/huang/code/NPP/output/lip/augment_64_16_2/384_384/checkpoint.pth"
        model_state_file = os.path.join(checkpoint_file)
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)  # distribute load
            print(checkpoint.keys())
            last_epoch = checkpoint['epoch'] + 1
            lr.load_state_dict(checkpoint['lr'])
            best_mIOU = checkpoint['perf_iou']
            best_acc = checkpoint['perf_pck']
            criterion1.load_state_dict(checkpoint['cri1'])
            criterion2.load_state_dict(checkpoint['cri2'])
            model.module.load_state_dict(checkpoint['best_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, last_epoch))
    for epoch in range(last_epoch, config.TRAIN.EPOCHS):
        if distributed:
            train_sampler.set_epoch(epoch)
        train(config, epoch, config.SEARCH.EPOCHS, lr, train_loader, optimizer, model, criterion1, criterion2,
              writer_dict, device)

        # validation
        valid_loss, mean_IoU, IoU_array, acc_avg = validate_sync(config, valid_loader, model, im_list, criterion1,
                                                                 criterion2, writer_dict, device)
        logger.info("mean_IoU of valdataset={:.4f}".format(mean_IoU))
        logger.info('acc_avg of valdataset={:.4f}'.format(acc_avg))
        lr.step()
        # save
        if best_mIOU < mean_IoU:
            if best_acc - 1 < acc_avg:
                best_mIOU = mean_IoU
                best_acc = acc_avg
                is_best = True
            else:
                is_best = False
        else:
            if best_acc + 1 < acc_avg:
                best_mIOU = mean_IoU
                best_acc = acc_avg
                is_best = True
            else:
                is_best = False
        print('is best ', is_best)
        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf_iou': best_mIOU,
                'perf_pck': best_acc,
                'lr': lr.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cri1': criterion1.state_dict(),
                'cri2': criterion2.state_dict(),
            }, is_best, final_output_dir)

            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}, ACC_AVG: {: 4.4f}, Best_ACC: {: 4.4f}'.format(
                valid_loss, mean_IoU, best_mIOU, acc_avg, best_acc)
            logger.info(msg)
            if epoch == config.TRAIN.EPOCHS - 1:
                final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
                logger.info('=> saving final model state to {}'.format(final_model_state_file))
                logger.info('=> best accuracy is {}'.format(best_mIOU))
                torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()

if __name__ == "__main__":
    main()




