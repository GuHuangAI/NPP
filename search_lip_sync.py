""" Search pose_cell and par_cell """
import os
import sys
import argparse
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler


from core.config import config
from core.config import update_config
from core.criterion import Criterion_pose, Criterion_par
from core.function import *
# from core.evaluate import accuracy
from models.model_search_interact import Network
from dataset.data_loader import LIPDataset as Dataset

from utils import utils
from utils.transforms import flip_back
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.utils import FullModel, get_rank
from utils.utils import pck_table_output_lip_dataset
from utils.utils import save_hpe_results_to_lip_format

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
        self.avg = self.sum / self.count if self.count != 0 else 0

print(torch.cuda.is_available())
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

    # searching
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)

    args = parser.parse_args()

    return args

def reset_config(config, args):

    if args.gpus:
        config.GPUS = args.gpus

def get_imlist(dataset):
    length = len(dataset)
    eval_im_name_list = []
    for i in range(length):
        im_name = dataset[i][3]['name']
        eval_im_name_list.append(im_name)
    return eval_im_name_list
def get_imlist2(dataloader):
    #length = len(dataloader)
    eval_im_name_list = []
    for i,batch in enumerate(dataloader):
        _,_,_,meta = batch
        for name in meta['name']:
            eval_im_name_list.append(name)
    return eval_im_name_list

def main():

    args = parse_args()
    reset_config(config, args)
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'search_PC_32_16_inter', 'train')


    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = True

    gpus = [int(i) for i in config.GPUS.split(',')]
    if not gpus == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS

    distributed = len(gpus) > 1
    if distributed:

        print(args.local_rank,args.global_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
        synchronize()

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
    data_transform = transforms.Compose([transforms.ToTensor(), normalize,])
    # im_root, pose_anno_file, parsing_anno_root
    train_dataset = Dataset(root=config.DATASET.ROOT, \
                        im_root=config.DATASET.TRAIN_IMROOT, \
                        pose_anno_file=config.SEARCH.TRAIN_SET, \
                        parsing_anno_root=config.DATASET.TRAIN_SEGROOT, \
                        transform=data_transform, \
                        # transform=None, \
                        pose_net_stride=4, \
                        parsing_net_stride=1, \
                        crop_size=crop_size, \
                        target_dist=1.171, scale_min=0.5, scale_max=1.5, \
                        max_rotate_degree=40, \
                        max_center_trans=40, \
                        flip_prob=0.5, \
                        pose_aux=True, \
                        is_visualization=False)

    mini_dataset = Dataset(root=config.DATASET.ROOT, \
                        im_root=config.DATASET.TRAIN_IMROOT, \
                        pose_anno_file=config.SEARCH.MINI_SET, \
                        parsing_anno_root=config.DATASET.TRAIN_SEGROOT, \
                        transform=data_transform, \
                        # transform=None, \
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
                        pose_anno_file=config.SEARCH.TEST_SET, \
                        parsing_anno_root=config.DATASET.VAL_SEGROOT, \
                        transform=data_transform, \
                        # transform=None, \
                        pose_net_stride=4, \
                        parsing_net_stride=1, \
                        crop_size=test_size, \
                        target_dist=1.171, scale_min=1.0, scale_max=1.0, \
                        max_rotate_degree=0, \
                        max_center_trans=0, \
                        flip_prob=0.5, \
                        pose_aux=True, \
                        is_visualization=False,
                        sample=5000,
                        is_train=False)

    #im_list=get_imlist(valid_dataset)
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        mini_sampler = DistributedSampler(mini_dataset)
        valid_sampler = DistributedSampler(valid_dataset)
    else:
        train_sampler = None
        mini_sampler = None
    print('batch', config.SEARCH.BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.SEARCH.BATCH_SIZE,
                                               shuffle=True and train_sampler is None,
                                               num_workers=config.WORKERS,
                                               pin_memory=True,
                                               sampler=train_sampler)
    mini_loader = torch.utils.data.DataLoader(mini_dataset,
                                               batch_size=config.SEARCH.BATCH_SIZE,
                                               shuffle=True and mini_sampler is None,
                                               num_workers=config.WORKERS,
                                               pin_memory=True,
                                               sampler=mini_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=2*config.SEARCH.BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=config.WORKERS,
                                               pin_memory=True,
                                               sampler=valid_sampler)
    '''
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.SEARCH.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=config.WORKERS,
                                               pin_memory=True)
    mini_loader = torch.utils.data.DataLoader(mini_dataset,
                                               batch_size=config.SEARCH.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=config.WORKERS,
                                               pin_memory=True)  
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=config.SEARCH.BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=config.WORKERS,
                                               pin_memory=True)
    '''

    im_list = get_imlist2(valid_loader)
    print(len(train_dataset),len(mini_dataset),len(im_list))


#   criterion = JointsMSELoss(use_target_weight = config.LOSS.USE_TARGET_WEIGHT).to(device)
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights)
    else:
        criterion1 = Criterion_pose(out_len=2, use_target_weight=False).cuda()
        criterion2 = Criterion_par(out_len=2).cuda()

    model = Network(config)
    model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #model._init_params()
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)

    arch_params = list(map(id, model.module.arch_parameters()))
    weight_params = filter(lambda p: id(p) not in arch_params, model.parameters())

    optimizer = torch.optim.Adam(weight_params, config.SEARCH.W_LR)
    optimizer.add_param_group({'params': criterion1.parameters(),'lr':0.0001})
    optimizer.add_param_group({'params': criterion2.parameters(),'lr':0.0001})
    a_optimizer = torch.optim.Adam(model.module.arch_parameters(),lr=config.SEARCH.APLHA_LR,betas=(0.5, 0.999), weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.SEARCH.LR_STEP, config.SEARCH.LR_FACTOR)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=2)


    logger.info("Logger is set - searching start")



    # search loop
    last_epoch = 0
    best_mIOU = 0
    best_acc = 0
    is_best = False
    if config.SEARCH.RESUME:
        #checkpoint_file = '/export/home/lg/huang/code/Auto_Par_Pose/output/lip/search_PC_32_16_1_model_hrnet2/384_384_parsing_PSP/checkpoint.pth'
        checkpoint_file = '/export/home/lg/huang/code/Auto_Par_Pose/output/lip/search_PC_32_16_inter/384_384_parsing_PSP/checkpoint.pth'
        model_state_file = os.path.join(checkpoint_file)
#        print('model_state_file')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
            print(checkpoint.keys())
            #best_mIOU = checkpoint['perf']
            last_epoch = checkpoint['epoch']+1
            lr_scheduler.load_state_dict(checkpoint['schedule'])
            best_mIOU=checkpoint['perf_iou']
            best_acc=checkpoint['perf_pck']
            criterion1.load_state_dict(checkpoint['cri1'])
            criterion2.load_state_dict(checkpoint['cri2'])
            model.module.load_state_dict(checkpoint['best_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            a_optimizer.load_state_dict(checkpoint['a_optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, last_epoch))
            print('load successful from {}'.format(checkpoint_file))
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.SEARCH.LR_STEP, config.SEARCH.LR_FACTOR)


    for epoch in range(last_epoch,config.SEARCH.EPOCHS):
    #for epoch in range(20, config.SEARCH.EPOCHS):
        if distributed:
            train_sampler.set_epoch(epoch)
            mini_sampler.set_epoch(epoch)
        # training

        #if epoch < 10:
        if epoch < 15:
            train(config, epoch, config.SEARCH.EPOCHS, lr_scheduler, train_loader, optimizer,model, criterion1, criterion2, writer_dict, device)
        else:
            train_with_alpha(config, train_loader, mini_loader, model, criterion1, criterion2, optimizer, a_optimizer, epoch, final_output_dir, tb_log_dir, writer_dict, device)

        # validation
        #valid_loss, mean_IoU, IoU_array, acc_avg= validate(config,valid_loader,model,im_list,criterion1, criterion2,writer_dict,device)
        valid_loss, mean_IoU, IoU_array, acc_avg= validate_sync(config,valid_loader,model,im_list,criterion1, criterion2,writer_dict,device)
        logger.info("mean_IoU of valdataset={:.4f}".format(mean_IoU))
        logger.info('acc_avg of valdataset={:.4f}'.format(acc_avg))

        genotype = model.module.genotype()

        if best_mIOU < mean_IoU:
            if best_acc-1 < acc_avg:
                best_mIOU = mean_IoU
                best_acc = acc_avg
                best_genotype = genotype
                is_best = True
            else:
                is_best = False
        else:
            if best_acc+1 < acc_avg:
                best_mIOU = mean_IoU
                best_acc = acc_avg
                best_genotype = genotype
                is_best = True
            else:
                is_best = False
        print('is best ',is_best)
        lr_scheduler.step()

        if args.local_rank == 0:
            logger.info("genotype = {}".format(genotype))
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch ,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf_iou': best_mIOU,
                'perf_pck': best_acc,
                'schedule':lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'a_optimizer': a_optimizer.state_dict(),
                'cri1': criterion1.state_dict(),
                'cri2': criterion2.state_dict(),
            }, is_best, final_output_dir)


            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}, ACC_AVG: {: 4.4f}, Best_ACC: {: 4.4f}'.format(valid_loss, mean_IoU, best_mIOU, acc_avg, best_acc)
            logger.info(msg)
            if epoch == 14:
                save_checkpoint({
                'epoch': epoch ,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf_iou': best_mIOU,
                'perf_pck': best_acc,
                'schedule':lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'a_optimizer': a_optimizer.state_dict(),
                'cri1': criterion1.state_dict(),
                'cri2': criterion2.state_dict(),
                }, False, final_output_dir,filename = 'warmed_state.pth')
                
            if epoch==config.SEARCH.EPOCHS-1:
                final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
                logger.info('=> saving final model state to {}'.format(final_model_state_file))  
                #logger.info('=> best accuracy is {}'.format(best_mIOU))
                torch.save(model.module.state_dict(), final_model_state_file)
                    
    logger.info("Final best moiu = {:.3f}".format(best_mIOU))
    logger.info("Final best pck = {:.3f}".format(acc_avg))
    logger.info("Best Genotype = {}".format(best_genotype))

if __name__ == "__main__":
    main()
