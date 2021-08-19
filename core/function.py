# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import cv2
import time
from PIL import Image
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank
from utils.utils import pck_table_output_lip_dataset
from core.evaluate import accuracy
from core.inference import get_final_preds
import glob
from utils.utils import save_hpe_results_to_lip_format
from scipy.ndimage.filters import gaussian_filter
import csv
from utils.calc_pckh import calc_pck_lip_dataset
import torch.distributions.categorical as cate
import math
from dataset.target_generation import get_affine_transform2
#import torchsnooper

pose_weight = 10
par_weight = 0.1

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp
        
#@torchsnooper.snoop()
#ml = MulLoss()
def train(config, epoch, num_epoch, lr, trainloader, optimizer, model, criterion_pose, criterion_par, writer_dict, device):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
  #  cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
#        images, labels, _, _ = batch
        images, labels_par, labels_pose, meta = batch
        pose_weight = meta['pose_weight'].to(device)
        images = images.to(device)
        labels_par[0] = labels_par[0].long().to(device)
        labels_par[1] = labels_par[1].long().to(device)
        #print('labels_par[0] ',labels_par[0].size())
        #print('labels_par[1] ',labels_par[1].size())
        
        if isinstance(labels_pose,list):
            labels_pose[0] = labels_pose[0][:,:-1,:,:].float().to(device)
            labels_pose[1] = labels_pose[1][:,:-1,:,:].float().to(device)
        else:
            labels_pose = labels_pose[:,:-1,:,:].float().to(device)
        #print('labels_pose ',labels_pose.size())
        #with torchsnooper.snoop():
        output_pose, output_par = model(images)
    
        losses_par=criterion_par(output_par,labels_par)
        losses_par = torch.unsqueeze(losses_par,0)
        losses_pose = criterion_pose(output_pose, labels_pose, target_weight=pose_weight) 
        losses_pose = torch.unsqueeze(losses_pose,0)
        #losses = losses_par*par_weight + losses_pose*pose_weight
        #ml = 
        #optimizer.add_param_group({'params': ml.parameters()})
        losses = losses_par + losses_pose
    
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)
        #for i in range(len(output_par)):
        #    print('pred_par {} size {}'.format(i,output_par[i][0].size()))
        #    print('pred_pose {} size {}'.format(i,output_pose[i].size()))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

    #    lr = adjust_learning_rate(optimizer,
     #                             base_lr,
     #                             num_iters,
     #                             i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            
#            msg1 = 'epcho {}'.format(epoch)
         #  # msg = 'Epoch: [{}/{}] , ' \
          #   #     'lr: {:.6f}, Loss: {:.6f}' .format(epoch, num_epoch, lr, print_loss           
            
#            msg2 = 'Loss: {:.6f}' .format(print_loss)
#            logging.info(msg1)
#            logging.info(msg2)
            #torch.save(labels_pose,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_labelpose{}.pth'.format(epoch,i_iter))
            #torch.save(labels_par,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_labelpar{}.pth'.format(epoch,i_iter))
            #torch.save(output_pose,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_predpose{}.pth'.format(epoch,i_iter))
            #torch.save(output_par,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_predpar{}.pth'.format(epoch,i_iter))
            msg1 = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i_iter, len(trainloader))
            msg2 =  'Loss: {:.6f}' .format(print_loss)                            
            msg3 = losses_par.mean()
            msg4 = losses_pose.mean()
                                           
            logging.info(msg1)
            logging.info(msg2) 
            logging.info(msg3)
            logging.info(msg4)         
            #genotype = model.module.genotype()
            #print(genotype)
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def train_pose(config, epoch, num_epoch, lr, trainloader, optimizer, model, criterion_pose,  writer_dict,
          device):
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    #  cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        #        images, labels, _, _ = batch
        images, labels_par, labels_pose, meta = batch
        pose_weight = meta['pose_weight'].to(device)
        images = images.to(device)
        # labels_par[0] = labels_par[0].long().to(device)
        # labels_par[1] = labels_par[1].long().to(device)
        # print('labels_par[0] ',labels_par[0].size())
        # print('labels_par[1] ',labels_par[1].size())

        if isinstance(labels_pose, list):
            labels_pose[0] = labels_pose[0][:, :-1, :, :].float().to(device)
            labels_pose[1] = labels_pose[1][:, :-1, :, :].float().to(device)
        else:
            labels_pose = labels_pose[:, :-1, :, :].float().to(device)
        # print('labels_pose ',labels_pose.size())
        output_pose = model(images)

        losses_pose = criterion_pose(output_pose, labels_pose, target_weight=pose_weight)
        losses_pose = torch.unsqueeze(losses_pose, 0)
        # losses = losses_par*par_weight + losses_pose*pose_weight
        # ml =

        loss = losses_pose.mean()

        reduced_loss = reduce_tensor(loss)
        # for i in range(len(output_par)):
        #    print('pred_par {} size {}'.format(i,output_par[i][0].size()))
        #    print('pred_pose {} size {}'.format(i,output_pose[i].size()))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        #    lr = adjust_learning_rate(optimizer,
        #                             base_lr,
        #                             num_iters,
        #                             i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size

            #            msg1 = 'epcho {}'.format(epoch)
            #  # msg = 'Epoch: [{}/{}] , ' \
            #   #     'lr: {:.6f}, Loss: {:.6f}' .format(epoch, num_epoch, lr, print_loss

            #            msg2 = 'Loss: {:.6f}' .format(print_loss)
            #            logging.info(msg1)
            #            logging.info(msg2)
            # torch.save(labels_pose,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_labelpose{}.pth'.format(epoch,i_iter))
            # torch.save(labels_par,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_labelpar{}.pth'.format(epoch,i_iter))
            # torch.save(output_pose,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_predpose{}.pth'.format(epoch,i_iter))
            # torch.save(output_par,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_predpar{}.pth'.format(epoch,i_iter))
            msg1 = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i_iter, len(trainloader))
            msg2 = 'Loss: {:.6f}'.format(print_loss)

            logging.info(msg1)
            logging.info(msg2)
            # genotype = model.module.genotype()
            # print(genotype)
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def train_par(config, epoch, num_epoch, lr, trainloader, optimizer, model, criterion_par, writer_dict,
          device):
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    #  cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        #        images, labels, _, _ = batch
        images, labels_par, labels_pose, meta = batch
        images = images.to(device)
        labels_par[0] = labels_par[0].long().to(device)
        labels_par[1] = labels_par[1].long().to(device)
        # print('labels_par[0] ',labels_par[0].size())
        # print('labels_par[1] ',labels_par[1].size())

        output_par = model(images)

        losses_par = criterion_par(output_par, labels_par)
        losses_par = torch.unsqueeze(losses_par, 0)
        # losses = losses_par*par_weight + losses_pose*pose_weight
        # ml =
        # optimizer.add_param_group({'params': ml.parameters()})

        loss = losses_par.mean()

        reduced_loss = reduce_tensor(loss)
        # for i in range(len(output_par)):
        #    print('pred_par {} size {}'.format(i,output_par[i][0].size()))
        #    print('pred_pose {} size {}'.format(i,output_pose[i].size()))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        #    lr = adjust_learning_rate(optimizer,
        #                             base_lr,
        #                             num_iters,
        #                             i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size

            #            msg1 = 'epcho {}'.format(epoch)
            #  # msg = 'Epoch: [{}/{}] , ' \
            #   #     'lr: {:.6f}, Loss: {:.6f}' .format(epoch, num_epoch, lr, print_loss

            #            msg2 = 'Loss: {:.6f}' .format(print_loss)
            #            logging.info(msg1)
            #            logging.info(msg2)
            # torch.save(labels_pose,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_labelpose{}.pth'.format(epoch,i_iter))
            # torch.save(labels_par,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_labelpar{}.pth'.format(epoch,i_iter))
            # torch.save(output_pose,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_predpose{}.pth'.format(epoch,i_iter))
            # torch.save(output_par,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_predpar{}.pth'.format(epoch,i_iter))
            msg1 = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i_iter, len(trainloader))
            msg2 = 'Loss: {:.6f}'.format(print_loss)

            logging.info(msg1)
            logging.info(msg2)
            # genotype = model.module.genotype()
            # print(genotype)
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def train_apex(config, epoch, num_epoch, lr, trainloader, optimizer, model, criterion_pose, criterion_par, writer_dict, device, amp):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
  #  cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
#        images, labels, _, _ = batch
        images, labels_par, labels_pose, _ = batch
        images = images.to(device)
        labels_par[0] = labels_par[0].long().to(device)
        labels_par[1] = labels_par[1].long().to(device)
        #print('labels_par[0] ',labels_par[0].size())
        #print('labels_par[1] ',labels_par[1].size())
        
        if isinstance(labels_pose,list):
            labels_pose[0] = labels_pose[0][:,:-1,:,:].float().to(device)
            labels_pose[1] = labels_pose[1][:,:-1,:,:].float().to(device)
        else:
            labels_pose = labels_pose[:,:-1,:,:].float().to(device)
        #print('labels_pose ',labels_pose.size())
        #with torchsnooper.snoop():
        output_pose, output_par = model(images)
    
        losses_par=criterion_par(output_par,labels_par)
        losses_par = torch.unsqueeze(losses_par,0)
        losses_pose = criterion_pose(output_pose, labels_pose) 
        losses_pose = torch.unsqueeze(losses_pose,0)
        #losses = losses_par*par_weight + losses_pose*pose_weight
        #ml = 
        #optimizer.add_param_group({'params': ml.parameters()})
        losses = losses_par + losses_pose
    
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)
        #for i in range(len(output_par)):
        #    print('pred_par {} size {}'.format(i,output_par[i][0].size()))
        #    print('pred_pose {} size {}'.format(i,output_pose[i].size()))

        model.zero_grad()
        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

    #    lr = adjust_learning_rate(optimizer,
     #                             base_lr,
     #                             num_iters,
     #                             i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            
#            msg1 = 'epcho {}'.format(epoch)
         #  # msg = 'Epoch: [{}/{}] , ' \
          #   #     'lr: {:.6f}, Loss: {:.6f}' .format(epoch, num_epoch, lr, print_loss           
            
#            msg2 = 'Loss: {:.6f}' .format(print_loss)
#            logging.info(msg1)
#            logging.info(msg2)
            #torch.save(labels_pose,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_labelpose{}.pth'.format(epoch,i_iter))
            #torch.save(labels_par,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_labelpar{}.pth'.format(epoch,i_iter))
            #torch.save(output_pose,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_predpose{}.pth'.format(epoch,i_iter))
            #torch.save(output_par,'/export/home/bbq/huang/code/Auto_Par_Pose/output/lip/search_PC_32_12_2_test6/384_384_parsing_PSP/train{}_predpar{}.pth'.format(epoch,i_iter))
            msg1 = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i_iter, len(trainloader))
            msg2 =  'Loss: {:.6f}' .format(print_loss)                            
            msg3 = losses_par.mean()
            msg4 = losses_pose.mean()
                                           
            logging.info(msg1)
            logging.info(msg2) 
            logging.info(msg3)
            logging.info(msg4)         
            #genotype = model.module.genotype()
            #print(genotype)
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
'''
def train_with_alpha(config, train_loader, mini_loader, model, criterion, optimizer, a_optimizer, epoch,
                     output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, ((input, target, target_weight, meta),(input1, target1, target_weight1, meta1)) in enumerate(zip(train_loader,mini_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        outputs = model(input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # optim alpha
        outputs1 = model(input1)
        target1 = target1.cuda(non_blocking=True)
        target_weight1 = target_weight1.cuda(non_blocking=True)

        if isinstance(outputs1, list):
            loss1 = criterion(outputs1[0], target1, target_weight1)
            for output1 in outputs1[1:]:
                loss1 += criterion(output1, target1, target_weight1)
        else:
            output1 = outputs1
            loss1 = criterion(output1, target1, target_weight1)

        a_optimizer.zero_grad()
        loss1.backward()
        a_optimizer.step()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
'''
def train_with_alpha(config, train_loader, mini_loader, model, criterion_pose, criterion_par, optimizer, a_optimizer, epoch,
                     output_dir, tb_log_dir, writer_dict,device):
        # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
  #  cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()
    for i_iter, (batch1,batch2) in enumerate(zip(train_loader,mini_loader)):
        images1, labels_par1, labels_pose1, meta1 = batch1
        images2, labels_par2, labels_pose2, meta2 = batch2
        images1 = images1.to(device)
        labels_par1[0] = labels_par1[0].long().to(device)
        labels_par1[1] = labels_par1[1].long().to(device)
        if isinstance(labels_pose1,list):
            #print('pose is list')
            labels_pose1[0] = labels_pose1[0][:,:-1,:,:].float().to(device)
            labels_pose1[1] = labels_pose1[1][:,:-1,:,:].float().to(device)
        else:
            #print('pose is not list')
            labels_pose1=labels_pose1[:,:-1,:,:].float().to(device)
        output_pose1, output_par1 = model(images1)
        #print(output_pose1[0].shape)
        pose_weight1 = meta1['pose_weight'].to(device)
        pose_weight2 = meta2['pose_weight'].to(device)
        losses_par1=criterion_par(output_par1,labels_par1)
        losses_par1 = torch.unsqueeze(losses_par1,0)
        losses_pose1 = criterion_pose(output_pose1, labels_pose1, target_weight=pose_weight1)  
        
        losses_pose1 = torch.unsqueeze(losses_pose1,0)
        
        #losses1 = losses_par1*par_weight+losses_pose1*pose_weight
        #losses1 = losses_par1 * torch.exp(-model.module.lamda2) + losses_pose1 * torch.exp(-model.module.lamda1) + model.module.lamda2 + model.module.lamda1
        #optimizer.add_param_group({'params': ml.parameters()})
        losses1 = losses_par1 + losses_pose1
        loss1 = losses1.mean()
        reduced_loss = reduce_tensor(loss1)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size            
            msg1 = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i_iter, len(train_loader))
            msg2 =  'Loss: {:.6f}' .format(print_loss)  
            msg3 = losses_par1.mean() 
            msg4 = losses_pose1.mean()                  
            logging.info(msg1)
            logging.info(msg2)
            logging.info(msg3)
            logging.info(msg4)  
        #optime alpha
        images2 = images2.to(device)
        # images1 = images1.to(device)
        labels_par2[0] = labels_par2[0].long().to(device)
        labels_par2[1] = labels_par2[1].long().to(device)
        if isinstance(labels_pose2,list):
            labels_pose2[0] = labels_pose2[0][:,:-1,:,:].float().to(device)
            labels_pose2[1] = labels_pose2[1][:,:-1,:,:].float().to(device)
        else:
            labels_pose2=labels_pose2[:,:-1,:,:].float().to(device)
        output_pose2, output_par2 = model(images2)
        losses_par2 = criterion_par(output_par2,labels_par2)
        losses_par2 = torch.unsqueeze(losses_par2,0)
        losses_pose2 = criterion_pose(output_pose2, labels_pose2, target_weight=pose_weight2)  
        losses_pose2 = torch.unsqueeze(losses_pose2,0)
        '''
        #######alpha entropy loss
        #wpo1 = F.softmax(model.module.alphas_pose,dim=-1)
        wpo1 = F.softmax(model.module._arch_parameters[0],dim=-1)
        en_po_b = model.module.entropy_beta(2,4,model.module._arch_parameters[4])
        #wpo2 = model.module.btw(2,4,model.module._arch_parameters[2])
        #wpo3 = torch.zeros_like(wpo1).to(device)
        #for ii in range(wpo1.shape[0]):
        #    wpo3[ii,:]=wpo1[ii,:]*wpo2[ii]
        #importance = torch.sum(wpo3, dim=-1)
        #prob=wpo3/importance[:,None]
        en = cate.Categorical(probs=wpo1).entropy() / math.log(wpo1.shape[1])
        en_po_a=en.mean(dim=0)
        #wpa1 = F.softmax(model.module.alphas_par.detach(),dim=-1)
        #wpa2 = model.module.btw(3,4,model.module.betas_par.detach())
        wpa1 = F.softmax(model.module._arch_parameters[1],dim=-1)
        #wpa2 = model.module.btw(2,4,model.module._arch_parameters[3])
        #wpa3 = torch.zeros_like(wpa1).to(device)
        #for ii in range(wpa1.shape[0]):
        #    wpa3[ii,:]=wpa1[ii,:]*wpa2[ii]
        #importance = torch.sum(wpa3, dim=-1)
        #prob=wpa1/importance[:,None]
        en_pa_b = model.module.entropy_beta(2,4,model.module._arch_parameters[5])
        en = cate.Categorical(probs=wpa1).entropy() / math.log(wpa1.shape[1])
        en_pa_a=en.mean(dim=0)
        #print('en_po:{},en_pa:{}'.format(en_po,en_pa))
        wpo1 = F.softmax(model.module._arch_parameters[2],dim=-1)
        en_po_b2 = model.module.entropy_beta(2,4,model.module._arch_parameters[6])
        en = cate.Categorical(probs=wpo1).entropy() / math.log(wpo1.shape[1])
        en_po_a2=en.mean(dim=0)
        wpa1 = F.softmax(model.module._arch_parameters[3],dim=-1)
        en_pa_b2 = model.module.entropy_beta(2,4,model.module._arch_parameters[7])
        en = cate.Categorical(probs=wpa1).entropy() / math.log(wpa1.shape[1])
        en_pa_a2=en.mean(dim=0)
        '''
        '''
        if epoch > 100:
            en_loss = 0
            losses2 = 5*(losses_par2*par_weight+losses_pose2*pose_weight)+en_loss
        elif epoch > 70:
            en_loss = 0
            losses2 = 10*(losses_par2*par_weight+losses_pose2*pose_weight)
        elif epoch > 40:
            en_loss = 0
            losses2 = 15*(losses_par2*par_weight+losses_pose2*pose_weight)
        else:
            en_loss = 0
            losses2 = 20*(losses_par2*par_weight+losses_pose2*pose_weight)
        '''
        #losses2 = losses_par2 * torch.exp(-model.module.lamda2) + losses_pose2 * torch.exp(-model.module.lamda1) + model.module.lamda2 + model.module.lamda1
        losses2 = losses_par2 + losses_pose2
        #print(losses2.shape)
        if epoch > 70 :
            en_loss = model.module.loss_entropy()
            if i_iter % config.PRINT_FREQ == 0 and rank == 0:
                print(en_loss)
            losses2 += 2*en_loss
        loss2 = 2 * losses2.mean()
#        reduced_loss = reduce_tensor(loss)
        a_optimizer.zero_grad()
        loss2.backward()
        a_optimizer.step()
        #if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            #msg5 = (en_po_a+en_pa_a+en_po_b+en_pa_b)/4  
            #msg5 = en_loss     
            #logging.info(msg5)
'''
def validate(config, testloader, model, criterion_pose, criterion_par, writer_dict, device):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    acc = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for i_iter, batch in enumerate(testloader):
#            image, label, _, _ = batch
            if i_iter==1:
                break
            image, label_par, label_pose, _ = batch
            # num_images = image.size(0)
            size = label_par[0].size()
            image = image.to(device)
            label_par[0] = label_par[0].long().to(device)
            label_par[1] = label_par[1].long().to(device)
            label_pose = label_pose[:,:-1,:,:].float().to(device)
            pred_pose, pred_par = model(image)
            losses_par = criterion_par(pred_par, label_par)
            losses_par = torch.unsqueeze(losses_par,0)  
            
            losses_pose = criterion_pose(pred_pose, label_pose)
            losses_pose = torch.unsqueeze(losses_pose,0)  
            losses = losses_par*par_weight + losses_pose*pose_weight
         #   losses, pred = model(image, label)
            
            if isinstance(pred_par,list):
                pred_par=pred_par[-1]
            if isinstance(pred_par,list):
                pred_par=pred_par[0]
            pred_par = F.interpolate(input=pred_par, size=(
                        size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            if i_iter % config.PRINT_FREQ == 0:
                            
                msg1 = 'Val: [{0}/{1}]'.format(i_iter, len(testloader))
                msg2 =  'Loss: {:.6f}' .format(ave_loss.average())                            
                logging.info(msg1)
                logging.info(msg2) 
            
            confusion_matrix += get_confusion_matrix(
                label_par[0],
                pred_par,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            if isinstance(pred_pose,list):
                pred_pose = pred_pose[-1]
            _, avg_acc, cnt, pred = accuracy(pred_pose.cpu().numpy(), 
                                             label_pose.cpu().numpy())
            acc.update(avg_acc, cnt)
            

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size
    
    


    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer.add_scalar('valid_acc', acc.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
        
    return print_loss, mean_IoU, IoU_array, acc
'''
def validate(config, testloader, model, im_list, criterion_pose, criterion_par, writer_dict, device, ml):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    # acc = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    # im_list=[]
    pred_list = np.zeros((len(im_list), 16, 3))
    n = 0
    pose_pred_path = config.POSE_PRED_PATH[:-4]+'{}.csv'.format(rank)
    pose_gt_path = config.POSE_GT_PATH
    with torch.no_grad():
        for i_iter, batch in enumerate(testloader):
#            image, label, _, _ = batch
            # if i_iter==1:
            #     break
            image, label_par, label_pose, meta = batch
            pose_weight = meta['pose_weight'].to(device)
            # flip_img = torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).to(device)
            flip_img = image.flip(3).to(device)
            num_images = image.size(0)
            size = label_par[0].size()
            image = image.to(device)
            label_par[0] = label_par[0].long().to(device)
            label_par[1] = label_par[1].long().to(device)
            if isinstance(label_pose,list):
                label_pose[0] = label_pose[0][:,:-1,:,:].float().to(device)
                label_pose[1] = label_pose[1][:,:-1,:,:].float().to(device)
            else:
                label_pose = label_pose[:,:-1,:,:].float().to(device)
            pred_pose, pred_par = model(image)
            flip_pred_pose, flip_pred_par = model(flip_img)
            flipped_poseidx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14, 15]
            
            
            losses_par = criterion_par(pred_par, label_par)
            losses_par = torch.unsqueeze(losses_par,0)
            
            losses_pose = criterion_pose(pred_pose, label_pose, target_weight=pose_weight)
            losses_pose = torch.unsqueeze(losses_pose,0)  
            #losses = losses_par*par_weight + losses_pose*pose_weight
            losses = ml.to(device)(losses_pose, losses_par)
         #   losses, pred = model(image, label)
            
            if isinstance(pred_par,list):
                pred_par=pred_par[-1]
                flip_pred_par=flip_pred_par[-1]
            if isinstance(pred_par,list):
                pred_par=pred_par[0]
                flip_pred_par=flip_pred_par[0]
            pred_par = F.interpolate(input=pred_par, size=(
                        size[-2], size[-1]), mode='bilinear')
            flip_pred_par = F.interpolate(input=flip_pred_par, size=(
                        size[-2], size[-1]), mode='bilinear')
            tmp = flip_pred_par
            flip_pred_par[:,14,:,:] = tmp[:,15,:,:]
            flip_pred_par[:,15,:,:] = tmp[:,14,:,:]
            flip_pred_par[:,16,:,:] = tmp[:,17,:,:]
            flip_pred_par[:,17,:,:] = tmp[:,16,:,:]
            flip_pred_par[:,18,:,:] = tmp[:,19,:,:]
            flip_pred_par[:,19,:,:] = tmp[:,18,:,:]
            flip_pred_par = flip_pred_par.flip(3)
            pred_par = 0.5*(pred_par + flip_pred_par)
            # a = torch.rand(3,3,3)
            # print(a)
            # b = a.flip(2)
            # print(b)
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            if i_iter % config.PRINT_FREQ == 0:
                            
                msg1 = 'Val: [{0}/{1}]'.format(i_iter, len(testloader))
                msg2 =  'Loss: {:.6f}' .format(ave_loss.average())                            
                logging.info(msg1)
                logging.info(msg2)
            
            confusion_matrix += get_confusion_matrix(
                label_par[0],
                pred_par,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            if isinstance(pred_pose,list):
                pred_pose = pred_pose[-1]
                flip_pred_pose = flip_pred_pose[-1]
            if isinstance(pred_pose,list):
                pred_pose = pred_pose[0]
                flip_pred_pose = flip_pred_pose[0]
            pred_pose = pred_pose.data.cpu().numpy()
            flip_pred_pose = flip_pred_pose.data.cpu().numpy()
            flip_pred_pose[:, :, :, 1:] = flip_pred_pose.copy()[:, :, :, 0:-1]
            pose = np.zeros((num_images,16, 3))
            # cropped_param = cropped_param_list[scale_multiplier.index(1)]
            for num in range(num_images):
                base_scale = meta['scale'].numpy()[num]
                ori_size = meta['size'].numpy()[num]
                center = np.array([ori_size[1]/2,ori_size[0]/2])
                long_size = max(ori_size[0], ori_size[1])
                scales = np.array([long_size, long_size])
                cropped_param = meta['crop_param'].numpy()[num,:,:]
                trans = get_affine_transform2(center, scales, 0, [pred_pose.shape[2], pred_pose.shape[3]], inv=1)
                for ji in range(0, 16):
                    heatmap = pred_pose[num, ji, :, :].copy()
                    #heatmap = cv2.resize(heatmap, (size[-2],size[-1]), interpolation=cv2.INTER_LINEAR)
                    flipped_heatmap = flip_pred_pose[num, flipped_poseidx[ji], :, :].copy()
                    #flipped_heatmap = cv2.resize(flipped_heatmap, (size[-2],size[-1]), interpolation=cv2.INTER_LINEAR)
                    flipped_heatmap = cv2.flip(flipped_heatmap,1)
                    heatmap += flipped_heatmap
                    heatmap *= 0.5
                    heatmap = gaussian_filter(heatmap, sigma=3)
                    pred_pos = np.unravel_index(heatmap.argmax(), np.shape(heatmap))

                    # if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    #     diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                    #                      hm[py + 1][px] - hm[py - 1][px]])
                    #     coords[n][p] += np.sign(diff) * .25

                    pred_x = (pred_pos[1] - cropped_param[0, 2] + cropped_param[0, 0]) / base_scale
                    pred_y = (pred_pos[0] - cropped_param[0, 3] + cropped_param[0, 1]) / base_scale
            
                    pose[num, ji, 0] = pred_x
                    pose[num, ji, 1] = pred_y
                    pose[num, ji, 2] = heatmap[pred_pos[0], pred_pos[1]]
            pred_list[n:n+num_images,:,:] = pose
            n = n+num_images
        print('gene pred csv...')
        print(len(im_list))
        save_hpe_results_to_lip_format(im_list, pred_list, save_path=pose_pred_path, eval_num=len(im_list))
        pck_avg = 0.0
        pck_all = calc_pck_lip_dataset(pose_gt_path, pose_pred_path, method_name='Ours', eval_num=len(im_list))
        pck_avg = pck_all[-1][-1]
        #pck_all_list.append(pck_all)
        #pck_avg_list.append(pck_avg)
            

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size
    
    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer.add_scalar('valid_acc', pck_avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
        
    return print_loss, mean_IoU, IoU_array, pck_avg

def validate_sync(config, testloader, model, im_list, criterion_pose, criterion_par, writer_dict, device):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    # acc = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    # im_list=[]
    pred_list = np.zeros((len(im_list), 16, 3))
    n = 0
    pose_pred_path = config.POSE_PRED_PATH[:-4]+'{}.csv'.format(rank)
    pose_gt_path = config.POSE_GT_PATH
    with torch.no_grad():
        for i_iter, batch in enumerate(testloader):
#            image, label, _, _ = batch
            # if i_iter==1:
            #     break
            image, label_par, label_pose, meta = batch
            pose_weight = meta['pose_weight'].to(device)
            # flip_img = torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).to(device)
            flip_img = image.flip(3).to(device)
            num_images = image.size(0)
            size = label_par[0].size()
            image = image.to(device)
            label_par[0] = label_par[0].long().to(device)
            label_par[1] = label_par[1].long().to(device)
            if isinstance(label_pose,list):
                label_pose[0] = label_pose[0][:,:-1,:,:].float().to(device)
                label_pose[1] = label_pose[1][:,:-1,:,:].float().to(device)
            else:
                label_pose = label_pose[:,:-1,:,:].float().to(device)
            pred_pose, pred_par = model(image)
            flip_pred_pose, flip_pred_par = model(flip_img)
            flipped_poseidx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14, 15]
            
            
            losses_par = criterion_par(pred_par, label_par)
            losses_par = torch.unsqueeze(losses_par,0)
            
            losses_pose = criterion_pose(pred_pose, label_pose, target_weight=pose_weight)
            losses_pose = torch.unsqueeze(losses_pose,0)  
            #losses = losses_par*par_weight + losses_pose*pose_weight
            #losses = losses_par * torch.exp(-model.module.lamda2) + losses_pose * torch.exp(-model.module.lamda1) + model.module.lamda2 + model.module.lamda1
            losses = losses_par + losses_pose
         #   losses, pred = model(image, label)
            
            if isinstance(pred_par,list):
                pred_par=pred_par[-1]
                flip_pred_par=flip_pred_par[-1]
            if isinstance(pred_par,list):
                pred_par=pred_par[0]
                flip_pred_par=flip_pred_par[0]
            pred_par = F.interpolate(input=pred_par, size=(
                        size[-2], size[-1]), mode='bilinear')
            flip_pred_par = F.interpolate(input=flip_pred_par, size=(
                        size[-2], size[-1]), mode='bilinear')
            tmp = flip_pred_par
            flip_pred_par[:,14,:,:] = tmp[:,15,:,:]
            flip_pred_par[:,15,:,:] = tmp[:,14,:,:]
            flip_pred_par[:,16,:,:] = tmp[:,17,:,:]
            flip_pred_par[:,17,:,:] = tmp[:,16,:,:]
            flip_pred_par[:,18,:,:] = tmp[:,19,:,:]
            flip_pred_par[:,19,:,:] = tmp[:,18,:,:]
            flip_pred_par = flip_pred_par.flip(3)
            pred_par = 0.5*(pred_par + flip_pred_par)
            # a = torch.rand(3,3,3)
            # print(a)
            # b = a.flip(2)
            # print(b)
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            if i_iter % config.PRINT_FREQ == 0:
                            
                msg1 = 'Val: [{0}/{1}]'.format(i_iter, len(testloader))
                msg2 =  'Loss: {:.6f}' .format(ave_loss.average())                            
                logging.info(msg1)
                logging.info(msg2)
            
            confusion_matrix += get_confusion_matrix(
                label_par[0],
                pred_par,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            if isinstance(pred_pose,list):
                pred_pose = pred_pose[-1]
                flip_pred_pose = flip_pred_pose[-1]
            if isinstance(pred_pose,list):
                pred_pose = pred_pose[0]
                flip_pred_pose = flip_pred_pose[0]
            pred_pose = pred_pose.data.cpu().numpy()
            flip_pred_pose = flip_pred_pose.data.cpu().numpy()
            pose = np.zeros((num_images, 16, 3))
            # cropped_param = cropped_param_list[scale_multiplier.index(1)]
            for num in range(num_images):
                base_scale = meta['scale'].numpy()[num]
                cropped_param = meta['crop_param'].numpy()[num,:,:]
                for ji in range(0, 16):
                    heatmap = pred_pose[num, ji, :, :].copy()
                    heatmap = cv2.resize(heatmap, (size[-2],size[-1]), interpolation=cv2.INTER_LINEAR)
                    flipped_heatmap = flip_pred_pose[num, flipped_poseidx[ji], :, :].copy()
                    flipped_heatmap = cv2.resize(flipped_heatmap, (size[-2],size[-1]), interpolation=cv2.INTER_LINEAR)
                    flipped_heatmap = cv2.flip(flipped_heatmap,1)
                    heatmap += flipped_heatmap
                    heatmap *= 0.5
                    heatmap = gaussian_filter(heatmap, sigma=3)
                    pred_pos = np.unravel_index(heatmap.argmax(), np.shape(heatmap))
                    pred_x = (pred_pos[1] - cropped_param[0, 2] + cropped_param[0, 0]) / base_scale
                    pred_y = (pred_pos[0] - cropped_param[0, 3] + cropped_param[0, 1]) / base_scale
            
                    pose[num, ji, 0] = pred_x
                    pose[num, ji, 1] = pred_y
                    pose[num, ji, 2] = heatmap[pred_pos[0], pred_pos[1]]
            pred_list[n:n+num_images,:,:] = pose
            n = n+num_images
        print('gene pred csv...')
        #torch.distributed.broadcast(im_list, src=[0,1,2,3])
        #torch.distributed.broadcast(pred_list, src=[0,1,2,3])
        print(len(im_list),len(pred_list))
        
        save_hpe_results_to_lip_format(im_list, pred_list, save_path=pose_pred_path, eval_num=len(im_list))
        torch.distributed.barrier()
        csv_list = glob.glob(config.POSE_PRED_PATH[:-4]+'*.csv')
        print(csv_list)
        list1 = []
        for i in csv_list:
            with open(i, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    list1.append(row)
        list2 = sorted(list1, key=lambda s: s[0])
        print(len(list2))
        pose_pred_path1 = config.POSE_PRED_PATH[:-6]+'{}.csv'.format(rank)
        with open(pose_pred_path1, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in list2:
                writer.writerow(line)
        
        pck_avg = 0.0
        pck_all = calc_pck_lip_dataset(pose_gt_path, pose_pred_path1, method_name='Ours', eval_num=len(list2))
        pck_avg = pck_all[-1][-1]
        #pck_all_list.append(pck_all)
        #pck_avg_list.append(pck_avg)
            

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size
    
    class_name = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
	              'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
	              'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
	              'rightShoe']
    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer.add_scalar('valid_acc', pck_avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
        # Num of correct pixels
        num_cor_pix = np.diag(confusion_matrix)
    
        # Num of gt pixels
        num_gt_pix = confusion_matrix.sum(1)
        print('=' * 50)
    
        # Evaluation 1: overall accuracy
        pixel_acc = num_cor_pix.sum() / confusion_matrix.sum()
        print('>>>', 'pixel accuracy', pixel_acc)
        print('-' * 50)
    
        # Evaluation 2: mean accuracy & per-class accuracy 
        print('Accuracy for each class (pixel accuracy):')
        per_class_acc = num_cor_pix / num_gt_pix
        mean_acc = np.nanmean(per_class_acc)
        for i in range(20):
            print('%-15s: %f' % (class_name[i], per_class_acc[i])) 
        print('>>>', 'mean accuracy', mean_acc)
        print('-' * 50)
    
        # Evaluation 3: mean IU & per-class IU
        union = num_gt_pix + confusion_matrix.sum(0) - num_cor_pix
        per_class_iou = num_cor_pix / union
        mean_iou = np.nanmean(per_class_iou)
        for i in range(20):
            print('%-15s: %f' % (class_name[i], per_class_iou[i]))
        print('>>>', 'mean IoU', mean_iou)
        print('-' * 50)
        '''
        # Evaluation 4: frequency weighted IU
        freq = num_gt_pix / confusion_matrix.sum()
        freq_w_iou = (freq[freq > 0] * per_class_iou[freq > 0]).sum()
        print('>>>', 'fwavacc', freq_w_iou)
        print('=' * 50)
        '''
        
    return print_loss, mean_IoU, IoU_array, pck_avg

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class MulAverageMeter(object):
    def __init__(self, length):
        self.mam = []
        self.length = length
        for i in range(length):
            self.mam.append(AverageMeter())
    def update(self, val, n=1):
        for i in range(self.length):
            self.mam[i].update(val[i], n)
    def reset(self):
        self.mem = []
    def val(self):
        val = []
        for i in range(self.length):
            val.append(self.mam[i].avg)
        return val

def validate_sync2(config, testloader, model, im_list, criterion_pose, criterion_par, writer_dict, device):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    acc = MulAverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    # im_list=[]
    pred_list = np.zeros((len(im_list), 16, 3))
    n = 0
    pose_pred_path = config.POSE_PRED_PATH[:-4]+'{}.csv'.format(rank)
    pose_gt_path = config.POSE_GT_PATH
    with torch.no_grad():
        for i_iter, batch in enumerate(testloader):
#            image, label, _, _ = batch
            # if i_iter==1:
            #     break
            image, label_par, label_pose, meta = batch
            pose_weight = meta['pose_weight'].to(device)
            # flip_img = torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).to(device)
            flip_img = image.flip(3).to(device)
            num_images = image.size(0)
            size = label_par[0].size()
            image = image.to(device)
            label_par[0] = label_par[0].long().to(device)
            label_par[1] = label_par[1].long().to(device)
            if isinstance(label_pose,list):
                label_pose[0] = label_pose[0][:,:-1,:,:].float().to(device)
                label_pose[1] = label_pose[1][:,:-1,:,:].float().to(device)
            else:
                label_pose = label_pose[:,:-1,:,:].float().to(device)
            pred_pose, pred_par = model(image)
            flip_pred_pose, flip_pred_par = model(flip_img)
            flipped_poseidx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14, 15]
            
            
            losses_par = criterion_par(pred_par, label_par)
            losses_par = torch.unsqueeze(losses_par,0)
            
            losses_pose = criterion_pose(pred_pose, label_pose, target_weight=pose_weight)
            losses_pose = torch.unsqueeze(losses_pose,0)  
            #losses = losses_par*par_weight + losses_pose*pose_weight
            #losses = losses_par * torch.exp(-model.module.lamda2) + losses_pose * torch.exp(-model.module.lamda1) + model.module.lamda2 + model.module.lamda1
            losses = losses_par + losses_pose
         #   losses, pred = model(image, label)
            
            if isinstance(pred_par,list):
                pred_par=pred_par[-1]
                flip_pred_par=flip_pred_par[-1]
            if isinstance(pred_par,list):
                pred_par=pred_par[0]
                flip_pred_par=flip_pred_par[0]
            pred_par = F.interpolate(input=pred_par, size=(
                        size[-2], size[-1]), mode='bilinear')
            flip_pred_par = F.interpolate(input=flip_pred_par, size=(
                        size[-2], size[-1]), mode='bilinear')
            tmp = flip_pred_par
            flip_pred_par[:,14,:,:] = tmp[:,15,:,:]
            flip_pred_par[:,15,:,:] = tmp[:,14,:,:]
            flip_pred_par[:,16,:,:] = tmp[:,17,:,:]
            flip_pred_par[:,17,:,:] = tmp[:,16,:,:]
            flip_pred_par[:,18,:,:] = tmp[:,19,:,:]
            flip_pred_par[:,19,:,:] = tmp[:,18,:,:]
            flip_pred_par = flip_pred_par.flip(3)
            pred_par = 0.5*(pred_par + flip_pred_par)
            # a = torch.rand(3,3,3)
            # print(a)
            # b = a.flip(2)
            # print(b)
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            if i_iter % config.PRINT_FREQ == 0:
                            
                msg1 = 'Val: [{0}/{1}]'.format(i_iter, len(testloader))
                msg2 =  'Loss: {:.6f}' .format(ave_loss.average())                            
                logging.info(msg1)
                logging.info(msg2)
            
            confusion_matrix += get_confusion_matrix(
                label_par[0],
                pred_par,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            if isinstance(pred_pose,list):
                pred_pose = pred_pose[-1]
                flip_pred_pose = flip_pred_pose[-1]
            if isinstance(pred_pose,list):
                pred_pose = pred_pose[0]
                flip_pred_pose = flip_pred_pose[0]
            pred_pose = pred_pose.data.cpu().numpy()
            flip_pred_pose = flip_pred_pose.data.cpu().numpy()
            flip_pred_pose[:, :, :, 1:] = flip_pred_pose.copy()[:, :, :, 0:-1]
            pose = np.zeros((num_images,16, 3))
            # cropped_param = cropped_param_list[scale_multiplier.index(1)]
            for num in range(num_images):
                #base_scale = meta['scale'].numpy()[num]
                ori_size = meta['size'].numpy()[num]
                center = np.array([ori_size[1] / 2, ori_size[0] / 2])
                long_size = max(ori_size[0], ori_size[1])
                scales = np.array([long_size, long_size])
                #cropped_param = meta['crop_param'].numpy()[num, :, :]
                trans = get_affine_transform2(center, scales, 0, [pred_pose.shape[2], pred_pose.shape[3]], inv=1)
                for ji in range(0, 16):
                    heatmap = pred_pose[num, ji, :, :].copy()
                    #heatmap = cv2.resize(heatmap, (size[-2],size[-1]), interpolation=cv2.INTER_LINEAR)
                    flipped_heatmap = flip_pred_pose[num, flipped_poseidx[ji], :, :].copy()
                    #flipped_heatmap = cv2.resize(flipped_heatmap, (size[-2],size[-1]), interpolation=cv2.INTER_LINEAR)
                    flipped_heatmap = cv2.flip(flipped_heatmap,1)
                    heatmap += flipped_heatmap
                    heatmap *= 0.5
                    #heatmap = gaussian_filter(heatmap, sigma=3)
                    pred_pos = np.unravel_index(heatmap.argmax(), np.shape(heatmap))
                    #pred_x = (pred_pos[1] - cropped_param[0, 2] + cropped_param[0, 0]) / base_scale
                    #pred_y = (pred_pos[0] - cropped_param[0, 3] + cropped_param[0, 1]) / base_scale
                    # pose[num, ji, 0] = pred_x
                    # pose[num, ji, 1] = pred_y
                    pose[num, ji, 0] = pred_pos[1]
                    pose[num, ji, 1] = pred_pos[0]
                    pose[num, ji, 2] = heatmap[pred_pos[0], pred_pos[1]]
                    if config.TEST.POST_PROCESS:
                        px = int(math.floor(pred_pos[1] + 0.5))
                        py = int(math.floor(pred_pos[0] + 0.5))
                        if 1 < px < heatmap.shape[1] - 1 and 1 < py < heatmap.shape[0] - 1:
                            diff = np.array([heatmap[py][px + 1] - heatmap[py][px - 1],
                                             heatmap[py + 1][px] - heatmap[py - 1][px]])
                            pose[num, ji, :] += np.sign(diff) * .25

                    pose[num, ji, 0:2] = affine_transform(pose[num, ji, 0:2].copy(), trans)

            #pose = affine_transform()
            # pred_list[n:n+num_images,:,:] = pose
            # n = n+num_images
            # for ji in range(0, 16):
            #     pred_pose[:, ji, :, :] = 0.5 * (pred_pose[:, ji, :, :] + flip_pred_pose[:, flipped_poseidx[ji], :, :])
            # acc1, avg_acc, cnt, _ = accuracy(pred_pose, label_pose.data.cpu().numpy())
            # acc.update(acc1, cnt)
            pck_table_output_lip_dataset(acc.val())
            
        print('gene pred csv...')
        #torch.distributed.broadcast(im_list, src=[0,1,2,3])
        #torch.distributed.broadcast(pred_list, src=[0,1,2,3])
        print(len(im_list),len(pred_list))

        save_hpe_results_to_lip_format(im_list, pred_list, save_path=pose_pred_path, eval_num=len(im_list))
        torch.distributed.barrier()
        csv_list = glob.glob(config.POSE_PRED_PATH[:-4]+'*.csv')
        print(csv_list)
        list1 = []
        for i in csv_list:
            with open(i, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    list1.append(row)
        list2 = sorted(list1, key=lambda s: s[0])
        print(len(list2))
        pose_pred_path1 = config.POSE_PRED_PATH[:-6]+'{}.csv'.format(rank)
        with open(pose_pred_path1, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in list2:
                writer.writerow(line)
        
        pck_avg = 0.0
        pck_all = calc_pck_lip_dataset(pose_gt_path, pose_pred_path1, method_name='Ours', eval_num=len(list2))
        pck_avg = pck_all[-1][-1]
        #pck_all_list.append(pck_all)
        #pck_avg_list.append(pck_avg)
            

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size
    
    class_name = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
	              'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
	              'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
	              'rightShoe']
    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer.add_scalar('valid_acc', pck_avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
        # Num of correct pixels
        num_cor_pix = np.diag(confusion_matrix)
    
        # Num of gt pixels
        num_gt_pix = confusion_matrix.sum(1)
        print('=' * 50)
    
        # Evaluation 1: overall accuracy
        pixel_acc = num_cor_pix.sum() / confusion_matrix.sum()
        print('>>>', 'pixel accuracy', pixel_acc)
        print('-' * 50)
    
        # Evaluation 2: mean accuracy & per-class accuracy 
        print('Accuracy for each class (pixel accuracy):')
        per_class_acc = num_cor_pix / num_gt_pix
        mean_acc = np.nanmean(per_class_acc)
        for i in range(20):
            print('%-15s: %f' % (class_name[i], per_class_acc[i])) 
        print('>>>', 'mean accuracy', mean_acc)
        print('-' * 50)
    
        # Evaluation 3: mean IU & per-class IU
        union = num_gt_pix + confusion_matrix.sum(0) - num_cor_pix
        per_class_iou = num_cor_pix / union
        mean_iou = np.nanmean(per_class_iou)
        for i in range(20):
            print('%-15s: %f' % (class_name[i], per_class_iou[i]))
        print('>>>', 'mean IoU', mean_iou)
        print('-' * 50)
        '''
        # Evaluation 4: frequency weighted IU
        freq = num_gt_pix / confusion_matrix.sum()
        freq_w_iou = (freq[freq > 0] * per_class_iou[freq > 0]).sum()
        print('>>>', 'fwavacc', freq_w_iou)
        print('=' * 50)
        '''
        
    return print_loss, mean_IoU, IoU_array, pck_avg

def testval(config, test_dataset, testloader, model, device,
        sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, label_edge, meta = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)
            label_edge=label_edge.long().to(device)
            pred = test_dataset.multi_scale_inference(
                        model,
                        image,
                        scales=config.TEST.SCALE_LIST,
                        flip=config.TEST.FLIP_TEST)
            # if isinstance(pred,list):
            #     pred=pred[0]
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(pred,(size[-2], size[-1]),
                                   mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
#                test_dataset.save_pred(pred, sv_path, meta['name'])
                palette = get_palette(config.DATASET.NUM_CLASSES)
                parsing_result_path = os.path.join(sv_path, meta['name'][:-4]+'.png')
                output_img = Image.fromarray(np.asarray(pred, dtype=np.uint8))
                output_img.putpalette(palette)
                output_img.save(parsing_result_path)
            
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model, device,logger,
        sv_dir='', sv_pred=False):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, label, label_edge, meta = batch
            # size = label.size()
            size=meta['size'].squeeze(0)
            print(size)
            # print(image.size())
            # print(meta)
            image = image.to(device)
            label = label.long().to(device)
            #print(type(label))
            label_edge=label_edge.long().to(device)
            #multi-scale
            # pred = test_dataset.multi_scale_inference(
            #             model, 
            #             image, 
            #             scales=config.TEST.SCALE_LIST, 
            #             flip=config.TEST.FLIP_TEST)
            label=np.asarray(label.cpu().squeeze(0),dtype=np.uint8)
            #singel scale
            pred = test_dataset.inference(
                        model, 
                        image, 
                        flip=config.TEST.FLIP_TEST)

            # print('b')
            # print(pred.size())
            # logger.info(pred)
            # print(label)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(pred, (size[-2], size[-1]), 
                                   mode='bilinear')
            # if label.size()[-2] != size[0] or label.size()[-1] != size[1]:
                # label = F.interpolate(label.unsqueeze(1), (size[-2], size[-1]), 
                #                    mode='bilinear')
            label = cv2.resize(label, (size[-1], size[-2]), interpolation=cv2.INTER_NEAREST)
            output = pred.cpu().numpy().transpose(0, 2, 3, 1)
            # print(output.shape)
            seg_pred = np.argmax(output, axis=3)
            # label=label.squeeze(1)
            # print(seg_pred)
            # print(seg_pred.squeeze(0).shape)
            
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
#                test_dataset.save_pred(pred, sv_path, meta['name'])
                palette = get_palette(config.DATASET.NUM_CLASSES)
                # palette1=get_palette(2)
                parsing_result_path = os.path.join(sv_path, meta['name'][0]+'.png')
                output_img = Image.fromarray(np.asarray(seg_pred.squeeze(0), dtype=np.uint8))
                output_img.putpalette(palette)
                output_img.save(parsing_result_path)
                label_img = Image.fromarray(label)
                label_img.putpalette(palette)
                label_img.save(os.path.join(sv_path, meta['name'][0]+'_label.png'))
                # plt.imshow(label_img)
                # plt.show(label_img)

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
if __name__ == '__main__':
    palatte = get_palette(150)
    pass