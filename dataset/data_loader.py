import matplotlib
# matplotlib.use('Qt5Agg')

import torch.utils.data as data
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import cv2
import numpy as np
import os
import os.path
import json
import random
import time

import dataset.data_augmentation as data_aug
import dataset.joint_transformation as joint_trans
import dataset.target_generation as target_gen
import dataset.vis_utils as vis_utils

# Use PIL to load image
def pil_loader(path):
    return Image.open(path).convert('RGB')

# Use opencv to load image
def opencv_loader(path):
    return cv2.imread(path, 1)

# LIP dataset Pose and Parsing
class LIPDataset(data.Dataset):
    def __init__(self,root, im_root, pose_anno_file, parsing_anno_root, transform=None, 
                                                                   target_transform=None, 
                                                                   loader=opencv_loader, 
                                                                   pose_net_stride=4, 
                                                                   sigma=7,
                                                                   parsing_net_stride=1,
                                                                   crop_size=256,
                                                                   target_dist=1.171, scale_min=0.7, scale_max=1.3,
                                                                   max_rotate_degree=40,
                                                                   max_center_trans=40,
                                                                   flip_prob=0.5,
                                                                   is_visualization=False,
                                                                   sample=-1,
                                                                   pose_aux=False,
                                                                   is_train=True,
                                                                   inv_order=False):
        
        self.root = root
        pose_anno_file = os.path.join(self.root, pose_anno_file)
        
        # Load train json file
        print('Loading training json file: {0}...'.format(pose_anno_file))
        train_list = []
        with open(pose_anno_file) as data_file:
            data_this = json.load(data_file)
            data_this = data_this['root']
            train_list = train_list + data_this
        print('Finished loading training json file')

        # Hyper-parameters
        self.im_root = os.path.join(self.root, im_root)
        self.parsing_anno_root = os.path.join(self.root, parsing_anno_root)
        self.pose_anno_list = train_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.pose_net_stride = pose_net_stride
        self.sigma = sigma
        self.parsing_net_stride = parsing_net_stride
        self.crop_size = crop_size
        self.target_dist = target_dist
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.max_rotate_degree = max_rotate_degree
        self.max_center_trans = max_center_trans
        self.flip_prob = flip_prob
        self.is_visualization = is_visualization
        self.is_train = is_train
        self.pose_aux = pose_aux
        if sample != -1:
            if not inv_order:
                self.pose_anno_list = self.pose_anno_list[0:sample]
            else:
                self.pose_anno_list = self.pose_anno_list[-sample:]

        # Number of train samples
        self.N_train = len(self.pose_anno_list)
    
    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.zeros((joints.shape[0], 1), dtype=np.float32)
        for i in range(len(joints_vis)):
            if joints_vis[i]:
                target_weight[:, 0] = 1

        # assert self.target_type == 'gaussian', \
            # 'Only support gaussian map now!'
        heatmap_size = (self.crop_size[1]//self.pose_net_stride,
                               self.crop_size[0]//self.pose_net_stride)
        if True:
            target = np.zeros((joints.shape[0]+1,
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma

            for joint_id in range(joints.shape[0]):
                feat_stride = self.pose_net_stride
                mu_x = int(joints[joint_id][0] / feat_stride + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            max_heatmap = target.max(0)
            target[-1, :, :] = 1 - max_heatmap

        # if self.use_different_joints_weight:
        # target_weight = np.multiply(target_weight)

        return target, target_weight
    
    def __getitem__(self, index):
        # Select a training sample
        # index = 0
        train_item = self.pose_anno_list[index]

        # Load training image
        im_name = train_item['im_name']
        im = self.loader(os.path.join(self.im_root, im_name))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        size = im.shape[:2]

        # Get parsing annotation 
        name_prefix = im_name.split('.')[0]
        parsing_anno_name = name_prefix + '.png'
        parsing_anno_path = os.path.join(self.parsing_anno_root, parsing_anno_name)
        parsing_anno = cv2.imread(parsing_anno_path, 0)

        # Get pose annotation 
        joints_all_info = np.array(train_item['joint_self'])
        joints_loc = np.zeros((joints_all_info.shape[0], 2))
        joints_loc[:, :] = joints_all_info[:, 0:2]

        # Reorder joints from MPI to ours
        joints_loc = joint_trans.transform_mpi_to_ours(joints_loc)

        # Get visibility of joints (The visibility information provided by the annotation is not accurate)
        coord_sum = np.sum(joints_loc, axis=1)
        visibility = coord_sum != 0

        # Get person center and scale
        person_center = np.array([train_item['objpos']])
        scale_provided = train_item['scale_provided']

        if self.is_train:
            # Random scaling
            scaled_im, scale_param = data_aug.augmentation_scale(im, scale_provided, target_dist=self.target_dist, scale_min=self.scale_min, scale_max=self.scale_max, istrain=self.is_train)
            scaled_joints, scaled_center = joint_trans.scale_coords(joints_loc, person_center, scale_param)
    
            # Random rotating
            rotated_im, rotate_param = data_aug.augmentation_rotate(scaled_im, max_rotate_degree=self.max_rotate_degree)
            rotated_joints, rotated_center = joint_trans.rotate_coords(scaled_joints, scaled_center, rotate_param)
    
            # Random cropping
            cropped_im, crop_param = data_aug.augmentation_cropped(rotated_im, rotated_center, crop_x=self.crop_size[0], crop_y=self.crop_size[1], max_center_trans=self.max_center_trans)
            cropped_joints, cropped_center = joint_trans.crop_coords(rotated_joints, rotated_center, crop_param)
            
            # Random flipping
            flipped_im, flip_param = data_aug.augmentation_flip(cropped_im, flip_prob=self.flip_prob)
            flipped_joints, flipped_center = joint_trans.flip_coords(cropped_joints, cropped_center, flip_param, flipped_im.shape[1])
    
            # If flip, then swap the visibility of left and right joints
            if flip_param:
                right_idx = [2, 3, 4, 8, 9, 10]
                left_idx = [5, 6, 7, 11, 12, 13]
                for i in range(0, 6):
                    temp_visibility = visibility[right_idx[i]]
                    visibility[right_idx[i]] = visibility[left_idx[i]]
                    visibility[left_idx[i]] = temp_visibility
            # Generate parsing target maps 
            parsing_target = target_gen.gen_parsing_target(parsing_anno, 
                                                           scale_param=scale_param, 
                                                           rotate_param=[rotate_param, rotated_im.shape[1], rotated_im.shape[0]],
                                                           crop_param=[crop_param, cropped_im.shape[1], cropped_im.shape[0]],
                                                           flip_param=flip_param,
                                                           stride=self.parsing_net_stride)
        else:
            scaled_im, scale_param = data_aug.augmentation_scale(im, scale_provided, target_dist=self.target_dist, scale_min=self.scale_min, scale_max=self.scale_max, istrain=self.is_train)
            scaled_joints, scaled_center = joint_trans.scale_coords(joints_loc, person_center, scale_param)
            
            cropped_im, crop_param = data_aug.augmentation_cropped(scaled_im, scaled_center, crop_x=self.crop_size[0], crop_y=self.crop_size[1], max_center_trans=0)
            cropped_joints, cropped_center = joint_trans.crop_coords(scaled_joints,scaled_center, crop_param)

            #flipped_im = cv2.resize(im,self.crop_size,cv2.INTER_LINEAR)
            #flipped_joints = data_aug.ori_joints_trans(joints_loc,size,self.crop_size)
            # flipped_joints[:,[0,1]] = flipped_joints[:,[1,0]]
            flipped_im = cropped_im
            flipped_joints = cropped_joints
            flipped_center = cropped_center
            # parsing_target = parsing_anno
            parsing_target = target_gen.gen_parsing_target(parsing_anno, 
                                                           scale_param=scale_param, 
                                                           crop_param=[crop_param, cropped_im.shape[1], cropped_im.shape[0]],
                                                           stride=self.parsing_net_stride)
        # Generate pose target maps
        grid_x = flipped_im.shape[1] // self.pose_net_stride
        grid_y = flipped_im.shape[0] // self.pose_net_stride
        BODY_PARTS = [[1, 0],
                      [1, 2], [2, 3], [3, 4],
                      [1, 5], [5, 6], [6, 7],
                      [1, 14], [14, 15],
                      [15, 8], [8, 9], [9, 10],
                      [15, 11], [11, 12], [12, 13]]
        pose_target, pose_target_aux = target_gen.gen_pose_target(flipped_joints, visibility, self.pose_net_stride,
                                                                  grid_x, grid_y, self.sigma, aux=self.pose_aux)
        # pose_target, pose_target_aux = target_gen.gen_pose_target2(flipped_joints, visibility, BODY_PARTS, self.pose_net_stride,
        #                                                           grid_x, grid_y, self.sigma, aux=self.pose_aux)
        pose_weight = np.zeros((len(visibility), 1))
        for i in range(len(visibility)):
            if visibility[i]:
                pose_weight[i, 0] = 1
        pose_weight = torch.from_numpy(pose_weight).float()

        # Generate parsing target maps 
        # parsing_target = target_gen.gen_parsing_target(parsing_anno, 
        #                                                scale_param=scale_param, 
        #                                                rotate_param=[rotate_param, rotated_im.shape[1], rotated_im.shape[0]],
        #                                                crop_param=[crop_param, cropped_im.shape[1], cropped_im.shape[0]],
        #                                                flip_param=flip_param,
        #                                                stride=self.parsing_net_stride)

        # Transform
        # flipped_im = flipped_im / 255.0
        # flipped_im = cv2.cvtColor(flipped_im, cv2.COLOR_BGR2RGB)
        flipped_im = np.array(flipped_im, dtype='uint8')
        if self.transform is not None:
            aug_im = self.transform(flipped_im)
            # print(100)
        else:
            aug_im = flipped_im
        
        # Visualize target maps
        if self.is_visualization:
            print('Visualize pose targets')
            vis_utils.vis_gaussian_maps(flipped_im, pose_target, self.pose_net_stride, save_im=True)
            vis_utils.vis_parsing_maps(flipped_im, parsing_target, self.parsing_net_stride, save_im=True)
        
        parsing_edge = target_gen.generate_edge(parsing_target)
        parsing_target = cv2.resize(parsing_target,self.crop_size,interpolation=cv2.INTER_NEAREST)
        parsing_edge = cv2.resize(parsing_edge,self.crop_size,interpolation=cv2.INTER_NEAREST)
        parsing_edge[parsing_target==255] = 255
        parsing_target = torch.from_numpy(parsing_target)
        parsing_edge = torch.from_numpy(parsing_edge)
        parsing_target = [parsing_target,parsing_edge]
        pose_target = torch.from_numpy(pose_target)
        if pose_target_aux is not None:
            pose_target_aux = torch.from_numpy(pose_target_aux)
            pose_target = [pose_target, pose_target_aux]
        
        meta = {
            'name':name_prefix,
            'size':size,
            'joints':flipped_joints,
            'visiable':visibility,
            'center':flipped_center,
            'scale':scale_param,
            'crop_param':crop_param,
            'pose_weight':pose_weight,
            }
        
        return aug_im, parsing_target, pose_target, meta
    
    def __len__(self):
        return self.N_train
    
    

if __name__ == '__main__':
    print('Data loader for Human Pose Estimation with Parsing Induced Learner on LIP dataset')
    # Image normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Data transform
    data_transform = transforms.Compose([transforms.ToTensor(), normalize,])
    lip_ds = LIPDataset(r'E:\database\LIP', \
                        r'val_images', \
                        r'E:\database\LIP\jsons\LIP_SP_VAL_annotations.json', \
                        r'val_segmentations', \
                        transform=data_transform, \
                        sigma=7, \
                        pose_net_stride=4, \
                        parsing_net_stride=1, \
                        crop_size=(384,384), \
                        target_dist=1.171, scale_min=0.7, scale_max=1.3, \
                        max_rotate_degree=40, \
                        max_center_trans=40, \
                        flip_prob=1, \
                        is_visualization=False,
                        pose_aux=True,
                        is_train=False)
    d_loader = torch.utils.data.DataLoader(lip_ds,
                                           batch_size=12,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True,
                                           # sampler = d_sampler,
                                           )

    for i, batch in enumerate(d_loader):
        # print(i)
        _, par, pose, meta = batch
        print(meta['name'])
        if '486491_1284095' in meta['name']:
            break
    len(lip_ds)
    a=lip_ds[0]
    image=a[0].numpy()
    b=a[1]
    tmp = b[0].numpy()
    
    c=a[2]
    d=c[0].numpy()
    d2=c[1].numpy()
    e=a[3]
    f= d[-1]
    f2 = d2[-1]
    # f3 = d[0]
    # plt.imshow(f3)
    
    im_path = r'E:\database\LIP\val_images\100034_483681.jpg'
    self = lip_ds
    im = self.loader(im_path)
    im2 = self.transform(im).numpy()
    
    from matplotlib import pyplot as plt
    # for i in [5,50,500,1000,1050,1350]:
    plt.imshow(tmp)
    a=lip_ds[16]
    image=a[0].numpy()
    plt.imshow(image.transpose((1,2,0)))
    # plt.imshow(image.transpose((1,2,0)))
    plt.imshow(cv2.flip(image.transpose((1,2,0)), 1))
    plt.imshow(b[0].numpy())
    # plt.imshow(a[0].numpy())
    plt.imshow(f)
    plt.imshow(f2)
    joints = a[3]['joints']
    vis = a[3]['visiable']
    pose_weight = a[3]['pose_weight']
    len(lip_ds)
    new_lipds = data.ConcatDataset([lip_ds, lip_ds])
    # newa = new_lipds[10000]
    # lip_ds.__add__(lip_ds)
    # target, target_weight = lip_ds.generate_target(joints, vis)
    # plt.imshow(target[0])
    # for i in range(len(lip_ds)):
    #     if i < 7000:
    #         pass
    #     else:
    #         a = lip_ds[i]
    #         print(a[3]['name'])
    #         if a[3]['name'] == '486491_1284095':
    #             break
    # # d.shape
    # #d_sampler = torch.utils.data.distributed.DistributedSampler(lip_ds)

    '''
    from matplotlib import pyplot as plt
    from core.evaluate import accuracy
    
    plt.imshow(d[-1])
    plt.imshow(d2[-1])
    j=e['joints']
    b[154,13]
    # plt.show()
    
    e=a[3]
    f=a[1][:-1]
    f=f.unsqueeze(0)
    f2=torch.cat([f,f],dim=0)
    f2.shape
    g=torch.rand((2,16,64,64))
    g[0,0,:,:]=f2[0,0,:,:]
    acc, avg_acc, cnt, pred = accuracy(g.cpu().numpy(), f2.cpu().numpy())

    f1,f12=get_max_preds(f2.numpy())
    f2,_=get_max_preds(f.numpy())
    h = f.shape[2]
    w = f.shape[3]
    norm = np.ones((f1.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(f1, f2, norm)
    dist_acc(dists[6])
    batch_heatmaps = a[1][:-1].unsqueeze(0).numpy()
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    # a[0].shape
    # data_transform = transforms.Compose([transforms.ToTensor()])
    # t=data_transform(a[0])  
    # t=transforms.ToTensor()(d)
    # print(t.numpy())
    output=g.numpy()
    target=f2.numpy()
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    
    
    '''