import matplotlib
# matplotlib.use('Qt5Agg')

import torch.utils.data as data
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import glob
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
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
import scipy.io as scio

# Use PIL to load image
def pil_loader(path):
    return Image.open(path).convert('RGB')


# Use opencv to load image
def opencv_loader(path):
    return cv2.imread(path, 1)

def IoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1) #C的面积
    garea = (gx2 - gx1) * (gy2 - gy1) #G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h #C∩G的面积

    iou = area / (carea + garea - area)

    return iou
# LIP dataset Pose and Parsing
class PPPDataset(data.Dataset):
    def __init__(self, root,
                 im_root,
                 im_list_path,
                 pose_anno_path,
                 parsing_anno_path,
                 mask_path,
                 transform=None,
                 target_transform=None,
                 pose_net_stride=4,
                 sigma=7,
                 parsing_net_stride=1,
                 crop_size=256,
                 target_dist=1.171,
                 scale_min=0.5,
                 scale_max=1.25,
                 max_rotate_degree=40,
                 max_center_trans=40,
                 flip_prob=0.5,
                 is_visualization=False,
                 sample=-1,
                 pose_aux=False,
                 is_train=True,
                 inv_order=False):

        self.root = root
        self.pose_anno_path = os.path.join(self.root, pose_anno_path)
        self.parsing_anno_path = os.path.join(self.root, parsing_anno_path)
        self.mask_path = os.path.join(self.root, mask_path)
        self.im_list = [line.strip() for line in open(os.path.join(root, im_list_path))]
        self.db = self.get_db(self.im_list)
        print('Finished preparing database')
        # self.pose_path_list = sorted(glob.glob(pose_anno_path+r'/*'))
        # self.par_path_list = sorted(glob.glob(parsing_anno_path+r'/*'))
        # Load train json file
        # print('Loading training json file: {0}...'.format(pose_anno_file))

        # Hyper-parameters
        self.im_root = os.path.join(self.root, im_root)
        self.transform = transform
        self.target_transform = target_transform
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
                self.db = self.db[0:sample]
            else:
                self.db = self.db[-sample:]

        # Number of train samples
        self.N_train = len(self.db)

    def get_db(self, im_list):
        db = []
        for i in range(len(im_list)):
            im_name = im_list[i]
            pose_path = os.path.join(self.pose_anno_path, im_name+'.mat')
            mask_path = os.path.join(self.mask_path, im_name+'.npy')
            mask_dict = np.load(mask_path, allow_pickle=True).item()
            parsing_path = os.path.join(self.parsing_anno_path, im_name+'.png')
            classes = mask_dict['pred_classes']
            masks = mask_dict['pred_masks']
            mask_boxes = mask_dict['boxes']
            person_mask = np.where(classes == 0)
            prior_boxes = mask_boxes[person_mask]
            prior_masks = masks[person_mask]

            if not os.path.isfile(pose_path):
                continue
            else:
                pose_labels = scio.loadmat(pose_path)
                boxes = pose_labels['boxes']
                joints = pose_labels['joints']
                assert boxes.shape[1] == joints.shape[1]
                cost = np.zeros((boxes.shape[1], prior_masks.shape[0]))
                for m in range(boxes.shape[1]):
                    for n in range(prior_masks.shape[0]):
                        cost[m, n] = 1 - IoU(boxes[0, m][0].astype(np.float32), prior_boxes[n])
                gt_index, prior_index = linear_assignment(cost)
                # for j in range(len(gt_index)):


                for j in range(len(gt_index)):
                    if cost[gt_index[j], prior_index[j]] >0.3:
                        continue
                    else:
                        data = {}
                        box = boxes[0, gt_index[j]]
                        joint = joints[0, gt_index[j]]
                        mask = prior_masks[prior_index[j]]
                        data['im_name'] = im_name
                        data['box'] = box
                        data['joint'] = joint
                        data['mask'] = mask
                        db.append(data)
        return db

    def __getitem__(self, index):
        # Select a training sample
        # background(id: 0), head(id: 1), torso(id: 2), upper - arm(id: 3), lower - arm(id: 4),
        # upper - leg(id: 5), and lower - leg(id: 6)
        # index = 0
        train_item = self.db[index]
        box = train_item['box'].astype(np.int32)
        mask = train_item['mask']
        # Load training image
        im_name = train_item['im_name']
        ori_im = cv2.imread(os.path.join(self.im_root, im_name+'.jpg'),1)
        ori_im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
        ori_size = ori_im.shape[:2]
        im = ori_im[box[0,1]:box[0,3], box[0,0]:box[0,2], :].copy()
        size = im.shape[:2]

        # Get Mask for parsing
        # mask_path = os.path.join(self.mask_path, im_name + '.npy')
        # mask_dict = np.load(mask_path, allow_pickle=True).item()
        # classes = mask_dict['pred_classes']
        # masks = mask_dict['pred_masks']
        # mask_boxes = mask_dict['boxes']
        # person_mask = np.where(classes==0)
        # prior_boxes = mask_boxes[person_mask]
        # prior_masks = masks[person_mask]
        # cost = np.zeros((box.shape[0], prior_masks.shape[0]))
        # for m in range(box.shape[0]):
        #     for n in range(prior_masks.shape[0]):
        #         cost[m, n] = 1 - IoU(box[m], prior_boxes[n])
        # gt_index, prior_index = linear_assignment(cost)

        # Get parsing annotation
        parsing_anno_path = os.path.join(self.parsing_anno_path, im_name+'.png')
        parsing_anno = cv2.imread(parsing_anno_path, 0)
        parsing_anno = parsing_anno * mask
        parsing_anno2 = parsing_anno[box[0, 1]:box[0, 3], box[0, 0]:box[0, 2]].copy()
        # if cost[gt_index, prior_index] <= 0.5:
        #     parsing_anno = parsing_anno * prior_masks[prior_index[0]]
        #     parsing_anno2 = parsing_anno[box[0, 1]:box[0, 3], box[0, 0]:box[0, 2]].copy()
        # else:
        #     parsing_anno2 = parsing_anno[box[0,1]:box[0,3], box[0,0]:box[0,2]].copy()

        # Get pose annotation
        # 0 forehead, 1 neck,
        # 2 left shoulder, 3 left elbow, 4 left wrist, 5 left hip, 6 left knee, 7 left ankle,
        # 8 right shoulder, 9 right elbow, 10 right wrist, 11 right hip, 12 right knee, 13right ankle
        joints_all_info = np.array(train_item['joint'])
        joints_loc = np.zeros((joints_all_info.shape[0], 2))
        # joints_loc[:, :] = joints_all_info[:, 0:2]
        joints_loc[:, 0] = joints_all_info[:, 0] - box[0, 0]
        joints_loc[:, 1] = joints_all_info[:, 1] - box[0, 1]

        # Reorder joints from MPI to ours
        # joints_loc = joint_trans.transform_mpi_to_ours(joints_loc)

        # Get visibility of joints (The visibility information provided by the annotation is not accurate)
        # coord_sum = np.sum(joints_loc, axis=1)
        visibility = joints_all_info[:, 2] != 0

        # Get person center and scale
        person_center = np.array([[(box[0, 2]-box[0, 0])/2, (box[0, 3]-box[0, 1])/2]])
        scale_provided = 1.

        if self.is_train:
            # Random scaling
            scaled_im, scale_param = data_aug.augmentation_scale(im, scale_provided, target_dist=self.target_dist,
                                                                 scale_min=self.scale_min, scale_max=self.scale_max,
                                                                 istrain=self.is_train)
            scaled_joints, scaled_center = joint_trans.scale_coords(joints_loc, person_center, scale_param)

            # Random rotating
            rotated_im, rotate_param = data_aug.augmentation_rotate(scaled_im, max_rotate_degree=self.max_rotate_degree)
            rotated_joints, rotated_center = joint_trans.rotate_coords(scaled_joints, scaled_center, rotate_param)

            # Random cropping
            cropped_im, crop_param = data_aug.augmentation_cropped(rotated_im, rotated_center, crop_x=self.crop_size[0],
                                                                   crop_y=self.crop_size[1],
                                                                   max_center_trans=self.max_center_trans)
            cropped_joints, cropped_center = joint_trans.crop_coords(rotated_joints, rotated_center, crop_param)

            # Random flipping
            flipped_im, flip_param = data_aug.augmentation_flip(cropped_im, flip_prob=self.flip_prob)
            flipped_joints, flipped_center = joint_trans.flip_coords(cropped_joints, cropped_center, flip_param,
                                                                     flipped_im.shape[1])

            # If flip, then swap the visibility of left and right joints
            if flip_param:
                right_idx = [2, 3, 4, 5, 6, 7]
                left_idx = [8, 9, 10, 11, 12, 13]
                for i in range(0, 6):
                    temp_visibility = visibility[right_idx[i]]
                    visibility[right_idx[i]] = visibility[left_idx[i]]
                    visibility[left_idx[i]] = temp_visibility
            # Generate parsing target maps
            parsing_target = target_gen.gen_parsing_target_ppp(parsing_anno2,
                                                           scale_param=scale_param,
                                                           rotate_param=[rotate_param, rotated_im.shape[1],
                                                                         rotated_im.shape[0]],
                                                           crop_param=[crop_param, cropped_im.shape[1],
                                                                       cropped_im.shape[0]],
                                                           flip_param=flip_param,
                                                           stride=self.parsing_net_stride)
        else:
            scaled_im, scale_param = data_aug.augmentation_scale(im, scale_provided, target_dist=self.target_dist,
                                                                 scale_min=self.scale_min, scale_max=self.scale_max,
                                                                 istrain=self.is_train)
            scaled_joints, scaled_center = joint_trans.scale_coords(joints_loc, person_center, scale_param)

            cropped_im, crop_param = data_aug.augmentation_cropped(scaled_im, scaled_center, crop_x=self.crop_size[0],
                                                                   crop_y=self.crop_size[1], max_center_trans=0)
            cropped_joints, cropped_center = joint_trans.crop_coords(scaled_joints, scaled_center, crop_param)

            # flipped_im = cv2.resize(im,self.crop_size,cv2.INTER_LINEAR)
            # flipped_joints = data_aug.ori_joints_trans(joints_loc,size,self.crop_size)
            # flipped_joints[:,[0,1]] = flipped_joints[:,[1,0]]
            flipped_im = cropped_im
            flipped_joints = cropped_joints
            flipped_center = cropped_center
            # parsing_target = parsing_anno
            parsing_target = target_gen.gen_parsing_target_ppp(parsing_anno2,
                                                           scale_param=scale_param,
                                                           crop_param=[crop_param, cropped_im.shape[1],
                                                                       cropped_im.shape[0]],
                                                           stride=self.parsing_net_stride)
        # Generate pose target maps
        grid_x = flipped_im.shape[1] // self.pose_net_stride
        grid_y = flipped_im.shape[0] // self.pose_net_stride
        pose_target, pose_target_aux = target_gen.gen_pose_target(flipped_joints, visibility, self.pose_net_stride,
                                                                  grid_x, grid_y, self.sigma, aux=self.pose_aux)
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
        parsing_target = cv2.resize(parsing_target, self.crop_size, interpolation=cv2.INTER_NEAREST)
        parsing_edge = cv2.resize(parsing_edge, self.crop_size, interpolation=cv2.INTER_NEAREST)
        parsing_edge[parsing_target == 255] = 255
        parsing_target = torch.from_numpy(parsing_target)
        parsing_edge = torch.from_numpy(parsing_edge)
        parsing_target = [parsing_target, parsing_edge]
        pose_target = torch.from_numpy(pose_target)
        if pose_target_aux is not None:
            pose_target_aux = torch.from_numpy(pose_target_aux)
            pose_target = [pose_target, pose_target_aux]

        meta = {
            'name': im_name,
            'box': box,
            'ori_size': ori_size,
            'croped_size': size,
            'joints': flipped_joints,
            'visiable': visibility,
            'center': flipped_center,
            'scale': scale_param,
            #'crop_param': crop_param,
            'pose_weight': pose_weight,
        }

        return aug_im, parsing_target, pose_target, meta

    def __len__(self):
        return self.N_train


if __name__ == '__main__':
    print('Data loader for Human Pose Estimation with Parsing Induced Learner on Pascal-person-part dataset')
    # Image normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Data transform
    data_transform = transforms.Compose([transforms.ToTensor(), normalize, ])
    ppp_ds = PPPDataset(r'G:\database\pascal_data', \
                        r'JPEGImages', \
                        r'val_id.txt', \
                        r'PersonJoints', \
                        r'SegmentationPart', \
                        r'Mask', \
                        transform=data_transform, \
                        sigma=6, \
                        pose_net_stride=4, \
                        parsing_net_stride=1, \
                        crop_size=(384, 384), \
                        target_dist=1.171, scale_min=0.7, scale_max=1.3, \
                        max_rotate_degree=40, \
                        max_center_trans=40, \
                        flip_prob=0.5, \
                        is_visualization=False,
                        pose_aux=True,
                        is_train=True)
    data_loader = torch.utils.data.DataLoader(ppp_ds,
                                               batch_size=10,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=True,
                                               pin_memory=True)
    for i, batch in enumerate(data_loader):
        _, _, _, meta = batch
        print(meta['name'])
    for i in range(len(ppp_ds)):
        a = ppp_ds[i]
    a = ppp_ds[10]
    image = a[0].numpy()
    b = a[1]
    tmp = b[0].numpy()

    c = a[2]
    d = c[0].numpy()
    d2 = c[1].numpy()
    e = a[3]
    f = d[-1]
    f2 = d2[-1]
    # f3 = d[0]
    # plt.imshow(f3)

    im_path = r'E:\database\LIP\val_images\100034_483681.jpg'
    self = ppp_ds
    im = self.loader(im_path)
    im2 = self.transform(im).numpy()

    from matplotlib import pyplot as plt

    # for i in [5,50,500,1000,1050,1350]:
    plt.imshow(tmp)
    a = ppp_ds[16]
    image = a[0].numpy()
    plt.imshow(image.transpose((1, 2, 0)))
    # plt.imshow(image.transpose((1,2,0)))
    plt.imshow(cv2.flip(image.transpose((1, 2, 0)), 1))
    plt.imshow(b[0].numpy())
    # plt.imshow(a[0].numpy())
    plt.imshow(f)
    plt.imshow(f2)
    joints = a[3]['joints']
    vis = a[3]['visiable']
    pose_weight = a[3]['pose_weight']
    len(ppp_ds)
    new_lipds = data.ConcatDataset([ppp_ds, ppp_ds])
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
    # d_loader = torch.utils.data.DataLoader(lip_ds,
    #                                        batch_size=12,
    #                                        shuffle=False,
    #                                        num_workers=0,
    #                                        pin_memory=True,
    #                                        # sampler = d_sampler,
    #                                        )

    # for i, batch in enumerate(d_loader):
    #     # print(i)
    #     if i <2000:
    #         pass
    #     else:
    #         _,par,pose,meta = batch
    #         print(meta['name'])
    #         if '486491_1284095' in meta['name']:
    #             break
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