import os
import sys
import numpy as np
import random
import cv2
# from joint_transformation import swap_left_and_right

def flip_joints(joints, im_w, r_joint = [0,1,2,10,11,12], l_joint = [3,4,5,13,14,15]):
    flipped_joints = joints.copy()
    flipped_joints[:, 0] = im_w - 1 - flipped_joints[:, 0]
    flipped_joints = swap_left_and_right(flipped_joints, r_joint, l_joint)
    return flipped_joints

def swap_left_and_right(joints, r_joint = [0,1,2,10,11,12], l_joint = [3,4,5,13,14,15]):

    swapped_joints = joints.copy()
    for i in range(len(r_joint)):
        temp_joint = np.zeros((1, 2))
        temp_joint[0, :] = swapped_joints[r_joint[i], :]
        swapped_joints[r_joint[i], :] = swapped_joints[l_joint[i], :]
        swapped_joints[l_joint[i], :] = temp_joint[0, :]

    return swapped_joints

def gen_parsing_target(parsing_anno, scale_param=None, rotate_param=None, crop_param=None, flip_param=None,stride=8):
	
	parsing_target = parsing_anno.copy()
	
	if scale_param is not None:
		parsing_target = cv2.resize(parsing_target, None, fx=scale_param, fy=scale_param, interpolation=cv2.INTER_NEAREST)
	
	if rotate_param is not None:
		parsing_target = cv2.warpAffine(parsing_target, rotate_param[0], dsize=(int(rotate_param[1]), int(rotate_param[2])), 
		                               flags=cv2.INTER_NEAREST, 
		                               borderMode=cv2.BORDER_CONSTANT, 
		                               borderValue=(255, ))

	if crop_param is not None:
		temp_crop_parsing_target = np.zeros((crop_param[1], crop_param[2])) + 255
		temp_crop_parsing_target[crop_param[0][0, 3]:crop_param[0][0, 7], crop_param[0][0, 2]:crop_param[0][0, 6]] = \
		                      parsing_target[crop_param[0][0, 1]:crop_param[0][0, 5], crop_param[0][0, 0]:crop_param[0][0, 4]]
		parsing_target = temp_crop_parsing_target.astype(np.uint8)

	if flip_param is not None:
		if flip_param:
			parsing_target = cv2.flip(parsing_target, 1)
			# Flip left and right parts
			# Right-arm: 15, Right-leg: 17, Right-shoe: 19
			# Left-arm: 14 , Left-leg: 16, Left-shoe: 18
			right_idx = [15, 17, 19]
			left_idx = [14, 16, 18]
			for i in range(0, 3):
				right_pos = np.where(parsing_target == right_idx[i])
				left_pos = np.where(parsing_target == left_idx[i])
				parsing_target[right_pos[0], right_pos[1]] = left_idx[i]
				parsing_target[left_pos[0], left_pos[1]] = right_idx[i]

	parsing_target = cv2.resize(parsing_target, None, fx=(1.0 / stride), fy=(1.0 / stride), interpolation=cv2.INTER_NEAREST)

	return parsing_target


def gen_parsing_target_ppp(parsing_anno, scale_param=None, rotate_param=None, crop_param=None, flip_param=None, stride=8):
    parsing_target = parsing_anno.copy()

    if scale_param is not None:
        parsing_target = cv2.resize(parsing_target, None, fx=scale_param, fy=scale_param,
                                    interpolation=cv2.INTER_NEAREST)

    if rotate_param is not None:
        parsing_target = cv2.warpAffine(parsing_target, rotate_param[0],
                                        dsize=(int(rotate_param[1]), int(rotate_param[2])),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(255,))

    if crop_param is not None:
        temp_crop_parsing_target = np.zeros((crop_param[1], crop_param[2])) + 255
        temp_crop_parsing_target[crop_param[0][0, 3]:crop_param[0][0, 7], crop_param[0][0, 2]:crop_param[0][0, 6]] = \
            parsing_target[crop_param[0][0, 1]:crop_param[0][0, 5], crop_param[0][0, 0]:crop_param[0][0, 4]]
        parsing_target = temp_crop_parsing_target.astype(np.uint8)

    if flip_param is not None:
        if flip_param:
            parsing_target = cv2.flip(parsing_target, 1)
            # Flip left and right parts
            # pascal-person data does not have flip pair

    parsing_target = cv2.resize(parsing_target, None, fx=(1.0 / stride), fy=(1.0 / stride),
                                interpolation=cv2.INTER_NEAREST)

    return parsing_target

def gen_pose_target(joints, visibility, stride=8, grid_x=46, grid_y=46, sigma=7, aux=False):
    #print "Target generation -- Gaussian maps"

    joint_num = joints.shape[0]
    gaussian_maps = np.zeros((joint_num + 1, grid_y, grid_x))
    for ji in range(0, joint_num):
        if visibility[ji]:
            gaussian_map = gen_single_gaussian_map(joints[ji, :], stride, grid_x, grid_y, sigma)
            gaussian_maps[ji, :, :] = gaussian_map[:, :]

    # Get background heatmap
    max_heatmap = gaussian_maps.max(0)
    gaussian_maps[joint_num, :, :] = 1 - max_heatmap
    
    
    if aux:
        gaussian_maps_aux = np.zeros((joint_num + 1, grid_y, grid_x))
        for ji in range(0, joint_num):
            if visibility[ji]:
                gaussian_map_aux = gen_single_gaussian_map(joints[ji, :], stride, grid_x, grid_y, 2*sigma)
                gaussian_maps_aux[ji, :, :] = gaussian_map_aux[:, :]
    
        # Get background heatmap
        max_heatmap = gaussian_maps_aux.max(0)
        gaussian_maps_aux[joint_num, :, :] = 1 - max_heatmap
        return gaussian_maps, gaussian_maps_aux
    else:
        return gaussian_maps, None


def gen_pose_target2(joints, visibility, BODY_PARTS, stride=8, grid_x=46, grid_y=46, sigma=7, aux=False):
    # print "Target generation -- Gaussian maps"

    joint_num = joints.shape[0]
    gaussian_maps = np.zeros((joint_num + 1, grid_y, grid_x))
    for ji in range(0, joint_num):
        if visibility[ji]:
            gaussian_map = gen_single_gaussian_map(joints[ji, :], stride, grid_x, grid_y, sigma)
            gaussian_maps[ji, :, :] = gaussian_map[:, :]

    # Get background heatmap
    max_heatmap = gaussian_maps.max(0)
    gaussian_maps[joint_num, :, :] = 1 - max_heatmap

    if aux:
        paf = get_paf_by_hm(gaussian_maps, visibility, BODY_PARTS, sigma_paf=2.5)
        paf = np.concatenate((paf, paf.sum(axis=0, keepdims=True)), axis=0)
        return gaussian_maps, paf
    else:
        return gaussian_maps, None

def gen_single_gaussian_map(center, stride, grid_x, grid_y, sigma):
    #print "Target generation -- Single gaussian maps"

    gaussian_map = np.zeros((grid_y, grid_x))
    start = stride / 2.0 - 0.5

    max_dist = np.ceil(np.sqrt(4.6052 * sigma * sigma * 2.0))
    start_x = int(max(0, np.floor((center[0] - max_dist - start) / stride)))
    end_x = int(min(grid_x, np.ceil((center[0] + max_dist - start) / stride)))
    start_y = int(max(0, np.floor((center[1] - max_dist - start) / stride)))
    end_y = int(min(grid_y, np.ceil((center[1] + max_dist - start) / stride)))

    for g_y in range(start_y, end_y):
        for g_x in range(start_x, end_x):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1])
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue
            gaussian_map[g_y, g_x] += np.exp(-exponent)
            if gaussian_map[g_y, g_x] > 1:
                gaussian_map[g_y, g_x] = 1
    return gaussian_map

def get_paf_by_hm(hm, vis, BODY_PARTS, sigma_paf=5, variable_width=False):
    size = hm.shape[-2:]
    out_pafs = np.zeros((len(BODY_PARTS), 2, size[0], size[1]))
    n_person_part = np.zeros((len(BODY_PARTS), size[0], size[1]))
    keypoints = np.zeros((hm.shape[0]-1, 2))
    for i in range(hm.shape[0] - 1):
        hm_tmp = hm[i]
        pos = np.unravel_index(hm_tmp.argmax(), np.shape(hm_tmp))
        keypoints[i, 0] = pos[1]
        keypoints[i, 1] = pos[0]

    for i in range(len(BODY_PARTS)):
        part = BODY_PARTS[i]
        keypoint_1 = keypoints[part[0]]
        keypoint_2 = keypoints[part[1]]
        if vis[part[0]] and vis[part[1]]:
            part_line_segment = keypoint_2 - keypoint_1
            # Notation from paper
            l = np.linalg.norm(part_line_segment)
            if l>1e-2:
                sigma = sigma_paf
                if variable_width:
                    sigma = sigma_paf *  l * 0.025
                v = part_line_segment/l
                v_per = v[1], -v[0]
                x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
                dist_along_part = v[0] * (x - keypoint_1[0]) + v[1] * (y - keypoint_1[1])
                dist_per_part = np.abs(v_per[0] * (x - keypoint_1[0]) + v_per[1] * (y - keypoint_1[1]))
                mask1 = dist_along_part >= 0
                mask2 = dist_along_part <= l
                mask3 = dist_per_part <= sigma
                mask = mask1 & mask2 & mask3
                out_pafs[i, 0] = out_pafs[i, 0] + mask.astype('float32') * v[0]
                out_pafs[i, 1] = out_pafs[i, 1] + mask.astype('float32') * v[1]
                n_person_part[i] += mask.astype('float32')
    n_person_part = n_person_part.reshape(out_pafs.shape[0], 1, size[0], size[1])
    out_pafs = out_pafs/(n_person_part + 1e-8)
    out_pafs = out_pafs.reshape(out_pafs.shape[0]*out_pafs.shape[1], size[0], size[1])
    return out_pafs

def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge
  
def _box2cs(box, aspect_ratio, pixel_std):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio, pixel_std)

def generate_pose(joints, visibility, trans, grid_x, grid_y, stride, sigma):
    joint_num = joints.shape[0] # 16 for lip
    tmp_size = sigma
    # get gaussian ma,p
    gaussian_maps = np.zeros((joint_num+1, grid_y, grid_x))
    target_weight = np.ones((joint_num, 1), dtype=np.float32)
    for i in range(len(joints)):
        if not joints[i]:
            target_weight[:, 0] = 0

    for joint_id in range(0, joint_num):
        mu_x = int(joints[joint_id][0] / stride + 0.5)
        mu_y = int(joints[joint_id][1] / stride + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= grid_x or ul[1] >= grid_y or br[0] < 0 or br[1] < 0:
            target_weight[joint_id] = 0
            continue

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], grid_x) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], grid_y) - ul[1]
        img_x = max(0, ul[0]), min(br[0], grid_x)
        img_y = max(0, ul[1]), min(br[1], grid_y)

        v = target_weight[joint_id]
        if v > 0.5:
            gaussian_maps[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    max_heatmap = gaussian_maps.max(0)
    gaussian_maps[-1, :, :] = 1 - max_heatmap
    return gaussian_maps

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        # print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_affine_transform2(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    #  print(src, dst)
    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def _xywh2cs(x, y, w, h, aspect_ratio, pixel_std):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    # if center[0] != -1:
    #    scale = scale * 1.25

    return center, scale
