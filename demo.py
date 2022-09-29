# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

'''
detector for single person detection:
    https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox

tracker for multi-person tracking and ReID:
    https://github.com/open-mmlab/mmtracking/tree/master/configs/mot/bytetrack
'''

import os
import os.path as osp
import cv2
import copy
import glob
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchgeometry as tgm
from torch.utils.data import DataLoader

import smplx

from models.smpl import SMPL
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50

from common import constants
from common.renderer_pyrd import Renderer
from common.mocap_dataset import MocapDataset
from common.utils import estimate_focal_length
from common.utils import strip_prefix_if_present, cam_crop2full, video_to_images

import mmcv
from mmtrack.apis import inference_mot, init_model
from mmtrack.core import results2outs
from mmdet.apis import inference_detector, init_detector


def perspective_projection(points, rotation, translation, focal_length,
                           camera_center):
    """This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)

    # ATTENTION: the line shoule be commented out as the points have been aligned
    # points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def main(args):
    
    if args.smooth:
        from mmhuman3d.utils.demo_utils import smooth_process
    
    device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')

    print("Input path:", args.input_path)
    print("Input type:", args.input_type)
    if args.input_type == "image":
        img_path_list = [args.input_path]
        base_dir = osp.dirname(osp.abspath(args.input_path))
        front_view_dir = side_view_dir = bbox_dir = base_dir
        result_filepath = f"{args.input_path[:-4]}_cliff_{args.backbone}.npz"
    else:
        if args.input_type == "video":
            basename = osp.basename(args.input_path).split('.')[0]
            base_dir = osp.join(osp.dirname(osp.abspath(args.input_path)), basename)
            img_dir = osp.join(base_dir, "imgs")
            front_view_dir = osp.join(base_dir, "front_view_%s" % args.backbone)
            side_view_dir = osp.join(base_dir, "side_view_%s" % args.backbone)
            bbox_dir = osp.join(base_dir, "bbox")
            result_filepath = osp.join(base_dir, f"{basename}_cliff_{args.backbone}.npz")
            if osp.exists(img_dir):
                print(f"Skip extracting images from video, because \"{img_dir}\" already exists")
            else:
                os.makedirs(img_dir, exist_ok=True)
                video_to_images(args.input_path, img_folder=img_dir)

        elif args.input_type == "folder":
            img_dir = osp.join(args.input_path, "imgs")
            front_view_dir = osp.join(args.input_path, "front_view_%s" % args.backbone)
            side_view_dir = osp.join(args.input_path, "side_view_%s" % args.backbone)
            bbox_dir = osp.join(args.input_path, "bbox")
            basename = args.input_path.split('/')[-1]
            result_filepath = osp.join(args.input_path, f"{basename}_cliff_{args.backbone}.npz")

        # get all image paths
        img_path_list = glob.glob(osp.join(img_dir, '*.jpg'))
        img_path_list.extend(glob.glob(osp.join(img_dir, '*.png')))
        img_path_list.sort()
    
    # load all images
    print("Loading images ...")
    orig_img_bgr_all = [cv2.imread(img_path) for img_path in tqdm(img_path_list)]
    print("Image number:", len(img_path_list))
    
    # multi-person
    if args.multi:
        
        # the number of tracked person
        # set N=2 for 2 person interactive videos
        N = -1
        
        # https://github.com/open-mmlab/mmtracking/tree/master/configs/mot/bytetrack
        mot_config = './mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private.py'
        checkpoint = './mmtracking/checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth'
        
        # load model
        mot_model = init_model(mot_config, checkpoint, device)
        
        # save results
        # [frame_id, x1, y1, x2, y2, conf_score, nms_threshold, person_id]
        detection_all = []
        
        # mmtracking procedure
        imgs = mmcv.VideoReader(args.input_path)
        prog_bar = mmcv.ProgressBar(len(imgs))

        for i, img in enumerate(imgs):
            
            result = inference_mot(mot_model, img, frame_id=i)

            track_masks = result.get('track_masks', None)
            track_bboxes = result.get('track_bboxes', None)
            outs_track = results2outs(bbox_results=track_bboxes,
                                      mask_results=track_masks,
                                      mask_shape=img.shape[:2])
            
            ids = outs_track.get('ids', None)
            bboxes = outs_track.get('bboxes', None)
            
            # make id starting from 0 in order
            ids, bboxes = (list(t) for t in zip(*sorted(zip(ids, bboxes))))
            
            # for convinience, just keep the bbox with the highest conf for each person
            existed_ids = []
            for j in range(len(bboxes)):
                if ids[j] in existed_ids:
                    continue
                if N != -1 and ids[j] > N:
                    continue
                x1, y1, x2, y2, score = bboxes[j]
                if score < 0.5:
                    continue
                detection_all.append([i, x1, y1, x2, y2, score, 0.99, ids[j]])
                existed_ids.append(ids[j])
        
        # list to array
        detection_all = np.array(detection_all)
    
    # single-person
    else:
        
        # https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox
        config = './mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py'
        checkpoint = './mmdetection/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
        
        # load detector
        model = init_detector(config, checkpoint, device)
            
        # save results
        detection_all = []
        
        # mmdetection procedure
        imgs = mmcv.VideoReader(args.input_path)
        prog_bar = mmcv.ProgressBar(len(imgs))
        
        # only take-out person (id=0)
        class_id = 0
        for i, img in enumerate(imgs):
            result = inference_detector(model, img)
            if len(result[class_id]) == 0:
                continue
            x1, y1, x2, y2, score = result[class_id][0]
            detection_all.append([i, x1, y1, x2, y2, score, 0.99, 0])
            
        # list to array
        detection_all = np.array(detection_all)
    
    print("--------------------------- 3D HPS estimation ---------------------------")
    # Create the model instance
    cliff = eval("cliff_" + args.backbone)
    cliff_model = cliff(constants.SMPL_MEAN_PARAMS).to(device)
    # Load the pretrained model
    print("Load the CLIFF checkpoint from path:", args.ckpt)
    state_dict = torch.load(args.ckpt)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    # Setup the SMPL model
    smpl_model = SMPL(constants.SMPL_MODEL_DIR).to(device)

    pred_vert_arr = []
    if args.save_results:
        smpl_pose = []
        smpl_betas = []
        smpl_trans = []
        smpl_joints = []
        cam_focal_l = []

    mocap_db = MocapDataset(orig_img_bgr_all, detection_all)
    mocap_data_loader = DataLoader(mocap_db, batch_size=min(args.batch_size, len(detection_all)), num_workers=0)
    for batch in tqdm(mocap_data_loader):
        norm_img = batch["norm_img"].to(device).float()
        center = batch["center"].to(device).float()
        scale = batch["scale"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        focal_length = batch["focal_length"].to(device).float()

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]
        
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)

        # convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)

        pred_output = smpl_model(betas=pred_betas,
                                 body_pose=pred_rotmat[:, 1:],
                                 global_orient=pred_rotmat[:, [0]],
                                 pose2rot=False,
                                 transl=pred_cam_full)
        pred_vertices = pred_output.vertices
        pred_vert_arr.extend(pred_vertices.cpu().numpy())
        
        # re-project to 2D keypoints on image plane for calculating reprojection loss
        '''
        # visualize
        for index, (px, py) in enumerate(pred_keypoints2d[0]):
            cv2.circle(img, (int(px), int(py)), 1, [255, 128, 0], 2)
        cv2.imwrite("front_view_kpt.jpg", img)
        '''
        pred_keypoints3d = pred_output.joints[:,:24,:]
        camera_center = torch.hstack((img_w[:,None], img_h[:,None])) / 2
        pred_keypoints2d = perspective_projection(
                pred_keypoints3d,
                rotation=torch.eye(3, device=device).unsqueeze(0).expand(pred_keypoints3d.shape[0], -1, -1),
                translation=pred_cam_full,
                focal_length=focal_length,
                camera_center=camera_center)
        
        if args.save_results:
            # default pose_format is rotation matrix instead of axis-angle
            if args.pose_format == "aa":
                rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
                rot_pad = rot_pad.expand(pred_rotmat.shape[0] * 24, -1, -1)
                rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad), dim=-1)
                pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)  # N*72
            else:
                pred_pose = pred_rotmat  # N*24*3*3

            smpl_pose.extend(pred_pose.cpu().numpy())
            smpl_betas.extend(pred_betas.cpu().numpy())
            smpl_trans.extend(pred_cam_full.cpu().numpy())
            smpl_joints.extend(pred_output.joints.cpu().numpy())
            cam_focal_l.extend(focal_length.cpu().numpy())

    if args.infill:
        print("Do motion interpolation.")
        
        from scipy.interpolate import interp1d
        
        # infilled smpl joints and vertices
        smpl_joints_fill = np.copy(smpl_joints)
        smpl_vertices_fill = np.copy(pred_vert_arr)
        detection_all_fill = np.copy(detection_all)
        
        # person number, only support infill for 1 or 2 persons now
        person_count = len(set(detection_all[:,-1]))
        
        # seperate multiple motions and infill each
        for person in range(person_count):
            
            choose_frame = []
            choose_index = []
            choose_joints = []
            choose_vertices = []
            
            for i in range(len(detection_all)):
                frame_id = detection_all[i][0]
                person_id = detection_all[i][-1]
                if person_id == person:
                    choose_frame.append(int(frame_id))
                    choose_index.append(len(choose_index))
                    choose_joints.append(smpl_joints[i])
                    choose_vertices.append(pred_vert_arr[i])
            
            if len(choose_frame) < 3:
                continue
            
            # existed frames that do not need infill
            existed_list = copy.copy(choose_frame)
            
            # chosen frame ids with interval
            interval = 10
            choose_frame = choose_frame[0::interval]
            choose_index = choose_index[0::interval]
            
            # stack results
            choose_joints = np.stack(choose_joints, axis=0) # (N, J_NUM, 3)
            choose_vertices = np.stack(choose_vertices, axis=0) # (N, V_NUM, 3)
            
            # linear interpolation
            choose_joints = interp1d(choose_frame, choose_joints[np.array(choose_index), :, :].transpose(1, 2, 0), 
                         kind='linear')(range(int(min(choose_frame)), int(max(choose_frame)))).transpose(2, 0, 1)    
            choose_vertices = interp1d(choose_frame, choose_vertices[np.array(choose_index), :, :].transpose(1, 2, 0), 
                                 kind='linear')(range(int(min(choose_frame)), int(max(choose_frame)))).transpose(2, 0, 1)
            
            if args.smooth:
                # limit memory, only smooth smpl joints
                print("Do motion smooth on person {}.".format(person))
                choose_joints = smooth_process(choose_joints, 
                                               smooth_type='smoothnet_windowsize8',
                                               cfg_base_dir='configs/_base_/post_processing/')
            
            infill_frame_ids = []
            for infill_frame_id in range(int(min(choose_frame)), int(max(choose_frame))):
                if infill_frame_id not in existed_list:
                    infill_frame_ids.append(infill_frame_id)
            print("Infill {} frames for person {}".format(len(infill_frame_ids), person))
            
            infill_seq = list(range(int(min(choose_frame)), int(max(choose_frame))))
            for infill_frame_id in infill_frame_ids:
                
                smpl_joints_fill_item = choose_joints[infill_seq.index(infill_frame_id)]
                smpl_joints_fill_item = smpl_joints_fill_item[np.newaxis, :]
                smpl_joints_fill = np.append(smpl_joints_fill, smpl_joints_fill_item, axis=0)
                
                smpl_vertices_fill_item = choose_vertices[infill_seq.index(infill_frame_id)]
                smpl_vertices_fill_item = smpl_vertices_fill_item[np.newaxis, :]
                smpl_vertices_fill = np.append(smpl_vertices_fill, smpl_vertices_fill_item, axis=0)
                
                detection_all_fill_item = np.array([infill_frame_id, 0, 0, 0, 0, 0, 0, person])
                detection_all_fill_item = detection_all_fill_item[np.newaxis, :]
                detection_all_fill = np.append(detection_all_fill, detection_all_fill_item, axis=0)
   
        smpl_joints = smpl_joints_fill
        pred_vert_arr = smpl_vertices_fill
        detection_all = detection_all_fill

    if args.save_results:
        if args.infill:
            result_filepath = result_filepath[:-4]+'_infill.npz'
        print(f"Save results to \"{result_filepath}\"")
        np.savez(result_filepath, imgname=img_path_list,
                 pose=smpl_pose, shape=smpl_betas, global_t=smpl_trans,
                 pred_joints=smpl_joints, focal_l=cam_focal_l,
                 detection_all=detection_all)

    print("--------------------------- Visualization ---------------------------")
    # make the output directory
    os.makedirs(front_view_dir, exist_ok=True)
    print("Front view directory:", front_view_dir)
    if args.show_sideView:
        os.makedirs(side_view_dir, exist_ok=True)
        print("Side view directory:", side_view_dir)
    if args.show_bbox:
        os.makedirs(bbox_dir, exist_ok=True)
        print("Bounding box directory:", bbox_dir)

    pred_vert_arr = np.array(pred_vert_arr)
    for img_idx, orig_img_bgr in enumerate(tqdm(orig_img_bgr_all)):
        chosen_mask = detection_all[:, 0] == img_idx
        chosen_vert_arr = pred_vert_arr[chosen_mask]
        
        # setup renderer for visualization
        img_h, img_w, _ = orig_img_bgr.shape
        focal_length = estimate_focal_length(img_h, img_w)
        
        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=smpl_model.faces,
                            same_mesh_color=False)
        front_view = renderer.render_front_view(chosen_vert_arr,
                                                bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())

        # save rendering results
        basename = osp.basename(img_path_list[img_idx]).split(".")[0]
        filename = basename + "_front_view_cliff_%s.jpg" % args.backbone
        front_view_path = osp.join(front_view_dir, filename)
        cv2.imwrite(front_view_path, front_view[:, :, ::-1])

        if args.show_sideView:
            side_view_img = renderer.render_side_view(chosen_vert_arr)
            filename = basename + "_side_view_cliff_%s.jpg" % args.backbone
            side_view_path = osp.join(side_view_dir, filename)
            cv2.imwrite(side_view_path, side_view_img[:, :, ::-1])

        # delete the renderer for preparing a new one
        renderer.delete()

        # draw the detection bounding boxes
        if args.show_bbox:
            chosen_detection = detection_all[chosen_mask]
            bbox_info = chosen_detection[:, 1:6]

            bbox_img_bgr = orig_img_bgr.copy()
            for min_x, min_y, max_x, max_y, conf in bbox_info:
                
                if conf == 0:
                    continue
                
                ul = (int(min_x), int(min_y))
                br = (int(max_x), int(max_y))
                cv2.rectangle(bbox_img_bgr, ul, br, color=(0, 255, 0), thickness=2)
                cv2.putText(bbox_img_bgr, "%.1f" % conf, ul,
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.0, color=(0, 0, 255), thickness=1)
            filename = basename + "_bbox.jpg"
            bbox_path = osp.join(bbox_dir, filename)
            cv2.imwrite(bbox_path, bbox_img_bgr)

    # make videos
    if args.make_video:
        print("--------------------------- Making videos ---------------------------")
        from common.utils import images_to_video
        images_to_video(front_view_dir, video_path=front_view_dir + ".mp4", frame_rate=args.frame_rate)
        if args.show_sideView:
            images_to_video(side_view_dir, video_path=side_view_dir + ".mp4", frame_rate=args.frame_rate)
        if args.show_bbox:
            images_to_video(bbox_dir, video_path=bbox_dir + ".mp4", frame_rate=args.frame_rate)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_type', default='image', choices=['image', 'folder', 'video'],
                        help='input type')
    parser.add_argument('--input_path', default='test_samples/nba.jpg', help='path to the input data')

    parser.add_argument('--ckpt',
                        default="data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt",
                        help='path to the pretrained checkpoint')
    parser.add_argument("--backbone", default="hr48", choices=['res50', 'hr48'],
                        help="the backbone architecture")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for detection and motion capture')

    parser.add_argument('--save_results', action='store_true',
                        help='save the results as a npz file')
    parser.add_argument('--pose_format', default='aa', choices=['aa', 'rotmat'],
                        help='aa for axis angle, rotmat for rotation matrix')

    parser.add_argument('--show_bbox', action='store_true',
                        help='show the detection bounding boxes')
    parser.add_argument('--show_sideView', action='store_true',
                        help='show the result from the side view')

    parser.add_argument('--make_video', action='store_true',
                        help='make a video of the rendering results')
    parser.add_argument('--frame_rate', type=int, default=30, help='frame rate')
    
    # NEW!
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--multi', action='store_true', help='multi-person')
    parser.add_argument('--infill', action='store_true', help='motion interpolation, only support linear interpolation now')
    parser.add_argument('--smooth', action='store_true', help='motion smooth, support oneeuro, gaus1d, savgol, smoothnet')
    
    args = parser.parse_args()
    main(args)
