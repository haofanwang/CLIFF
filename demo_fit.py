'''
# CLIFF + SMPLify

This demo further applys SMPLify fitting after CLIFF, OpenPose format 2D Keypoints are required for convinence.

RUN:
    python3 demo_fit.py --img=examples/im1010.jpg --openpose=examples/im1010_openpose.json
'''


import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pytorch3d import transforms

from models.smpl import SMPL
from common import constants

from losses import *
from smplify import SMPLify

from common.renderer_pyrd import Renderer
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from common.utils import strip_prefix_if_present, cam_crop2full
from common.mocap_dataset import MocapDataset

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')


if __name__ == '__main__':

    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load pretrained model
    cliff = eval("cliff_hr48")
    cliff_model = cliff('./data/smpl_mean_params.npz').to(device)
    state_dict = torch.load('./data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    # Load SMPL model
    smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1).to(device)
    
    orig_img_bgr_all = [cv2.imread('./examples/im1010.jpg')]
    detection_all = np.array([[0, 17, 37, 117, 142, 0.99, 0.99, 0]])
    
    mocap_db = MocapDataset(orig_img_bgr_all, detection_all)
    mocap_data_loader = DataLoader(mocap_db, batch_size=len(detection_all), num_workers=0)
    
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
    
        # load 2D keypoints
        keypoints = json.load(open(args.openpose))
        keypoints = np.array(keypoints['people'][0]['pose_keypoints_2d']).reshape((25,3))
        kpts = np.zeros((1, 49, 3))
        kpts[0, :25, :] = keypoints
        keypoints = torch.from_numpy(kpts).to(device)
        
        # run CLIFF model
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)

        # Convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)

        pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
        
        smpl_poses = transforms.matrix_to_axis_angle(pred_rotmat).contiguous().view(-1, 72) # N*72
        camera_center = torch.hstack((img_w[:,None], img_h[:,None])) / 2
        
        pred_output = smpl(betas=pred_betas,
                           body_pose=smpl_poses[:, 3:],
                           global_orient=smpl_poses[:, :3],
                           pose2rot=True,
                           transl=pred_cam_full)

        flag = True
        if flag:
            
            # re-project to 2D keypoints on image plane
            pred_keypoints3d = pred_output.joints
            rotation = torch.eye(3, device=device).unsqueeze(0).expand(pred_keypoints3d.shape[0], -1, -1)
            pred_keypoints2d = perspective_projection(pred_keypoints3d,
                                                      rotation,
                                                      pred_cam_full,
                                                      focal_length,
                                                      camera_center) # (N, 49, 2)
            
            op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
            op_joints_ind = np.array([constants.JOINT_IDS[joint] for joint in op_joints])

            # visualize GT (Openpose) 2D kpts
            orig_img_bgr = orig_img_bgr_all[0].copy()
            keypoints_gt = json.load(open(args.openpose))
            keypoints_gt = np.array(keypoints_gt['people'][0]['pose_keypoints_2d']).reshape((25,3))
            kpts = np.zeros((1, 49, 3))
            kpts[0, :25, :] = keypoints_gt
            keypoints_gt = kpts

            for index, (px, py,_) in enumerate(keypoints_gt[0][op_joints_ind]):
                cv2.circle(orig_img_bgr, (int(px), int(py)), 1, [255, 128, 0], 2)
            cv2.imwrite("kpt2d_gt.jpg", orig_img_bgr)

            # visualize predicted re-project 2D kpts
            orig_img_bgr = orig_img_bgr_all[0].copy()
            for index, (px, py) in enumerate(pred_keypoints2d[0][op_joints_ind]):
                cv2.circle(orig_img_bgr, (int(px), int(py)), 1, [255, 128, 0], 2)
            cv2.imwrite("kpt2d.jpg", orig_img_bgr)

            # calculate re-projection loss
            reprojection_error_op = (keypoints_gt[0][op_joints_ind][:,:2] - pred_keypoints2d[0][op_joints_ind].cpu().numpy()) ** 2
            print(reprojection_error_op.sum())

            # visualize predicted mesh
            renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                faces=smpl.faces,
                                same_mesh_color=False)

            front_view = renderer.render_front_view(pred_output.vertices.cpu().numpy(),
                                                    bg_img_rgb=orig_img_bgr_all[0][:, :, ::-1].copy())

            cv2.imwrite('mesh.jpg', front_view[:, :, ::-1])
        
        # be careful: the estimated focal_length should be used here instead of the default constant
        smplify = SMPLify(step_size=1e-2, batch_size=1, num_iters=100, focal_length=focal_length)
        
        results = smplify(smpl_poses.detach(), 
                          pred_betas.detach(),
                          pred_cam_full.detach(),
                          camera_center,
                          keypoints)
        
        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = results
    
        with torch.no_grad():
            pred_output = smpl(betas=new_opt_betas,
                               body_pose=new_opt_pose[:, 3:],
                               global_orient=new_opt_pose[:, :3],
                               pose2rot=True,
                               transl=new_opt_cam_t)
            pred_vertices = pred_output.vertices
        
        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=smpl.faces,
                            same_mesh_color=False)
        
        front_view = renderer.render_front_view(pred_vertices.cpu().numpy(),
                                                bg_img_rgb=orig_img_bgr_all[0][:, :, ::-1].copy())

        cv2.imwrite('mesh_fit.jpg', front_view[:, :, ::-1])