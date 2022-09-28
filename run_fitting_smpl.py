import os
import pickle

import numpy as np

import torch
import torch.nn as nn

import smplx
from common import constants

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from prior import MaxMixturePrior


def perspective_projection(points, 
                           rotation, 
                           translation, 
                           focal_length,
                           camera_center):
    
    """
    This function computes the perspective projection of a set of points.

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

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2

def body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior,
                      focal_length=5000, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='sum'):
    """
    Loss function for body fitting
    """

    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)
    
    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)
    
    print(body_pose.shape, betas.shape)
    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss

def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)
    
    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]
    gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]
    
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
                             projected_joints[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:,None,None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op + (1-is_valid) * reprojection_error_gt).sum(dim=(1,2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()


class SMPLify():
    
    """
    Implementation of single-stage SMPLify.
    """ 
    
    def __init__(self, 
                 step_size=1e-2,
                 num_iters=10000,
                 focal_length=5000,
                 device=torch.device('cuda')):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder='data',
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        
        
        
        self.smpl = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(self.device)
        
    def __call__(self, init_betas, init_pose, init_global_orient, init_cam_t, camera_center, keypoints_2d):
        
        """
        Perform body fitting.
        """
        
        init_betas = init_betas.to(self.device)
        init_pose = init_pose.to(self.device)
        init_global_orient = init_global_orient.to(self.device)
        init_cam_t = init_cam_t.to(self.device)
        camera_center = camera_center.to(self.device)
        keypoints_2d = keypoints_2d.to(self.device)
        
        # SMPL params
        betas = init_betas.detach().clone()
        body_pose = init_pose.detach().clone()
        global_orient = init_global_orient.detach().clone()

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get 2D joints & confidence, (N, J_NUM, 3)
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        betas.requires_grad=False
        body_pose.requires_grad=False
        global_orient.requires_grad=True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            
            smpl_output = self.smpl(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    transl=camera_translation)
            
            model_joints = smpl_output.joints
            loss = camera_fitting_loss(model_joints, 
                                       camera_translation,
                                       init_cam_t, 
                                       camera_center,
                                       joints_2d, 
                                       joints_conf, 
                                       focal_length=self.focal_length)
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad=True
        betas.requires_grad=True
        global_orient.requires_grad=True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.
        
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters):
            
            smpl_output = self.smpl(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    transl=camera_translation)

            model_joints = smpl_output.joints
            loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                     joints_2d, joints_conf, self.pose_prior,
                                     focal_length=self.focal_length)
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            
            smpl_output = self.smpl(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    transl=camera_translation)

            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        
        betas = betas.detach()
        body_pose = body_pose.detach()
        global_orient = global_orient.detach()

        return vertices, joints, betas, body_pose, global_orient, camera_translation, reprojection_loss

    
def main():
    
    step_size = 1e-2
    num_smplify_iters = 10
    focal_length = 5000
  
    smplify = SMPLify(step_size=step_size, 
                      num_iters=num_smplify_iters, 
                      focal_length=focal_length)
        
    batch_size = 1
    init_betas = torch.ones(batch_size, 10) # (N,10)
    init_pose = torch.ones(batch_size, 69) # (N, 69)
    init_global_orient = torch.ones(batch_size, 3) # (N, 3)
    init_cam_t = torch.ones(batch_size, 3) # (N, 3)
    camera_center = torch.ones(batch_size, 2) # (N, 2)
    keypoints_2d = torch.ones(batch_size, 45, 3) # (N, 45, 3)
        
    # Run SMPLify optimization starting from the network prediction
    results = smplify(init_betas, init_pose, init_global_orient, init_cam_t, camera_center, keypoints_2d)
    vertices, joints, betas, body_pose, global_orient, camera_translation, reprojection_loss = results
    
    
if __name__ == '__main__':
    
    main()
    