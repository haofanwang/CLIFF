import os
import cv2
import torch
import numpy as np

import smplx
from smplx import SMPL

import trimesh
import pyrender
import colorsys

def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree

class Renderer(object):

    def __init__(self, focal_length=600, img_w=512, img_h=512, faces=None,
                 same_mesh_color=False):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)
        
        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            if self.same_mesh_color:
                mesh_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
            else:
                mesh_color = colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        if bg_img_rgb is None:
            return color_rgb
        else:
            mask = depth_map > 0
            bg_img_rgb[mask] = color_rgb[mask]
            return bg_img_rgb

    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()

        
if __name__ == '__main__':
    
    device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')

    # motion data dir
    file_dir = '/share/wanghaofan/research/OpenDance-10K/data/2nd_data/clips_results/luGqEHIO_WrZoL2LvlN3z1uoLN1V_compress_L2/luGqEHIO_WrZoL2LvlN3z1uoLN1V_compress_L2_cliff_hr48_infill.npz'
    
    # load motion
    results = np.load(file_dir)

    pred_betas = torch.from_numpy(results['shape']).float().to(device)
    pred_rotmat = torch.from_numpy(results['pose']).float().to(device)
    pred_cam_full = torch.from_numpy(results['global_t']).float().to(device)

    # load smpl model
    # smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)
    smpl_model = SMPL(model_path='./data/smpl', gender='MALE', batch_size=1).eval().to(device)
    
    # shape params: betas=pred_betas or beta=None
    pred_output = smpl_model(body_pose=pred_rotmat[:, 1:],
                             global_orient=pred_rotmat[:, 0:1],
                             pose2rot=True,
                             transl=pred_cam_full)
    
    # get vertices
    pred_vertices = pred_output.vertices.detach().cpu().numpy()[:150]

    # setup renderer for visualization
    img_h, img_w, _ = 512, 512, 3
    background = np.zeros((img_h, img_w, 3))
    focal_length = estimate_focal_length(img_h, img_w)

    video = cv2.VideoWriter('temp.mp4',
                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                            60, (img_h, img_w))

    for i, mesh in enumerate(pred_vertices):

        renderer = Renderer(focal_length=focal_length, 
                            img_w=img_w, img_h=img_h,
                            faces=smpl_model.faces,
                            same_mesh_color=False)

        front_view = renderer.render_front_view(mesh[None,:], bg_img_rgb=background.copy())
        video.write((front_view[:, :, ::-1]*255).astype(np.uint8))
        renderer.delete()
    video.release()
