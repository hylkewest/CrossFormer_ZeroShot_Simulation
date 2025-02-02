import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from datetime import datetime

from network.hardware.device import get_device
#from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.visualisation.plot import plot_results
from network.utils.dataset_processing.grasp import detect_grasps
from environment.env import Environment
from trained_models.CrossFormer.crossformer_wrapper import CrossFormerWrapper
from crossformer.model.crossformer_model import CrossFormerModel
from skimage.filters import gaussian
import pybullet as p
import os
import jax
import jax.numpy as jnp
import cv2

class GraspGenerator:
    IMG_WIDTH = 224
    IMG_ROTATION = -np.pi * 0.5
    CAM_ROTATION = 0
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path, camera, depth_radius, fig, IMG_WIDTH=224, network='GR_ConvNet', device='cpu'):

        if (device=='cpu' and network != "CrossFormer"):
            self.net = torch.load(net_path, map_location=device)
            self.device = get_device(force_cpu=True)
        elif network == "CrossFormer":
            # model = CrossFormerModel.load_pretrained(net_path)
            # self.net = CrossFormerWrapper(model, model.params)
            # self.device = get_device(force_cpu=True)
            print("hello")
        else:
            #self.net = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(1))
            #self.device = get_device()
            print ("GPU is not supported yet! :( -- continuing experiment on CPU!" )
            self.net = torch.load(net_path, map_location='cpu')
            self.device = get_device(force_cpu=True)

        np.set_printoptions(suppress=True, precision=8)
    
        self.near = camera.near
        self.far = camera.far
        self.depth_r = depth_radius
        
        self.fig = fig
        self.network = network

        self.PIX_CONVERSION = 277 * IMG_WIDTH/224

        self.IMG_WIDTH = IMG_WIDTH
        # print (self.IMG_WIDTH)

        # Get rotation matrix
        img_center = self.IMG_WIDTH / 2 - 0.5
        self.img_to_cam = self.get_transform_matrix(-img_center/self.PIX_CONVERSION,
                                                    img_center/self.PIX_CONVERSION,
                                                    0,
                                                    self.IMG_ROTATION)
        self.cam_to_robot_base = self.get_transform_matrix(
            camera.x, camera.y, camera.z, self.CAM_ROTATION)

    def get_transform_matrix(self, x, y, z, rot):
        return np.array([
                        [np.cos(rot),   -np.sin(rot),   0,  x],
                        [np.sin(rot),   np.cos(rot),    0,  y],
                        [0,             0,              1,  z],
                        [0,             0,              0,  1]
                        ])

    def grasp_to_robot_frame(self, grasp, depth_img):
        """
        return: x, y, z, roll, opening length gripper, object height
        """
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]

        # Get area of depth values around center pixel
        x_min = int(np.clip(x_p - self.depth_r, 0, self.IMG_WIDTH))
        x_max = int(np.clip(x_p + self.depth_r, 0, self.IMG_WIDTH))
        y_min = int(np.clip(y_p - self.depth_r, 0, self.IMG_WIDTH))
        y_max = int(np.clip(y_p + self.depth_r, 0, self.IMG_WIDTH))
        depth_values = depth_img[x_min:x_max, y_min:y_max]

        # Get minimum depth value from selected area
        z_p = np.amin(depth_values)

        # Convert pixels to meters
        x_p /= self.PIX_CONVERSION
        y_p /= self.PIX_CONVERSION
        z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)

        # Convert image space to camera's 3D space
        img_xyz = np.array([x_p, y_p, -z_p, 1])
        cam_space = np.matmul(self.img_to_cam, img_xyz)

        # Convert camera's 3D space to robot frame of reference
        robot_frame_ref = np.matmul(self.cam_to_robot_base, cam_space)

        # Change direction of the angle and rotate by alpha rad
        roll = grasp.angle * -1 + (self.IMG_ROTATION)
        if roll < -np.pi / 2:
            roll += np.pi

        # Covert pixel width to gripper width
        opening_length = (grasp.length / int(self.MAX_GRASP *
                          self.PIX_CONVERSION)) * self.MAX_GRASP

        obj_height = self.DIST_BACKGROUND - z_p

        # return x, y, z, roll, opening length gripper
        return robot_frame_ref[0], robot_frame_ref[1], robot_frame_ref[2], roll, opening_length, obj_height

    def post_process_output(self, q_img, cos_img, sin_img, width_img, pixels_max_grasp):
        """
        Post-process the raw output of the network, convert to numpy arrays, apply filtering.
        :param q_img: Q output of network (as torch Tensors)
        :param cos_img: cos output of network
        :param sin_img: sin output of network
        :param width_img: Width output of network
        :return: Filtered Q output, Filtered Angle output, Filtered Width output
        """
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * pixels_max_grasp

        q_img = gaussian(q_img, 1.0, preserve_range=True)
        ang_img = gaussian(ang_img, 1.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        return q_img, ang_img, width_img

    def predict(self, rgb, depth, n_grasps=1, show_output=False):

        max_val = np.max(depth)
        depth = depth * (255 / max_val)
        depth = np.clip((depth - depth.mean())/175, -1, 1)

        print(self.network)
        
        if (self.network == 'GR_ConvNet'):
            ##### GR-ConvNet #####
            depth = np.expand_dims(np.array(depth), axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
        elif self.network == 'GGCNN':
            ##### GGCNN #####
            depth = np.expand_dims(np.array(depth), axis=2)
            depth_img = torch.tensor(depth, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        elif self.network == "CrossFormer":
            ##### CrossFormer #####
            print("CrossFormer")
            # actions_np = self.net.single_step_forward(rgb, depth)
            # action_tokens = actions_np[0, 0]  # shape => (action_horizon, action_dim)

        else:
            print("The selected network has not been implemented yet -- please choose another network!")
            exit() 

        with torch.no_grad():
            if (self.network == 'GR_ConvNet'):
                ##### GR-ConvNet #####
                xc = x.to(self.device)
                pred = self.net.predict(xc)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(
                    pred['pos'], pred['cos'], pred['sin'], pred['width'], pixels_max_grasp
                )
            elif self.network == 'GGCNN':
                ##### GGCNN #####
                pred = self.net(depth_img)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(
                    pred[0], pred[1], pred[2], pred[3], pixels_max_grasp
                )
            elif self.network == "CrossFormer":
                ##### CrossFormer #####
                actions = self.net.single_step_forward(rgb, depth)
                time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                save_name = 'network_output/{}'.format(time)
                return actions, save_name
            else: 
                print ("you need to add your function here!")        
        
        save_name = None
        if show_output:
            #fig = plt.figure(figsize=(10, 10))
            im_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb is not None else None
            plot = plot_results(self.fig,
                                rgb_img=im_bgr,
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                depth_img=depth,
                                no_grasps=3,
                                grasp_width_img=width_img)

            if not os.path.exists('network_output'):
                os.mkdir('network_output')
            time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            save_name = 'network_output/{}'.format(time)
            plot.savefig(save_name + '.png')
            plot.clf()

        grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
        return grasps, save_name

    def predict_grasp(self, rgb, depth, n_grasps=1, show_output=False):
        # Pass only the required data based on the network type
        if self.network == 'GGCNN':
            # GGCNN uses depth-only data
            predictions, save_name = self.predict(None, depth, n_grasps=n_grasps, show_output=show_output)            
        else:
            # GR_ConvNet and CrossFormer both use RGB and depth
            predictions, save_name = self.predict(rgb, depth, n_grasps=n_grasps, show_output=show_output)

        grasps = []
        for grasp in predictions:
            if self.network == "CrossFormer":
                grasp
                starting_pos = np.array([0.4958126339091726, 0.10914993600132894, 1.21110183716901, -3.140626890894155, 1.5446709877086156, -3.141564660437764])
                new_pos = starting_pos + np.array(grasp[0:6])
                print("----------------")
                print("new_pos: ")
                print(new_pos)
                print("----------------")
                print("----------------")

                grasps.append(new_pos)
            else: 
                x, y, z, roll, opening_len, obj_height = self.grasp_to_robot_frame(grasp, depth)

                grasps.append((x, y, z, roll, opening_len, obj_height))

        return grasps, save_name


    # def tokenize_input(self, rgb, depth, patch_size=16):
    #     """
    #     Tokenize RGB and depth input into patches.
    #     :param rgb: RGB image as NumPy array.
    #     :param depth: Depth image as NumPy array.
    #     :param patch_size: Size of patches for tokenization.
    #     :return: Torch tensor of tokenized patches.
    #     """
    #     # Combine RGB and Depth
    #     rgb_depth = np.concatenate((rgb, np.expand_dims(depth, axis=-1)), axis=-1)  # Shape: (H, W, 4)

    #     # Reshape into patches
    #     H, W, C = rgb_depth.shape
    #     patches = [
    #         rgb_depth[i:i+patch_size, j:j+patch_size]
    #         for i in range(0, H, patch_size)
    #         for j in range(0, W, patch_size)
    #     ]

    #     patches = np.stack(patches, axis=0)  # (num_patches, patch_size, patch_size, C)
    #     return torch.tensor(patches, dtype=torch.float32).permute(0, 3, 1, 2)  # (num_patches, C, patch_size, patch_size)


    # def parse_single_pose(self, unnorm_tokens):
    #     """
    #     Suppose your final single action is 5 floats: (x_px, y_px, angle, width, conf).
    #     Or you might have 7 floats. 
    #     We'll parse them accordingly.
    #     """
    #     # Example:
    #     x_px = unnorm_tokens[0]
    #     y_px = unnorm_tokens[1]
    #     angle = unnorm_tokens[2]
    #     width = unnorm_tokens[3]
    #     confidence = unnorm_tokens[4]  # if present

    #     # Return as a simple tuple
    #     return (x_px, y_px, angle, width, confidence)
    
    # def unnormalize_action(self, action_tokens):
    #     """
    #     If your model returns coords in [-1, 1], convert them to [0, IMG_WIDTH].
    #     For instance, if self.IMG_WIDTH=224, then -1 -> 0, +1 -> 224.
    #     """
    #     # Suppose action_tokens = [x_raw, y_raw, angle, width, etc.]
    #     x_raw = action_tokens[0]
    #     y_raw = action_tokens[1]
        
    #     x_pixel = (x_raw + 1) / 2 * self.IMG_WIDTH
    #     y_pixel = (y_raw + 1) / 2 * self.IMG_WIDTH

    #     # Then keep angle, width, or the rest of the vector as is,
    #     # or do a similar unnormalization if they're also in [-1,1].
    #     angle = action_tokens[2]
    #     width = action_tokens[3]
    #     confidence = action_tokens[4] if len(action_tokens) > 4 else 1.0

    #     return [x_pixel, y_pixel, angle, width, confidence]

    # def single_pose_to_fake_prediction(self, single_pose):
    #     """
    #     Convert (x_px, y_px, angle, width, confidence) into a 'grasp-like' object
    #     that can be passed to grasp_to_robot_frame(...).
    #     """
    #     class FakeGrasp:
    #         pass

    #     fake = FakeGrasp()
    #     # Typically, your code expects .center, .angle, .length
    #     fake.center = (single_pose[0], single_pose[1])  # (x_px, y_px) in pixel coords
    #     fake.angle = single_pose[2]
    #     fake.length = single_pose[3]
    #     # confidence = single_pose[4]  # not used by 'grasp_to_robot_frame', but you can store if you want

    #     return fake
        



