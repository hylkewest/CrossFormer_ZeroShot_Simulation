import time
import numpy as np
import jax

import cv2
import pybullet as p
from PIL import Image
from scipy.spatial.transform import Rotation as R

from environment.utilities import Camera, MultiCams


class TrajectoryGenerator:
    def __init__(
        self,
        crossformer_model,
        task,
        camera_cfg,
        img_size=224,
        max_steps=500,
        window_size=5,
        make_video=False
    ):
        self.img_size = img_size
        self.window_size = window_size
        self.max_steps = max_steps
        self.current_observation = 1
        self.camera_cfg = camera_cfg

        #model = CrossFormerModel.load_pretrained(network_path)
        self.crossformer = crossformer_model

        #self.task = self.crossformer.create_tasks(texts=[f"pick up the object on the white table"])
        self.task = task

        self.obs_buffer = []
        self.test_array = []

        self.make_video = make_video

    def add_image_to_buffer(self, rgb_img):
        if not self.obs_buffer:
            for _ in range(self.window_size):
                self.obs_buffer.append(rgb_img)
                self.test_array.append(self.current_observation)
            return
        
        self.current_observation += 1

        if self.current_observation < 6:
            if len(self.obs_buffer) == self.window_size:
                slice_index = self.current_observation - 1
                self.obs_buffer = self.obs_buffer[:slice_index]
                self.test_array = self.test_array[:slice_index]
                while len(self.obs_buffer) < self.window_size:
                    self.obs_buffer.append(rgb_img)
                    self.test_array.append(self.current_observation)
                return
        else:
            self.obs_buffer.pop(0)
            self.obs_buffer.append(rgb_img)

            self.test_array.pop(0)
            self.test_array.append(self.current_observation)


    def get_observations(self):
        images_np = np.stack(self.obs_buffer, axis=0)
        images_np = images_np[None]

        observation = {
            "image_primary": images_np,
            "timestep_pad_mask": np.full(
                (1, images_np.shape[1]), True, dtype=bool
            ),
        }
        return observation


    def parse_action(self, action_vector):
        dx = action_vector[0]
        dy = action_vector[1]
        dz = action_vector[2]
        dyaw = action_vector[3]
        dpitch = action_vector[4]
        droll = action_vector[5]
        grasp = action_vector[6]
        return dx, dy, dz, dyaw, dpitch, droll, grasp

    def render(self, env):
        
        def transform_image(image):
            h, w = image.shape[:2]
            # Center crop to 480x480
            start_x = w//2 - 240
            start_y = h//2 - 240
            image = image[start_y:start_y+480, start_x:start_x+480]
            # Resize to 224x224
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            return image

        camera = MultiCams(env.physicsClient, cfg=self.camera_cfg, n_cams=1)
        rgb, _ = camera.cams[0].get_images(get_rgb=True, get_depth=False, get_seg=False)
        rgb_transformed = transform_image(rgb)
        return rgb_transformed

    # @staticmethod
    # def transform_to_policy_frame(xyz, rpy):
    #     """
    #     Transform poses from robot frame to policy frame convention.
        
    #     Robot:   X-forward, Y-right, Z-down
    #     Policy:  X-forward, Y-left,  Z-up
    #     """
    #     policy_xyz = np.array([
    #         xyz[0],   # X stays the same
    #         -xyz[1],  # Y flipped (right → left)
    #         -xyz[2]   # Z flipped (down → up)
    #     ])
    #     policy_rpy = np.copy(rpy)
    #     # policy_rpy = np.array([
    #     #     rpy[0],    # Roll: rotate around X (stays same)
    #     #     -rpy[1],   # Pitch: flip due to Z flip
    #     #     -rpy[2]    # Yaw: flip due to Y flip
    #     # ])
    #     return policy_xyz, policy_rpy

    def apply_delta_action(self, current_xyz, current_rpy, delta_action):
        """
        Apply delta actions to current end-effector pose and return new target pose in world frame.
        
        Args:
            current_xyz (np.ndarray): Current ee position [x, y, z]
            current_rpy (np.ndarray): Current ee orientation in roll-pitch-yaw [r, p, y]
            delta_action (np.ndarray): Policy prediction [dx, dy, dz, dyaw, dpitch, droll, grasp]
        
        Returns:
            tuple: (target_xyz, target_quat, grasp_state)
                - target_xyz: New position [x, y, z]
                - target_quat: New orientation as quaternion [x, y, z, w]
                - grasp_state: New gripper state [0-1]
        """
        # Extract components from delta_action
        dx, dy, dz, dyaw, dpitch, droll, _ = self.parse_action(delta_action)
        delta_xyz =[dx, dy, dz]
        delta_rpy = [droll, dpitch, dyaw]
        
        # current_xyz_fix, current_rpy_fix = self.transform_to_policy_frame(
        #     current_xyz, current_rpy)
        current_xyz_fix = current_xyz
        current_rpy_fix = current_rpy

        # 1. Handle position: Simply add delta to current position
        target_xyz = np.asarray(current_xyz_fix) + np.asarray(delta_xyz)
        
        # 2. Handle orientation
        # Convert current RPY to rotation matrix
        current_rot = R.from_euler('xyz', current_rpy_fix)
        current_matrix = current_rot.as_matrix()
        
        # Convert delta RPY to rotation matrix
        delta_rot = R.from_euler('xyz', delta_rpy)
        delta_matrix = delta_rot.as_matrix()
        
        # Compose rotations: new_R = current_R * delta_R
        target_matrix = current_matrix @ delta_matrix
        
        # Convert to quaternion for move_ee
        target_rot = R.from_matrix(target_matrix)
        target_quat = target_rot.as_quat()  # Returns [x, y, z, w]
        
        return target_xyz, target_quat

    def predict_trajectory(self, env):
        step_count = 0

        # _, ee_orn_0 = env.get_ee_pose()
        # ee_orn_0 = p.getQuaternionFromEuler(ee_orn_0)
        # env.move_ee([-0.15, -0.52, 1.3, ee_orn_0])
        # for _ in range(30):
        #     p.stepSimulation()

        if self.make_video:
            video_frames= []

        while step_count < self.max_steps:
            step_count += 1

            rgb = self.render(env)
            
            #Image.fromarray(rgb).save("rgb_image_image.png")

            self.add_image_to_buffer(rgb)

            #print(self.test_array)

            observation = self.get_observations()

            actions = self.crossformer.sample_actions(
                observation,
                self.task,
                head_name="single_arm",
                unnormalization_statistics=self.crossformer.dataset_statistics["bridge_dataset"]["action"],
                rng=jax.random.PRNGKey(0)
            )
            actions = actions[0]

            success = False
            for j, action_vector in enumerate(actions): # action horizon 4
                ee_xyz, ee_rpy = env.get_ee_pose() # current ee pose
                target_xyz, target_quat = self.apply_delta_action(
                    ee_xyz, ee_rpy, action_vector) # target ee pose after applying deltas

                env.move_ee([*target_xyz, target_quat])
                for _ in range(10):
                    p.stepSimulation()
                    time.sleep(env.SIMULATION_STEP_DELAY)

                print(f"Action {j+1}/{4} in step {step_count}/{self.max_steps}.\nTook delta action {action_vector}\n\n")
                if self.make_video:
                    video_frames.append(observation['image_primary'][0][-1])

                # grasp if predicted
                grasp_state = action_vector[-1]
                if grasp_state <= 0.01:
                    print(f"[TrajectoryGenerator] Grasp triggered at step {step_count}.")
                    env.auto_close_gripper(check_contact=True)
                    for _ in range(10):
                        p.stepSimulation()
                        time.sleep(env.SIMULATION_STEP_DELAY)

                    # for now I hard-core True, you should use `env.check_grasped()` to check if object is indeed grasped.
                    success = True
                    break

            if success:
                print("[TrajectoryGenerator] successfully grasped the object Exitting.")
                break
                
        if step_count == self.max_steps:
            print("[TrajectoryGenerator] Reached max steps without grasp=1. Stopping.")

        if self.make_video and success:
            output_file = './temp_video.mp4'
            height, width, channels = video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
            out = cv2.VideoWriter(output_file, fourcc, 20, (width, height))
            for frame in video_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f'Video saved at {output_file}')