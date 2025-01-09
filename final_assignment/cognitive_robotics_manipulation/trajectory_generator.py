import time
import numpy as np
import jax

import cv2
import pybullet as p

from crossformer.model.crossformer_model import CrossFormerModel

from trained_models.CrossFormer.crossformer_wrapper import CrossFormerWrapper


class TrajectoryGenerator:
    def __init__(
        self,
        network_path,
        camera,
        fig,
        img_size=224,
        device="cpu",
        max_steps=300,
        window_size=5
    ):
        self.network_path = network_path
        self.device = device
        self.img_size = img_size
        self.window_size = window_size
        self.max_steps = max_steps
        self.fig = fig
        self.current_observation = 1

        model = CrossFormerModel.load_pretrained(network_path)
        self.crossformer = model

        self.obs_buffer = []
        self.test_array = []


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


    def predict_trajectory(self, env, obj_name):
        step_count = 0
        print(obj_name)

        self.task = self.crossformer.create_tasks(texts=[f"pick up the {obj_name}"])

        while step_count < self.max_steps:
            step_count += 1

            bgr, depth, _ = env.camera.get_cam_img()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            #Debugging
            cv2.imwrite("rgb_image.png", rgb)

            self.add_image_to_buffer(rgb)

            print(self.test_array)

            observation = self.get_observations()

            actions = self.crossformer.sample_actions(
                observation,
                self.task,
                head_name="single_arm",
                unnormalization_statistics=self.crossformer.dataset_statistics["bridge_dataset"]["action"],
                rng=jax.random.PRNGKey(0)
            )


            action = actions[0][0]

            dx, dy, dz, dyaw, dpitch, droll, grasp = self.parse_action(action)

            print(
                f"dx={dx}, "
                f"dy={dy}, "
                f"dz={dz}, "
                f"dyaw={dyaw}, "
                f"dpitch={dpitch}, "
                f"droll={droll}, "
                f"grasp={grasp}"
            )

            # if grasp >= 0.01:
            #     print(f"[TrajectoryGenerator] Grasp triggered at step {step_count}.")
            #     env.auto_close_gripper(check_contact=True)
            #     break

            ee_xyz, ee_rpy = env.get_ee_pose()
            x_new = ee_xyz[0] + dx
            y_new = ee_xyz[1] + dy
            z_new = ee_xyz[2] + dz

            roll = ee_rpy[0] + droll
            pitch = ee_rpy[1] + dpitch
            yaw = ee_rpy[2] + dyaw
            orn = p.getQuaternionFromEuler([roll, pitch, yaw])

            env.move_ee([x_new, y_new, z_new, orn], max_step=200)

            for _ in range(1):
                p.stepSimulation()
                time.sleep(env.SIMULATION_STEP_DELAY)

        if step_count == self.max_steps:
            print("[TrajectoryGenerator] Reached max steps without grasp=1. Stopping.")