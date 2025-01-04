import jax
import jax.numpy as jnp
import numpy as np

class CrossFormerWrapper:
    def __init__(self, model, params, device='cpu'):
        """
        :param model: A pretrained CrossFormerModel instance (Flax/JAX)
        :param params: JAX model parameters
        :param device: Not strictly needed for JAX, but kept for API compatibility
        """
        self.model = model
        self.params = params
        self.device = device

    def single_step_forward(self, rgb, depth):
        goal_image = rgb[None]

        task = self.model.create_tasks(
            goals={"image_primary": goal_image}
        )

        goal_image_with_time_dim = goal_image[:, None, ...]

        goal_image_sequence = np.tile(goal_image_with_time_dim, (1, 5, 1, 1, 1))  # Shape: (1, 5, 224, 224, 3)

        observation = {
          "image_primary": goal_image_sequence,
          "timestep_pad_mask": np.full((1, goal_image_sequence.shape[1]), True, dtype=bool),
        }

        actions = self.model.sample_actions(
            observation,
            task,
            head_name="single_arm",
            rng=jax.random.PRNGKey(0),
        )

        return actions[0]