# ----------------------------------------------------
# crossformer_wrapper.py
# ----------------------------------------------------
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
        # 1) If your rgb is (224,224,4), trim to (224,224,3)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        # 2) Build observations
        observations = build_observations(rgb)  # => shape (1,5,224,224,3) etc.

        # 3) Build tasks
        tasks = build_tasks_dummy(rgb)  # => shape (1,224,224,3) etc.

        # 4) Now call sample_actions
        actions_jax = self.model.sample_actions(
            observations=observations,
            tasks=tasks,
            # use the model’s required pad mask for timesteps:
            timestep_pad_mask=observations["timestep_pad_mask"], 
            train=False,
            argmax=True,
            rng=jax.random.PRNGKey(0)
        )
        actions_np = np.array(actions_jax)
        return actions_np
    

def build_observations(example_rgb):
    """
    Build an observations dict that matches exactly:
      {
        'image_high': (1, 5, 224, 224, 3),
        'image_left_wrist': (1, 5, 224, 224, 3),
        'image_nav': (1, 5, 224, 224, 3),
        'image_primary': (1, 5, 224, 224, 3),
        'image_right_wrist': (1, 5, 224, 224, 3),
        'pad_mask_dict': {
            'image_high': (1, 5),
            'image_left_wrist': (1, 5),
            'image_nav': (1, 5),
            'image_primary': (1, 5),
            'image_right_wrist': (1, 5),
            'proprio_bimanual': (1, 5),
            'proprio_quadruped': (1, 5),
            'timestep': (1, 5)
        },
        'proprio_bimanual':  (1, 5, 14),
        'proprio_quadruped': (1, 5, 59),
        'task_completed':    (1, 5, 100),
        'timestep':          (1, 5),
        'timestep_pad_mask': (1, 5)
      }

    Args:
      example_rgb: A single (224,224,3) NumPy array (RGB). We'll tile it to (1,5,224,224,3).

    Returns:
      observations: a dict with all required keys and shapes.
    """

    # 1) Tile your single RGB (224,224,3) into (1,5,224,224,3)
    #    to mimic 5 timesteps.
    repeated_rgb = np.tile(example_rgb[None, None, ...], (1, 5, 1, 1, 1))

    # 2) For the other images (image_high, image_left_wrist, etc.), use zeros
    #    but keep the exact shape. E.g. (1,5,224,224,3)
    dummy_img = np.zeros_like(repeated_rgb)
    
    observations = {
        "image_high":        dummy_img.copy(),
        "image_left_wrist":  dummy_img.copy(),
        "image_nav":         dummy_img.copy(),
        "image_primary":     repeated_rgb,   # We'll actually put the real top-down image here
        "image_right_wrist": dummy_img.copy(),
    }

    # 3) pad_mask_dict => each key => shape (1,5)
    pad_mask_dict = {
        "image_high":         np.ones((1,5), dtype=bool),
        "image_left_wrist":   np.ones((1,5), dtype=bool),
        "image_nav":          np.ones((1,5), dtype=bool),
        "image_primary":      np.ones((1,5), dtype=bool),
        "image_right_wrist":  np.ones((1,5), dtype=bool),
        "proprio_bimanual":   np.ones((1,5), dtype=bool),
        "proprio_quadruped":  np.ones((1,5), dtype=bool),
        "timestep":           np.ones((1,5), dtype=bool),
    }

    # 4) proprio_bimanual => shape (1,5,14)
    dummy_proprio_bimanual = np.zeros((1,5,14), dtype=np.float32)

    # 5) proprio_quadruped => shape (1,5,59)
    dummy_proprio_quadruped = np.zeros((1,5,59), dtype=np.float32)

    # 6) task_completed => shape (1,5,100)
    dummy_task_completed = np.zeros((1,5,100), dtype=np.float32)

    # 7) timestep => shape (1,5)
    dummy_timestep = np.zeros((1,5), dtype=np.float32)

    # 8) timestep_pad_mask => shape (1,5)
    #    The model wants a separate "timestep_pad_mask" key, so let's set it all True
    #    to indicate these are real timesteps (not padded).
    dummy_timestep_pad_mask = np.ones((1,5), dtype=bool)

    # Add them all to observations
    observations["pad_mask_dict"]   = pad_mask_dict
    observations["proprio_bimanual"] = dummy_proprio_bimanual
    observations["proprio_quadruped"] = dummy_proprio_quadruped
    observations["task_completed"]  = dummy_task_completed
    observations["timestep"]        = dummy_timestep
    observations["timestep_pad_mask"] = dummy_timestep_pad_mask

    return observations


def build_tasks_dummy(example_rgb_224_224_3):
    """
    Creates a tasks dict that matches the shape:
      {
        'image_high':         (1,224,224,3)
        'image_left_wrist':   (1,224,224,3)
        'image_nav':          (1,224,224,3)
        'image_primary':      (1,224,224,3)
        'image_right_wrist':  (1,224,224,3)
        'language_instruction': (1,512)
        'pad_mask_dict': {
           'image_high': (1,),
           'image_left_wrist': (1,),
           'image_nav': (1,),
           'image_primary': (1,),
           'image_right_wrist': (1,),
           'language_instruction': (1,),
           'proprio_bimanual': (1,),
           'proprio_quadruped': (1,),
           'timestep': (1,)
        },
        'proprio_bimanual':   (1,14),
        'proprio_quadruped':  (1,59),
        'timestep':           (1,)
      }

    We'll store your real top-down image into 'image_primary', and zero for the rest.
    """

    # 1) Real top-down image => shape (1,224,224,3)
    real_image_task = example_rgb_224_224_3[None, ...]  # shape => (1,224,224,3)

    # 2) Zero arrays for the other cameras
    dummy_img = np.zeros_like(real_image_task)  # (1,224,224,3)

    tasks = {
        "image_high":         dummy_img.copy(),
        "image_left_wrist":   dummy_img.copy(),
        "image_nav":          dummy_img.copy(),
        "image_primary":      real_image_task,   # put the real camera here
        "image_right_wrist":  dummy_img.copy(),
    }

    # 3) language_instruction => shape (1,512)
    #    If your model used text instructions, you can encode something meaningful.
    #    Otherwise, just fill with zeros.
    tasks["language_instruction"] = np.zeros((1,512), dtype=np.float32)

    # 4) pad_mask_dict => each key => (1,)
    pad_mask_dict = {
      "image_high":          np.ones((1,), dtype=bool),
      "image_left_wrist":    np.ones((1,), dtype=bool),
      "image_nav":           np.ones((1,), dtype=bool),
      "image_primary":       np.ones((1,), dtype=bool),
      "image_right_wrist":   np.ones((1,), dtype=bool),
      "language_instruction":np.ones((1,), dtype=bool),
      "proprio_bimanual":    np.ones((1,), dtype=bool),
      "proprio_quadruped":   np.ones((1,), dtype=bool),
      "timestep":            np.ones((1,), dtype=bool),
    }

    tasks["pad_mask_dict"] = pad_mask_dict

    # 5) proprio_bimanual => shape (1,14)
    tasks["proprio_bimanual"] = np.zeros((1,14), dtype=np.float32)

    # 6) proprio_quadruped => shape (1,59)
    tasks["proprio_quadruped"] = np.zeros((1,59), dtype=np.float32)

    # 7) timestep => shape (1,)
    tasks["timestep"] = np.zeros((1,), dtype=np.float32)

    return tasks