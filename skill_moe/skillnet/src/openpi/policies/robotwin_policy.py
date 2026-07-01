import ast
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_skill_ids(value) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.int64)


@dataclasses.dataclass(frozen=True)
class RoboTwinInputs(transforms.DataTransformFn):
    """Map RoboTwin LeRobot samples to the SkillNet model input format."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    max_skills: int = 4

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(data["head_color"]),
                "left_wrist_0_rgb": _parse_image(data["hand_left_color"]),
                "right_wrist_0_rgb": _parse_image(data["hand_right_color"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "skills" in data:
            skill_ids = _parse_skill_ids(data["skills"])
            num_skills = min(skill_ids.shape[0], self.max_skills)
            padded = np.zeros((self.max_skills,), dtype=np.int64)
            padded[:num_skills] = skill_ids[:num_skills] + 1
            inputs["skills"] = padded
            inputs["skill_mask"] = np.arange(self.max_skills) < num_skills

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboTwinOutputs(transforms.DataTransformFn):
    """Return the 16-D dual-arm joint action used by RoboTwin."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :16])}
