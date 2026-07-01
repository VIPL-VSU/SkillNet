import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
import ast


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0
    
    offline_sampling: bool = False

    def __call__(self, data: dict) -> dict:
        # state : torch.Size([53]) -> [16] = (D)
        state_base_pos = data["state"][:3]
        state_base_rot = data["state"][3:7] #quaternion
        state_end_eff_pos_rel = data["state"][10:13]
        state_end_eff_rot_rel = data["state"][17:21] #quaternion
        state_gripper_qpos = data["state"][21:23]
        
        state = np.concatenate([state_base_pos, state_base_rot, state_end_eff_pos_rel, state_end_eff_rot_rel, state_gripper_qpos])
         # since the robocasa action_dim = 12, which is < state_dim = 16, so pad is skipped.
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        left_image = _parse_image(data["video.left_view"])
        right_image = _parse_image(data["video.right_view"])
        wrist_image = _parse_image(data["video.wrist_view"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                #NOTE : ordered from left arm to right arm
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (left_image, right_image, wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (left_image, right_image, wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST_RSCL:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (left_image, right_image, wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        
        if "annotation.human.action.task_description" in data:
            inputs["prompt"] = data["annotation.human.action.task_description"]
        
        if "action" in data: #SFT Data
            # action : torch.Size([16, 12]) = (H, D)
            action_base_motion = data["action"][:,:4]
            action_control_mode = data["action"][:,4:5] #binary
            action_end_eff_pos = data["action"][:,5:8]
            action_end_eff_rot = data["action"][:,8:11] #axis angle
            action_gripper_close = data["action"][:,11:12] #binary
            
            inputs["actions"] = np.concatenate([action_base_motion, action_control_mode, action_end_eff_pos, action_end_eff_rot, action_gripper_close], axis=1)

        if "skills" in data:
            # print(data["skills"])
            lst = ast.literal_eval(data["skills"])  # 把字符串安全地解析为 Python 列表
            arr = np.array(lst)
            arr = arr + 1
            if len(arr) < 4:
                pad_len = 4 - len(arr)
                arr = np.concatenate([arr, np.zeros(pad_len, dtype=arr.dtype)])
            
            # 生成 mask，1 表示有效，0 表示补的部分
            skill_mask = np.arange(len(arr)) < len(lst)

            inputs["skills"] = arr
            inputs["skill_mask"] = skill_mask        

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Robocasa, return 12 actions.
        return {**data, "actions": np.asarray(data["actions"][..., :12])}