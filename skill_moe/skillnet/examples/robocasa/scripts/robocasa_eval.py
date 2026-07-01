# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import dataclasses
import logging
import os
from pathlib import Path
import sys

import numpy as np
from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm
from tqdm import trange
import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

openpi_client_src = REPO_ROOT / "packages" / "openpi-client" / "src"
if openpi_client_src.exists():
    sys.path.insert(0, str(openpi_client_src))

groot_path = Path(os.environ.get("GROOT_PATH", REPO_ROOT / "examples" / "robocasa" / "gr00t"))
if groot_path.exists():
    sys.path.insert(0, str(groot_path))

from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper
from gr00t.eval.wrappers.robocasa_wrapper import load_robocasa_gym_env
from openpi_client import websocket_client_policy_robocasa as _websocket_client_policy

PALIGEMMA_ACTION_END_TOKEN = 235371


def get_env_horizon(env_name):
    if env_name in SINGLE_STAGE_TASK_DATASETS:
        ds_config = SINGLE_STAGE_TASK_DATASETS[env_name]
    elif env_name in MULTI_STAGE_TASK_DATASETS:
        ds_config = MULTI_STAGE_TASK_DATASETS[env_name]
    else:
        raise ValueError(f"Environment {env_name} not found in dataset registry")
    return ds_config["horizon"]


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# postprocess function of action, to handle the case where number of dimensions are not the same
def postprocess_action(action):
    new_action = {}
    for k, v in action.items():
        if v.ndim == 1:
            new_action[k] = v[..., None]
        else:
            new_action[k] = v
    return new_action


@dataclasses.dataclass
class Args:
    # Model/Policy
    port: int = 8000
    host: str = "0.0.0.0"

    seed: int = 7

    # Setup
    env_name: str = "<ENV_NAME>"  # Name of the environment to run. -> Task Name
    video_dir: str = "data/robocasa/videos"
    n_episodes: int = 50
    max_episode_steps: int = 750

    # Robocasa env parameters
    controller: str | None = (
        None  # Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file"
    )
    robots: list[str] = dataclasses.field(default_factory=lambda: ["PandaOmron"])  # Which robot(s) to use in the env
    config: str = "single-arm-opposed"  # Specified environment configuration if necessary
    arm: str = "right"
    obj_groups: list[str] | None = (
        None  # In kitchen environments, either the name of a group to sample object from or path to an .xml file"
    )
    layout: list[int] = dataclasses.field(default_factory=lambda: [-1])
    style: list[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11])

    generative_textures: bool = False  # Use generative textures for the environment

    # Data collection parameters
    collect_data: bool = False  # Whether to collect data


def eval_robocasa(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # logging 설정
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load Model from Server
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    available_policies = client.get_available_policies()
    action_horizon = client.get_action_horizon()
    logging.info(f"Available policies: {available_policies}")
    logging.info(f"Action horizon: {action_horizon}")

    # ROBOCASA ENV SETUP
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    env_name = args.env_name
    logging.info(f"Environment name: {env_name}")

    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
        "generative_textures": "100p",
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Mirror actions if using a kitchen environment -> Default : Not Use
    if env_name in ["Lift"]:  # add other non-kitchen tasks here
        if args.obj_groups is not None:
            logging.warning("Specifying 'obj_groups' in non-kitchen environment does not have an effect.")
    else:
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        ### update config for kitchen envs ###
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})

        # by default use obj instance split A
        config["obj_instance_split"] = "A"
        # config["obj_instance_split"] = None
        # config["obj_registries"] = ("aigen",)

    env = load_robocasa_gym_env(
        args.env_name,
        seed=args.seed,
        # robosuite-related configs
        robots=args.robots,
        camera_widths=256,
        camera_heights=256,
        render_onscreen=False,
        # robocasa-related configs
        obj_instance_split="B",
        generative_textures="100p" if args.generative_textures else None,
        randomize_cameras=False,
        layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
        # data collection configs
        collect_data=args.collect_data,
    )
    logging.info(f"Environment {args.env_name} loaded successfully.")

    env = RoboCasaWrapper(env)

    # logging is_success
    stats = defaultdict(list)
    evaluated_episodes = set()
    if os.path.exists(f"{args.video_dir}/prediction.txt"):
        with open(f"{args.video_dir}/prediction.txt") as f:
            for line in f:
                if line.startswith("episode"):
                    try:
                        ep_num = int(line.split()[1])
                        evaluated_episodes.add(ep_num)
                        # is_success 값도 파싱해서 stats에 추가
                        is_success_str = line.strip().split(":")[-1].strip()
                        # [True] 또는 [False] 형태 처리
                        if is_success_str.startswith("[") and is_success_str.endswith("]"):
                            is_success_val = is_success_str[1:-1].strip()
                            if is_success_val.lower() == "true":
                                is_success = True
                            elif is_success_val.lower() == "false":
                                is_success = False
                            else:
                                is_success = False
                        else:
                            # True/False 문자열 처리
                            is_success = is_success_str.lower() == "true"
                        add_to(stats, flatten({"is_success": is_success}))
                    except Exception as e:
                        logging.warning(f"prediction.txt 파싱 에러: {e}, line: {line}")

    def episode_trigger(episode_id):
        # episode_id는 실제 episode 번호 (0-49)
        # 모든 episode에 대해 비디오 생성 (n_save=1이므로)
        return True

    if args.video_dir is not None:
        env = RecordVideo(
            env,
            Path(args.video_dir),
            disable_logger=True,
            episode_trigger=episode_trigger,
            fps=20,
            name_prefix=f"{args.env_name}",
        )

    max_episode_steps = get_env_horizon(args.env_name)
    env = MultiStepWrapper(
        env,
        video_delta_indices=np.arange(1),  # t
        state_delta_indices=np.arange(1),  # t
        n_action_steps=action_horizon,
        max_episode_steps=max_episode_steps,
    )

    logging.info(f"Starting evaluation for {args.env_name} with {args.n_episodes} episodes...")
    # main evaluation loop
    for i in trange(args.n_episodes):
        pbar = tqdm(
            total=max_episode_steps, desc=f"Episode {i} / {env.unwrapped.get_ep_meta()['lang']}", leave=False
        )

        # RecordVideo wrapper에 episode 번호 설정 (env.reset() 이전에)
        if hasattr(env, "set_episode_number"):
            env.set_episode_number(i)

        # 1. Reset Environment and Get Observation
        obs, info = env.reset()

        # env.reset() 이후에 episode skip 체크
        if i in evaluated_episodes:
            logging.info(f"Skipping episode {i} as it has already been evaluated.")
            continue

        # Episode 실행을 위한 변수 초기화 (skip 이후에)
        done = False
        step = 0

        while not done:
            # NOTE: The observation is vertically flipped, uint8
            obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
            obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
            obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)

            # TODO : Train때 state 전체를 받는 것을 가정했어서 eval할때 concat해서 넘겨줘야함..
            state = np.concatenate(
                [
                    obs["state.base_position"],
                    obs["state.base_rotation"],
                    obs["state.end_effector_position_absolute"],
                    obs["state.end_effector_position_relative"],
                    obs["state.end_effector_rotation_absolute"],
                    obs["state.end_effector_rotation_relative"],
                    obs["state.gripper_qpos"],
                    obs["state.gripper_qvel"],
                    obs["state.joint_position"],
                    obs["state.joint_position_cos"],
                    obs["state.joint_position_sin"],
                    obs["state.joint_velocity"],
                ],
                axis=1,
            )

            assert state.shape == (1, 53)

            # Element
            # video.left_view: shape=(256, 256, 3), dtype=uint8
            # video.right_view: shape=(256, 256, 3), dtype=uint8
            # video.wrist_view: shape=(256, 256, 3), dtype=uint8
            # state : shape=(53,)
            # prompt : str

            element = {
                "video.left_view": obs["video.left_view"][0],
                "video.right_view": obs["video.right_view"][0],
                "video.wrist_view": obs["video.wrist_view"][0],
                "state": state[0],
                "annotation.human.action.task_description": obs["annotation.human.action.task_description"][0],
            }

            # 2. Forward Model
            import time

            # t0 = time.time()
            outputs = client.infer(element)
            # t1 = time.time()
            # print(f"Forward time: {(t1 - t0)*1000:.4f} \n")
            # # 将耗时追加写入一个文件，排除异常数据点的影响
            # if t1 - t0 < 1:  # 仅记录小于1000ms的正常数据点
            #     with open("forward_time_log_moe.txt", "a") as f:
            #     # 仅写入浮点数
            #         f.write(f"{(t1 - t0) * 1000:.4f}\n")

            action_chunk = outputs["actions"]
            action_chunk = action_chunk[:action_horizon, :]

            action_chunk = outputs["actions"]
            action_chunk = action_chunk[:action_horizon, :]

            # assert action_chunk.shape == (action_horizon, 12), f"{action_chunk.shape=} expected (16,12)"  # (16,12)

            action_dict = {
                "action.base_motion": action_chunk[..., :4],
                "action.control_mode": (action_chunk[..., 4:5] > 0.5).astype(action_chunk.dtype),  # should be binary
                "action.end_effector_position": action_chunk[..., 5:8],
                "action.end_effector_rotation": action_chunk[..., 8:11],
                "action.gripper_close": (action_chunk[..., 11:12] > 0.5).astype(action_chunk.dtype),  # should be binary
            }

            post_action = postprocess_action(action_dict)

            # 3. Execute action in environment
            obs, reward, terminated, truncated, info = env.step(post_action)
            done = terminated or truncated
            step += action_horizon

            pbar.update(action_horizon)

        add_to(stats, flatten({"is_success": info["is_success"]}))
        with open(f"{args.video_dir}/prediction.txt", "a") as f:
            f.write(f"episode {i} is_success: {info['is_success']} \n")
        pbar.close()

    # 4. Close Environment
    env.close()

    # 통계 계산 - 안전한 방식으로 처리
    # 먼저 계산할 키들을 미리 결정
    keys_to_process = list(stats.keys())
    additional_stats = {}

    for k in keys_to_process:
        try:
            # boolean 값들을 정수로 변환하여 평균 계산
            if k == "is_success":
                # True/False를 1/0으로 변환
                success_count = sum(1 for x in stats[k] if x)
                total_count = len(stats[k])
                if total_count > 0:
                    stats[k] = success_count / total_count
                    # 맞춘 개수 / 전체 개수 형태로 저장
                    additional_stats[f"{k}_count"] = f"{success_count}/{total_count}"
                else:
                    stats[k] = 0.0
                    additional_stats[f"{k}_count"] = "0/0"
            # 다른 통계값들에 대해서는 기존 방식 사용
            elif len(stats[k]) > 0:
                stats[k] = float(np.mean(stats[k]))
            else:
                stats[k] = 0.0
        except Exception as e:
            logging.warning(f"통계 계산 에러 for {k}: {e}")
            stats[k] = 0.0

    # 추가 통계를 stats에 병합
    stats.update(additional_stats)

    # 파일에 결과 작성
    for k, v in stats.items():
        with open(f"{args.video_dir}/prediction.txt", "a") as f:
            if k == "is_success":
                f.write(f"{k}: {v:.2f} ({stats[f'{k}_count']}) \n")
            # is_success만 저장하므로 다른 키들은 무시
    logging.info(stats)

    sys.exit()


if __name__ == "__main__":
    tyro.cli(eval_robocasa)
