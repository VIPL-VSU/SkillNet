"""Evaluate a SkillNet policy in a local RoboTwin-2.0 checkout.

This runner intentionally keeps the RoboTwin simulator and the SkillNet model
runtime decoupled. Start `scripts/serve_policy_moe_skill.py` from the SkillNet
source root, then run this script against a RoboTwin checkout. Observations are
sent to the SkillNet websocket server using the same field names consumed by
the RoboTwin LeRobot training config.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SKILLNET_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_SKILL_PLAN = REPO_ROOT / "data_process" / "robotwin" / "robotwin_plan.json"

PRETRAIN_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "click_bell",
    "grab_roller",
    "handover_block",
    "lift_pot",
    "move_can_pot",
    "move_playingcard_away",
    "open_microwave",
    "place_burger_fries",
    "place_object_basket",
    "rotate_qrcode",
    "shake_bottle_horizontally",
    "stack_blocks_two",
]

TRANSFER_TASKS = [
    "blocks_ranking_size",
    "hanging_mug",
    "move_pillbottle_pad",
    "open_laptop",
    "place_a2b_left",
    "place_bread_basket",
    "place_bread_skillet",
    "place_cans_plasticbox",
    "place_fan",
    "press_stapler",
    "scan_object",
    "shake_bottle",
    "stack_blocks_three",
    "stack_bowls_two",
    "stamp_seal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True, help="Path to a RoboTwin-2.0 checkout.")
    parser.add_argument("--host", default="127.0.0.1", help="SkillNet policy server host.")
    parser.add_argument("--port", type=int, default=8098, help="SkillNet policy server port.")
    parser.add_argument("--task-config", default="demo_clean", help="RoboTwin task_config YAML stem.")
    parser.add_argument("--task-set", choices=["pretrain", "transfer", "paper", "all"], default="transfer")
    parser.add_argument("--tasks", nargs="*", default=None, help="Explicit RoboTwin task names. Overrides --task-set.")
    parser.add_argument("--skill-plan", type=Path, default=DEFAULT_SKILL_PLAN)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seed-attempts", type=int, default=10000)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--action-horizon", type=int, default=10, help="Number of actions to execute per server call.")
    parser.add_argument("--result-dir", type=Path, default=Path("data/robotwin/eval_results"))
    parser.add_argument("--expert-seed-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-render-test", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def add_import_paths(robotwin_root: Path) -> None:
    for path in (
        SKILLNET_ROOT / "packages" / "openpi-client" / "src",
        robotwin_root,
        robotwin_root / "policy",
        robotwin_root / "description" / "utils",
    ):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return data


def config_path(configs_path: Any, fallback_root: Path, name: str) -> Path:
    path = Path(str(configs_path)) / name
    if path.exists():
        return path
    return fallback_root / "task_config" / name


def load_task_args(robotwin_root: Path, task_name: str, task_config: str, *, save_video: bool) -> tuple[dict[str, Any], str]:
    from envs import CONFIGS_PATH

    args = load_yaml(robotwin_root / "task_config" / f"{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["policy_name"] = "skillnet_websocket"
    args["ckpt_setting"] = "skillnet_server"

    camera_config = load_yaml(config_path(CONFIGS_PATH, robotwin_root, "_camera_config.yml"))
    embodiment_types = load_yaml(config_path(CONFIGS_PATH, robotwin_root, "_embodiment_config.yml"))

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]
    video_size = f"{camera_config[head_camera_type]['w']}x{camera_config[head_camera_type]['h']}"

    embodiment = args.get("embodiment")
    def resolve_robot_file(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else robotwin_root / path

    if len(embodiment) == 1:
        left_robot_file = resolve_robot_file(embodiment_types[embodiment[0]]["file_path"])
        right_robot_file = resolve_robot_file(embodiment_types[embodiment[0]]["file_path"])
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left_robot_file = resolve_robot_file(embodiment_types[embodiment[0]]["file_path"])
        right_robot_file = resolve_robot_file(embodiment_types[embodiment[1]]["file_path"])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("RoboTwin embodiment must contain either one item or three items.")

    args["left_robot_file"] = str(left_robot_file)
    args["right_robot_file"] = str(right_robot_file)
    args["left_embodiment_config"] = load_yaml(left_robot_file / "config.yml")
    args["right_embodiment_config"] = load_yaml(right_robot_file / "config.yml")
    args["eval_video_log"] = bool(save_video and args.get("eval_video_log", False))
    args["eval_mode"] = True
    return args, video_size


def load_skill_plan(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def resolve_tasks(task_set: str, tasks: list[str] | None, plan: dict[str, Any]) -> list[str]:
    if tasks:
        return tasks
    if task_set == "pretrain":
        return PRETRAIN_TASKS
    if task_set == "transfer":
        return TRANSFER_TASKS
    if task_set == "paper":
        return PRETRAIN_TASKS + TRANSFER_TASKS
    return sorted(plan)


def make_env(task_name: str):
    env_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(env_module, task_name)
    return env_class()


def observation_to_skillnet(observation: dict[str, Any], *, instruction: str, skills: list[int]) -> dict[str, Any]:
    obs = observation["observation"]
    return {
        "head_color": obs["head_camera"]["rgb"],
        "hand_left_color": obs["left_camera"]["rgb"],
        "hand_right_color": obs["right_camera"]["rgb"],
        "state": observation["joint_action"]["vector"],
        "prompt": instruction,
        "skills": skills,
    }


def choose_instruction(
    *,
    task_name: str,
    env: Any,
    episode_info: dict[str, Any] | None,
    instruction_type: str,
    plan_description: str,
    rng: Any,
) -> str:
    if instruction_type in {"task", "plan", "description"}:
        return plan_description
    if instruction_type in {"env", "default"}:
        return str(env.get_instruction())

    try:
        from generate_episode_instructions import generate_episode_descriptions

        info = episode_info.get("info", episode_info) if episode_info else {}
        generated = generate_episode_descriptions(task_name, [info], 1)
        choices = generated[0].get(instruction_type)
        if choices:
            return str(rng.choice(choices))
    except Exception as exc:  # pragma: no cover - depends on RoboTwin checkout.
        logging.debug("Falling back to RoboTwin environment instruction: %s", exc)

    try:
        return str(env.get_instruction())
    except Exception:
        return plan_description


def close_env(env: Any, *, clear_cache: bool = False) -> None:
    try:
        env.close_env(clear_cache=clear_cache)
    except TypeError:
        env.close_env()
    except Exception as exc:  # pragma: no cover - simulator cleanup best effort.
        logging.debug("Ignoring environment cleanup error: %s", exc)


def find_stable_seed(env: Any, env_args: dict[str, Any], seed: int, episode_index: int) -> tuple[bool, dict[str, Any] | None]:
    render_freq = env_args.get("render_freq", 0)
    env_args["render_freq"] = 0
    try:
        env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **env_args)
        episode_info = env.play_once()
        stable = bool(getattr(env, "plan_success", False) and env.check_success())
        return stable, episode_info
    except Exception as exc:  # pragma: no cover - simulator-specific instability.
        logging.debug("Seed %s rejected by expert check: %s", seed, exc)
        return False, None
    finally:
        env_args["render_freq"] = render_freq
        close_env(env)


def infer_action_chunk(policy: Any, payload: dict[str, Any], action_horizon: int) -> np.ndarray:
    import numpy as np

    response = policy.infer(payload)
    if isinstance(response, (list, tuple)):
        response = response[0]
    actions = np.asarray(response["actions"])[..., :16]
    if actions.ndim == 1:
        actions = actions[None, :]
    if action_horizon > 0:
        actions = actions[:action_horizon]
    return actions


def run_trial(
    *,
    task_name: str,
    env: Any,
    env_args: dict[str, Any],
    policy: Any,
    instruction: str,
    skills: list[int],
    seed: int,
    episode_index: int,
    action_horizon: int,
) -> dict[str, Any]:
    env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **env_args)
    if hasattr(env, "set_instruction"):
        env.set_instruction(instruction=instruction)

    policy.reset()
    success = False
    try:
        while env.take_action_cnt < env.step_lim:
            observation = env.get_obs()
            payload = observation_to_skillnet(observation, instruction=instruction, skills=skills)
            actions = infer_action_chunk(policy, payload, action_horizon)
            for action in actions:
                env.take_action(action)
                if getattr(env, "eval_success", False):
                    success = True
                    break
            if success:
                break
    finally:
        steps = int(getattr(env, "take_action_cnt", -1))
        close_env(env, clear_cache=((episode_index + 1) % int(env_args.get("clear_cache_freq", 5)) == 0))

    return {
        "task": task_name,
        "episode_index": episode_index,
        "seed": seed,
        "success": success,
        "steps": steps,
        "instruction": instruction,
        "skills": skills,
    }


def evaluate_task(
    *,
    task_name: str,
    env_args: dict[str, Any],
    policy: Any,
    plan: dict[str, Any],
    num_trials: int,
    seed_start: int,
    max_seed_attempts: int,
    instruction_type: str,
    action_horizon: int,
    expert_seed_filter: bool,
    result_dir: Path,
) -> dict[str, Any]:
    if task_name not in plan:
        raise KeyError(f"Task {task_name!r} is missing from the skill plan.")
    task_plan = plan[task_name]
    skills = [int(x) for x in task_plan.get("skills", [])]
    description = str(task_plan.get("description", task_name))
    task_dir = result_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    episode_path = task_dir / "episodes.jsonl"

    rng = np.random.default_rng(seed_start)
    seed = seed_start
    attempts = 0
    records = []
    with episode_path.open("w", encoding="utf-8") as f:
        while len(records) < num_trials:
            if attempts >= max_seed_attempts:
                raise RuntimeError(f"Could not collect {num_trials} stable trials for {task_name}.")
            attempts += 1
            episode_info = None
            env = make_env(task_name)
            if expert_seed_filter:
                stable, episode_info = find_stable_seed(env, env_args, seed, len(records))
                if not stable:
                    seed += 1
                    continue

            env = make_env(task_name)
            instruction = choose_instruction(
                task_name=task_name,
                env=env,
                episode_info=episode_info,
                instruction_type=instruction_type,
                plan_description=description,
                rng=rng,
            )
            record = run_trial(
                task_name=task_name,
                env=env,
                env_args=env_args,
                policy=policy,
                instruction=instruction,
                skills=skills,
                seed=seed,
                episode_index=len(records),
                action_horizon=action_horizon,
            )
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            logging.info(
                "%s %d/%d success=%s seed=%d",
                task_name,
                len(records),
                num_trials,
                record["success"],
                seed,
            )
            seed += 1

    successes = sum(1 for record in records if record["success"])
    summary = {
        "task": task_name,
        "num_trials": num_trials,
        "successes": successes,
        "success_rate": successes / num_trials if num_trials else 0.0,
        "result_file": str(episode_path),
    }
    (task_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    robotwin_root = args.robotwin_root.resolve()
    add_import_paths(robotwin_root)

    if not args.skip_render_test:
        from test_render import Sapien_TEST

        Sapien_TEST()

    from openpi_client import websocket_client_policy
    import numpy as np

    plan = load_skill_plan(args.skill_plan)
    tasks = resolve_tasks(args.task_set, args.tasks, plan)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for task_name in tasks:
        env_args, _video_size = load_task_args(robotwin_root, task_name, args.task_config, save_video=False)
        summary = evaluate_task(
            task_name=task_name,
            env_args=env_args,
            policy=policy,
            plan=plan,
            num_trials=args.num_trials,
            seed_start=args.seed,
            max_seed_attempts=args.max_seed_attempts,
            instruction_type=args.instruction_type,
            action_horizon=args.action_horizon,
            expert_seed_filter=args.expert_seed_filter,
            result_dir=result_dir,
        )
        summaries.append(summary)

    total_trials = sum(item["num_trials"] for item in summaries)
    total_successes = sum(item["successes"] for item in summaries)
    aggregate = {
        "tasks": summaries,
        "num_tasks": len(summaries),
        "num_trials": total_trials,
        "successes": total_successes,
        "success_rate": total_successes / total_trials if total_trials else 0.0,
    }
    output_path = result_dir / "summary.json"
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Wrote RoboTwin SkillNet evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
