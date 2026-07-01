import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
DEFAULT_SKILL_ANNOTATION_PATH = (
    pathlib.Path(__file__).resolve().parent / "annotations" / "libero_skill_obj_annotations.json"
)
DEFAULT_LIBERO_SKILL_MANIFEST_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "third_party"
    / "libero"
    / "libero"
    / "libero"
    / "bddl_files"
    / "libero_skill_obj"
    / "public_task_manifest.json"
)


def _load_expected_libero_skill_tasks() -> list[str]:
    manifest = json.loads(DEFAULT_LIBERO_SKILL_MANIFEST_PATH.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ValueError(f"Invalid LIBERO-Skill manifest: {DEFAULT_LIBERO_SKILL_MANIFEST_PATH}")
    return tasks


EXPECTED_LIBERO_SKILL_TASKS = _load_expected_libero_skill_tasks()


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8056
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_skill_obj.
    # The alias "libero_skill" is also accepted when the installed LIBERO registers that name directly.
    task_suite_name: str = "libero_skill_obj"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos/skillnet_libero_skill_obj"
    skill_annotation_path: pathlib.Path = DEFAULT_SKILL_ANNOTATION_PATH

    seed: int = 7  # Random Seed (for reproducibility)
    zero_shot: bool = True
    fail_fast: bool = False


def _validate_libero_skill_tasks(task_suite) -> None:
    observed = [pathlib.Path(task_suite.get_task(task_id).bddl_file).stem for task_id in range(task_suite.n_tasks)]
    if observed != EXPECTED_LIBERO_SKILL_TASKS:
        expected = "\n".join(f"  {idx + 1}. {task}" for idx, task in enumerate(EXPECTED_LIBERO_SKILL_TASKS))
        actual = "\n".join(f"  {idx + 1}. {task}" for idx, task in enumerate(observed))
        raise ValueError(
            "The registered LIBERO-Skill task list does not match SkillNet's public 9-task benchmark.\n"
            f"Expected:\n{expected}\nActual:\n{actual}\n"
            "Run examples/libero/install_libero_skill_assets.py --install or set --task-suite-name to the "
            "benchmark key that contains these 9 tasks."
        )


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = args.task_suite_name
    if task_suite_name not in benchmark_dict and task_suite_name == "libero_skill_obj" and "libero_skill" in benchmark_dict:
        task_suite_name = "libero_skill"
    elif task_suite_name not in benchmark_dict and task_suite_name == "libero_skill" and "libero_skill_obj" in benchmark_dict:
        task_suite_name = "libero_skill_obj"
    if task_suite_name not in benchmark_dict:
        available = ", ".join(sorted(benchmark_dict))
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. Available task suites: {available}")
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {task_suite_name}")
    if task_suite_name in ("libero_skill", "libero_skill_obj"):
        _validate_libero_skill_tasks(task_suite)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    elif task_suite_name in ("libero_skill", "libero_skill_obj"):
        max_steps = 800  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.zero_shot:
        max_steps = max_steps * 2

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    skill_anno_file = args.skill_annotation_path.expanduser()
    if not skill_anno_file.exists():
        raise FileNotFoundError(
            f"Skill annotation file not found: {skill_anno_file}. "
            "Pass --skill-annotation-path or keep examples/libero/annotations/libero_skill_obj_annotations.json in place."
        )
    with open(skill_anno_file, "r", encoding="utf-8") as f:
        all_classes_list = json.load(f)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # if task_id not in [2, 10, 11, 21, 24, 25, 26, 27, 32]: 
        #     continue

        # if task_id not in [25, 26, 31, 32, 33, 34, 35, 36]: 
        #     continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        if task_description[0] == ' ':
            task_description = task_description[1:]
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if task_description not in all_classes_list:
                        raise KeyError(f"Task description not found in skill annotations: {task_description}")
                    task_annotation = all_classes_list[str(task_description)]
                    all_classes = str(task_annotation["all_classes"])
                    objects = str(task_annotation["objects"]) if "objects" in task_annotation else None
                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "skills": all_classes,
                        }
                        if objects is not None:
                            element["objects"] = objects

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception:
                    logging.exception("Caught exception during rollout")
                    if args.fail_fast:
                        raise
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            # if done: ### TODO
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"{task_id:02d}_{task_segment}_{suffix}_ep{episode_idx+1:02d}_{suffix}.mp4", # 
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    import datetime

    args = tyro.cli(Args)

    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_libero_{args.task_suite_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Logging to {log_file}")

    eval_libero(args)
