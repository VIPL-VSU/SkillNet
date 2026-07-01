import os
import json
import tqdm_loggable.auto as tqdm
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import openpi.policies.policy as _policy
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import dataclasses
import numpy as np
import random
from collections import deque
import tyro
import logging

@dataclasses.dataclass
class Checkpoint:
    policy_config: str
    policy_dir: str
    best_of_n: int = 1
    temperature: float = 0.0

@dataclasses.dataclass
class Args:
    checkpoint: Checkpoint
    output_dir: str
    batch_size: int = 1
    num_workers: int = 16
    subset_size: int = 414110 # all 60 demos
    chunk_size: int = 10000
    negative_sample_prob: float = 0.1
    
    default_prompt: str | None = None
    seed: int = 7

def make_preference_dataset(args: Args, policy: _policy.Policy, config):
    
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = sharding.jax.sharding.NamedSharding(mesh, sharding.jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=False,
        subset_size=args.subset_size, #all 60 demos
        num_batches=args.subset_size // args.batch_size,
    )
    
    data_size = len(data_loader._data_loader._dataset)
    logging.info(f"Data loader created with {data_size} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    chunk = []
    chunk_idx = 0
    action_queue = deque()
    prev_episode_index = None

    for batch_idx, (observation, og_state, og_action, timestamp, frame_index, episode_index, index, task_index) in enumerate(tqdm.tqdm(data_loader, desc="Merging GT and model negative actions")):
        
        # 에피소드가 바뀌면 queue를 비움
        if prev_episode_index is not None and episode_index[0] != prev_episode_index:
            action_queue.clear()
        prev_episode_index = episode_index[0]
        
        if not action_queue and random.random() < args.negative_sample_prob:
            if hasattr(observation, 'to_dict'):
                obs_dict = observation.to_dict()
            else:
                obs_dict = observation
                
            outputs = policy.infer(obs=obs_dict, action_tokens=None)
            
            pred_actions = outputs['actions']  # (N, horizon, 12) or (horizon, 12) (Batch=1 전제)
            action_masks = outputs['action_masks']  # (N, 1) or [True or False]
            
            if pred_actions.ndim == 3:
                action_horizon = pred_actions.shape[1]
            else:
                action_horizon = pred_actions.shape[0]
                
            
            for t in range(action_horizon):
                if pred_actions.ndim == 3:
                    # NOTE: decoding error -> zero action -> all drop (for better dataset)
                    if action_masks[0][0] == True:
                        action_queue.append(np.array(pred_actions[0, t], dtype=np.float64).reshape(12).tolist())
                else:
                    if action_masks[0] == True:
                        action_queue.append(np.array(pred_actions[t], dtype=np.float64).reshape(12).tolist())
            

        # negative action: queue에서 pop, 없으면 None
        negative_action = action_queue.popleft() if action_queue else None
        
        if negative_action is not None:
            arr = np.array(negative_action)
            # numpy 변환 후 binary 처리, 다시 list로 변환
            arr[4] = arr[4] > 0.5
            arr[11] = arr[11] > 0.5
            negative_action = arr.astype(np.float64).tolist()
            
            
        # GT action: 항상 첫번째만 저장 (batch=1, (1, 16, 12))
        gt_action_arr = np.array(og_action[0], dtype=np.float64)
        positive_action = gt_action_arr[0].reshape(12).tolist()
        
        frame_info = {
            'timestamp': np.array(timestamp[0], dtype=np.float32).reshape(1).tolist(),
            'frame_index': np.array(frame_index[0], dtype=np.int64).reshape(1).tolist(),
            'episode_index': np.array(episode_index[0], dtype=np.int64).reshape(1).tolist(),
            'index': np.array(index[0], dtype=np.int64).reshape(1).tolist(),
            'task_index': np.array(task_index[0], dtype=np.int64).reshape(1).tolist(),
            
            'observation.state': np.array(og_state[0], dtype=np.float64).reshape(53).tolist(),
            'positive_action': positive_action,
            'negative_action': negative_action,
        }
        
        chunk.append(frame_info)
        if len(chunk) == args.chunk_size:
            with open(os.path.join(args.output_dir, f"preference_dataset_chunk_{chunk_idx:04d}.json"), "w") as f:
                json.dump(chunk, f)
            logging.info(f"Saved chunk {chunk_idx:04d} ({len(chunk)} samples)")
            chunk = []
            chunk_idx += 1
            
    if chunk:
        with open(os.path.join(args.output_dir, f"preference_dataset_chunk_{chunk_idx:04d}.json"), "w") as f:
            json.dump(chunk, f)
        logging.info(f"Saved chunk {chunk_idx:04d} ({len(chunk)} samples)")
        
        
    logging.info(f"Saved {chunk_idx+1} merged action json chunk files to {args.output_dir}")

def main(args: Args):
    logging.basicConfig(level=logging.INFO, force=True)
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    config = _config.get_config(args.checkpoint.policy_config)
    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data=dataclasses.replace(config.data, offline_sampling=True)
    )
    
    logging.info(config)
    
    policy = _policy_config.create_trained_policy(
        config,
        args.checkpoint.policy_dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"temperature": args.checkpoint.temperature, "best_of_n": args.checkpoint.best_of_n},
    )
    
    logging.info("Policy 생성 완료!")
    
    make_preference_dataset(args, policy, config)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args) 