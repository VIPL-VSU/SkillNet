import os
import json
import logging
import dataclasses
import multiprocessing
import random

import numpy as np
import sentencepiece
import tyro
import tqdm_loggable.auto as tqdm

# JAX/SkillNet compatibility imports
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
import openpi.shared.download as download

from openpi.models.fast_tokenizer import UniversalActionProcessor
from torch.utils.data import Subset

# 멀티프로세싱 시작 방식 고정 (HPC/Slurm 환경 일관성)
multiprocessing.set_start_method('spawn', force=True)


@dataclasses.dataclass
class Args:
    config_name: str
    output_dir: str
    batch_size: int = 1
    num_workers: int = 16
    max_frames: int | None = None
    sample_ratio: float = 1.0
    seed: int = 42


def create_dataset(config: _config.TrainConfig, sample_ratio: float = 1.0, seed: int = 42) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    
    # 단일 데이터셋만 처리
    dataset = _data_loader.create_dataset(data_config, config.model)
    
    # Transform 적용
    final_dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                data_config.norm_stats, 
                use_quantiles=data_config.use_quantile_norm,
                key_mapping=data_config.normalize_key_mapping
            ),
            *data_config.model_transforms.inputs,
        ],
    )
        
    # 샘플링 적용 (sample_ratio < 1.0일 때만)
    if sample_ratio < 1.0:
        n = len(final_dataset)
        sample_size = int(n * sample_ratio)
        rng = random.Random(seed)
        indices = rng.sample(range(n), sample_size)
        final_dataset = Subset(final_dataset, indices)
        logging.info(f"Sampled {sample_size} from {n} total samples (ratio: {sample_ratio})")
    
    return data_config, final_dataset


def measure_tokenize_detokenize_loss(args: Args):
    config = _config.get_config(args.config_name)
    data_config, dataset = create_dataset(config, sample_ratio=args.sample_ratio, seed=args.seed)

    num_frames = len(dataset)
    
    # shuffle와 num_batches 결정
    shuffle = False
    if args.sample_ratio < 1.0:
        shuffle = True  # 샘플링 시 shuffle
    
    if args.max_frames is not None and args.max_frames < num_frames:
        num_batches = args.max_frames // args.batch_size
        shuffle = True  # max_frames 제한 시 shuffle
    else:
        num_batches = num_frames // args.batch_size
    
    data_loader = _data_loader.TorchDataLoader(
        dataset=dataset,
        local_batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    # DataLoaderImpl로 감싸서 SFTBatch 형태로 변환
    class DataLoaderImpl:
        def __init__(self, data_config, torch_data_loader):
            self._data_config = data_config
            self._data_loader = torch_data_loader

        def data_config(self):
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                # SFT 형태로 변환
                from openpi.models import model as _model
                observation = _model.Observation.from_dict(batch)
                actions = batch["actions"]
                yield _model.SFTBatch(observation=observation, actions=actions)

    wrapped_data_loader = DataLoaderImpl(data_config, data_loader)

    # 로깅 정보
    logging.info(
        f"Dataset: {num_frames} samples, batch_size={args.batch_size}, "
        f"num_batches={num_batches}, max_frames={args.max_frames}, shuffle={shuffle}"
    )

    # ----- Output 준비 -----
    os.makedirs(args.output_dir, exist_ok=True)

    # Config에서 action dimensions 가져오기
    action_horizon = config.model.action_horizon  # 16
    action_dim = config.model.action_dim  # 12

    # UniversalActionProcessor를 from_pretrained로 초기화
    fast_tok = UniversalActionProcessor.from_pretrained(
        "physical-intelligence/fast",
    )
    
    # PaliGemma tokenizer 별도 초기화 (tokenizer.py와 동일한 방식)
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    
    fast_skip_tokens = 128  # FASTTokenizer에서 사용하던 값
    
    # Loss 계산용 변수들
    full_token_losses = []  # 전체 토큰으로 detokenize한 경우의 loss (전체 평균)
    first_10_token_losses = []  # 처음 10개 토큰으로 detokenize한 경우의 loss (전체 평균)
    
    # 차원별 loss 계산용 변수들
    full_token_losses_per_dim = []  # 전체 토큰: 각 차원별 loss (N, 7)
    first_10_token_losses_per_dim = []  # 처음 10개 토큰: 각 차원별 loss (N, 7)
    
    # 복원률 계산을 위한 원본 variance 계산용
    original_actions_list = []  # 원본 action들 (뒤 7개 차원만)
    
    total_sequences = 0
    skipped_sequences = 0

    # 메인 처리 루프
    for batch_idx, sft_batch in enumerate(
        tqdm.tqdm(wrapped_data_loader, desc="Measuring tokenize->detokenize loss")
    ):
        obs = sft_batch.observation
        actions = sft_batch.actions  # 원본 normalized actions (B, H, D)
        tokens = np.asarray(obs.tokenized_prompt)              # (B, L)
        token_mask = np.asarray(obs.tokenized_prompt_mask)     # (B, L)
        loss_mask = np.asarray(obs.token_loss_mask)            # (B, L); postfix(True)

        B = tokens.shape[0]

        for i in range(B):
            valid = token_mask[i].astype(bool)
            post = loss_mask[i].astype(bool)

            seq = tokens[i][valid]
            post_seq_mask = post[valid]

            post_idxs = np.where(post_seq_mask)[0]
            if post_idxs.size == 0:
                skipped_sequences += 1
                continue

            postfix_segment = seq[post_idxs]
            if postfix_segment.size == 0:
                skipped_sequences += 1
                continue

            # extract_actions 방식을 따라 텍스트 레벨에서 처리
            # 1. postfix segment를 텍스트로 디코딩
            decoded_tokens = paligemma_tokenizer.decode(postfix_segment.tolist())
            
            # 2. "Action: " 체크 및 추출
            if "Action: " not in decoded_tokens:
                skipped_sequences += 1
                continue
            
            # 3. Action: 이후 부분을 추출하고 | 이전까지만 가져오기
            try:
                action_part = decoded_tokens.split("Action: ")[1].split("|")[0].strip()
            except (IndexError, AttributeError):
                skipped_sequences += 1
                continue
                
            if not action_part:
                skipped_sequences += 1
                continue
            
            # 4. 텍스트를 다시 토큰화
            raw_action_tokens = np.array(paligemma_tokenizer.encode(action_part))
            if raw_action_tokens.size == 0:
                skipped_sequences += 1
                continue

            # 원본 action chunk 가져오기 (B, H, D) -> (H, D)
            original_actions = actions[i]  # (16, 12)
            
            # 5. PaliGemma token을 FAST token으로 변환 (extract_actions와 동일)
            fast_action_tokens = paligemma_tokenizer.vocab_size() - 1 - fast_skip_tokens - raw_action_tokens
            
            # 1. 전체 토큰으로 detokenize
            try:
                decoded_full = fast_tok.decode(
                    [fast_action_tokens.tolist()], 
                    time_horizon=action_horizon, 
                    action_dim=action_dim
                )[0]  # (16, 12)
                
                # Loss 계산: 뒤의 7개 차원에 대해서만 (dim 5~11)
                diff = original_actions[:, 5:] - decoded_full[:, 5:]  # (16, 7)
                loss_per_timestep_dim = diff ** 2  # (16, 7)
                
                # 전체 평균 loss (기존 방식)
                loss_full = np.mean(loss_per_timestep_dim)
                full_token_losses.append(loss_full)
                
                # 차원별 loss (timestep에 대해 평균, 차원별로 유지)
                loss_per_dim = np.mean(loss_per_timestep_dim, axis=0)  # (7,)
                full_token_losses_per_dim.append(loss_per_dim)
                
                # 원본 action 저장 (복원률 계산용)
                original_actions_list.append(original_actions[:, 5:].flatten())
                
            except Exception as e:
                logging.warning(f"Failed to decode full tokens for batch {batch_idx}, sample {i}: {e}")
                logging.warning(f"Tokens: {fast_action_tokens.tolist()}")
                skipped_sequences += 1
                continue
            
            # 2. 처음 10개 토큰만으로 detokenize (또는 가능한 만큼만)
            if raw_action_tokens.size > 0:  # 토큰이 있으면 처리
                try:
                    # 최대 10개까지만 사용 (토큰이 부족하면 가능한 만큼만)
                    num_tokens_to_use = min(10, raw_action_tokens.size)
                    first_tokens = fast_action_tokens[:num_tokens_to_use]
                    
                    # FAST tokenizer가 자동으로 패딩 처리함
                    decoded_partial = fast_tok.decode(
                        [first_tokens.tolist()], 
                        time_horizon=action_horizon, 
                        action_dim=action_dim
                    )[0]  # (16, 12)
                    
                    # Loss 계산: 뒤의 7개 차원에 대해서만 (dim 5~11)
                    diff_partial = original_actions[:, 5:] - decoded_partial[:, 5:]  # (16, 7)
                    loss_per_timestep_dim_partial = diff_partial ** 2  # (16, 7)
                    
                    # 전체 평균 loss (기존 방식)
                    loss_partial = np.mean(loss_per_timestep_dim_partial)
                    first_10_token_losses.append(loss_partial)
                    
                    # 차원별 loss (timestep에 대해 평균, 차원별로 유지)
                    loss_per_dim_partial = np.mean(loss_per_timestep_dim_partial, axis=0)  # (7,)
                    first_10_token_losses_per_dim.append(loss_per_dim_partial)
                    
                except Exception as e:
                    logging.warning(f"Failed to decode partial tokens for batch {batch_idx}, sample {i}: {e}")
                    logging.warning(f"Tokens: {first_tokens.tolist()}")
                    # 이 경우는 전체 시퀀스 카운트에 영향 주지 않음 (이미 full decode는 성공했으므로)
                    continue
            
            total_sequences += 1

    # 복원률 계산 (원본 데이터의 variance 대비 MSE loss)
    if original_actions_list:
        all_original_actions = np.concatenate(original_actions_list, axis=0)  # (N*16*7,)
        original_variance = np.var(all_original_actions)
        
        # 차원별 원본 variance 계산
        all_original_reshaped = np.array(original_actions_list)  # (N, 16*7)
        all_original_per_dim = all_original_reshaped.reshape(-1, 16, 7)  # (N, 16, 7)
        original_variance_per_dim = np.var(all_original_per_dim, axis=(0, 1))  # (7,) - 각 차원별 variance
        
        # 전체 복원률
        full_recovery_rate = (1 - np.mean(full_token_losses) / original_variance) * 100 if original_variance > 0 else 0.0
        partial_recovery_rate = (1 - np.mean(first_10_token_losses) / original_variance) * 100 if original_variance > 0 and first_10_token_losses else 0.0
        
        # 차원별 복원률 계산
        if full_token_losses_per_dim and len(full_token_losses_per_dim) > 0:
            full_losses_per_dim_array = np.array(full_token_losses_per_dim)  # (N, 7)
            full_mean_loss_per_dim = full_losses_per_dim_array.mean(axis=0)  # (7,)
            full_recovery_rate_per_dim = []
            for dim_idx in range(7):
                if original_variance_per_dim[dim_idx] > 0:
                    recovery_rate = (1 - full_mean_loss_per_dim[dim_idx] / original_variance_per_dim[dim_idx]) * 100
                    full_recovery_rate_per_dim.append(recovery_rate)
                else:
                    full_recovery_rate_per_dim.append(0.0)
        else:
            full_recovery_rate_per_dim = [0.0] * 7
            
        if first_10_token_losses_per_dim and len(first_10_token_losses_per_dim) > 0:
            partial_losses_per_dim_array = np.array(first_10_token_losses_per_dim)  # (N, 7)
            partial_mean_loss_per_dim = partial_losses_per_dim_array.mean(axis=0)  # (7,)
            partial_recovery_rate_per_dim = []
            for dim_idx in range(7):
                if original_variance_per_dim[dim_idx] > 0:
                    recovery_rate = (1 - partial_mean_loss_per_dim[dim_idx] / original_variance_per_dim[dim_idx]) * 100
                    partial_recovery_rate_per_dim.append(recovery_rate)
                else:
                    partial_recovery_rate_per_dim.append(0.0)
        else:
            partial_recovery_rate_per_dim = [0.0] * 7
        
        # 차원별 통계 계산
        if full_token_losses_per_dim:
            full_losses_per_dim_array = np.array(full_token_losses_per_dim)  # (N, 7)
            full_losses_per_dim_stats = {
                "mean": full_losses_per_dim_array.mean(axis=0).tolist(),  # 각 차원별 평균
                "std": full_losses_per_dim_array.std(axis=0).tolist(),    # 각 차원별 표준편차
                "min": full_losses_per_dim_array.min(axis=0).tolist(),    # 각 차원별 최소값
                "max": full_losses_per_dim_array.max(axis=0).tolist(),    # 각 차원별 최대값
            }
        else:
            full_losses_per_dim_stats = {"mean": [0.0]*7, "std": [0.0]*7, "min": [0.0]*7, "max": [0.0]*7}
            
        if first_10_token_losses_per_dim:
            partial_losses_per_dim_array = np.array(first_10_token_losses_per_dim)  # (N, 7)
            partial_losses_per_dim_stats = {
                "mean": partial_losses_per_dim_array.mean(axis=0).tolist(),
                "std": partial_losses_per_dim_array.std(axis=0).tolist(),
                "min": partial_losses_per_dim_array.min(axis=0).tolist(),
                "max": partial_losses_per_dim_array.max(axis=0).tolist(),
            }
        else:
            partial_losses_per_dim_stats = {"mean": [0.0]*7, "std": [0.0]*7, "min": [0.0]*7, "max": [0.0]*7}
    else:
        original_variance = 0.0
        full_recovery_rate = 0.0
        partial_recovery_rate = 0.0
        full_recovery_rate_per_dim = [0.0] * 7
        partial_recovery_rate_per_dim = [0.0] * 7
        original_variance_per_dim = [0.0] * 7
        full_losses_per_dim_stats = {"mean": [0.0]*7, "std": [0.0]*7, "min": [0.0]*7, "max": [0.0]*7}
        partial_losses_per_dim_stats = {"mean": [0.0]*7, "std": [0.0]*7, "min": [0.0]*7, "max": [0.0]*7}

    # 결과 저장 및 출력
    results = {
        "total_sequences": total_sequences,
        "skipped_sequences": skipped_sequences,
        "valid_full_sequences": len(full_token_losses),
        "valid_partial_sequences": len(first_10_token_losses),
        "full_token_loss": {
            "mean": float(np.mean(full_token_losses)) if full_token_losses else 0.0,
            "std": float(np.std(full_token_losses)) if full_token_losses else 0.0,
            "min": float(np.min(full_token_losses)) if full_token_losses else 0.0,
            "max": float(np.max(full_token_losses)) if full_token_losses else 0.0,
        },
        "first_10_token_loss": {
            "mean": float(np.mean(first_10_token_losses)) if first_10_token_losses else 0.0,
            "std": float(np.std(first_10_token_losses)) if first_10_token_losses else 0.0,
            "min": float(np.min(first_10_token_losses)) if first_10_token_losses else 0.0,
            "max": float(np.max(first_10_token_losses)) if first_10_token_losses else 0.0,
        },
        "recovery_rates": {
            "full_token_recovery_rate_percent": float(full_recovery_rate),
            "first_10_token_recovery_rate_percent": float(partial_recovery_rate),
            "original_variance": float(original_variance),
            "full_token_recovery_rate_per_dim_percent": [float(x) for x in full_recovery_rate_per_dim],
            "first_10_token_recovery_rate_per_dim_percent": [float(x) for x in partial_recovery_rate_per_dim],
            "original_variance_per_dim": [float(x) for x in original_variance_per_dim],
        },
        "config": {
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "loss_dimensions": "5:12 (7 dims)",  # 뒤의 7개 차원
        },
        "full_losses_per_dim_stats": full_losses_per_dim_stats,
        "first_10_losses_per_dim_stats": partial_losses_per_dim_stats,
    }

    # 결과 저장
    results_path = os.path.join(args.output_dir, "tokenize_detokenize_loss_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Saved results to {os.path.abspath(results_path)}")
    
    # 결과 출력
    logging.info("=== Tokenize -> Detokenize Loss Results ===")
    logging.info(f"Total sequences: {total_sequences}, Skipped: {skipped_sequences}")
    logging.info(f"Valid full sequences: {len(full_token_losses)}, Valid partial sequences: {len(first_10_token_losses)}")
    logging.info(f"Original data variance: {results['recovery_rates']['original_variance']:.6f}")
    logging.info(f"Full token loss (mean±std): {results['full_token_loss']['mean']:.6f}±{results['full_token_loss']['std']:.6f}")
    logging.info(f"First 10 token loss (mean±std): {results['first_10_token_loss']['mean']:.6f}±{results['first_10_token_loss']['std']:.6f}")
    logging.info(f"🎯 Full token recovery rate: {results['recovery_rates']['full_token_recovery_rate_percent']:.2f}%")
    logging.info(f"🎯 First 10 token recovery rate: {results['recovery_rates']['first_10_token_recovery_rate_percent']:.2f}%")
    
    # 차원별 loss 출력
    logging.info("=== Per-Dimension Loss Analysis ===")
    if full_losses_per_dim_stats["mean"]:
        logging.info("Full token losses per dimension (5~11):")
        for dim_idx, (mean_loss, std_loss) in enumerate(zip(full_losses_per_dim_stats["mean"], full_losses_per_dim_stats["std"])):
            logging.info(f"  Dim {dim_idx+5}: {mean_loss:.6f}±{std_loss:.6f}")
    
    if partial_losses_per_dim_stats["mean"]:
        logging.info("First 10 token losses per dimension (5~11):")
        for dim_idx, (mean_loss, std_loss) in enumerate(zip(partial_losses_per_dim_stats["mean"], partial_losses_per_dim_stats["std"])):
            logging.info(f"  Dim {dim_idx+5}: {mean_loss:.6f}±{std_loss:.6f}")
    
    # 차원별 복원률 출력
    logging.info("=== Per-Dimension Recovery Rates ===")
    if full_recovery_rate_per_dim:
        logging.info("Full token recovery rates per dimension (5~11):")
        for dim_idx, recovery_rate in enumerate(full_recovery_rate_per_dim):
            logging.info(f"  Dim {dim_idx+5}: {recovery_rate:.2f}%")
    
    if partial_recovery_rate_per_dim:
        logging.info("First 10 token recovery rates per dimension (5~11):")
        for dim_idx, recovery_rate in enumerate(partial_recovery_rate_per_dim):
            logging.info(f"  Dim {dim_idx+5}: {recovery_rate:.2f}%")
    
    # 상세한 분포 저장
    detailed_results = {
        "full_token_losses": [float(x) for x in full_token_losses],
        "first_10_token_losses": [float(x) for x in first_10_token_losses],
        "full_token_losses_per_dim": [x.tolist() for x in full_token_losses_per_dim],  # 차원별 loss
        "first_10_token_losses_per_dim": [x.tolist() for x in first_10_token_losses_per_dim],  # 차원별 loss
    }
    
    detailed_path = os.path.join(args.output_dir, "tokenize_detokenize_loss_detailed.json")
    with open(detailed_path, "w") as f:
        json.dump(detailed_results, f)
    
    logging.info(f"Saved detailed results to {os.path.abspath(detailed_path)}")


def main(args: Args):
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Starting tokenize->detokenize loss measurement")
    measure_tokenize_detokenize_loss(args)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
