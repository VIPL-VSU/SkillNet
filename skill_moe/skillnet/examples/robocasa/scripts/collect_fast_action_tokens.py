import os
import json
import time
import signal
import logging
import dataclasses
import multiprocessing
from collections import Counter
import random

import numpy as np
import tyro
import tqdm_loggable.auto as tqdm

# JAX/OpenPI 관련
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

from openpi.models.tokenizer import FASTTokenizer
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
    stream_flush_every: int = 2000     # 버퍼 크기 임계치(개수)
    stream_sample_every: int = 1       # 1이면 전부, 10이면 10개마다 1개
    time_flush_sec: float = 5.0        # 시간 기반 flush 주기(초). 0이면 비활성화.


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


def collect_fast_action_tokens(args: Args):
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

    # Tokenizer 초기화
    fast_tok = FASTTokenizer()
    pg = fast_tok._paligemma_tokenizer
    action_prefix_tokens = pg.encode("Action: ")
    eos_suffix_tokens = pg.encode("|", add_eos=True)
    pipe_suffix_tokens = pg.encode("|")

    prefix_len = len(action_prefix_tokens)
    eos_suffix_len = len(eos_suffix_tokens)
    pipe_suffix_len = len(pipe_suffix_tokens)

    eos_suffix_arr = np.asarray(eos_suffix_tokens, dtype=np.int32)
    pipe_suffix_arr = np.asarray(pipe_suffix_tokens, dtype=np.int32)

    # 통계 변수들
    counts = Counter()
    total_sequences = 0
    skipped_short = 0
    unique_tokens = set()
    
    # 프리픽스 검증 변수들
    PREFIX_TOKEN = 256554
    REQUIRED_PREFIX_REPEAT = 5
    prefix_checked = 0
    prefix_true = 0
    prefix_short = 0

    # ----- 스트리밍 파일 -----
    stream_path = os.path.abspath(os.path.join(args.output_dir, "real_action_token_stream.ndjson"))
    logging.info(f"[stream] writing to: {stream_path}")

    stream_f = None
    try:
        stream_f = open(stream_path, "w", buffering=1)  # 'w' 모드로 변경 (덮어쓰기)
        stream_buf = []
        seq_counter = 0

        # 종료 신호 처리: 마지막 flush 보장
        should_exit = {"flag": False}

        def _force_flush(reason: str):
            """버퍼와 파일을 강제 플러시하고 fsync로 가시성 보장."""
            try:
                if stream_buf and stream_f:
                    stream_f.write("\n".join(stream_buf) + "\n")
                    stream_buf.clear()
                    stream_f.flush()
                    os.fsync(stream_f.fileno())
                    logging.info(f"[stream] {reason} flush done.")
            except Exception as e:
                logging.exception(f"[stream] flush failed during {reason}: {e}")

        def _handle_term(sig, frm):
            logging.warning(f"[stream] caught signal {sig}, forcing final flush...")
            should_exit["flag"] = True
            try:
                _force_flush("signal")
            finally:
                try:
                    if stream_f:
                        stream_f.close()
                except Exception as e:
                    logging.exception(f"[stream] close failed after signal: {e}")
            # 즉시 종료
            raise SystemExit(0)

        # 메인 스레드에서만 등록 가능
        for _sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(_sig, _handle_term)
            except Exception:
                # 워커 프로세스/스레드 환경에서는 무시
                pass

        last_time_flush = time.time()
        FLUSH_SEC = float(args.time_flush_sec) if args.time_flush_sec and args.time_flush_sec > 0 else None

        # 메인 처리 루프
        for batch_idx, sft_batch in enumerate(
            tqdm.tqdm(wrapped_data_loader, desc="Counting real action tokens (SFT)")
        ):
            obs = sft_batch.observation
            tokens = np.asarray(obs.tokenized_prompt)              # (B, L)
            token_mask = np.asarray(obs.tokenized_prompt_mask)     # (B, L)
            loss_mask = np.asarray(obs.token_loss_mask)            # (B, L); postfix(True)

            B = tokens.shape[0]
            batch_token_arrays = []  # 배치 내 시퀀스 토큰들 모아 일괄 집계

            for i in range(B):
                valid = token_mask[i].astype(bool)
                post = loss_mask[i].astype(bool)

                seq = tokens[i][valid]
                post_seq_mask = post[valid]

                post_idxs = np.where(post_seq_mask)[0]
                if post_idxs.size == 0:
                    continue

                postfix_segment = seq[post_idxs]
                if postfix_segment.size <= prefix_len:
                    skipped_short += 1
                    continue

                # 'Action: ' prefix 제거
                content = postfix_segment[prefix_len:]
                if content.size == 0:
                    continue

                # suffix 트림: 우선 '|'+EOS, 없으면 bare '|'
                suffix_trim = 0
                if eos_suffix_len > 0 and content.size >= eos_suffix_len and np.array_equal(
                    content[-eos_suffix_len:], eos_suffix_arr
                ):
                    suffix_trim = eos_suffix_len
                elif pipe_suffix_len > 0 and content.size >= pipe_suffix_len and np.array_equal(
                    content[-pipe_suffix_len:], pipe_suffix_arr
                ):
                    suffix_trim = pipe_suffix_len

                real_action_tokens = content[: content.size - suffix_trim] if suffix_trim > 0 else content
                if real_action_tokens.size == 0:
                    continue

                arr = real_action_tokens.astype(np.int32)
                batch_token_arrays.append(arr)

                has_prefix_256554_x5 = (
                    arr.size >= REQUIRED_PREFIX_REPEAT and np.all(arr[:REQUIRED_PREFIX_REPEAT] == PREFIX_TOKEN)
                )
                if arr.size >= REQUIRED_PREFIX_REPEAT:
                    prefix_checked += 1
                    if has_prefix_256554_x5:
                        prefix_true += 1
                else:
                    prefix_short += 1

                # 스트림 샘플링/버퍼링
                if args.stream_sample_every == 1 or (seq_counter % args.stream_sample_every == 0):
                    stream_buf.append(
                        json.dumps({
                            "batch_idx": int(batch_idx),
                            "i_in_batch": int(i),
                            "token_ids": arr.astype(int).tolist(),
                            "prefix_256554_x5": bool(has_prefix_256554_x5),
                        })
                    )

                # 크기 기반 flush
                if len(stream_buf) >= args.stream_flush_every:
                    try:
                        stream_f.write("\n".join(stream_buf) + "\n")
                        stream_buf.clear()
                        stream_f.flush()
                        os.fsync(stream_f.fileno())
                        logging.info(f"[stream] size-flush at batch={batch_idx}, seq_counter={seq_counter}")
                    except Exception as e:
                        logging.exception(f"[stream] size-flush failed: {e}")

                # 시간 기반 flush (옵션)
                if FLUSH_SEC is not None:
                    now = time.time()
                    if (now - last_time_flush) >= FLUSH_SEC and stream_buf:
                        try:
                            stream_f.write("\n".join(stream_buf) + "\n")
                            stream_buf.clear()
                            stream_f.flush()
                            os.fsync(stream_f.fileno())
                            last_time_flush = now
                            logging.info(f"[stream] time-flush at batch={batch_idx}, seq_counter={seq_counter}")
                        except Exception as e:
                            logging.exception(f"[stream] time-flush failed: {e}")

                seq_counter += 1
                total_sequences += 1

                if should_exit["flag"]:
                    break  # 안전하게 탈출

            # 배치 단위 집계 (벡터화)
            if len(batch_token_arrays) > 0:
                batch_all_tokens = np.concatenate(batch_token_arrays)
                uniq_tokens, uniq_counts = np.unique(batch_all_tokens, return_counts=True)
                # counts 누적
                for t, c in zip(uniq_tokens.tolist(), uniq_counts.tolist()):
                    counts[int(t)] += int(c)
                # unique 토큰 누적
                unique_tokens.update(int(t) for t in uniq_tokens.tolist())

            if should_exit["flag"]:
                break

        # ----- 에필로그: 남은 버퍼 무조건 flush -----
        _force_flush("epilogue")
        
    finally:
        try:
            if stream_f:
                stream_f.close()
        except Exception as e:
            logging.exception(f"[stream] close failed: {e}")

    # ----- 분포 저장 -----
    dist_path = os.path.join(args.output_dir, "real_action_token_distribution.json")
    with open(dist_path, "w") as f:
        json.dump({str(k): int(v) for k, v in counts.items()}, f)
    logging.info(f"Saved distribution to {os.path.abspath(dist_path)}")

    # unique 저장 및 검증
    unique_path = os.path.join(args.output_dir, "real_action_unique_tokens.json")
    with open(unique_path, "w") as f:
        json.dump(sorted(int(x) for x in unique_tokens), f)
    logging.info(f"Saved {len(unique_tokens)} unique tokens to {os.path.abspath(unique_path)}")

    counts_keys_set = set(int(k) for k in counts.keys())
    if counts_keys_set == unique_tokens:
        logging.info("unique_tokens set matches counts keys.")
    else:
        only_in_unique = sorted(unique_tokens - counts_keys_set)[:10]
        only_in_counts = sorted(counts_keys_set - unique_tokens)[:10]
        logging.warning(
            f"Mismatch between unique tokens and counts keys. "
            f"examples only_in_unique={only_in_unique}, only_in_counts={only_in_counts}"
        )

    # Summary 로그
    distinct = len(counts)
    total_count = sum(counts.values())
    logging.info(f"Distinct real action tokens: {distinct}")
    logging.info(f"Total real action tokens counted: {total_count}")
    logging.info(f"Sequences processed: {total_sequences}, skipped (too short): {skipped_short}")
    logging.info(
        f"Prefix {PREFIX_TOKEN} x{REQUIRED_PREFIX_REPEAT}: true={prefix_true}, "
        f"checked={prefix_checked}, too_short(<{REQUIRED_PREFIX_REPEAT})={prefix_short}"
    )

    # Top-N 미리보기
    top_preview = counts.most_common(20)
    preview_path = os.path.join(args.output_dir, "real_action_token_top20.json")
    with open(preview_path, "w") as f:
        json.dump([{"token": k, "count": v} for k, v in top_preview], f)
    logging.info(f"Saved top-20 preview to {os.path.abspath(preview_path)}")


def main(args: Args):
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Starting dataset iteration with SFT batches (no model sampling)")
    collect_fast_action_tokens(args)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)