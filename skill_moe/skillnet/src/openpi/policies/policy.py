from collections.abc import Sequence
import datetime
import json
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from skillnet_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _get_skillnet_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"SKILLNET_{name}", os.environ.get(f"OPENPI_{name}", default))


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._subtask_trace_step = 0
        self._subtask_tokenizer = None
        self._cached_subtask_tokens = None
        self._cached_subtask_prompt = None
        self._cached_subtask_max_tokens = None

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._sample_actions_from_prefix_cache = (
                nnx_utils.module_jit(model.sample_actions_from_prefix_cache)
                if hasattr(model, "sample_actions_from_prefix_cache")
                else None
            )
            self._rng = rng or jax.random.key(0)

    def _decode_subtask_tokens(self, tokens: np.ndarray, max_len: int) -> str:
        if self._subtask_tokenizer is None:
            self._subtask_tokenizer = _tokenizer.PaligemmaTokenizer(max_len=max_len)
        return self._subtask_tokenizer.detokenize(tokens)

    def _write_subtask_trace(
        self,
        *,
        trace_dir: str,
        prompt: Any,
        tokens: np.ndarray,
        trace: dict[str, np.ndarray],
        max_len: int,
        step: int,
    ) -> None:
        trace_path = pathlib.Path(trace_dir)
        trace_path.mkdir(parents=True, exist_ok=True)

        token_ids = [int(x) for x in tokens.tolist()]
        token_texts = [self._decode_subtask_tokens(np.asarray([x], dtype=np.int32), max_len) for x in token_ids]
        generated_text = self._decode_subtask_tokens(tokens, max_len)
        selected_probs = [float(x) for x in trace["selected_probs"].tolist()]
        top_token_ids = trace["top_token_ids"].astype(np.int32)
        top_probs = trace["top_probs"].astype(np.float32)

        steps = []
        for i, token_id in enumerate(token_ids):
            candidates = []
            for candidate_id, candidate_prob in zip(top_token_ids[i], top_probs[i], strict=True):
                candidate_id = int(candidate_id)
                candidates.append(
                    {
                        "token_id": candidate_id,
                        "token_text": self._decode_subtask_tokens(np.asarray([candidate_id], dtype=np.int32), max_len),
                        "prob": float(candidate_prob),
                    }
                )
            steps.append(
                {
                    "step": i,
                    "selected_token_id": token_id,
                    "selected_token_text": token_texts[i],
                    "selected_prob": selected_probs[i],
                    "top_candidates": candidates,
                }
            )

        payload = {
            "trace_index": step,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "prompt": None if prompt is None else str(prompt),
            "generated_text": generated_text,
            "generated_token_ids": token_ids,
            "generated_token_texts": token_texts,
            "steps": steps,
        }

        base = trace_path / f"subtask_{step:06d}"
        with (base.with_suffix(".json")).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logging.info(
            "SUBTASK_TRACE[%06d] prompt=%r generated=%r tokens=%s",
            step,
            payload["prompt"],
            generated_text,
            token_ids,
        )

        if (_get_skillnet_env("SUBTASK_TRACE_PNG", "1") or "1").lower() in {"0", "false", "no"}:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(max(8, len(token_ids) * 0.8), 6), constrained_layout=True)
            x = np.arange(len(token_ids))
            axes[0].plot(x, selected_probs, marker="o")
            axes[0].set_ylim(0.0, 1.05)
            axes[0].set_title(f"Generated subtask: {generated_text}")
            axes[0].set_ylabel("selected prob")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([str(t) for t in token_ids], rotation=45, ha="right")

            image = axes[1].imshow(top_probs.T, aspect="auto", vmin=0.0, vmax=1.0)
            axes[1].set_title("top-k probability per decoding step")
            axes[1].set_xlabel("decode step")
            axes[1].set_ylabel("rank")
            axes[1].set_yticks(np.arange(top_probs.shape[1]))
            axes[1].set_yticklabels([f"top-{i + 1}" for i in range(top_probs.shape[1])])
            axes[1].set_xticks(x)
            for i in range(top_token_ids.shape[0]):
                for j in range(top_token_ids.shape[1]):
                    axes[1].text(i, j, str(int(top_token_ids[i, j])), ha="center", va="center", fontsize=7)
            fig.colorbar(image, ax=axes[1], label="probability")
            fig.savefig(base.with_suffix(".png"), dpi=160)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - visualization is best-effort.
            logging.warning("Failed to write subtask trace PNG: %s", exc)

    def _maybe_generate_subtask(
        self,
        observation: _model.Observation,
        *,
        prompt: Any,
    ) -> _model.Observation:
        if self._is_pytorch_model:
            return observation
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            return observation
        if observation.token_loss_mask is not None:
            return observation
        if not hasattr(self._model, "generate_subtask") or not hasattr(self._model, "build_full_observation"):
            return observation

        trace_dir = _get_skillnet_env("SUBTASK_TRACE_DIR")
        max_tokens = int(_get_skillnet_env("SUBTASK_MAX_TOKENS", "50") or "50")
        top_k = int(_get_skillnet_env("SUBTASK_TRACE_TOP_K", "5") or "5")
        trace_every = int(_get_skillnet_env("SUBTASK_TRACE_EVERY", "1") or "1")
        generate_every = max(1, int(_get_skillnet_env("SUBTASK_GENERATE_EVERY", "1") or "1"))
        trace_step = self._subtask_trace_step
        self._subtask_trace_step += 1

        prompt_key = None if prompt is None else str(prompt)
        cache_valid = (
            self._cached_subtask_tokens is not None
            and self._cached_subtask_prompt == prompt_key
            and self._cached_subtask_max_tokens == max_tokens
        )
        should_generate = not cache_valid or trace_step % generate_every == 0
        should_trace = (
            should_generate
            and trace_dir
            and trace_every > 0
            and trace_step % trace_every == 0
            and hasattr(self._model, "generate_subtask_trace")
        )

        if should_trace:
            subtask_tokens, trace = self._model.generate_subtask_trace(
                observation,
                max_tokens=max_tokens,
                top_k=top_k,
            )
            trace_np = jax.tree.map(lambda x: np.asarray(x[0, ...]), trace)
            tokens_np = np.asarray(subtask_tokens[0, ...])
            self._write_subtask_trace(
                trace_dir=trace_dir,
                prompt=prompt,
                tokens=tokens_np,
                trace=trace_np,
                max_len=int(observation.tokenized_prompt.shape[-1]),
                step=trace_step,
            )
        elif should_generate:
            subtask_tokens = self._model.generate_subtask(observation, max_tokens=max_tokens)
        else:
            subtask_tokens = self._cached_subtask_tokens

        self._cached_subtask_tokens = subtask_tokens
        self._cached_subtask_prompt = prompt_key
        self._cached_subtask_max_tokens = max_tokens

        return self._model.build_full_observation(observation, subtask_tokens)

    def _maybe_generate_subtask_causal_cache(
        self,
        observation: _model.Observation,
        *,
        prompt: Any,
    ):
        if self._is_pytorch_model:
            return None
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            return None
        if observation.token_loss_mask is not None:
            return None
        required = ("generate_subtask_with_cache", "build_subtask_causal_cache")
        if not all(hasattr(self._model, name) for name in required):
            return None
        if self._sample_actions_from_prefix_cache is None:
            return None

        trace_dir = _get_skillnet_env("SUBTASK_TRACE_DIR")
        max_tokens = int(_get_skillnet_env("SUBTASK_MAX_TOKENS", "50") or "50")
        top_k = int(_get_skillnet_env("SUBTASK_TRACE_TOP_K", "5") or "5")
        trace_every = int(_get_skillnet_env("SUBTASK_TRACE_EVERY", "1") or "1")
        generate_every = max(1, int(_get_skillnet_env("SUBTASK_GENERATE_EVERY", "1") or "1"))
        trace_step = self._subtask_trace_step
        self._subtask_trace_step += 1

        prompt_key = None if prompt is None else str(prompt)
        cache_valid = (
            self._cached_subtask_tokens is not None
            and self._cached_subtask_prompt == prompt_key
            and self._cached_subtask_max_tokens == max_tokens
        )
        should_generate = not cache_valid or trace_step % generate_every == 0
        should_trace = (
            should_generate
            and trace_dir
            and trace_every > 0
            and trace_step % trace_every == 0
            and hasattr(self._model, "generate_subtask_trace")
        )

        if should_trace:
            subtask_tokens, trace = self._model.generate_subtask_trace(
                observation,
                max_tokens=max_tokens,
                top_k=top_k,
            )
            trace_np = jax.tree.map(lambda x: np.asarray(x[0, ...]), trace)
            tokens_np = np.asarray(subtask_tokens[0, ...])
            self._write_subtask_trace(
                trace_dir=trace_dir,
                prompt=prompt,
                tokens=tokens_np,
                trace=trace_np,
                max_len=int(observation.tokenized_prompt.shape[-1]),
                step=trace_step,
            )
            action_observation, prefix_kv_cache, action_prefix_mask = self._model.build_subtask_causal_cache(
                observation,
                subtask_tokens,
            )
        elif should_generate:
            (
                subtask_tokens,
                action_observation,
                prefix_kv_cache,
                action_prefix_mask,
            ) = self._model.generate_subtask_with_cache(observation, max_tokens=max_tokens)
        else:
            subtask_tokens = self._cached_subtask_tokens
            action_observation, prefix_kv_cache, action_prefix_mask = self._model.build_subtask_causal_cache(
                observation,
                subtask_tokens,
            )

        self._cached_subtask_tokens = subtask_tokens
        self._cached_subtask_prompt = prompt_key
        self._cached_subtask_max_tokens = max_tokens

        return action_observation, prefix_kv_cache, action_prefix_mask

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        prompt = obs.get("prompt")
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        action_prefix_mode = (_get_skillnet_env("SUBTASK_ACTION_PREFIX_MODE", "") or "").lower()
        use_causal_cache = action_prefix_mode in {"causal_cache", "causal-cache", "causal"}
        causal_cache_payload = None
        if use_causal_cache:
            causal_cache_payload = self._maybe_generate_subtask_causal_cache(observation, prompt=prompt)
        if causal_cache_payload is None:
            observation = self._maybe_generate_subtask(observation, prompt=prompt)
        start_time = time.monotonic()

        if causal_cache_payload is not None:
            action_observation, prefix_kv_cache, action_prefix_mask = causal_cache_payload
            actions_output = self._sample_actions_from_prefix_cache(
                sample_rng_or_pytorch_device,
                action_observation,
                prefix_kv_cache,
                action_prefix_mask,
                **sample_kwargs,
            )
        else:
            actions_output = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        if isinstance(actions_output, tuple):
            # 新接口：actions + stats
            actions, attn_traj, router_traj, skills = actions_output
        else:
            # 旧接口：只返回 actions
            actions = actions_output
            attn_traj = None
            router_traj = None

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if attn_traj != None:
            return outputs, attn_traj, router_traj, skills
        else:
            return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
