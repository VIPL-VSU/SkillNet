import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import flax.traverse_util as traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PartialCheckpointWeightLoader(WeightLoader):
    """
    Loads a possibly partial set of weights and merges missing keys from reference params.
    """

    params_path: str
    # Regex for keys to backfill from reference params when missing from the checkpoint.
    missing_regex: str = ".*"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")

@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader_MoE:
    """Enhanced loader that adapts FFN checkpoints to MoE model structures."""

    params_path: str

    # =====================================================
    # Main entry
    # =====================================================
    def load(self, params: at.Params) -> at.Params:
        """Loads checkpoint weights and adapts FFN → MoE if needed."""
        print(f"[INFO] Loading checkpoint from {self.params_path} ...")

        # Step 1. Load FFN-style checkpoint weights
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path),
            restore_type=np.ndarray,
        )

        # Step 2. Detect whether target model uses MoE
        num_experts = self._detect_num_experts(params)
        if num_experts:
            print(f"[INFO] Detected MoE structure with {num_experts} experts.")
            loaded_params_ = self._convert_ffn_to_moe(loaded_params, num_experts)

        # Step 3. Merge params and fill missing keys (e.g., router/lora)
        merged = self._merge_params(loaded_params_, params, missing_regex=".*lora.*")

        # Step 4. Debug output for verification
        self._debug_missing_keys(merged, params)

        self._verify_copy(loaded_params, merged)

        print("[INFO] Checkpoint loading complete.\n")
        return merged

    # =====================================================
    # Internal helpers
    # =====================================================
    def _detect_num_experts(self, params: dict) -> int | None:
        """Infer number of experts from param tree."""
        flat = traverse_util.flatten_dict(params, sep="/")
        expert_keys = [k for k in flat if "moe_ffn_1" in k and "expert_" in k]
        if not expert_keys:
            return None
        print(expert_keys)
        expert_ids = set(int(k.split("expert_")[-1].split("/")[0]) for k in expert_keys)
        num_experts = max(expert_ids) + 1
        return num_experts

    def _convert_ffn_to_moe(self, params: dict, num_experts: int) -> dict:
        """Duplicate FFN weights for each MoE expert."""
        flat = traverse_util.flatten_dict(params, sep="/")
        new_flat = dict(flat)
        copied_keys = []

        for path, value in flat.items():
            if "mlp_1" in path and "moe_ffn_1" not in path:
                for i in range(num_experts):
                    new_path = path.replace("mlp_1", f"moe_ffn_1/expert_{i}")
                    new_flat[new_path] = value.copy()
                    copied_keys.append(new_path)
                new_path = path.replace("mlp_1", f"moe_ffn_1/shared_expert")
                new_flat[new_path] = value.copy()
                copied_keys.append(new_path)
                # print(f"{path}: shape={value.shape}, mean={value.mean():.4f}, std={value.std():.4f}, max={value.max():.4f}, max={value.min():.4f}")    

        print(f"[INFO] FFN → MoE duplication done ({num_experts} experts).")
        print(f"[DEBUG] Example copied keys (first 5): {copied_keys[:5]}")
        return traverse_util.unflatten_dict(new_flat, sep="/")

    def _merge_params(self, loaded: dict, target: dict, missing_regex: str):
        """Merges loaded and target params, keeping random init for missing ones."""
        # mimic _merge_params from upstream if needed
        flat_loaded = traverse_util.flatten_dict(loaded, sep="/")
        flat_target = traverse_util.flatten_dict(target, sep="/")

        merged = {}
        for k, v in flat_target.items():
            merged[k] = flat_loaded.get(k, v)  # use checkpoint if exists, else keep init
        return traverse_util.unflatten_dict(merged, sep="/")

    def _debug_missing_keys(self, loaded_params: dict, target_params: dict):
        """Print parameters that remain random-initialized (not found in checkpoint)."""
        flat_loaded = set(traverse_util.flatten_dict(loaded_params, sep="/").keys())
        flat_target = set(traverse_util.flatten_dict(target_params, sep="/").keys())
        missing = sorted(flat_target - flat_loaded)
        if missing:
            print("[DEBUG] Random-initialized parameters (not found in checkpoint):")
            for k in missing:
                if "moe" in k:
                    print("   ", k)
        else:
            print("[DEBUG] All parameters matched checkpoint.")

    def _verify_copy(self, loaded_params, new_params):
        flat_loaded = traverse_util.flatten_dict(loaded_params, sep="/")
        flat_new = traverse_util.flatten_dict(new_params, sep="/")
        for path, value in flat_loaded.items():
            if "mlp_1" in path:
                moe_path = path.replace("mlp_1", "moe_ffn_1/expert_0")
                if moe_path in flat_new:
                    diff = np.abs(flat_new[moe_path] - value).max()
                    print(diff)
                    if diff > 1e-7:
                        print(f"[WARN] FFN→MoE param mismatch: {path} (max diff {diff})")
