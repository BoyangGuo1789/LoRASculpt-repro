# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from deepspeed.utils import \
    safe_get_full_fp32_param, safe_get_local_fp32_param, \
    safe_get_full_grad, safe_get_local_grad,\
    safe_set_full_fp32_param,safe_set_local_fp32_param

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from llava.train.llava_trainer import LLaVATrainer

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_dataloader_sampler,
    get_model_param_count,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.utils.quantization_config import QuantizationMethod

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"



STEP_THRESHOLD=int(os.environ.get('STEP_THRESHOLD', 100))
AB_PRESERVE_RATIO=float(os.environ.get('AB_PRESERVE_RATIO', 0.1))
CMR_LAMBDA=float(os.environ.get('CMR_LAMBDA', 1e-3))
OMEGA=float(os.environ.get('OMEGA', 1.0))





class LoRASculptMIGDIS(LLaVATrainer):

    def _migdis_bool(self, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).lower() in ("1", "true", "yes", "y", "on")

    def _migdis_enabled(self):
        return self._migdis_bool(getattr(self.args, "migdis_enable", os.environ.get("MIGDIS_ENABLE", "true")))

    def _migdis_arg(self, name, env_name, default, cast):
        value = getattr(self.args, name, None)
        if value is None:
            value = os.environ.get(env_name, default)
        return cast(value)

    def _migdis_grad_ema_beta(self):
        return self._migdis_arg("migdis_grad_ema_beta", "MIGDIS_GRAD_EMA_BETA", 0.95, float)

    def _migdis_grad_mix(self):
        return self._migdis_arg("migdis_grad_mix", "MIGDIS_GRAD_MIX", 0.50, float)

    def _migdis_source_margin(self):
        return self._migdis_arg("migdis_source_margin", "MIGDIS_SOURCE_MARGIN", 0.35, float)

    def _migdis_source_scope(self):
        return str(getattr(self.args, "migdis_source_scope", os.environ.get("MIGDIS_SOURCE_SCOPE", "qkv"))).lower()

    def _migdis_norm_mode(self):
        return str(getattr(self.args, "migdis_norm", os.environ.get("MIGDIS_NORM", "robust"))).lower()

    def _migdis_norm_q_low(self):
        return self._migdis_arg("migdis_norm_q_low", "MIGDIS_NORM_Q_LOW", 0.05, float)

    def _migdis_norm_q_high(self):
        return self._migdis_arg("migdis_norm_q_high", "MIGDIS_NORM_Q_HIGH", 0.95, float)

    def _migdis_eps(self):
        return self._migdis_arg("migdis_eps", "MIGDIS_EPS", 1e-6, float)

    def _migdis_source_chunk_rows(self):
        return self._migdis_arg("migdis_source_chunk_rows", "MIGDIS_SOURCE_CHUNK_ROWS", 2048, int)

    def _migdis_final_gamma(self):
        return self._migdis_arg("migdis_final_gamma", "MIGDIS_FINAL_GAMMA", 1.0, float)

    def _migdis_debug_dump(self):
        return self._migdis_bool(getattr(self.args, "migdis_debug_dump", os.environ.get("MIGDIS_DEBUG_DUMP", "true")))

    def _migdis_selection_mode(self):
        mode = str(getattr(self.args, "migdis_selection_mode", os.environ.get("MIGDIS_SELECTION_MODE", "global"))).lower()
        if mode not in ("global", "tgsr", "dqss"):
            raise ValueError(f"Unsupported MIG-DIS selection mode: {mode}")
        return mode

    def _migdis_tgsr_candidate_ratio(self):
        return self._migdis_arg("migdis_tgsr_candidate_ratio", "MIGDIS_TGSR_CANDIDATE_RATIO", 0.50, float)

    def _migdis_tgsr_core_source_margin(self):
        return self._migdis_arg("migdis_tgsr_core_source_margin", "MIGDIS_TGSR_CORE_SOURCE_MARGIN", 0.35, float)

    def _migdis_tgsr_debug_overlap(self):
        return self._migdis_bool(
            getattr(self.args, "migdis_tgsr_debug_overlap", os.environ.get("MIGDIS_TGSR_DEBUG_OVERLAP", "true"))
        )

    def _migdis_dqss_aux_grad_mix(self):
        return self._migdis_arg("migdis_dqss_aux_grad_mix", "MIGDIS_DQSS_AUX_GRAD_MIX", 0.25, float)

    def _migdis_dqss_aux_source_margin(self):
        return self._migdis_arg("migdis_dqss_aux_source_margin", "MIGDIS_DQSS_AUX_SOURCE_MARGIN", 0.70, float)

    def _migdis_dqss_rho(self):
        return self._migdis_arg("migdis_dqss_rho", "MIGDIS_DQSS_RHO", 0.25, float)

    def _migdis_dqss_debug_overlap(self):
        return self._migdis_bool(
            getattr(self.args, "migdis_dqss_debug_overlap", os.environ.get("MIGDIS_DQSS_DEBUG_OVERLAP", "true"))
        )

    def _migdis_dqss_module_scope(self):
        return str(
            getattr(self.args, "migdis_dqss_module_scope", os.environ.get("MIGDIS_DQSS_MODULE_SCOPE", "qkv"))
        ).lower()

    def _migdis_dqss_anti_collapse(self):
        return self._migdis_bool(
            getattr(self.args, "migdis_dqss_anti_collapse", os.environ.get("MIGDIS_DQSS_ANTI_COLLAPSE", "true"))
        )

    def _migdis_dqss_max_aux_overlap(self):
        return self._migdis_arg("migdis_dqss_max_aux_overlap", "MIGDIS_DQSS_MAX_AUX_OVERLAP", 0.98, float)

    def _migdis_dqss_min_core_overlap(self):
        return self._migdis_arg("migdis_dqss_min_core_overlap", "MIGDIS_DQSS_MIN_CORE_OVERLAP", 0.70, float)

    def _migdis_log(self, message):
        logger.info(message)
        try:
            is_zero = self.is_world_process_zero()
        except Exception:
            is_zero = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        if is_zero:
            print(message, flush=True)

    def _migdis_should_track_param(self, name):
        if not ("lora_A" in name or "lora_B" in name):
            return False
        blocked = ("mm_projector", "vision_tower", "vision_resampler")
        return not any(key in name for key in blocked)

    def _migdis_full_param(self, param):
        full_param = safe_get_full_fp32_param(param)
        if full_param is None:
            full_param = param.detach()
        return full_param.float().clone()

    def _migdis_full_grad(self, param):
        grad = param.grad
        if grad is None or grad.shape != param.shape:
            return None
        return grad.detach().float().clone()

    def _migdis_make_grad_hook(self, name):
        def hook(grad):
            if grad is None or not self._migdis_enabled() or getattr(self, "migdis_masks_built", False):
                return grad
            with torch.no_grad():
                abs_grad = grad.detach().float().abs()
                if name in self.migdis_grad_ema and self.migdis_grad_ema[name].shape != abs_grad.shape:
                    return grad
                beta = self._migdis_grad_ema_beta()
                if name not in self.migdis_grad_ema:
                    self.migdis_grad_ema[name] = torch.zeros_like(abs_grad, dtype=torch.float32, device=abs_grad.device)
                    self.migdis_grad_steps[name] = 0
                    self.migdis_grad_hook_inits = getattr(self, "migdis_grad_hook_inits", 0) + 1
                self.migdis_grad_ema[name].mul_(beta).add_(abs_grad, alpha=(1.0 - beta))
                self.migdis_grad_steps[name] += 1
                self.migdis_grad_hook_updates = getattr(self, "migdis_grad_hook_updates", 0) + 1
            return grad
        return hook

    def _migdis_register_grad_hooks_once(self, model):
        if not self._migdis_enabled() or getattr(self, "migdis_masks_built", False):
            return
        if getattr(self, "migdis_grad_hooks_registered", False):
            return

        registered = 0
        self.migdis_grad_hooks = []
        for name, param in model.named_parameters():
            if not param.requires_grad or not self._migdis_should_track_param(name):
                continue
            self.migdis_grad_hooks.append(param.register_hook(self._migdis_make_grad_hook(name)))
            registered += 1
        self.migdis_grad_hooks_registered = True
        self._migdis_log(f"MIG-DIS grad hooks registered for {registered} tensors")

    def _migdis_update_grad_ema(self, model):
        if not self._migdis_enabled() or getattr(self, "migdis_masks_built", False):
            return

        if getattr(self, "migdis_grad_hooks_registered", False):
            if self.migdis_grad_ema and not getattr(self, "migdis_logged_init", False):
                self._migdis_log(
                    f"MIG-DIS grad EMA initialized for {len(self.migdis_grad_ema)} tensors via backward hooks"
                )
                self.migdis_logged_init = True
            return

        beta = self._migdis_grad_ema_beta()
        initialized = 0
        updated = 0
        for name, param in model.named_parameters():
            if not param.requires_grad or not self._migdis_should_track_param(name):
                continue
            grad = self._migdis_full_grad(param)
            if grad is None:
                continue
            abs_grad = grad.abs()
            if name not in self.migdis_grad_ema:
                self.migdis_grad_ema[name] = torch.zeros_like(abs_grad, dtype=torch.float32, device=abs_grad.device)
                self.migdis_grad_steps[name] = 0
                initialized += 1
            self.migdis_grad_ema[name].mul_(beta).add_(abs_grad, alpha=(1.0 - beta))
            self.migdis_grad_steps[name] += 1
            updated += 1

        if initialized and not getattr(self, "migdis_logged_init", False):
            self._migdis_log(f"MIG-DIS grad EMA initialized for {initialized} tensors")
            self.migdis_logged_init = True
        if updated:
            self.migdis_grad_ema_updates = getattr(self, "migdis_grad_ema_updates", 0) + 1

    def _migdis_sync_grad_ema(self):
        if not dist.is_available() or not dist.is_initialized():
            return
        for grad_ema in self.migdis_grad_ema.values():
            dist.all_reduce(grad_ema, op=dist.ReduceOp.SUM)
            grad_ema.div_(dist.get_world_size())

    def _migdis_norm(self, tensor):
        x = tensor.detach().float().abs()
        if x.numel() == 0:
            return x
        if self._migdis_norm_mode() == "mean":
            return x / (x.mean() + self._migdis_eps())

        flat = x.flatten()
        lo = torch.quantile(flat, self._migdis_norm_q_low())
        hi = torch.quantile(flat, self._migdis_norm_q_high())
        return ((x - lo) / (hi - lo + self._migdis_eps())).clamp_(0.0, 1.0)

    def _migdis_normalize_lora_key(self, name):
        if name.startswith("module."):
            name = name[len("module."):]
        return (name
                .replace(".lora_A.default.weight", ".weight")
                .replace(".lora_B.default.weight", ".weight")
                .replace(".base_layer.weight", ".weight"))

    def _migdis_should_apply_source(self, name):
        mode = self._migdis_source_scope()
        if mode == "none":
            return False
        if mode == "all":
            return True
        return any(proj in name for proj in ("q_proj", "k_proj", "v_proj"))

    def _migdis_is_qkv_module(self, name):
        return any(proj in name for proj in ("q_proj", "k_proj", "v_proj"))

    def _migdis_dqss_should_apply(self, name):
        scope = self._migdis_dqss_module_scope()
        if scope == "all":
            return True
        if scope == "none":
            return False
        tokens = [token.strip() for token in scope.replace("+", ",").split(",") if token.strip()]
        aliases = {
            "qkv": ("q_proj", "k_proj", "v_proj"),
            "qk": ("q_proj", "k_proj"),
            "qv": ("q_proj", "v_proj"),
            "kv": ("k_proj", "v_proj"),
            "q": ("q_proj",),
            "k": ("k_proj",),
            "v": ("v_proj",),
            "o": ("o_proj",),
            "mlp": ("gate_proj", "up_proj", "down_proj"),
        }
        if not tokens:
            tokens = ["qkv"]
        projections = []
        for token in tokens:
            projections.extend(aliases.get(token, (token,)))
        return any(proj in name for proj in projections)

    def _migdis_source_row_col_stats(self, weight_param):
        weight = self._migdis_full_param(weight_param)
        if weight.dim() != 2:
            return None, None

        with torch.no_grad():
            norm = torch.norm(weight, p=2) + self._migdis_eps()
            out_dim, in_dim = weight.shape
            row_mean = torch.empty(out_dim, device=weight.device, dtype=torch.float32)
            col_sum = torch.zeros(in_dim, device=weight.device, dtype=torch.float32)
            chunk_rows = max(1, self._migdis_source_chunk_rows())
            for start in range(0, out_dim, chunk_rows):
                end = min(start + chunk_rows, out_dim)
                x = weight[start:end].abs() / norm + self._migdis_eps()
                s = torch.abs(1.0 / torch.log(x))
                m = torch.tanh(OMEGA * s)
                row_mean[start:end] = m.mean(dim=1)
                col_sum += m.sum(dim=0)
            return row_mean, col_sum / out_dim

    def _migdis_source_penalty(self, name, lora_param, base_param):
        if base_param is None or not self._migdis_should_apply_source(name):
            return None, None

        row_mean, col_mean = self._migdis_source_row_col_stats(base_param)
        if row_mean is None or col_mean is None:
            return None, None

        try:
            if "lora_B" in name:
                penalty = row_mean[:, None].expand_as(lora_param).to(lora_param.device)
            elif "lora_A" in name:
                penalty = col_mean[None, :].expand_as(lora_param).to(lora_param.device)
            else:
                return None, None
            return penalty, {
                "source_row_mean": float(row_mean.mean().item()),
                "source_col_mean": float(col_mean.mean().item()),
            }
        except RuntimeError:
            logger.warning(
                f"MIG-DIS source projection shape mismatch for {name}: "
                f"row={tuple(row_mean.shape)} col={tuple(col_mean.shape)} lora={tuple(lora_param.shape)}"
            )
            return None, None

    def _migdis_grad_hat(self, name, lora_param):
        grad_ema = self.migdis_grad_ema.get(name)
        steps = self.migdis_grad_steps.get(name, 0)
        if grad_ema is None or grad_ema.shape != lora_param.shape or steps <= 0:
            return torch.zeros_like(lora_param, dtype=torch.float32, device=lora_param.device)
        beta = self._migdis_grad_ema_beta()
        return grad_ema.to(lora_param.device) / (1.0 - beta ** steps + self._migdis_eps())

    def _migdis_build_score_bundle(
        self,
        name,
        lora_param,
        base_param,
        grad_mix=None,
        source_margin=None,
        core_source_margin=None,
    ):
        grad_mix = self._migdis_grad_mix() if grad_mix is None else float(grad_mix)
        source_margin = self._migdis_source_margin() if source_margin is None else float(source_margin)
        core_source_margin = (
            self._migdis_tgsr_core_source_margin() if core_source_margin is None else float(core_source_margin)
        )
        task_score = (1.0 - grad_mix) * self._migdis_norm(lora_param.abs())
        grad_hat = self._migdis_grad_hat(name, lora_param)
        task_score = task_score + grad_mix * self._migdis_norm(grad_hat)

        penalty, source_stats = self._migdis_source_penalty(name, lora_param, base_param)
        source_applied = penalty is not None
        if source_applied:
            penalty_score = self._migdis_norm(penalty)
            score = task_score - source_margin * penalty_score
            core_score = task_score - core_source_margin * penalty_score
        else:
            score = task_score
            core_score = task_score

        stats = {
            "grad_mix": float(grad_mix),
            "source_margin": float(source_margin),
            "core_source_margin": float(core_source_margin),
            "grad_ema_mean": float(grad_hat.float().mean().item()),
            "score_mean": float(score.float().mean().item()),
            "task_score_mean": float(task_score.float().mean().item()),
            "Q_core_mean": float(core_score.float().mean().item()),
            "Q_src_mean": float(score.float().mean().item()),
            "source_scope_applied": bool(source_applied),
        }
        if source_stats:
            stats.update(source_stats)
        return {
            "score": score.detach(),
            "core_score": core_score.detach(),
            "stats": stats,
        }

    def _migdis_make_topk_mask(self, score, keep_ratio):
        flat_score = score.float().flatten()
        k = max(1, int(math.ceil(float(keep_ratio) * flat_score.numel())))
        top_indices = torch.topk(flat_score, k=k, largest=True, sorted=False).indices
        mask = torch.zeros_like(flat_score, dtype=score.dtype, device=score.device)
        mask[top_indices] = 1
        return top_indices, mask.reshape(score.shape)

    def _migdis_make_tgsr_mask(self, core_score, source_score, keep_ratio, candidate_ratio):
        flat_core = core_score.detach().float().flatten()
        flat_src = source_score.detach().float().flatten()
        n = flat_core.numel()
        k = max(1, int(math.ceil(float(keep_ratio) * n)))
        kc = max(k, int(math.ceil(float(candidate_ratio) * n)))
        kc = min(kc, n)

        candidate_idx = torch.topk(flat_core, k=kc, largest=True, sorted=False).indices
        candidate_scores = flat_src[candidate_idx]
        local_idx = torch.topk(candidate_scores, k=k, largest=True, sorted=False).indices
        final_idx = candidate_idx[local_idx]

        flat_mask = torch.zeros_like(flat_core, dtype=source_score.dtype, device=source_score.device)
        flat_mask[final_idx] = 1

        stats = {
            "candidate_ratio_effective": float(kc / max(n, 1)),
            "retain_ratio": float(k / max(n, 1)),
        }
        if self._migdis_tgsr_debug_overlap():
            core_topk = torch.topk(flat_core, k=k, largest=True, sorted=False).indices
            src_topk = torch.topk(flat_src, k=k, largest=True, sorted=False).indices
            final_bool = torch.zeros(n, dtype=torch.bool, device=flat_core.device)
            core_bool = torch.zeros(n, dtype=torch.bool, device=flat_core.device)
            src_bool = torch.zeros(n, dtype=torch.bool, device=flat_core.device)
            final_bool[final_idx] = True
            core_bool[core_topk] = True
            src_bool[src_topk] = True
            stats["overlap_with_core_topk"] = float((final_bool & core_bool).sum().item() / k)
            stats["overlap_with_src_global_topk"] = float((final_bool & src_bool).sum().item() / k)

        return final_idx, flat_mask.reshape(source_score.shape), stats

    def _migdis_make_dqss_mask(self, core_score, aux_score, keep_ratio, rho):
        flat_core = core_score.detach().float().flatten()
        flat_aux = aux_score.detach().float().flatten()
        n = flat_core.numel()
        k = max(1, int(math.ceil(float(keep_ratio) * n)))
        if k <= 1:
            k_aux = 0
        else:
            k_aux = int(math.ceil(float(rho) * k))
            k_aux = min(max(1, k_aux), k - 1)
        k_core = k - k_aux

        core_idx = torch.topk(flat_core, k=k_core, largest=True, sorted=False).indices
        if k_aux > 0:
            aux_score_for_pick = flat_aux.clone()
            aux_score_for_pick[core_idx] = -torch.inf
            aux_idx = torch.topk(aux_score_for_pick, k=k_aux, largest=True, sorted=False).indices
            final_idx = torch.cat([core_idx, aux_idx], dim=0)
        else:
            aux_idx = torch.empty(0, dtype=core_idx.dtype, device=core_idx.device)
            final_idx = core_idx

        guard_replacements = 0
        core_topk = None
        aux_topk = None
        if self._migdis_dqss_debug_overlap() or self._migdis_dqss_anti_collapse():
            core_topk = torch.topk(flat_core, k=k, largest=True, sorted=False).indices
            aux_topk = torch.topk(flat_aux, k=k, largest=True, sorted=False).indices

        if self._migdis_dqss_anti_collapse() and k_aux > 0 and aux_topk is not None and core_topk is not None:
            max_aux_overlap = min(max(self._migdis_dqss_max_aux_overlap(), 0.0), 1.0)
            min_core_overlap = min(max(self._migdis_dqss_min_core_overlap(), 0.0), 1.0)
            aux_bool = torch.zeros(n, dtype=torch.bool, device=flat_core.device)
            core_bool = torch.zeros(n, dtype=torch.bool, device=flat_core.device)
            final_bool = torch.zeros(n, dtype=torch.bool, device=flat_core.device)
            aux_bool[aux_topk] = True
            core_bool[core_topk] = True
            final_bool[final_idx] = True

            aux_overlap_count = int((final_bool & aux_bool).sum().item())
            core_overlap_count = int((final_bool & core_bool).sum().item())
            max_aux_count = int(math.floor((max_aux_overlap - 1e-6) * k))
            min_core_count = int(math.ceil(min_core_overlap * k))
            required_replace = max(0, aux_overlap_count - max_aux_count)
            core_budget = max(0, core_overlap_count - min_core_count)
            if required_replace > 0 and core_budget > 0:
                core_overlap_positions = torch.nonzero(aux_bool[core_idx], as_tuple=False).flatten()
                candidate_mask = torch.ones(n, dtype=torch.bool, device=flat_core.device)
                candidate_mask[final_idx] = False
                candidate_mask[aux_topk] = False
                candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                num_replace = min(required_replace, core_budget, int(core_overlap_positions.numel()), int(candidate_idx.numel()))
                if num_replace > 0:
                    replacement_local = torch.topk(
                        flat_core[candidate_idx], k=num_replace, largest=True, sorted=False
                    ).indices
                    replacement_idx = candidate_idx[replacement_local]
                    replace_scores = flat_core[core_idx[core_overlap_positions]]
                    replace_local = torch.topk(-replace_scores, k=num_replace, largest=True, sorted=False).indices
                    replace_positions = core_overlap_positions[replace_local]
                    core_idx = core_idx.clone()
                    core_idx[replace_positions] = replacement_idx
                    final_idx = torch.cat([core_idx, aux_idx], dim=0)
                    guard_replacements = int(num_replace)

        flat_mask = torch.zeros_like(flat_core, dtype=core_score.dtype, device=core_score.device)
        flat_mask[final_idx] = 1

        stats = {
            "retain_ratio": float(k / max(n, 1)),
            "dqss_core_count": int(k_core),
            "dqss_aux_count": int(k_aux),
            "dqss_aux_fraction": float(k_aux / max(k, 1)),
            "dqss_guard_replacements": guard_replacements,
            "dqss_guard_max_aux_overlap": float(self._migdis_dqss_max_aux_overlap()),
            "dqss_guard_min_core_overlap": float(self._migdis_dqss_min_core_overlap()),
        }
        if self._migdis_dqss_debug_overlap() and core_topk is not None and aux_topk is not None:
            final_bool = torch.zeros(n, dtype=torch.bool, device=flat_core.device)
            core_bool = torch.zeros(n, dtype=torch.bool, device=flat_core.device)
            aux_bool = torch.zeros(n, dtype=torch.bool, device=flat_core.device)
            final_bool[final_idx] = True
            core_bool[core_topk] = True
            aux_bool[aux_topk] = True
            stats["overlap_with_core_global_topk"] = float((final_bool & core_bool).sum().item() / k)
            stats["overlap_with_aux_global_topk"] = float((final_bool & aux_bool).sum().item() / k)

        return final_idx, flat_mask.reshape(core_score.shape), stats

    def _migdis_build_masks_once(self, model):
        if getattr(self, "migdis_masks_built", False):
            return

        self._migdis_sync_grad_ema()
        self._migdis_log(
            f"MIG-DIS prune triggered at global_step={self.state.global_step}; "
            f"grad_mix={self._migdis_grad_mix()} source_margin={self._migdis_source_margin()} "
            f"source_scope={self._migdis_source_scope()} norm={self._migdis_norm_mode()} "
            f"selection_mode={self._migdis_selection_mode()}"
        )
        self._migdis_log(f"MIG-DIS selection_mode={self._migdis_selection_mode()}")
        if self._migdis_selection_mode() == "tgsr":
            self._migdis_log(
                f"MIG-DIS-TGSR candidate_ratio={self._migdis_tgsr_candidate_ratio()} "
                f"core_source_margin={self._migdis_tgsr_core_source_margin()}"
            )
        if self._migdis_selection_mode() == "dqss":
            self._migdis_log(
                f"MIG-DIS-DQSS rho={self._migdis_dqss_rho()} "
                f"aux_grad_mix={self._migdis_dqss_aux_grad_mix()} "
                f"aux_source_margin={self._migdis_dqss_aux_source_margin()} "
                f"module_scope={self._migdis_dqss_module_scope()}"
            )
        self._migdis_log("MIG-DIS connector hard masked=false")

        dict_pt = {}
        for name, param in model.named_parameters():
            key = self._migdis_normalize_lora_key(name)
            if "base_layer" in name or key.endswith(".weight"):
                dict_pt.setdefault(key, param)

        active_a = []
        active_b = []
        debug_by_module = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or not self._migdis_should_track_param(name):
                continue

            lora_param = self._migdis_full_param(param)
            base_param = dict_pt.get(self._migdis_normalize_lora_key(name))
            if self._migdis_enabled():
                score_bundle = self._migdis_build_score_bundle(name, lora_param, base_param)
            else:
                score_bundle = {
                    "score": lora_param.abs().detach(),
                    "core_score": lora_param.abs().detach(),
                    "stats": {},
                }
            score_stats = score_bundle["stats"]
            selection_stats = {}
            if self._migdis_enabled() and self._migdis_selection_mode() == "tgsr":
                top_indices, coef_matrix, selection_stats = self._migdis_make_tgsr_mask(
                    score_bundle["core_score"],
                    score_bundle["score"],
                    AB_PRESERVE_RATIO,
                    self._migdis_tgsr_candidate_ratio(),
                )
            elif (
                self._migdis_enabled()
                and self._migdis_selection_mode() == "dqss"
                and self._migdis_dqss_should_apply(name)
                and score_stats.get("source_scope_applied", False)
            ):
                aux_bundle = self._migdis_build_score_bundle(
                    name,
                    lora_param,
                    base_param,
                    grad_mix=self._migdis_dqss_aux_grad_mix(),
                    source_margin=self._migdis_dqss_aux_source_margin(),
                    core_source_margin=self._migdis_dqss_aux_source_margin(),
                )
                top_indices, coef_matrix, selection_stats = self._migdis_make_dqss_mask(
                    score_bundle["score"],
                    aux_bundle["score"],
                    AB_PRESERVE_RATIO,
                    self._migdis_dqss_rho(),
                )
                aux_stats = aux_bundle["stats"]
                score_stats["Q_aux_mean"] = aux_stats.get("score_mean", 0.0)
                score_stats["aux_grad_ema_mean"] = aux_stats.get("grad_ema_mean", 0.0)
                score_stats["aux_grad_mix"] = aux_stats.get("grad_mix", self._migdis_dqss_aux_grad_mix())
                score_stats["aux_source_margin"] = aux_stats.get(
                    "source_margin", self._migdis_dqss_aux_source_margin()
                )
            else:
                top_indices, coef_matrix = self._migdis_make_topk_mask(score_bundle["score"], AB_PRESERVE_RATIO)
            scaled_lora_param = coef_matrix.to(lora_param.device) * lora_param * self._migdis_final_gamma()
            safe_set_full_fp32_param(param, scaled_lora_param)
            self.AB_masks[name] = top_indices

            density = float(coef_matrix.sum().item() / coef_matrix.numel())
            key = self._migdis_normalize_lora_key(name)
            module_stats = debug_by_module.setdefault(key, {"module": key})
            module_stats["selection_mode"] = self._migdis_selection_mode()
            module_stats["source_scope"] = self._migdis_source_scope()
            module_stats["grad_mix"] = score_stats.get("grad_mix", self._migdis_grad_mix())
            module_stats["core_source_margin"] = score_stats.get(
                "core_source_margin", self._migdis_tgsr_core_source_margin()
            )
            module_stats["source_margin"] = score_stats.get("source_margin", self._migdis_source_margin())
            module_stats["candidate_ratio_requested"] = self._migdis_tgsr_candidate_ratio()
            module_stats["dqss_module_scope"] = self._migdis_dqss_module_scope()
            module_stats["dqss_rho"] = self._migdis_dqss_rho()
            module_stats["dqss_aux_grad_mix"] = score_stats.get(
                "aux_grad_mix", self._migdis_dqss_aux_grad_mix()
            )
            module_stats["dqss_aux_source_margin"] = score_stats.get(
                "aux_source_margin", self._migdis_dqss_aux_source_margin()
            )
            if "lora_A" in name:
                active_a.append(density)
                module_stats["A_shape"] = list(lora_param.shape)
                module_stats["mask_density_A"] = density
                module_stats["retain_ratio_A"] = selection_stats.get("retain_ratio", density)
                module_stats["candidate_ratio_effective_A"] = selection_stats.get("candidate_ratio_effective", 1.0)
                if "dqss_aux_fraction" in selection_stats:
                    module_stats["dqss_aux_fraction_A"] = selection_stats["dqss_aux_fraction"]
                    module_stats["dqss_core_count_A"] = selection_stats["dqss_core_count"]
                    module_stats["dqss_aux_count_A"] = selection_stats["dqss_aux_count"]
                    module_stats["dqss_guard_replacements_A"] = selection_stats.get("dqss_guard_replacements", 0)
                module_stats["grad_ema_mean_A"] = score_stats.get("grad_ema_mean", 0.0)
                module_stats["score_A_mean"] = score_stats.get("score_mean", 0.0)
                module_stats["Q_core_A_mean"] = score_stats.get("Q_core_mean", 0.0)
                module_stats["Q_src_A_mean"] = score_stats.get("Q_src_mean", score_stats.get("score_mean", 0.0))
                module_stats["Q_aux_A_mean"] = score_stats.get("Q_aux_mean", 0.0)
                if "overlap_with_core_topk" in selection_stats:
                    module_stats["overlap_with_core_topk_A"] = selection_stats["overlap_with_core_topk"]
                if "overlap_with_src_global_topk" in selection_stats:
                    module_stats["overlap_with_src_global_topk_A"] = selection_stats["overlap_with_src_global_topk"]
                if "overlap_with_core_global_topk" in selection_stats:
                    module_stats["overlap_with_core_global_topk_A"] = selection_stats[
                        "overlap_with_core_global_topk"
                    ]
                if "overlap_with_aux_global_topk" in selection_stats:
                    module_stats["overlap_with_aux_global_topk_A"] = selection_stats[
                        "overlap_with_aux_global_topk"
                    ]
            elif "lora_B" in name:
                active_b.append(density)
                module_stats["B_shape"] = list(lora_param.shape)
                module_stats["mask_density_B"] = density
                module_stats["retain_ratio_B"] = selection_stats.get("retain_ratio", density)
                module_stats["candidate_ratio_effective_B"] = selection_stats.get("candidate_ratio_effective", 1.0)
                if "dqss_aux_fraction" in selection_stats:
                    module_stats["dqss_aux_fraction_B"] = selection_stats["dqss_aux_fraction"]
                    module_stats["dqss_core_count_B"] = selection_stats["dqss_core_count"]
                    module_stats["dqss_aux_count_B"] = selection_stats["dqss_aux_count"]
                    module_stats["dqss_guard_replacements_B"] = selection_stats.get("dqss_guard_replacements", 0)
                module_stats["grad_ema_mean_B"] = score_stats.get("grad_ema_mean", 0.0)
                module_stats["score_B_mean"] = score_stats.get("score_mean", 0.0)
                module_stats["Q_core_B_mean"] = score_stats.get("Q_core_mean", 0.0)
                module_stats["Q_src_B_mean"] = score_stats.get("Q_src_mean", score_stats.get("score_mean", 0.0))
                module_stats["Q_aux_B_mean"] = score_stats.get("Q_aux_mean", 0.0)
                if "overlap_with_core_topk" in selection_stats:
                    module_stats["overlap_with_core_topk_B"] = selection_stats["overlap_with_core_topk"]
                if "overlap_with_src_global_topk" in selection_stats:
                    module_stats["overlap_with_src_global_topk_B"] = selection_stats["overlap_with_src_global_topk"]
                if "overlap_with_core_global_topk" in selection_stats:
                    module_stats["overlap_with_core_global_topk_B"] = selection_stats[
                        "overlap_with_core_global_topk"
                    ]
                if "overlap_with_aux_global_topk" in selection_stats:
                    module_stats["overlap_with_aux_global_topk_B"] = selection_stats[
                        "overlap_with_aux_global_topk"
                    ]
            module_stats["source_scope_applied"] = bool(score_stats.get("source_scope_applied", False))
            if "source_row_mean" in score_stats:
                module_stats["source_row_mean"] = score_stats["source_row_mean"]
            if "source_col_mean" in score_stats:
                module_stats["source_col_mean"] = score_stats["source_col_mean"]

        self.migdis_masks_built = True
        self.migdis_grad_ema = {}
        self.migdis_grad_steps = {}
        avg_a = sum(active_a) / max(len(active_a), 1)
        avg_b = sum(active_b) / max(len(active_b), 1)
        self._migdis_log(f"MIG-DIS active ratio A={avg_a:.6f}, B={avg_b:.6f}")
        self._migdis_log(f"MIG-DIS source_scope={self._migdis_source_scope()}")

        if self._migdis_debug_dump() and self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
            debug_path = os.path.join(self.args.output_dir, "migdis_mask_stats.json")
            with open(debug_path, "w") as f:
                json.dump(list(debug_by_module.values()), f, indent=2)

    def _migdis_apply_masks_to_params_and_grads(self, model, apply_grads=False):
        for name, param in model.named_parameters():
            if name not in self.AB_masks:
                continue
            lora_param = self._migdis_full_param(param)
            flat_lora_param = lora_param.flatten()
            top_indices = self.AB_masks[name]
            restored_tensor = torch.zeros_like(flat_lora_param, device=lora_param.device)
            restored_tensor[top_indices] = flat_lora_param[top_indices]
            restored_tensor = restored_tensor.reshape(lora_param.shape)
            safe_set_full_fp32_param(param, restored_tensor)

            if apply_grads and param.grad is not None:
                grad_mask = torch.zeros_like(param.grad)
                flat_mask = grad_mask.flatten()
                local_top_indices = top_indices.to(flat_mask.device)
                local_top_indices = local_top_indices[local_top_indices < flat_mask.numel()]
                flat_mask[local_top_indices] = 1
                param.grad.mul_(grad_mask)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        
        self.AB_masks = {}
        self.migdis_grad_ema = {}
        self.migdis_grad_steps = {}
        self.migdis_masks_built = False
        self.migdis_logged_init = False
        self.migdis_grad_ema_updates = 0
        self.migdis_grad_hook_inits = 0
        self.migdis_grad_hook_updates = 0
        self.migdis_grad_hooks_registered = False
        self.migdis_grad_hooks = []
        

        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {'use_reentrant':False}
                

            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1



            for step, inputs in enumerate(epoch_iterator):
                

                
                if self.state.global_step == STEP_THRESHOLD:
                    self._migdis_build_masks_once(model)


                def restore_lora_param(param_name, param):
                    if param_name in self.AB_masks:
                        lora_param = (safe_get_full_fp32_param(param)).clone()
                        flat_lora_param = lora_param.flatten()

                        top_indices = self.AB_masks[param_name]
                        restored_tensor = torch.zeros_like(flat_lora_param, device=param.device)
                        restored_tensor[top_indices] = flat_lora_param[top_indices]

                        restored_tensor = restored_tensor.reshape(param.shape)
                        safe_set_full_fp32_param(param, restored_tensor)

                for name, param in model.named_parameters():
                    if param.requires_grad and 'lora' in name:
                        restore_lora_param(name, param)

                

                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model): # automatically perform the gradient accumulation
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):  # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):  # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping
                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    self._migdis_apply_masks_to_params_and_grads(model, apply_grads=True)
                    self.optimizer.step()
                    self._migdis_apply_masks_to_params_and_grads(model, apply_grads=False)



                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)



                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    






    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        self._migdis_register_grad_hooks_once(model)
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)


        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        self.accelerator.backward(loss)
        self._migdis_update_grad_ema(model)

        return loss.detach() / self.args.gradient_accumulation_steps
    

    
    def comput_custom_reg(self, model, reg_lambda=0.1):
        if reg_lambda <= 0:
            return torch.zeros((), device=next(model.parameters()).device)

        reg_loss = 0.0
        param_count = 0

        def normalize_lora_key(name):
            if name.startswith("module."):
                name = name[len("module."):]
            return (name
                    .replace(".lora_A.default.weight", ".weight")
                    .replace(".lora_B.default.weight", ".weight")
                    .replace(".base_layer.weight", ".weight"))

        dict_A={}
        dict_B={}
        dict_PT={}
        target_projs = ("q_proj", "k_proj", "v_proj", "mm_projector")
        for name, param in model.named_parameters():
            if not any(proj in name for proj in target_projs):
                continue
            key = normalize_lora_key(name)
            if "lora_A" in name:
                dict_A[key] = param
            elif "lora_B" in name:
                dict_B[key] = param
            elif "base_layer" in name or key.endswith(".weight"):
                dict_PT.setdefault(key, param)

        
        for key, A in dict_A.items():
            B = dict_B.get(key)
            W = dict_PT.get(key)
            if B is None or W is None:
                continue
            
            
            with torch.no_grad():
                M_pre = W / torch.norm(W, p=2)
                epsilon = 1e-15
                S = torch.abs((1 / torch.log(M_pre.abs() + epsilon)))
                M = torch.tanh(OMEGA * S)

            reg_loss += torch.norm(M * (B @ A), p=2)   
            param_count += 1

        if param_count == 0:
            return torch.zeros((), device=next(model.parameters()).device)
        reg_loss = reg_lambda * reg_loss / param_count

        return reg_loss



    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]

        reg_loss = self.comput_custom_reg(model, reg_lambda=CMR_LAMBDA)
        loss += reg_loss

        return (loss, outputs) if return_outputs else loss
    
