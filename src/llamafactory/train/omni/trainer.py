# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ...data.omni_dataset import MODALITY_NAMES, MODALITY_PAD_ID
from ..callbacks import SaveProcessorCallback

# from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class OmniTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        # if model_args is not None and model_args.fp8:
        #     configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        # Force-disable model_accepts_loss_kwargs regardless of processor.
        # Our model forward declares **kwargs: Unpack[TransformersKwargs] which the
        # Trainer auto-detects as loss-kwargs-accepting; combined with HF passing
        # num_items_in_batch, that path SKIPS the `loss / grad_accum_steps`
        # division, making logged loss scale linearly with grad_accum.
        # ref: https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
        self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        # if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
        #     verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_train_dataloader(self):
        """Skip accelerator.prepare to avoid double-sharding on IterableDataset.
        DDP sharding is already done via ds.shard() before .map() in _build_dataset.
        """
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
        return dataloader

    @override
    def get_eval_dataloader(self, eval_dataset=None):
        """Skip accelerator.prepare to avoid double-sharding.
        DDP sharding is already done via ds.shard() in _build_dataset.
        """
        from torch.utils.data import DataLoader

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )
        return dataloader

    def _ensure_hidden_hook(self, model):
        """Register a forward hook on the inner Qwen3_5AEModel so we can read the
        last hidden state even when Liger's fused CE hides logits. Idempotent."""
        if getattr(self, "_hidden_hook_installed", False):
            return
        target = None
        # `named_modules` traverses DeepSpeed / PEFT / Lora wrappers. Match by
        # class name so this keeps working if the module is accessed through
        # DeepSpeedEngine.module.base_model.model.model etc.
        for _, mod in model.named_modules():
            if type(mod).__name__ == "Qwen3_5AEModel":
                target = mod
                break
        if target is None:
            logger.warning_once(
                "per-modality hidden-state hook: Qwen3_5AEModel not found in model tree; "
                "per-task loss logging will remain off under Liger."
            )
            self._hidden_hook_installed = True  # don't retry
            return
        self._last_hidden = None

        def _hook(_module, _inputs, output):
            # output is Qwen3_5AEModelOutputWithPast; last_hidden_state via attr
            # or tuple-index 0.
            try:
                if hasattr(output, "last_hidden_state"):
                    self._last_hidden = output.last_hidden_state
                else:
                    self._last_hidden = output[0]
            except Exception:
                pass

        self._hidden_hook_handle = target.register_forward_hook(_hook)
        self._hidden_hook_installed = True

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # modality_ids is data-side metadata only; strip before model forward.
        modality_ids = inputs.pop("modality_ids", None)
        labels = inputs.get("labels")
        # Lazy-register hidden-state hook once (needed when Liger suppresses logits).
        if modality_ids is not None:
            self._ensure_hidden_hook(model)
            self._last_hidden = None  # reset so we can detect whether hook fired
        outputs = model(**inputs)

        if getattr(outputs, "ce_loss", None) is not None:
            _keys = ["ce_loss", "calm_loss", "router_loss", "student_recon_mse"]
            if not hasattr(self, "_custom_losses"):
                self._custom_losses = dict.fromkeys(_keys, 0.0)
                self._custom_loss_count = 0
            for k in _keys:
                val = getattr(outputs, k, None)
                if val is not None:
                    self._custom_losses[k] += val.detach().item()
            self._custom_loss_count += 1

        # Per-modality loss decomposition. Computes CE on the shifted logits/labels
        # separately for each modality id present in the batch, accumulates in
        # _modality_losses for the next log() flush.
        logits = getattr(outputs, "logits", None)
        # Liger-kernel fused CE leaves outputs.logits=None to save memory — fall
        # back to projecting the hook-captured last hidden state through lm_head.
        # Runs in no_grad so the large logits tensor never enters the autograd graph.
        if logits is None and modality_ids is not None:
            last_hidden = getattr(self, "_last_hidden", None)
            if last_hidden is None:
                logger.warning_once(
                    "per-modality logits fallback: hook did not capture last hidden state."
                )
            else:
                # Find lm_head by walking the model tree (match by name, same as hook).
                lm_head = None
                for name, mod in model.named_modules():
                    if name.endswith("lm_head") and hasattr(mod, "weight"):
                        lm_head = mod
                        break
                if lm_head is None:
                    logger.warning_once(
                        "per-modality logits fallback: lm_head not found in model tree."
                    )
                else:
                    try:
                        with torch.no_grad():
                            logits = lm_head(last_hidden)
                    except Exception as e:
                        logger.warning_once(f"per-modality logits recompute skipped: {e}")

        if (
            modality_ids is not None
            and logits is not None
            and labels is not None
        ):
            try:
                with torch.no_grad():
                    # shift by 1 for causal LM: predict token at position t+1 using logits at t
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_modality = modality_ids[..., 1:].contiguous()
                    # token-level CE (no reduction) so we can slice by modality
                    tok_ce = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=IGNORE_INDEX,
                        reduction="none",
                    ).view_as(shift_labels)
                    label_mask = shift_labels != IGNORE_INDEX
                    if not hasattr(self, "_modality_losses"):
                        self._modality_losses: dict[int, tuple[float, int]] = {}
                    for mid in shift_modality.unique().tolist():
                        if mid == MODALITY_PAD_ID:
                            continue
                        sel = (shift_modality == mid) & label_mask
                        n_tok = int(sel.sum().item())
                        if n_tok == 0:
                            continue
                        loss_sum = float(tok_ce[sel].sum().item())
                        acc_sum, acc_n = self._modality_losses.get(mid, (0.0, 0))
                        self._modality_losses[mid] = (acc_sum + loss_sum, acc_n + n_tok)
            except Exception as e:  # defensive — never break training on logging
                logger.warning_once(f"per-modality loss decomposition skipped: {e}")

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    @override
    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_custom_loss_count") and self._custom_loss_count > 0:
            for k, v in self._custom_losses.items():
                logs[k] = v / self._custom_loss_count
            for k in self._custom_losses:
                self._custom_losses[k] = 0.0
            self._custom_loss_count = 0
        # Per-modality loss (token-weighted mean since last log() call).
        if hasattr(self, "_modality_losses") and self._modality_losses:
            for mid, (loss_sum, n_tok) in self._modality_losses.items():
                if n_tok == 0:
                    continue
                name = MODALITY_NAMES.get(mid, f"modality_{mid}")
                logs[f"loss/{name}"] = loss_sum / n_tok
                logs[f"tokens/{name}"] = n_tok
            self._modality_losses = {}
        super().log(logs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self,
        dataset: "Dataset",
        predict_results: "PredictionOutput",
        skip_special_tokens: bool = True,
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX,
            predict_results.label_ids,
            self.processing_class.pad_token_id,
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
