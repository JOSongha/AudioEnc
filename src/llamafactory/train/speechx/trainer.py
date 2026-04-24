# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
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

import accelerate
import os
import json
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from functools import partial

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from transformers.training_args import OptimizerNames
from transformers.utils import is_sagemaker_mp_enabled
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)

class SpeechXTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        unit_token_ids,
        enable_liger_kernel=False,
        **kwargs
    ) -> None:
        self.unit_token_ids = unit_token_ids
        self.finetuning_args = finetuning_args
        self.unit_ppl = {'train': [], 'eval': []}
        self.text_ppl = {'train': [], 'eval': []}

        if 'model' in kwargs and os.getenv("FREEZE_WEIGHT", False):
            logger.info_rank0(f"Freeze model weights except unit embeddings.")
            model = kwargs.pop('model')
            for name, param in model.named_parameters():
                if "embed_tokens" not in name and "lm_head" not in name:
                    param.requires_grad = False
                else:
                    freeze_hook = partial(self._freeze_indices_hook,
                                          param=param,
                                          unit_token_ids=self.unit_token_ids)
                    param.register_hook(freeze_hook)

            kwargs['model'] = model

        super().__init__(**kwargs)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def _freeze_indices_hook(self, grad, param, unit_token_ids):
        # unfreeze unit embeddings and apply L2_reg not to use weight decay
        mask = torch.zeros_like(grad)
        mask[unit_token_ids] = 1.0
        return (grad + (2 * 0.0001 * param)) * mask

    @property
    def total_trained_tokens(self):
        return (self.args.generation_max_length *
            self.args.per_device_train_batch_size *
            self.args.gradient_accumulation_steps *
            self.args.world_size *
            self.state.global_step)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)

        if (emb_lr := os.getenv("EMB_LR_MULTIPLIER")) and (
            self.optimizer is None or isinstance(self.optimizer, accelerate.utils.deepspeed.DummyOptim)):
            opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {"params": [], "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate}, 
                {"params": [], "weight_decay": 0.0, "lr": self.args.learning_rate},
                {"params": [], "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate * float(emb_lr)}, 
            ]

            for n, p in opt_model.named_parameters():
                if not p.requires_grad: continue
                v = 2 if ("embed_tokens" in n or "lm_head" in n) else (0 if n in decay_parameters else 1)
                optimizer_grouped_parameters[v]['params'].append(p)

            logger.info_rank0(f"Using {emb_lr} mulitiplier for embedding layers.")
            if self.args.optim == OptimizerNames.ADAMW_APEX_FUSED: # hijacking apex fusedadam -> deepspeed fusedadam
                import sys
                from deepspeed.ops.adam.fused_adam import FusedAdam
                sys.modules["apex.optimizers"] = type(sys)("apex.optimizers")
                sys.modules["apex.optimizers"].FusedAdam = FusedAdam

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")  

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, torch.nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

            self.optimizer = self.accelerator.prepare(self.optimizer)

        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)

        if self.args.lr_scheduler_type=='warmup_stable_decay':
            if not self.args.lr_scheduler_kwargs:
                self.args.lr_scheduler_kwargs = {
                    'num_stable_steps': int(0.8 * num_training_steps) - self.args.get_warmup_steps(num_training_steps),
                    'num_decay_steps': num_training_steps - int(0.8 * num_training_steps)
                }

        return super().create_scheduler(num_training_steps, optimizer)

    def get_unit_text_ppl(self, ce_loss, shift_labels, bsz):
        unit_mask = (shift_labels >= self.unit_start) & (shift_labels < self.unit_end)
        text_mask = (~unit_mask) & (shift_labels != -100)

        unit_loss = (ce_loss * unit_mask).view(bsz, -1).sum(-1) / unit_mask.view(bsz, -1).sum(-1)
        text_loss = (ce_loss * text_mask).view(bsz, -1).sum(-1) / text_mask.view(bsz, -1).sum(-1)

        unit_ppl = torch.exp(unit_loss).detach()
        text_ppl = torch.exp(text_loss).detach()

        return unit_ppl, text_ppl

    @override
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        kwargs.update({"return_outputs": return_outputs})
        return super().compute_loss(model, inputs, *args, **kwargs)  
    
        # TODO remove ppl calculating
        loss = None
        labels = inputs.pop("labels") if "labels" in inputs else None
        inputs.update({"return_dict": True})
        outputs = model(**inputs)
        logits = outputs.logits

        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss = self.loss_fct(shift_logits, shift_labels)
            mask = (shift_labels != -100)
            loss = (ce_loss * mask).sum() / mask.sum()
        
        with torch.no_grad():
            unit_ppl_step, text_ppl_step = self.get_unit_text_ppl(
                ce_loss, shift_labels, bsz=labels.size(0))
            stage = 'train' if self.model.training else 'eval'
            self.unit_ppl[stage].append(unit_ppl_step)
            self.text_ppl[stage].append(text_ppl_step)

        return (loss, outputs) if return_outputs else loss

    # @override
    # def training_step(self, model, inputs):
    #     results = super().training_step(model, inputs)
    #     return results

    #     # TODO: remove ppl calculating
    #     if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
    #         text_ppls = torch.cat(self.text_ppl['train'])
    #         gathered_text_ppl = self._nested_gather(text_ppls).nanmean().item()

    #         unit_ppls = torch.cat(self.unit_ppl['train'])
    #         gathered_unit_ppl = self._nested_gather(unit_ppls).nanmean().item()

    #         if self.state.is_world_process_zero:
    #             log = {'text_ppl': gathered_text_ppl,
    #                    'unit_ppl': gathered_unit_ppl}
    #             self.log(log)
            
    #         self.text_ppl['train'], self.unit_ppl['train'] = [], []

    #     return results

    @override
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        flip_flatten = False
        if getattr(self.data_collator, "flatten", False):
            self.data_collator.flatten = False
            flip_flatten = True

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        metrics = {}
        text_ppls, unit_ppls = {}, {}

        if isinstance(eval_dataset, dict):
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                metric_key = f"{metric_key_prefix}_{eval_dataset_name}"
                dataset_metrics = super().evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key,
                )
                text_ppls[metric_key] = torch.cat(self.text_ppl['eval'])
                unit_ppls[metric_key] = torch.cat(self.unit_ppl['eval'])
                self.text_ppl['eval'], self.unit_ppl['eval'] = [], []
                metrics.update(dataset_metrics)
        else:
            metric_key = metric_key_prefix
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key)
            text_ppls[metric_key] = torch.cat(self.text_ppl['eval'])
            unit_ppls[metric_key] = torch.cat(self.unit_ppl['eval'])
            self.unit_ppl['eval'], self.text_ppl['eval'] = [], []

        gathered_text_ppl = {k: v.nanmean().item() for k, v in self._nested_gather(text_ppls).items()}
        gathered_unit_ppl = {k: v.nanmean().item() for k, v in self._nested_gather(unit_ppls).items()}

        if self.state.is_world_process_zero:
            metrics.update(**{"token": self.total_trained_tokens,
                              "global_step": self.state.global_step})
            for metric_key in gathered_text_ppl:
                log = {
                    f"{metric_key}_text_ppl": gathered_text_ppl[metric_key],
                    f"{metric_key}_unit_ppl": gathered_unit_ppl[metric_key]
                }
                metrics.update(**log)
                self.log(log)

        torch.cuda.empty_cache()

        if flip_flatten:
            self.data_collator.flatten = True

        return metrics

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


    def save_metrics(self, split, metrics, combined=True):
        """
        Save metrics into a json file for that split, e.g. `train_results.json`.

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (`str`):
                Mode/split name: one of `train`, `eval`, `test`, `all`
            metrics (`Dict[str, float]`):
                The metrics returned from train/evaluate/predict
            combined (`bool`, *optional*, defaults to `True`):
                Creates combined metrics by updating `all_results.json` with metrics of this call

        To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
        unformatted numbers are saved in the current method.

        """
        if not self.is_world_process_zero():
            return

        if split != 'eval':
            super().save_metrics(split, metrics, combined)
        else:
            path = os.path.join(self.args.output_dir, f"{split}_results.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(metrics, sort_keys=True, ensure_ascii=False) + "\n")

        if combined:
            path = os.path.join(self.args.output_dir, "all_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)
