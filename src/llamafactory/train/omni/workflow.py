# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional

from transformers import TrainerCallback

from ...data import OmniCollator, get_omni_dataset
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import OmniTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments

    from ...hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
    )

logger = get_logger(__name__)


def run_omni(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    dataset_module = get_omni_dataset(model_args, data_args, training_args, tokenizer)

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # Freeze all parameters except:
    #   - audio_encoder.projector (always trainable — the adapter Stage 1 aligned)
    #   - LoRA adapters (peft inserts these as `base_model.model.*.lora_A/lora_B.*` when
    #     finetuning_type=lora; the substring `lora_` matches both and is absent under
    #     full-FT, so the filter is safe for both paths)
    # Stage 1: only projector trains.
    # Stage 2: projector + LoRA adapters train, everything else frozen.
    if training_args.do_train:
        for name, param in model.named_parameters():
            require_grad = ("audio_encoder.projector" in name) or ("lora_" in name)
            param.requires_grad = require_grad

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    attn_impl = getattr(model.config, "_attn_implementation", "eager")

    # Pick collator matching the audio encoder type (auto-detected from model config).
    audio_cfg = getattr(model.config, "audio_config", None)
    is_whisper = audio_cfg is not None and getattr(audio_cfg, "whisper_model_id", None) is not None
    if is_whisper:
        from ...data.omni_dataset_whisper import WhisperOmniCollator

        logger.info_rank0("[omni] using WhisperOmniCollator (mel stack)")
        data_collator = WhisperOmniCollator(
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=attn_impl,
            block_diag_attn=model_args.block_diag_attn,
            compute_dtype=model_args.compute_dtype,
        )
    else:
        data_collator = OmniCollator(
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=attn_impl,
            block_diag_attn=model_args.block_diag_attn,
            compute_dtype=model_args.compute_dtype,
        )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Initialize our Trainer
    trainer = OmniTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        # Sanitize NaN/Inf gradients before any DeepSpeed/NCCL collective sees
        # them. Without this, a NaN grad on one rank wedges that rank's NCCL
        # reduce-scatter; the others trip the 600 s collective timeout. See
        # llamafactory/data/fault_tolerant.py::install_grad_nan_guard.
        from ...data.fault_tolerant import install_grad_nan_guard

        n_guards = install_grad_nan_guard(trainer.model)
        logger.info_rank0(f"[grad-nan-guard] installed {n_guards} post-accumulate hooks")

        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        if model_args.global_step is not None:
            trainer.state.global_step = model_args.global_step

        metrics = trainer.evaluate(metric_key_prefix="eval")
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
    trainer.accelerator.free_memory(trainer.model, trainer.optimizer, trainer.lr_scheduler)
