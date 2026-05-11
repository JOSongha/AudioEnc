# ruff: noqa: UP006
# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    template: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which template to use for constructing prompts in training and inference."
        },
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."
        },
    )
    omni_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the JSONL manifest for omni dataset (each line: {wav_path, text})."
        },
    )
    omni_eval_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSONL manifest for omni evaluation dataset."},
    )
    omni_sample_rate: Optional[int] = field(
        default=48000,
        metadata={"help": "Target sample rate for omni audio loading. Default: 48000."},
    )
    omni_hop_length: Optional[int] = field(
        default=1920,
        metadata={"help": "Hop length of the audio encoder (for computing audio token count)."},
    )
    omni_max_audio_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max audio samples per utterance (length limit). None for unlimited."},
    )
    omni_shuffle_buffer_size: int = field(
        default=10000,
        metadata={"help": "Buffer size for IterableDataset shuffle in omni pipeline."},
    )
    omni_packing_bucket_size: int = field(
        default=256,
        metadata={
            "help": "Number of samples per bucket for greedy knapsack packing in omni pipeline."
        },
    )
    omni_per_modality_manifests: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "{modality: manifest_path} for interleaved per-modality streaming. "
                    "Each source is loaded, sharded, and shuffled independently, then "
                    "combined via interleave_datasets with omni_per_modality_probs. Lets "
                    "a small pool (e.g. emotion) cycle while larger pools (asr / env / "
                    "text superset) are effectively re-sampled across epochs. When set, "
                    "overrides omni_manifest for training."
        },
    )
    omni_per_modality_probs: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": "Per-modality sampling probabilities for interleave_datasets. Keys "
                    "must match omni_per_modality_manifests. Should sum to ~1."
        },
    )
    omni_per_modality_stopping: str = field(
        default="all_exhausted",
        metadata={
            "help": "stopping_strategy for interleave_datasets. 'all_exhausted' cycles "
                    "shorter pools until the longest one is exhausted (recommended when a "
                    "modality pool like emotion is much smaller than the others); "
                    "'first_exhausted' ends the epoch at the shortest source."
        },
    )
    load_from_nubes: bool = field(
        default=False,
        metadata={"help": "Load from nubes directly rather than load from local file."},
    )
    init_ag_from_text: Optional[bool] = field(
        default=False,
        metadata={"help": "Initialize MoT _ag (audio pathway) weights from text pathway weights."},
    )
    init_consistency_zero: Optional[bool] = field(
        default=False,
        metadata={"help": "Zero-initialize consistency head & weight net last layers."},
    )
    eval_dataset: Optional[Union[Dict[str, str], str]] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."
        },
    )
    eval_tokenized_path: Optional[str] = field(
        default=None,
        metadata={"help": "tokenized path for eval dataset"},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    media_dir: str | None = field(
        default=None,
        metadata={
            "help": "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`."
        },
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={
            "help": "Size of the buffer to randomly sample examples from in dataset streaming."
        },
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={
            "help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."
        },
    )
    interleave_probs: str | None = field(
        default=None,
        metadata={
            "help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: int | None = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each dataset."
        },
    )
    eval_num_beams: int | None = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."
        },
    )
    val_size: float = field(
        default=0.0,
        metadata={
            "help": "Size of the validation set, should be an integer or a float in range `[0,1)`."
        },
    )
    eval_on_each_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to evaluate on each dataset separately."},
    )
    packing: bool | None = field(
        default=None,
        metadata={
            "help": "Enable sequences packing in training. Will automatically enable in pre-training."
        },
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    tool_format: str | None = field(
        default=None,
        metadata={"help": "Tool format to use for constructing function calling examples."},
    )
    default_system: str | None = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    enable_thinking: bool | None = field(
        default=True,
        metadata={"help": "Whether or not to enable thinking mode for reasoning models."},
    )
    tokenized_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
            )
        },
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={"help": "Whether or not to use a shared file system for the datasets."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)

        if self.media_dir is None:
            self.media_dir = self.dataset_dir

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

        if self.interleave_probs is not None:
            if self.mix_strategy == "concat":
                raise ValueError("`interleave_probs` is only valid for interleaved mixing.")

            self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            if self.dataset is not None and len(self.dataset) != len(self.interleave_probs):
                raise ValueError("The length of dataset and interleave probs should be identical.")

            if self.eval_dataset is not None and len(self.eval_dataset) != len(
                self.interleave_probs
            ):
                raise ValueError(
                    "The length of eval dataset and interleave probs should be identical."
                )

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

        if self.neat_packing:
            self.packing = True

        if self.packing:
            self.cutoff_len -= 1  # avoid pad_to_multiple_of, needs improve

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
