# Copyright 2025 the LlamaFactory team.
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

import os
import sys
from math import ceil
from typing import TYPE_CHECKING, Literal, Optional, Union

import datasets
import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    Value,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    load_from_disk,
)
from tqdm import tqdm

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import (
    convert_to_hf_iterable_dataset,
    get_dataset_module,
    merge_dataset,
    read_cloud_json,
    split_dataset,
)
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSpeechXDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SpeechXDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)
from .speechx_utils import (
    RatioSplitDataset,
    WrapperMTIterableDataset,
    optimized_typed_sequence_init,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: list[str] | None,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Union["Dataset", "IterableDataset", dict[str, "Dataset"]] | None:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "speechx":
        from datasets.arrow_writer import OptimizedTypedSequence

        OptimizedTypedSequence.__init__ = optimized_typed_sequence_init
        if data_args.packing:
            dataset_processor_class = PackedSpeechXDatasetProcessor
        else:
            dataset_processor_class = SpeechXDatasetProcessor

    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset", "DatasetDict"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "speechx"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
    remove_columns: bool = True,
    split: str = None,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None
    if isinstance(dataset, DatasetDict):
        for d_name, d_val in dataset.items():
            dataset[d_name] = _get_preprocessed_dataset(
                d_val,
                data_args,
                training_args,
                stage,
                template,
                tokenizer,
                processor,
                is_eval,
                remove_columns,
                split,
            )
        return dataset

    dataset_processor = _get_dataset_processor(
        data_args,
        stage,
        template,
        tokenizer,
        processor,
        do_generate=(training_args.predict_with_generate and is_eval),
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names if remove_columns else None,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        # move front to make sure eval_dataset(if contain or split) can preprocessed appropriately
        train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)

        if "train" in train_dict:
            train_dict["train"] = _get_preprocessed_dataset(
                train_dict["train"],
                data_args,
                training_args,
                stage,
                template,
                tokenizer,
                processor,
                is_eval=False,
            )

        for key in eval_dict:
            eval_dict[key] = _get_preprocessed_dataset(
                eval_dict[key],
                data_args,
                training_args,
                stage,
                template,
                tokenizer,
                processor,
                is_eval=True,
            )

        # Combine train and eval dictionaries
        dataset_dict = DatasetDict({**train_dict, **eval_dict})

        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        return get_dataset_module(dataset_dict)


def merge_datasets_in_directory(directory_path: str) -> "Dataset":
    dataset_paths = []
    if os.path.exists(os.path.join(directory_path, "dataset_info.json")):
        dataset_paths.append(directory_path)  # root 디렉터리 체크
    else:
        for root, dirs, files in os.walk(directory_path):
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, "dataset_info.json")):
                    dataset_paths.append(os.path.join(root, dir))

    if not dataset_paths:
        raise ValueError(f"No valid datasets found in {directory_path}")

    datasets_list = []
    for path in tqdm(dataset_paths):
        try:
            dataset = load_from_disk(path)
            datasets_list.append(dataset)
            logger.info_rank0(f"Loaded dataset from {path}")
        except Exception as e:
            logger.info_rank0(f"Error loading dataset from {path}: {str(e)}")

    if not datasets_list:
        raise ValueError("No datasets could be loaded successfully")

    merged_dataset = concatenate_datasets(datasets_list)
    logger.info_rank0(f"Successfully merged {len(datasets_list)} datasets")
    logger.info_rank0(f"Merged dataset size: {len(merged_dataset)} rows")

    return merged_dataset


def sample_datasets_by_ratio(dataset_list, ratios) -> "Dataset":
    """datasets: list of HuggingFace datasets
    ratios: list of sampling ratios
    """
    assert len(dataset_list) == len(ratios)
    if len(dataset_list) == 1:
        return dataset_list[0]

    def find_common_indices(a, b):
        # O(m log n)
        indices = np.searchsorted(a, b)
        mask = (indices < len(a)) & (a[indices] == b)
        return indices[mask]

    logger.info_rank0(f"Sample {len(dataset_list)} datasets from the dataset_list by ratio")

    l = sorted(range(len(dataset_list)), key=lambda i: len(dataset_list[i]))
    dataset_list[:], ratios[:] = [dataset_list[i] for i in l], [ratios[i] for i in l]
    idx_array = np.array(dataset_list[-1]["idx"], dtype="int64")
    np.random.shuffle(idx_array)

    for i, ratio in enumerate(tqdm(ratios)):
        size = ceil(ratio * len(dataset_list[i]))
        sample_idx = np.sort(idx_array[:size])
        dataset_list[i] = dataset_list[i].sort("idx")
        data_idx = np.array(dataset_list[i]["idx"])
        dataset_list[i] = dataset_list[i].select(find_common_indices(data_idx, sample_idx))
        idx_array = idx_array[size:]

    logger.info_rank0("Successfully sampled datasets by ratio")

    return concatenate_datasets(dataset_list)


def get_speechx_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "speechx"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Gets the train dataset and optionally gets the evaluation dataset."""
    assert not data_args.streaming, "Currently not support streaming speechx dataset"

    # Load tokenized dataset
    if data_args.speechx_tokenized_path is not None and has_tokenized_data(data_args.speechx_tokenized_path):
        speechx_datasets = []
        if isinstance(data_args.speechx_tokenized_path, (list, tuple)):
            assert (
                len(data_args.speechx_tokenized_path) == len(data_args.speechx_dataset_ratios)
                or not data_args.speechx_dataset_ratios
            )
            for d in data_args.speechx_tokenized_path:
                speechx_datasets.append(load_from_disk(d))
        else:
            speechx_datasets.append(load_from_disk(data_args.speechx_tokenized_path))
        logger.info_rank0(f"Loaded tokenized dataset from {data_args.speechx_tokenized_path}.")

        dataset_module: dict[str, Dataset] = {
            "train_dataset": RatioSplitDataset(datasets=speechx_datasets, ratios=data_args.speechx_dataset_ratios)
            if len(speechx_datasets) > 1
            else speechx_datasets[0]
        }

        if data_args.eval_tokenized_path and has_tokenized_data(data_args.eval_tokenized_path):
            dataset_module["eval_dataset"] = load_from_disk(data_args.eval_tokenized_path)
        else:
            logger.warning_rank0("validation dataset not exist.")

        return dataset_module

    # Load dataset
    with training_args.main_process_first(desc="load dataset"):
        # Load SpeechX train set
        if isinstance(data_args.speechx_dataset, str):
            data_args.speechx_dataset = [data_args.speechx_dataset]
            data_args.speechx_tokenized_path = [data_args.speechx_tokenized_path]

        spx_datasets = {"train": [], "validation": []}
        if not data_args.speechx_tokenized_path:
            data_args.speechx_tokenized_path = [None] * len(data_args.speechx_dataset)

        for d, tok_path in zip(data_args.speechx_dataset, data_args.speechx_tokenized_path):
            spx_dataset = merge_datasets_in_directory(d)

            spx_dataset = spx_dataset.select_columns("input_ids")
            if spx_dataset.features["input_ids"] != datasets.Sequence(feature=Value(dtype="int32")):
                spx_dataset = spx_dataset.cast_column(
                    "input_ids",
                    datasets.Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                )

            spx_datasets["train"].append((spx_dataset, tok_path))

        # Load valid dataset
        if data_args.eval_dataset and isinstance(data_args.eval_dataset, dict):
            spx_datasets["validation"].append(
                (
                    DatasetDict(
                        {
                            data_name: load_from_disk(data_path)
                            for data_name, data_path in data_args.eval_dataset.items()
                        }
                    ),
                    data_args.eval_tokenized_path,
                )
            )
        elif data_args.eval_dataset and isinstance(data_args.eval_dataset, str):
            spx_datasets["validation"].append((load_from_disk(data_args.eval_dataset), data_args.eval_tokenized_path))
        else:
            logger.warning_rank0("eval_dataset not provided.")
            if data_args.val_size:
                raise ValueError("Splitting training set into validation set is not supported!")

    # Preprocess dataset
    with training_args.main_process_first(desc="pre-process dataset"):
        # Packing
        do_packing = data_args.packing
        for split, dataset in spx_datasets.items():
            if not training_args.do_train and split == "train":
                continue
            if split == "validation":
                data_args.packing = False
            for d, tok_path in dataset:
                preprocessed = _get_preprocessed_dataset(
                    d,
                    data_args,
                    training_args,
                    stage,
                    template,
                    tokenizer,
                    processor,
                    is_eval=split != "train",
                    remove_columns=False,
                    split=split,
                )
                if tok_path is not None and training_args.should_save:
                    preprocessed.save_to_disk(
                        tok_path,
                        max_shard_size="2048MB",
                        num_proc=data_args.preprocessing_num_workers,
                    )
                    preprocessed.cleanup_cache_files()
                    logger.info_rank0(f"Packed dataset saved as {tok_path}.")

            data_args.packing = do_packing

        if data_args.speechx_tokenized_path is not None and training_args.should_save:
            logger.info_rank0("Save Completed! Terminate.")
            sys.exit(0)

        raise NotImplementedError

        return dataset_module


def get_omni_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: "PreTrainedTokenizer",
) -> "DatasetModule":
    """Build IterableDataset-based omni pipeline from JSONL manifest."""
    import torch.distributed as dist
    from transformers import AutoConfig

    from .omni_dataset import (
        create_omni_packer,
        create_omni_processor,
        resolve_jsonl_files,
    )

    if data_args.omni_manifest is None:
        raise ValueError("`omni_manifest` is required for omni pipeline.")

    # Resolve audio_pad_token_id from tokenizer
    audio_pad_token_id = getattr(tokenizer, "audio_pad_token_id", None)
    if audio_pad_token_id is None:
        audio_pad_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

    if audio_pad_token_id is None or audio_pad_token_id == tokenizer.unk_token_id:
        raise ValueError(
            "Cannot resolve audio_pad_token_id. "
            "Ensure the tokenizer has `audio_pad_token_id` attribute or `<|audio_pad|>` token."
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if data_args.load_from_nubes:
        logger.info_rank0("[omni] Load from nubes on-the-fly data-processing")

    # Auto-detect audio encoder type from model config (DAC vs Whisper).
    # Whisper variant adds `whisper_model_id` to the audio sub-config; DAC variant has `dac_*` fields.
    ae_cfg = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    audio_cfg = getattr(ae_cfg, "audio_config", None)
    is_whisper = audio_cfg is not None and getattr(audio_cfg, "whisper_model_id", None) is not None

    if is_whisper:
        from .omni_dataset_whisper import create_omni_processor_whisper

        whisper_id = audio_cfg.whisper_model_id
        logger.info_rank0(f"[omni] audio encoder = whisper ({whisper_id})")
        processor_fn = create_omni_processor_whisper(
            tokenizer=tokenizer,
            audio_pad_token_id=audio_pad_token_id,
            whisper_model_id=whisper_id,
            max_audio_samples=data_args.omni_max_audio_samples or (30 * 16000),
            load_from_nubes=data_args.load_from_nubes,
        )
    else:
        logger.info_rank0("[omni] audio encoder = dac (legacy)")
        processor_fn = create_omni_processor(
            tokenizer=tokenizer,
            audio_pad_token_id=audio_pad_token_id,
            sample_rate=data_args.omni_sample_rate,
            hop_length=data_args.omni_hop_length,
            max_audio_samples=data_args.omni_max_audio_samples,
            load_from_nubes=data_args.load_from_nubes,
        )
    packer_fn = create_omni_packer(
        cutoff_len=data_args.cutoff_len,
        pad_token_id=tokenizer.pad_token_id,
        neat_packing=data_args.neat_packing,
    )

    def _load_source(
        manifest_path: str,
        shuffle: bool,
        streaming: bool,
        seed: int,
        tag: str = "",
    ):
        """Load one manifest → sharded, (optionally) shuffled IterableDataset/Dataset.

        Factored out so per-modality interleaving can build multiple sources with
        distinct shuffle seeds before combining them via interleave_datasets.
        """
        jsonl_files = resolve_jsonl_files(manifest_path)
        prefix = f"[omni{(':' + tag) if tag else ''}]"
        logger.info_rank0(
            f"{prefix} Found {len(jsonl_files)} jsonl files, rank={rank}, world_size={world_size}, streaming={streaming}"
        )
        sds = datasets.load_dataset(
            "json",
            data_files=jsonl_files,
            split="train",
            streaming=streaming,
        )
        if streaming:
            logger.info_rank0(f"{prefix} Before split_dataset_by_node: num_shards={sds.num_shards}")

        if world_size > 1:
            sds = sds.shard(num_shards=world_size, index=rank, contiguous=False)
            if streaming:
                logger.info_rank0(f"{prefix} After split_dataset_by_node: num_shards={sds.num_shards}")
            else:
                logger.info_rank0(f"{prefix} After shard: {len(sds)} samples for rank={rank}")

        if shuffle:
            if streaming:
                sds = sds.shuffle(seed=seed, buffer_size=data_args.omni_shuffle_buffer_size)
                logger.info_rank0(f"{prefix} After shuffle (seed={seed}): num_shards={sds.num_shards}")
            else:
                sds = sds.shuffle(seed=seed)

        return sds

    def _build_dataset(
        manifest_path: str,
        shuffle: bool = False,
        streaming: bool = True,
        keep_in_memory: bool = False,
    ):
        per_mod_manifests = data_args.omni_per_modality_manifests
        if streaming and per_mod_manifests:
            # Per-modality interleave path: each modality is its own stream with
            # an independent shuffle seed; interleave_datasets draws per the
            # configured probabilities. Smaller pools cycle (all_exhausted) so a
            # small corpus like emotion repeats while larger ones are re-sampled.
            per_mod_probs = data_args.omni_per_modality_probs or {}
            missing_probs = [m for m in per_mod_manifests if m not in per_mod_probs]
            if missing_probs:
                raise ValueError(
                    f"[omni] omni_per_modality_probs missing entries for: {missing_probs}. "
                    f"Must cover all modalities in omni_per_modality_manifests."
                )
            modalities = list(per_mod_manifests.keys())
            probs = [float(per_mod_probs[m]) for m in modalities]
            prob_sum = sum(probs)
            if abs(prob_sum - 1.0) > 1e-3:
                logger.warning_rank0(
                    f"[omni] omni_per_modality_probs sums to {prob_sum:.4f}, normalizing to 1.0."
                )
                probs = [p / prob_sum for p in probs]

            sub_datasets = []
            for idx, mod_name in enumerate(modalities):
                # Distinct seed per source so modality streams shuffle independently.
                sub_seed = training_args.seed + idx * 1_000_003
                sub_datasets.append(
                    _load_source(
                        per_mod_manifests[mod_name],
                        shuffle=shuffle,
                        streaming=True,
                        seed=sub_seed,
                        tag=mod_name,
                    )
                )

            logger.info_rank0(
                f"[omni] interleave_datasets: modalities={modalities}, "
                f"probs={[round(p, 4) for p in probs]}, "
                f"stopping={data_args.omni_per_modality_stopping}, seed={training_args.seed}"
            )
            ds = interleave_datasets(
                sub_datasets,
                probabilities=probs,
                seed=training_args.seed,
                stopping_strategy=data_args.omni_per_modality_stopping,
            )
            logger.info_rank0(f"[omni] After interleave: num_shards={ds.num_shards}")
        else:
            ds = _load_source(
                manifest_path,
                shuffle=shuffle,
                streaming=streaming,
                seed=training_args.seed,
            )

        map_kwargs = {"keep_in_memory": keep_in_memory} if not streaming else {}
        # IterableDataset.column_names is None for streaming; peek first example to get columns
        if not (col_names := ds.column_names):
            col_names = list(next(iter(ds)).keys())

        ds = ds.map(
            processor_fn,
            batched=True,
            batch_size=32,
            remove_columns=col_names,
            **map_kwargs,
        )
        if streaming:
            logger.info_rank0(f"[omni] After processor map: num_shards={ds.num_shards}")

        if data_args.packing:
            ds = ds.map(
                packer_fn,
                batched=True,
                batch_size=data_args.omni_packing_bucket_size,
                **map_kwargs,
            )
            if streaming:
                logger.info_rank0(f"[omni] After packer map: num_shards={ds.num_shards}")

        if streaming:
            logger.info_rank0(f"[omni] Final ex_iterable.num_shards={ds._ex_iterable.num_shards}")

        return ds

    dataset_module: dict = {
        "train_dataset": _build_dataset(data_args.omni_manifest, shuffle=True, streaming=True),
    }
    if data_args.omni_eval_manifest:
        dataset_module["eval_dataset"] = _build_dataset(
            data_args.omni_eval_manifest, shuffle=False, streaming=False, keep_in_memory=True
        )

    return dataset_module


if __name__ == "__main__":
    spx_dataset = merge_datasets_in_directory("/mnt/ddn/audiollm/datasets/train_db/units_word_aligned/datasets")
    hcx_dataset = convert_to_hf_iterable_dataset(WrapperMTIterableDataset("/mnt/ddn/audiollm/datasets/hcx"))

    print(len(spx_dataset))
    print(len(hcx_dataset))
