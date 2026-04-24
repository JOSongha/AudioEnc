from typing import List
from copy import copy
from collections import defaultdict
import os
from glob import glob
import torch.utils.data as tud
from datasets.features import Value
from datasets.arrow_writer import TypedSequence

# to hack OptimizedTypedSequence when packing
def optimized_typed_sequence_init(
    self,
    data,
    type = None,
    try_type = None,
    col= None,
    optimized_int_type = None):

    optimized_int_type_by_col = {
        "attention_mask": Value("int16"),  # binary tensor
        "special_tokens_mask": Value("int8"),
        "input_ids": Value("int32"),  # typical vocab size: 0-50k (max ~500k, never > 1M)
        "labels": Value("int32"),
        "token_type_ids": Value(
            "int8"
        ),  # binary mask; some (XLNetModel) use an additional token represented by a 2
    }
    if type is None and try_type is None:
        optimized_int_type = optimized_int_type_by_col.get(col, None)
    return TypedSequence.__init__(self, data, type=type, try_type=try_type, optimized_int_type=optimized_int_type)


class WrapperMTDataset(tud.Dataset):
    def __init__(self, dataset_dir_path: str | List[str]):
        """
        Args:
            dataset_dir_path (str | List[str]): The path or a list of the paths to text pretraining corpus.
        """
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        path_list = [path[:-4] for path in glob(os.path.join(dataset_dir_path, "*.idx"))]
        self.dataset_list = [(i, IndexedDataset(path)) for i, path in enumerate(path_list)]

        self.total = 0
        for i, d in enumerate(self.dataset_list):
            l = len(d[-1])
            self.dataset_list[i] += (l, ) # (i, data, len)
            self.total += l
        
    def __len__(self):
        return self.total
    
    def __getitem__(self, idx):
        if idx >= self.total:
            raise IndexError("Index out of range")
        # 전체 인덱스 idx에 대해 round-robin으로 dataset을 선택
        num_datasets = len(self.dataset_list)
        start_did = idx % num_datasets
        local_idx = idx // num_datasets

        # start_did 부터 시작해서 순환적으로 다음 dataset에서 local_idx 에 해당하는 아이템이 있는지 확인
        for offset in range(num_datasets):
            did = (start_did + offset) % num_datasets
            _, dataset, length = self.dataset_list[did]
            if local_idx < length:
                return {"input_ids": dataset[local_idx]}

        # 여기까지 왔다면 해당 idx에 대해 사용할 수 있는 데이터가 없는 경우
        raise IndexError(f"Index {idx} is out of range")


class WrapperMTIterableDataset(tud.IterableDataset):
    def __init__(self, 
                 dataset_dir_path: str | List[str], 
                 chunk: int = 4096
                 ):
        """
        Args:
            dataset_dir_path (str | List[str]): The path or a list of the paths to text pretraining corpus.
            max_sample (int): The maximum number of samples to use. If None, use all dataset.
        """
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        # Generate the list of datasets
        path_list = [path[:-4] for path in glob(os.path.join(dataset_dir_path, "*.idx"))]
        self.dataset_list = [(i, IndexedDataset(path)) for i, path in enumerate(path_list)]

        self.total = 0
        for i, d in enumerate(self.dataset_list):
            l = len(d[-1])
            self.dataset_list[i] += (l, ) # (i, data, len)
            self.total += l

        self.chunk = chunk

    def __len__(self):
        return self.total

    def __iter__(self):
        """
        Iterates over the remapped indices and yields data.
        """
        dataset_list = copy(self.dataset_list) # shallow copy
        idx = defaultdict(int)

        for _ in range(self.total):
            for i, (j, dataset, length) in enumerate(dataset_list):
                input_ids = dataset[idx[j]].tolist()
                yield {
                    "input_ids": input_ids
                }
                idx[j] = idx[j]+self.chunk if idx[j]+self.chunk < length else (idx[j]+1) % min(length, self.chunk)
                if idx[j]==0: dataset_list.pop(i) # pop inside iteration, safe

        # Optional: raise an error if iteration fails (should not occur normally)
        # raise StopIteration("Finished iterating over the dataset")


class RatioSplitDataset(tud.Dataset):
    def __init__(self, datasets, ratios=None):
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.lengths = [len(dataset) for dataset in datasets]
        self.total = sum(self.lengths)

        if ratios is None:
            self.ratios = [1] * self.num_datasets
        else:
            assert len(datasets) == len(ratios)
            self.ratios = ratios

        self.total_ratio = sum(self.ratios)
        self._len = int(self.total_ratio * max([self.lengths[i] / self.ratios[i] for i in range(len(datasets))]))

    def __len__(self):
        """
        가장 긴 데이터가 소진되면 1epoch로 계산되도록 되어있음.
        """
        return self._len

    def __getitem__(self, idx):
        """
        단일 인덱스로 2차원 데이터에 접근하는 함수

        각 데이터셋이 소진되면 다시 처음부터 순환적으로 접근

        idx를 통해 어떤 데이터의 어떤 항목에 접근할지 계산:
        - cycle_idx: 몇 번째 사이클인지 (sum(ratios)로 나눈 몫)
        - inner_idx: 현재 사이클 내에서의 인덱스 (sum(ratios)로 나눈 나머지)
        """
        if idx < 0:
            raise IndexError("음수 인덱스는 지원하지 않습니다")

        # idx를 사이클과 내부 인덱스로 분해
        cycle_idx = idx // self.total_ratio
        inner_idx = idx % self.total_ratio

        # inner_idx를 사용하여 접근할 데이터와 해당 데이터 내 인덱스 계산
        cumulative = 0

        for i, ratio in enumerate(self.ratios):
            if inner_idx < cumulative + ratio:
                data_idx = i  # 접근할 데이터 리스트의 인덱스
                item_idx = cycle_idx * ratio + (inner_idx - cumulative)  # 해당 데이터 내 항목 인덱스

                # 순환적 접근을 위해 모듈로 연산 사용
                dataset_length = len(self.datasets[data_idx])
                if dataset_length > 0:  # 빈 데이터셋 방지
                    item_idx = item_idx % dataset_length  # 데이터셋 길이로 나머지 연산하여 순환 접근
                    return self.datasets[data_idx][item_idx]
                else:
                    raise IndexError(f"데이터셋 {data_idx}이(가) 비어 있습니다")

            cumulative += ratio

        raise IndexError(f"인덱스 {idx}가 유효하지 않습니다")


if __name__ == "__main__":
    from datasets import Dataset

    hcx = Dataset.load_from_disk("/mnt/fr20tb/audiollm/datasets/hcx/hf_dataset")
    print(len(hcx))
    hcx = hcx.select(range(226715501, 226715501 + 16384))
    hcx_sampled = hcx.save_to_disk("/mnt/fr20tb/audiollm/datasets/testset/hcx_textonly", 
                                   max_shard_size="2048MB", num_proc=16)



# TESTIING1_SAVE
# if __name__ == "__main__":
#     from datasets import Dataset, IterableDataset

#     def convert_to_hf_iterable_dataset(iterable_dataset, max_row=2147483647):
#         def generator():
#             for i, item in iterable_dataset:
#                 yield item  # Each `item` is already in dictionary format
#                 if i >= min(len(iterable_dataset), max_row):
#                     break
#         return IterableDataset.from_generator(generator)

#     def convert_to_hf_dataset(idx_dataset, max_row=None):
#         def generator(): 
#             for i in range(min(len(idx_dataset), max_row)):
#                 yield idx_dataset[i]
#         return Dataset.from_generator(generator)

#     dataset_mt = convert_to_hf_dataset(
#         WrapperMTDataset(
#             dataset_dir_path="/mnt/fr20tb/audiollm/datasets/hcx",
#             max_row=226715500
#         )
#     )

#     dataset_mt.save_to_disk(dataset_path="/mnt/fr20tb/audiollm/datasets/hcx_sampled")

# TESTING2
# if __name__ == "__main__":
#     import time
#     from transformers import AutoTokenizer
#     from datasets import load_dataset, Dataset, IterableDataset, interleave_datasets

#     tokenizer = AutoTokenizer.from_pretrained(
#         "/mnt/ddn/audiollm/models/speechx-jessica-000")
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     tokenizer.truncation_side = "left"
#     tokenizer.padding_side = "right"

#     dataset_mt = convert_to_hf_iterable_dataset(
#         WrapperMTIterableDataset(
#             dataset_dir_path="/mnt/ddn/audiollm/datasets/hcx"))
#     # dataset_mt = dataset_mt.shuffle(buffer_size=4096)
    
#     # dataset_mt = convert_to_hf_iterable_dataset(
#     #     WrapperMTDataset(
#     #         dataset_dir_path="/mnt/ddn/audiollm/datasets/hcx"))

#     start = time.time()
#     for i, d in enumerate(dataset_mt):
#         if i>100000:
#             break
#     end = time.time()

#     print(f"{end - start:.5f} sec")

#     start = time.time()
#     for i, d in enumerate(dataset_mt):
#         if i>100000:
#             break
#     end = time.time()

#     print(f"{end - start:.5f} sec")


    # dataset_speechx = get_dataset(
    #     "/mnt/ddn/audiollm/datasets/train_db/units_word_aligned/241204/train/*.jsonl", 
    #     "/mnt/ddn/audiollm/datasets/.cache", 
    #     tokenizer)
    
    # dataset_mt = convert_to_hf_iterable_dataset([{'d': d} for d in [1,2,3,4,5,6,7,8,9]])
    # dataset_speechx = convert_to_hf_iterable_dataset([{'d': d} for d in [10,11,12,13]])

    # # 끝난 데이터는 처음부터, 안 끝난 데이터는 계속 이어서 yield.
    # dataset_combined = interleave_datasets(
    #     [dataset_mt, dataset_speechx], 
    #      probabilities=[0.5, 0.5], 
    #      seed=42, 
    #      stopping_strategy="all_exhausted")
    
    # for epoch in range(2):
    #     dataset_combined.set_epoch(epoch)
    #     for d in dataset_combined:
    #         print(d)
