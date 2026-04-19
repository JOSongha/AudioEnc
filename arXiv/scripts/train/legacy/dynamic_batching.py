"""
Dynamic Batching — arXiv 아카이브 모듈.

현재 파이프라인(train_pipeline_override.py)은 Sequence Packing을 사용하며
이 모듈은 사용되지 않음. arXiv 스크립트(train.py, train_debug.py, train_stage1.py 등)
의존성 보존 및 기술 비교 목적으로 보존.

참고: docs/dynamic_batching.md
"""

import io
import os
import pickle
import re

import torch
import torchaudio
from torch.utils.data import ConcatDataset, Subset

from dataset import (
    LibriSpeechDataset,
    MLSDataset,
    GigaSpeechDataset,
    VoxPopuliDataset,
)


# ──────────────────────────────────────────────────────────────────────────────
# 길이 계산 유틸리티 (DynamicBatchSampler 전처리용)
# ──────────────────────────────────────────────────────────────────────────────

def get_dataset_lengths(dataset, samples_per_token: float, max_text_len: int,
                        cache_dir: str = None, cache_tag: str = "") -> list:
    """
    LLM 토큰 수 기준 길이를 반환.
      token_len = audio_samples_16k / samples_per_token + max_text_len

    raw 오디오 샘플 수는 cache_dir에 캐싱되어 재사용됨.
    samples_per_token / max_text_len 이 바뀌어도 캐시를 다시 쓰지 않아도 됨
    (변환은 캐시 로드 후 런타임에 적용).
    cache_tag: 데이터셋 구성 식별자 (동일 len의 다른 구성 간 충돌 방지)
    """
    audio_lengths = _get_raw_lengths(dataset, cache_dir)
    return [int(l / samples_per_token) + max_text_len for l in audio_lengths]


def _get_raw_lengths(dataset, cache_dir: str = None) -> list:
    """오디오 샘플 수(16kHz 기준)를 반환. cache_dir 지정 시 캐싱."""
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"bucket_lengths_{len(dataset)}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                print(f"[DEBUG] Loaded raw lengths from {cache_path}. dataset: {type(dataset).__name__}, samples: {len(dataset)}")
                return pickle.load(f)

    lengths = _collect_lengths(dataset)

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            print(f"[DEBUG] Saving lengths to {cache_path}. dataset: {type(dataset).__name__}, samples: {len(dataset)}")
            pickle.dump(lengths, f)

    return lengths


def _collect_lengths(dataset) -> list:
    if isinstance(dataset, ConcatDataset):
        lengths = []
        for sub in dataset.datasets:
            lengths.extend(_collect_lengths(sub))
        print(f"[DEBUG] Collected lengths for ConcatDataset with {len(lengths)} samples (subsets {[type(s).__name__ for s in dataset.datasets]})")
        return lengths
    elif isinstance(dataset, Subset):
        parent_lengths = _collect_lengths(dataset.dataset)
        print(f"[DEBUG] Collecting lengths for Subset with {len(dataset)} samples (parent {len(parent_lengths)})...")
        return [parent_lengths[i] for i in dataset.indices]
    elif isinstance(dataset, LibriSpeechDataset):
        print(f"[DEBUG] Collecting lengths for LibriSpeechDataset with {len(dataset)} samples...")
        return _librispeech_lengths(dataset)
    elif isinstance(dataset, MLSDataset):
        print(f"[DEBUG] Collecting lengths for MLSDataset with {len(dataset)} samples...")
        return _mls_lengths(dataset)
    elif isinstance(dataset, GigaSpeechDataset):
        print(f"[DEBUG] Collecting lengths for GigaSpeechDataset with {len(dataset)} samples...")
        return _gigaspeech_lengths(dataset)
    elif isinstance(dataset, VoxPopuliDataset):
        print(f"[DEBUG] Collecting lengths for VoxPopuliDataset with {len(dataset)} samples...")
        return _voxpopuli_lengths(dataset)
    raise ValueError(f"Unknown dataset type: {type(dataset)}")


def _librispeech_lengths(dataset: "LibriSpeechDataset") -> list:
    """BytesIO + torchaudio.info로 FLAC 헤더만 읽어 길이 계산 (오디오 디코딩 없음).
    dataset.dataset은 이미 Audio(decode=False)로 캐스팅된 상태."""
    lengths = []
    for item in dataset.dataset:
        audio_bytes = item["audio"]["bytes"]
        info = torchaudio.info(io.BytesIO(audio_bytes))
        n = int(info.num_frames * dataset.target_sr / info.sample_rate)
        lengths.append(min(n, dataset.max_len))
    return lengths


def _voxpopuli_lengths(dataset: "VoxPopuliDataset") -> list:
    """transcript 글자 수 → 오디오 샘플 수로 변환 (오디오 디코딩 없음)."""
    selected = dataset.dataset.select(dataset.indices)
    chars_per_sec = 14.0
    texts = [
        (row.get("normalized_text") or row.get("raw_text") or "")
        for row in selected
    ]
    return [
        min(int(len(t) / chars_per_sec * dataset.target_sr), dataset.max_len)
        for t in texts
    ]


def _mls_lengths(dataset: "MLSDataset") -> list:
    """transcript 글자 수 → 오디오 샘플 수로 변환 (오디오 디코딩 없음).
    영어 평균 발화 속도 ~14자/초를 기준으로 LibriSpeech 길이와 동일한 스케일 유지."""
    selected = dataset.dataset.select(dataset.indices)
    chars_per_sec = 14.0
    return [
        min(int(len(t) / chars_per_sec * dataset.target_sr), dataset.max_len)
        for t in selected["transcript"]
    ]


def _gigaspeech_lengths(dataset: "GigaSpeechDataset") -> list:
    """transcript 글자 수 → 오디오 샘플 수로 변환 (오디오 디코딩 없음).
    MLS와 동일한 방식: 영어 평균 발화 속도 ~14자/초 기준."""
    selected = dataset.dataset.select(dataset.indices)
    chars_per_sec = 14.0
    return [
        min(int(len(re.sub(r"<[^>]+>", "", t).strip()) / chars_per_sec * dataset.target_sr),
            dataset.max_len)
        for t in selected["text"]
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Sampler 클래스
# ──────────────────────────────────────────────────────────────────────────────

class BucketBatchSampler(torch.utils.data.Sampler):
    """
    길이 기반 버킷 배치 샘플러.
    비슷한 길이의 샘플을 같은 배치로 묶어 패딩 낭비를 최소화.

    DDP 환경: num_replicas / rank 로 각 프로세스에 배치를 분배.
    set_epoch(epoch) 호출로 에폭마다 셔플 시드 변경.
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        bucket_size_multiplier: int = 100,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.lengths   = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank      = rank
        self.bucket_size = batch_size * bucket_size_multiplier
        self.drop_last = drop_last
        self.seed      = seed
        self.epoch     = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        sorted_idx = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        all_batches = []
        for start in range(0, len(sorted_idx), self.bucket_size):
            bucket = sorted_idx[start : start + self.bucket_size]
            perm   = torch.randperm(len(bucket), generator=g).tolist()
            bucket = [bucket[p] for p in perm]
            for b in range(0, len(bucket), self.batch_size):
                batch = bucket[b : b + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        perm        = torch.randperm(len(all_batches), generator=g).tolist()
        all_batches = [all_batches[p] for p in perm]

        # DDP: num_replicas 배수로 패딩 (앞 배치 반복) — truncate 없음
        remainder = len(all_batches) % self.num_replicas
        if remainder:
            all_batches += all_batches[: self.num_replicas - remainder]

        yield from all_batches[self.rank :: self.num_replicas]

    def __len__(self):
        total = (len(self.lengths) + self.batch_size - 1) // self.batch_size
        # DDP 패딩 후 각 rank가 받는 수
        remainder = total % self.num_replicas
        if remainder:
            total += self.num_replicas - remainder
        return total // self.num_replicas


class DynamicBatchSampler(torch.utils.data.Sampler):
    """
    token budget 기반 동적 배치 샘플러.

    배치 내 총 LLM 토큰 수(오디오 토큰 + 텍스트 토큰)가
    max_batch_tokens를 넘지 않도록 배치 크기를 동적으로 결정.

        짧은 샘플이 모인 배치 → batch_size 커짐
        긴 샘플이 모인 배치  → batch_size 줄어듦
        → VRAM 사용량이 에폭 내내 일정하게 유지됨

    안전성:
        - DDP 각 rank가 정확히 같은 수의 배치를 갖도록 앞 배치를 반복 패딩.
          (버리는 배치 없음, NCCL hang 방지)
        - 너무 긴 단일 샘플도 min_batch_size=1로 단독 처리.
    """

    def __init__(
        self,
        lengths,
        max_batch_tokens: int,
        min_batch_size: int = 1,
        num_replicas: int = 1,
        rank: int = 0,
        bucket_size_multiplier: int = 100,
        seed: int = 0,
    ):
        self.lengths           = lengths
        self.max_batch_tokens  = max_batch_tokens
        self.min_batch_size    = min_batch_size
        self.num_replicas      = num_replicas
        self.rank              = rank
        self.seed              = seed
        self.epoch             = 0
        max_len = max(lengths) if lengths else 1
        approx_min_bs = max(1, int(max_batch_tokens) // int(max_len))
        self.bucket_size = approx_min_bs * bucket_size_multiplier

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        sorted_idx = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        all_batches = []
        for start in range(0, len(sorted_idx), self.bucket_size):
            bucket = sorted_idx[start : start + self.bucket_size]
            perm   = torch.randperm(len(bucket), generator=g).tolist()
            bucket = [bucket[p] for p in perm]

            # 버킷 내 greedy 배치 구성
            cur_batch, cur_max = [], 0
            for idx in bucket:
                l       = self.lengths[idx]
                new_max = max(cur_max, l)
                if cur_batch and (len(cur_batch) + 1) * new_max > self.max_batch_tokens:
                    if len(cur_batch) >= self.min_batch_size:
                        all_batches.append(cur_batch)
                    cur_batch, cur_max = [idx], l
                else:
                    cur_batch.append(idx)
                    cur_max = new_max
            if len(cur_batch) >= self.min_batch_size:
                all_batches.append(cur_batch)

        perm        = torch.randperm(len(all_batches), generator=g).tolist()
        all_batches = [all_batches[p] for p in perm]

        # DDP: num_replicas 배수로 패딩 (앞 배치 반복) — truncate 없음
        remainder = len(all_batches) % self.num_replicas
        if remainder:
            all_batches += all_batches[: self.num_replicas - remainder]

        yield from all_batches[self.rank :: self.num_replicas]

    def __len__(self):
        if not hasattr(self, '_cached_len'):
            self._cached_len = self._count_batches()
        return self._cached_len

    def _count_batches(self) -> int:
        """실제 greedy packing 로직으로 배치 수를 정확히 계산 (epoch=0 기준, 캐싱)."""
        g = torch.Generator()
        g.manual_seed(self.seed)  # epoch 0

        sorted_idx = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        count = 0
        for start in range(0, len(sorted_idx), self.bucket_size):
            bucket = sorted_idx[start : start + self.bucket_size]
            perm   = torch.randperm(len(bucket), generator=g).tolist()
            bucket = [bucket[p] for p in perm]

            cur_count, cur_max = 0, 0
            for idx in bucket:
                l       = self.lengths[idx]
                new_max = max(cur_max, l)
                if cur_count > 0 and (cur_count + 1) * new_max > self.max_batch_tokens:
                    if cur_count >= self.min_batch_size:
                        count += 1
                    cur_count, cur_max = 1, l
                else:
                    cur_count += 1
                    cur_max = new_max
            if cur_count >= self.min_batch_size:
                count += 1

        remainder = count % self.num_replicas
        if remainder:
            count += self.num_replicas - remainder
        return count // self.num_replicas
