import bisect
import io
import os
import pickle
import random
import re
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, ConcatDataset, Subset

IGNORE_INDEX = -100


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech 단일 split 래퍼.
    - 출력: (waveform @ 16kHz mono, transcript 소문자)
    - max_len 초과분은 truncate
    """

    def __init__(self, root: str, url: str = "train-clean-100",
                 target_sr: int = 16000, max_len: int = 160000):
        self.target_sr = target_sr
        self.max_len   = max_len
        self.dataset   = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]

        if sample_rate != self.target_sr:
            waveform = torchaudio.transforms.Resample(sample_rate, self.target_sr)(waveform)

        # 다채널 → 모노
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (T,)

        if waveform.shape[0] > self.max_len:
            waveform = waveform[: self.max_len]

        return waveform, transcript.lower()


class MLSDataset(Dataset):
    """
    parler-tts/mls_eng_10k — MLS English 10k시간 (HuggingFace datasets, decode=False).
    - audio bytes를 io.BytesIO + torchaudio.load로 디코딩 (torchcodec 불필요)
    - num_samples 개만큼 랜덤 샘플링 (재현성을 위해 seed 고정)
    - 출력: (waveform @ 16kHz mono, transcript 소문자)
    """

    def __init__(self, cache_dir: str, num_samples: int = 4_050_000,
                 target_sr: int = 16000, max_len: int = 160000,
                 seed: int = 42):
        from datasets import load_dataset, Audio
        self.target_sr = target_sr
        self.max_len   = max_len
        ds = load_dataset(
            "parler-tts/mls_eng_10k",
            split="train",
            cache_dir=cache_dir,
        )
        self.dataset = ds.cast_column("audio", Audio(decode=False))

        total = len(self.dataset)
        if num_samples is None or num_samples >= total:
            self.indices = list(range(total))
        else:
            rng = random.Random(seed)
            self.indices = rng.sample(range(total), num_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        audio_bytes = item["audio"]["bytes"]
        transcript  = item["transcript"]

        try:
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        except Exception as e:
            print("ERROR SAMPLE:", idx)
            print(len(audio_bytes))
            print(audio_bytes[:20])
            raise e

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (T,)

        if sample_rate != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sr)

        if waveform.shape[0] > self.max_len:
            waveform = waveform[: self.max_len]

        return waveform, transcript.lower()


class VoxPopuliDataset(Dataset):
    """
    facebook/voxpopuli — VoxPopuli (HuggingFace datasets).
    language: "en" (기본값) 등 언어 코드
    split: "train" / "validation" / "test"
    num_samples: None이면 전체, 정수면 랜덤 서브샘플 (seed 고정)
    출력: (waveform @ 16kHz mono, normalized_text 소문자)
    """

    def __init__(self, cache_dir: str, language: str = "en",
                 split: str = "train", num_samples: int = None,
                 target_sr: int = 16000, max_len: int = 160000,
                 seed: int = 42):
        from datasets import load_dataset
        self.target_sr = target_sr
        self.max_len   = max_len

        ds = load_dataset(
            "facebook/voxpopuli",
            language,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        total = len(ds)
        if num_samples is None or num_samples >= total:
            self.indices = list(range(total))
        else:
            rng = random.Random(seed)
            self.indices = rng.sample(range(total), num_samples)
        self.dataset = ds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item        = self.dataset[self.indices[idx]]
        audio       = item["audio"]
        waveform    = torch.tensor(audio["array"], dtype=torch.float32)
        sample_rate = audio["sampling_rate"]

        if waveform.dim() > 1:
            waveform = waveform.mean(0)

        if sample_rate != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sr)

        if waveform.shape[0] > self.max_len:
            waveform = waveform[: self.max_len]

        transcript = item.get("normalized_text") or item.get("raw_text") or ""
        return waveform, transcript.lower()


class GigaSpeechDataset(Dataset):
    """
    speechcolab/gigaspeech — GigaSpeech English (HuggingFace datasets).
    subset: "xs"(10h) / "s"(250h) / "m"(1000h) / "l"(2500h) / "xl"(10000h)
    num_samples: None이면 전체, 정수면 랜덤 서브샘플 (seed 고정)
    """

    def __init__(self, cache_dir: str, subset: str = "l",
                 num_samples: int = None,
                 target_sr: int = 16000, max_len: int = 160000,
                 seed: int = 42):
        from datasets import load_dataset
        self.target_sr = target_sr
        self.max_len   = max_len

        ds = load_dataset(
            "speechcolab/gigaspeech",
            subset,
            split="train",
            # cache_dir=cache_dir,
            cache_dir="/mnt/tmp/cache/hf/datasets",
            trust_remote_code=True,
        )
        total = len(ds)
        if num_samples is None or num_samples >= total:
            self.indices = list(range(total))
        else:
            rng = random.Random(seed)
            self.indices = rng.sample(range(total), num_samples)
        self.dataset = ds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item      = self.dataset[self.indices[idx]]
        audio     = item["audio"]
        waveform  = torch.tensor(audio["array"], dtype=torch.float32)
        sample_rate = audio["sampling_rate"]

        if waveform.dim() > 1:
            waveform = waveform.mean(0)

        if sample_rate != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sr)

        if waveform.shape[0] > self.max_len:
            waveform = waveform[: self.max_len]

        transcript = item["text"].lower()
        # GigaSpeech 태그 제거: <COMMA>, <PERIOD>, <NOISE> 등
        import re
        transcript = re.sub(r"<[^>]+>", "", transcript).strip()

        return waveform, transcript


_LS_URL = {
    "ls100": "train-clean-100",
    "ls360": "train-clean-360",
    "ls500": "train-other-500",
}


def build_datasets(cfg: dict):
    """
    train: cfg["datasets"] 목록에 지정된 데이터셋만 사용.
           datasets 예: ["ls100", "ls360", "ls500", "mls", "gs"]
           *_num_samples 로 각 양을 제한.
    val:   LibriSpeech dev-clean
    """
    root        = cfg["data_path"]
    mls_root    = cfg.get("mls_data_path", root)
    max_len     = cfg["max_audio_len"]
    datasets    = cfg.get("datasets", ["ls100", "ls360", "ls500", "mls"])

    parts = []

    # LibriSpeech splits
    ls_splits = [name for name in ["ls100", "ls360", "ls500"] if name in datasets]
    if ls_splits:
        ls_parts = [LibriSpeechDataset(root=root, url=_LS_URL[name], max_len=max_len)
                    for name in ls_splits]
        librispeech = ConcatDataset(ls_parts) if len(ls_parts) > 1 else ls_parts[0]
        ls_num = cfg.get("librispeech_num_samples", None)
        if ls_num and ls_num < len(librispeech):
            indices = random.Random(42).sample(range(len(librispeech)), ls_num)
            librispeech = Subset(librispeech, sorted(indices))
        parts.append(librispeech)
        print (f"Loaded LibriSpeech splits {ls_splits}, total samples: {len(librispeech)}")

    # MLS
    if "mls" in datasets:
        parts.append(MLSDataset(
            cache_dir=mls_root,
            num_samples=cfg.get("mls_num_samples", None),
            max_len=max_len,
        ))
        print (f"Loaded MLS, total samples: {len(parts[-1])}")

    # GigaSpeech
    if "gs" in datasets:
        parts.append(GigaSpeechDataset(
            cache_dir=mls_root,
            subset=cfg.get("gs_subset", "xl"),
            num_samples=cfg.get("gs_num_samples", None),
            max_len=max_len,
        ))
        print (f"Loaded GigaSpeech subset {cfg.get('gs_subset', 'l')}, total samples: {len(parts[-1])}")

    # VoxPopuli
    if "vp" in datasets:
        parts.append(VoxPopuliDataset(
            cache_dir=mls_root,
            language=cfg.get("vp_language", "en"),
            num_samples=cfg.get("vp_num_samples", None),
            max_len=max_len,
        ))
        print (f"Loaded VoxPopuli language {cfg.get('vp_language', 'en')}, total samples: {len(parts[-1])}")

    if not parts:
        raise ValueError(f"datasets에 유효한 항목이 없습니다: {datasets}")

    train_dataset = ConcatDataset(parts) if len(parts) > 1 else parts[0]
    val_dataset   = LibriSpeechDataset(root=root, url="dev-clean", max_len=max_len)
    return train_dataset, val_dataset


def build_train_eval_dataset(cfg, n_per_ds: int = 10):
    """
    WER 평가용 train subset.
    cfg["datasets"]에 있는 각 데이터셋에서 앞 n_per_ds개씩 가져와 concat.
    """
    root     = cfg["data_path"]
    mls_root = cfg.get("mls_data_path", root)
    max_len  = cfg["max_audio_len"]
    datasets = cfg.get("datasets", ["ls100", "ls360", "ls500", "mls"])

    parts = []
    for name in ["ls100", "ls360", "ls500"]:
        if name in datasets:
            ds = LibriSpeechDataset(root=root, url=_LS_URL[name], max_len=max_len)
            parts.append(Subset(ds, range(min(n_per_ds, len(ds)))))

    if "mls" in datasets:
        ds = MLSDataset(cache_dir=mls_root, max_len=max_len)
        parts.append(Subset(ds, range(min(n_per_ds, len(ds)))))

    if "gs" in datasets:
        ds = GigaSpeechDataset(
            cache_dir=mls_root,
            subset=cfg.get("gs_subset", "l"),
            max_len=max_len,
        )
        parts.append(Subset(ds, range(min(n_per_ds, len(ds)))))

    if "vp" in datasets:
        ds = VoxPopuliDataset(
            cache_dir=mls_root,
            language=cfg.get("vp_language", "en"),
            max_len=max_len,
        )
        parts.append(Subset(ds, range(min(n_per_ds, len(ds)))))

    if not parts:
        raise ValueError(f"datasets에 유효한 항목이 없습니다: {datasets}")
    return ConcatDataset(parts) if len(parts) > 1 else parts[0]


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
    # audio_lengths = _get_raw_lengths(dataset, cache_dir, cache_tag=cache_tag)
    # return [int(l / samples_per_token) + max_text_len for l in audio_lengths]
    audio_lengths = _get_raw_lengths(dataset, cache_dir)
    return [int(l / samples_per_token) + max_text_len for l in audio_lengths]


def _get_raw_lengths(dataset, cache_dir: str = None) -> list:
    """오디오 샘플 수(16kHz 기준)를 반환. cache_dir 지정 시 캐싱."""
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"bucket_lengths_{len(dataset)}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                print (f"[DEBUG] Loaded raw lengths from {cache_path}. dataset: {type(dataset).__name__}, samples: {len(dataset)}")
                return pickle.load(f)

    lengths = _collect_lengths(dataset)

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            print (f"[DEBUG] Saving lengths to {cache_path}. dataset: {type(dataset).__name__}, samples: {len(dataset)}")
            pickle.dump(lengths, f)

    return lengths


def _collect_lengths(dataset) -> list:
    if isinstance(dataset, ConcatDataset):
        lengths = []
        for sub in dataset.datasets:
            lengths.extend(_collect_lengths(sub))
        print (f"[DEBUG] Collected lengths for ConcatDataset with {len(lengths)} samples (subsets {[type(s).__name__ for s in dataset.datasets]})")
        return lengths
    elif isinstance(dataset, Subset):
        parent_lengths = _collect_lengths(dataset.dataset)
        print (f"[DEBUG] Collecting lengths for Subset with {len(dataset)} samples (parent {len(parent_lengths)})...")
        return [parent_lengths[i] for i in dataset.indices]
    elif isinstance(dataset, LibriSpeechDataset):
        print (f"[DEBUG] Collecting lengths for LibriSpeechDataset with {len(dataset)} samples...")
        return _librispeech_lengths(dataset)
    elif isinstance(dataset, MLSDataset):
        print (f"[DEBUG] Collecting lengths for MLSDataset with {len(dataset)} samples...")
        return _mls_lengths(dataset)
    elif isinstance(dataset, GigaSpeechDataset):
        print (f"[DEBUG] Collecting lengths for GigaSpeechDataset with {len(dataset)} samples...")
        return _gigaspeech_lengths(dataset)
    elif isinstance(dataset, VoxPopuliDataset):
        print (f"[DEBUG] Collecting lengths for VoxPopuliDataset with {len(dataset)} samples...")
        return _voxpopuli_lengths(dataset)
    elif isinstance(dataset, GigaSpeechDataset):
        return _gigaspeech_lengths(dataset)
    raise ValueError(f"Unknown dataset type: {type(dataset)}")


def _librispeech_lengths(dataset: "LibriSpeechDataset") -> list:
    """torchaudio.info()로 FLAC 헤더만 읽어 길이 계산 (오디오 디코딩 없음)."""
    inner = dataset.dataset  # torchaudio.datasets.LIBRISPEECH
    base  = inner._path      # .../LibriSpeech/train-clean-100 등
    lengths = []
    for fileid in inner._walker:
        speaker_id, chapter_id = fileid.split("-")[:2]
        path = os.path.join(base, speaker_id, chapter_id, f"{fileid}.flac")
        info = torchaudio.info(path)
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
    import re
    return [
        min(int(len(re.sub(r"<[^>]+>", "", t).strip()) / chars_per_sec * dataset.target_sr),
            dataset.max_len)
        for t in selected["text"]
    ]


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


def collate_fn_factory(tokenizer, max_text_len: int = 256, eos_token_id: int = None, eos_before_pad: bool = False):
    """
    batch: [(waveform, transcript), ...]
    반환: (audios_padded, audio_lengths, input_ids)
      - audios_padded: (B, T_max)  0-padded
      - audio_lengths: (B,)        실제 샘플 수
      - input_ids:     (B, L+1)    텍스트 토큰 + EOS
    eos_token_id: None이면 tokenizer.eos_token_id 사용 (기본값).
                  --debug n 시 model.asr_eos_token_id를 전달.
    eos_before_pad: True  → [tok1, tok2, EOS, PAD, PAD]  (--debug o)
                    False → [tok1, tok2, PAD, PAD, EOS]  (기본)
    """
    _eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    _pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        audios = [item[0] for item in batch]
        texts  = [item[1] for item in batch]

        audio_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
        audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)

        if eos_before_pad and _eos_id is not None:
            seqs = []
            for t in texts:
                ids = tokenizer(
                    t,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_text_len - 1,
                    add_special_tokens=False,
                ).input_ids[0]
                ids = torch.cat([ids, torch.tensor([_eos_id], dtype=torch.long)])
                seqs.append(ids)
            max_len = max(s.shape[0] for s in seqs)
            input_ids = torch.full((len(seqs), max_len), _pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, :s.shape[0]] = s
        else:
            text_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_text_len,
                add_special_tokens=False,
            )
            input_ids = text_inputs.input_ids
            if _eos_id is not None:
                eos = torch.full((input_ids.shape[0], 1), _eos_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, eos], dim=1)

        return audios_padded, audio_lengths, input_ids

    return collate_fn


# ==========================================
# Sequence Packing 파이프라인
# ==========================================

def build_packed_processor(tokenizer, cfg: dict):
    """
    (waveform: Tensor[T], transcript: str) → dict:
      input_ids:      list[int]  [p1] + [audio_pad × t_audio] + [p2] + [text] + [EOS]
      labels:         list[int]  [IGNORE × (p1+t_audio+p2)] + [text] + [EOS]
      audio_features: Tensor[T]  raw waveform (16kHz)
      audio_lengths:  int        t_audio (audio token 수, projector 출력 기준)

    모델 forward 시 audio_pad 위치를 encoder+projector 출력으로 교체.
    t_audio = int(num_samples / samples_per_token) — config와 동일한 공식.
    """
    audio_pad_id      = cfg["audio_pad_token_id"]
    samples_per_token = cfg["samples_per_token"]
    max_text_len      = cfg["max_text_len"]
    llm_type          = cfg.get("llm_type", "base")

    if llm_type == "instruct":
        p1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        p2 = "\nTranscribe the audio to text.<|im_end|>\n<|im_start|>assistant\n"
    else:
        p1 = "Audio:\n"
        p2 = "\nTranscript:\n"

    p1_ids = tokenizer.encode(p1, add_special_tokens=False)
    p2_ids = tokenizer.encode(p2, add_special_tokens=False)

    def process(waveform: torch.Tensor, transcript: str) -> dict:
        num_samples = waveform.shape[0]
        t_audio     = max(1, int(num_samples / samples_per_token))

        text_ids = tokenizer.encode(
            transcript, add_special_tokens=False,
            truncation=True, max_length=max_text_len - 1,
        )
        if tokenizer.eos_token_id is not None:
            text_ids = text_ids + [tokenizer.eos_token_id]

        input_ids = p1_ids + [audio_pad_id] * t_audio + p2_ids + text_ids
        labels    = [IGNORE_INDEX] * (len(p1_ids) + t_audio + len(p2_ids)) + text_ids

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "audio_features": waveform,
            "audio_lengths":  t_audio,
        }

    return process


def _build_packs(token_lengths: list, cutoff_len: int, seed: int = 42) -> list:
    """
    greedy knapsack packing: cutoff_len 내에 최대한 많은 샘플을 이어붙임.
    bisect로 O(log n) 탐색. cutoff_len 초과 샘플은 드롭.
    반환: [[orig_idx, ...], [orig_idx, ...], ...]
    """
    rng = random.Random(seed)

    indexed = sorted(
        [(l, i) for i, l in enumerate(token_lengths) if l <= cutoff_len],
        key=lambda x: x[0],
    )
    sorted_lengths = [l for l, _ in indexed]
    sorted_indices = [i for _, i in indexed]

    packs = []
    while sorted_lengths:
        pack      = []
        remaining = cutoff_len
        while True:
            pos = bisect.bisect_right(sorted_lengths, remaining) - 1
            if pos < 0:
                break
            remaining -= sorted_lengths.pop(pos)
            pack.append(sorted_indices.pop(pos))
        if pack:
            packs.append(pack)

    rng.shuffle(packs)
    return packs


class PackedDataset(Dataset):
    """
    기존 (waveform, transcript) Dataset을 greedy knapsack sequence packing으로 변환.

    초기화 시 token_lengths로 packing 계획을 수립 (오디오 미로드),
    __getitem__ 에서 실제 샘플을 로드·처리·이어붙여 반환.

    출력:
      input_ids:      Tensor[cutoff_len]  audio_pad placeholder 포함
      labels:         Tensor[cutoff_len]  audio/prompt 위치 IGNORE_INDEX
      attention_mask: Tensor[cutoff_len]  0=pad, 1=sample1, 2=sample2, ...
      audio_features: list[Tensor[S_k]]  각 오디오 waveform
      audio_lengths:  list[int]           각 오디오 token 수 (t_audio)
    """

    def __init__(
        self,
        source_dataset,
        processor_fn,
        token_lengths: list,
        cutoff_len: int,
        pad_token_id: int,
        seed: int = 42,
    ):
        self.source       = source_dataset
        self.processor    = processor_fn
        self.cutoff_len   = cutoff_len
        self.pad_token_id = pad_token_id
        self.packs        = _build_packs(token_lengths, cutoff_len, seed)

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, idx: int) -> dict:
        orig_indices          = self.packs[idx]
        packed_input_ids      = []
        packed_labels         = []
        packed_attention_mask = []
        packed_audio_features = []
        packed_audio_lengths  = []

        for seq_idx, orig_idx in enumerate(orig_indices):
            waveform, transcript = self.source[orig_idx]
            item = self.processor(waveform, transcript)

            packed_input_ids.extend(item["input_ids"])
            packed_labels.extend(item["labels"])
            packed_attention_mask.extend([seq_idx + 1] * len(item["input_ids"]))
            packed_audio_features.append(item["audio_features"])
            packed_audio_lengths.append(item["audio_lengths"])

        # cutoff_len까지 패딩
        pad_len = self.cutoff_len - len(packed_input_ids)
        if pad_len > 0:
            packed_input_ids.extend([self.pad_token_id] * pad_len)
            packed_labels.extend([IGNORE_INDEX] * pad_len)
            packed_attention_mask.extend([0] * pad_len)

        return {
            "input_ids":      torch.tensor(packed_input_ids[:self.cutoff_len], dtype=torch.long),
            "labels":         torch.tensor(packed_labels[:self.cutoff_len], dtype=torch.long),
            "attention_mask": torch.tensor(packed_attention_mask[:self.cutoff_len], dtype=torch.long), # 샘플 인덱스 (sample_idx+1, pad=0) 
            "audio_features": packed_audio_features, # list of waveforms (샘플별 개별 tensor)
            "audio_lengths":  packed_audio_lengths,  # list of t_audio
        }


def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    neat_packing attention_mask (B, T, values=sample_idx) →
    position_ids (B, T): 샘플마다 0부터 리셋, pad=0.
    FA2 varlen 경계 인식에 사용.
    """
    B, T = attention_mask.shape
    position_ids = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        cnt, prev = 0, -1
        for t in range(T):
            grp = int(attention_mask[b, t])
            if grp == 0:
                cnt, prev = 0, 0
            else:
                if grp != prev:
                    cnt, prev = 0, grp
                position_ids[b, t] = cnt
                cnt += 1
    return position_ids


def _make_block_causal_mask(
    attention_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """
    neat_packing attention_mask (B, T, values=sample_idx) →
    4D block-diagonal causal additive mask (B, 1, T, T).

    같은 샘플 내 + causal 방향만 attend. pad(0) 위치는 완전 차단.
    eager / sdpa attention에 직접 전달 가능.
    """
    B, T   = attention_mask.shape
    device = attention_mask.device

    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    same   = (
        (attention_mask.unsqueeze(2) == attention_mask.unsqueeze(1))
        & (attention_mask.unsqueeze(2) != 0)
        & (attention_mask.unsqueeze(1) != 0)
    )
    mask_bool = causal.unsqueeze(0) & same                            # (B, T, T)

    additive = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    additive.masked_fill_(~mask_bool.unsqueeze(1), float("-inf"))
    return additive


class PackedCollator:
    """
    PackedDataset 출력을 배치로 묶어 model._forward_packed() 입력 형태로 변환.

    flash_attention_2:
      - 패딩 제거 → input_ids/labels: (1, sum_nonpad)
      - position_ids: 샘플마다 0부터 리셋 (FA2 varlen 경계 인식)

    eager / sdpa:
      - input_ids/labels: (B, cutoff_len)
      - attention_mask: 4D block-diagonal causal mask (B, 1, T, T)
    """

    def __init__(
        self,
        pad_token_id: int,
        attn_implementation: str = "eager",
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        self.pad_token_id        = pad_token_id
        self.attn_implementation = attn_implementation
        self.compute_dtype       = compute_dtype

    def __call__(self, features: list) -> dict:
        input_ids      = torch.stack([f["input_ids"]      for f in features])  # (B, T)
        labels         = torch.stack([f["labels"]         for f in features])  # (B, T)
        attention_mask = torch.stack([f["attention_mask"] for f in features])  # (B, T)

        # 배치 내 모든 오디오를 flatten + zero-pad → (N_audio, 1, S_max)
        all_wavs, all_t_audio = [], []
        for f in features:
            for wav in f["audio_features"]:
                all_wavs.append(
                    wav if isinstance(wav, torch.Tensor)
                    else torch.tensor(wav, dtype=torch.float32)
                )
            all_t_audio.extend(f["audio_lengths"])

        if all_wavs:
            max_s = max(w.shape[0] for w in all_wavs)
            audio_features = torch.stack([
                torch.nn.functional.pad(w, (0, max_s - w.shape[0])) for w in all_wavs
            ]).unsqueeze(1)  # (N, 1, S_max)
        else:
            audio_features = torch.zeros(0, 1, 0, dtype=torch.float32)
        audio_lengths = torch.tensor(all_t_audio, dtype=torch.long)

        result = {
            "audio_features": audio_features,
            "audio_lengths": audio_lengths,
            "num_items_in_batch": int((labels != -100).sum()),
        }

        if self.attn_implementation == "flash_attention_2":
            non_pad             = attention_mask != 0
            result["input_ids"] = input_ids[non_pad].unsqueeze(0)    # (1, sum_nonpad)
            result["labels"]    = labels[non_pad].unsqueeze(0)
            position_ids        = _build_position_ids(attention_mask)
            result["position_ids"] = position_ids[non_pad].unsqueeze(0)
        else:
            result["input_ids"]      = input_ids
            result["labels"]         = labels
            result["attention_mask"] = _make_block_causal_mask(
                attention_mask, self.compute_dtype
            )

        return result


def build_packed_datasets(cfg: dict, tokenizer, is_main_process: bool = False) -> tuple:
    """
    Sequence packing 파이프라인용 데이터셋 빌드.

    train: PackedDataset (cfg["datasets"] 기준, greedy knapsack 패킹)
    val:   LibriSpeech dev-clean (패킹 없음 — 기존 collate_fn_factory 사용)
    """
    train_base, val_dataset = build_datasets(cfg)

    spt       = cfg["samples_per_token"]
    mt        = cfg["max_text_len"]
    cache_dir = cfg["model_cache_dir"]
    cutoff    = cfg["packing_cutoff_len"]

    if is_main_process:
        print("Computing token lengths for sequence packing...")
    token_lengths = get_dataset_lengths(train_base, spt, mt, cache_dir=cache_dir)

    processor    = build_packed_processor(tokenizer, cfg)
    train_packed = PackedDataset(
        source_dataset=train_base,
        processor_fn=processor,
        token_lengths=token_lengths,
        cutoff_len=cutoff,
        pad_token_id=tokenizer.pad_token_id,
    )
    return train_packed, val_dataset


def collate_fn_factory_eos_first(tokenizer, max_text_len: int = 256):
    """
    EOS-first collate: truncate to max_text_len-1, append EOS, then right-pad.
    결과: [tok1, tok2, tok3, EOS, PAD, PAD]
    → loss가 tok3 위치에서 EOS를 예측 (추론과 일치)
    """
    def collate_fn(batch):
        audios = [item[0] for item in batch]
        texts  = [item[1] for item in batch]

        audio_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
        audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)

        text_inputs = tokenizer(
            texts,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=max_text_len - 1,
            add_special_tokens=False,
        )
        sequences = []
        for ids in text_inputs["input_ids"]:
            seq = torch.tensor(ids + [tokenizer.eos_token_id], dtype=torch.long)
            sequences.append(seq)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        return audios_padded, audio_lengths, input_ids

    return collate_fn
