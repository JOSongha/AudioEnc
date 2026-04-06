import io
import os
import pickle
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, ConcatDataset, Subset


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

        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))  # (C, T)

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

    # MLS
    if "mls" in datasets:
        parts.append(MLSDataset(
            cache_dir=mls_root,
            num_samples=cfg.get("mls_num_samples", None),
            max_len=max_len,
        ))

    # GigaSpeech
    if "gs" in datasets:
        parts.append(GigaSpeechDataset(
            cache_dir=mls_root,
            subset=cfg.get("gs_subset", "l"),
            num_samples=cfg.get("gs_num_samples", None),
            max_len=max_len,
        ))

    # VoxPopuli
    if "vp" in datasets:
        parts.append(VoxPopuliDataset(
            cache_dir=mls_root,
            language=cfg.get("vp_language", "en"),
            num_samples=cfg.get("vp_num_samples", None),
            max_len=max_len,
        ))

    if not parts:
        raise ValueError(f"datasets에 유효한 항목이 없습니다: {datasets}")

    train_dataset = ConcatDataset(parts) if len(parts) > 1 else parts[0]
    val_dataset   = LibriSpeechDataset(root=root, url="dev-clean", max_len=max_len)
    return train_dataset, val_dataset


def get_dataset_lengths(dataset, samples_per_token: float, max_text_len: int,
                        cache_dir: str = None) -> list:
    """
    LLM 토큰 수 기준 길이를 반환.
      token_len = audio_samples_16k / samples_per_token + max_text_len

    raw 오디오 샘플 수는 cache_dir에 캐싱되어 재사용됨.
    samples_per_token / max_text_len 이 바뀌어도 캐시를 다시 쓰지 않아도 됨
    (변환은 캐시 로드 후 런타임에 적용).
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
                return pickle.load(f)

    lengths = _collect_lengths(dataset)

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(lengths, f)

    return lengths


def _collect_lengths(dataset) -> list:
    if isinstance(dataset, ConcatDataset):
        lengths = []
        for sub in dataset.datasets:
            lengths.extend(_collect_lengths(sub))
        return lengths
    elif isinstance(dataset, Subset):
        parent_lengths = _collect_lengths(dataset.dataset)
        return [parent_lengths[i] for i in dataset.indices]
    elif isinstance(dataset, LibriSpeechDataset):
        return _librispeech_lengths(dataset)
    elif isinstance(dataset, MLSDataset):
        return _mls_lengths(dataset)
    elif isinstance(dataset, VoxPopuliDataset):
        return _voxpopuli_lengths(dataset)
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
