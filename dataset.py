import io
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, ConcatDataset


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
        n = min(num_samples, total)
        rng = random.Random(seed)
        self.indices = rng.sample(range(total), n)

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


def build_datasets(cfg: dict):
    """
    train: LibriSpeech 전체 (clean-100 + clean-360 + other-500, ~960h)
           + MLS English 샘플링 (~9000h)
    val:   LibriSpeech dev-clean
    """
    root        = cfg["data_path"]
    mls_root    = cfg.get("mls_data_path", root)
    max_len     = cfg["max_audio_len"]
    mls_samples = cfg.get("mls_num_samples", 900_000)

    train_dataset = ConcatDataset([
        LibriSpeechDataset(root=root, url="train-clean-100", max_len=max_len),
        LibriSpeechDataset(root=root, url="train-clean-360", max_len=max_len),
        LibriSpeechDataset(root=root, url="train-other-500", max_len=max_len),
        MLSDataset(cache_dir=mls_root, num_samples=mls_samples, max_len=max_len),
    ])
    val_dataset = LibriSpeechDataset(root=root, url="dev-clean", max_len=max_len)
    return train_dataset, val_dataset


def collate_fn_factory(tokenizer, max_text_len: int = 256):
    """
    batch: [(waveform, transcript), ...]
    반환: (audios_padded, audio_lengths, input_ids)
      - audios_padded: (B, T_max)  0-padded
      - audio_lengths: (B,)        실제 샘플 수
      - input_ids:     (B, L+1)    텍스트 토큰 + EOS
    """
    def collate_fn(batch):
        audios = [item[0] for item in batch]
        texts  = [item[1] for item in batch]

        audio_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
        audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)

        text_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_len,
            add_special_tokens=False,
        )
        input_ids = text_inputs.input_ids
        if tokenizer.eos_token_id is not None:
            eos = torch.full((input_ids.shape[0], 1), tokenizer.eos_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, eos], dim=1)

        return audios_padded, audio_lengths, input_ids

    return collate_fn
