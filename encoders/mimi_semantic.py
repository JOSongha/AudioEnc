from typing import Optional, Tuple

import torch
import torchaudio.functional as AF

from .base import BaseAudioEncoder


class MimiSemanticEncoder(BaseAudioEncoder):
    """
    kyutai/mimi — acoustic encoder + encoder_transformer (semantic).

    파이프라인:
        16kHz → 24kHz resample
        → mimi.encoder()              (B, 512, T_enc)   acoustic
        → mimi.encoder_transformer()  (B, T_enc, 512)   semantic ← 이게 핵심

    출력 특성:
        - 고수준 semantic 피처 (언어적 내용, 음소 등)
        - 25fps (hop=960 @ 24kHz)  ← encoder_transformer가 fps를 바꾸지 않음
        - out_dim=512
        - q_ming.py의 MimiEncoder와 동일한 동작

    MimiAcousticEncoder와의 차이:
        - encoder_transformer를 통과해 semantic 표현으로 변환
        - ASR에서 acoustic보다 높은 성능 기대
        - 파라미터 수가 더 많아 메모리 소비 증가
    """

    def __init__(self, cfg: dict, cache_dir: str):
        super().__init__()
        try:
            from transformers import MimiModel
        except ImportError:
            raise ImportError("transformers >= 4.45 필요. pip install --upgrade transformers")

        print("Loading Mimi encoder — acoustic + semantic (encoder_transformer)...")
        self.mimi    = MimiModel.from_pretrained(cfg["model_id"], cache_dir=cache_dir)
        self.out_dim = cfg["out_dim"]   # 512
        self.tgt_sr  = cfg["tgt_sr"]   # 24000
        self.hop     = cfg["hop"]       # 960 (25fps)

        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.mimi.eval()
        return self

    def forward(
        self,
        audio_waveform: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = audio_waveform.device
        B = audio_waveform.shape[0]

        audio_tgt = AF.resample(audio_waveform.float(), self.src_sr, self.tgt_sr)

        if audio_lengths is not None:
            lengths_tgt = (audio_lengths.float() * self.tgt_sr / self.src_sr).long()
        else:
            lengths_tgt = torch.full((B,), audio_tgt.shape[1], dtype=torch.long, device=device)

        audio_in = audio_tgt.unsqueeze(1).float().contiguous()  # (B, 1, T_24k)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                # Step 1: acoustic encoder → (B, 512, T_enc)
                enc = self.mimi.encoder(audio_in)
                if not isinstance(enc, torch.Tensor):
                    enc = enc.last_hidden_state  # ModelOutput 대비

                # (B, C, T) → (B, T, C) for transformer input
                # shape[1] == out_dim 이고 T != C 인 경우만 transpose
                if enc.ndim == 3 and enc.shape[1] == self.out_dim and enc.shape[1] != enc.shape[2]:
                    enc = enc.transpose(1, 2)

                # Step 2: semantic transformer → (B, T_enc, 512)
                sem = self.mimi.encoder_transformer(enc)
                feats = sem.last_hidden_state  # (B, T_enc, 512)

        T_enc = feats.shape[1]
        lengths_enc = (lengths_tgt.float() / self.hop).ceil().long().clamp(max=T_enc)
        enc_mask = torch.arange(T_enc, device=device).unsqueeze(0) < lengths_enc.unsqueeze(1)

        return feats, enc_mask
