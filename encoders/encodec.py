from typing import Optional, Tuple

import torch
import torchaudio.functional as AF
from transformers import EncodecModel

from .base import BaseAudioEncoder


class EnCodecEncoder(BaseAudioEncoder):
    """
    facebook/encodec-24khz — pre-RVQ continuous acoustic latent.

    - 입력: (B, T) @ 16kHz
    - 내부에서 24kHz로 리샘플 후 model.encoder() 직접 호출 (RVQ 없음)
    - 출력: (B, T_enc, 128) @ 75fps, 항상 frozen fp32

    V100 주의: cuDNN LSTM이 non-contiguous fp16 hidden state를 거부함.
    autocast 비활성화 + float().contiguous() 로 fp32 강제.
    """

    def __init__(self, cfg: dict, cache_dir: str):
        super().__init__()
        print("Loading EnCodec encoder (24kHz)...")
        self.encodec = EncodecModel.from_pretrained(
            cfg["model_id"], cache_dir=cache_dir
        )
        self.out_dim = cfg["out_dim"]
        self.tgt_sr  = cfg["tgt_sr"]
        self.hop     = cfg["hop"]

        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.encodec.eval()  # 항상 eval 유지
        return self

    def forward(
        self,
        audio_waveform: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = audio_waveform.device
        B = audio_waveform.shape[0]

        # 16kHz → 24kHz
        audio_tgt = AF.resample(audio_waveform.float(), self.src_sr, self.tgt_sr)

        if audio_lengths is not None:
            lengths_tgt = (audio_lengths.float() * self.tgt_sr / self.src_sr).long()
        else:
            lengths_tgt = torch.full((B,), audio_tgt.shape[1], dtype=torch.long, device=device)

        # (B, 1, T_24k) → model.encoder() → (B, 128, T_enc)
        audio_in = audio_tgt.unsqueeze(1).float().contiguous()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                feats = self.encodec.encoder(audio_in)  # (B, 128, T_enc)
        feats = feats.transpose(1, 2)  # (B, T_enc, 128)

        T_enc = feats.shape[1]
        lengths_enc = (lengths_tgt.float() / self.hop).ceil().long().clamp(max=T_enc)
        enc_mask = torch.arange(T_enc, device=device).unsqueeze(0) < lengths_enc.unsqueeze(1)

        return feats, enc_mask
