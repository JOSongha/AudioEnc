from typing import Optional, Tuple

import torch
import torchaudio.functional as AF

from .base import BaseAudioEncoder


class MimiAcousticEncoder(BaseAudioEncoder):
    """
    kyutai/mimi — acoustic encoder만 사용 (encoder_transformer 없음).

    파이프라인:
        16kHz → 24kHz resample → mimi.encoder() → (B, T_enc, 512)

    출력 특성:
        - 저수준 acoustic 피처 (스펙트럼 패턴, 운율 등)
        - 25fps (hop=960 @ 24kHz)
        - out_dim=512

    MimiSemanticEncoder와의 차이:
        - encoder_transformer를 통과하지 않음
        - semantic 정보 없이 acoustic 표면 정보만 포함
        - 일반적으로 ASR에서 semantic보다 성능이 낮을 것으로 예상
    """

    def __init__(self, cfg: dict, cache_dir: str):
        super().__init__()
        try:
            from transformers import MimiModel
        except ImportError:
            raise ImportError("transformers >= 4.45 필요. pip install --upgrade transformers")

        print("Loading Mimi encoder — acoustic only (no encoder_transformer)...")
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
                feats = self.mimi.encoder(audio_in)  # (B, 512, T_enc)

        if not isinstance(feats, torch.Tensor):
            feats = feats.last_hidden_state

        # (B, C, T) → (B, T, C)
        # shape[1] == out_dim 이고 T != C 인 경우만 transpose (ambiguous square case 방어)
        if feats.ndim == 3 and feats.shape[1] == self.out_dim and feats.shape[1] != feats.shape[2]:
            feats = feats.transpose(1, 2)

        T_enc = feats.shape[1]
        lengths_enc = (lengths_tgt.float() / self.hop).ceil().long().clamp(max=T_enc)
        enc_mask = torch.arange(T_enc, device=device).unsqueeze(0) < lengths_enc.unsqueeze(1)

        return feats, enc_mask
