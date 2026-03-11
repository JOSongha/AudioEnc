from typing import Optional, Tuple

import torch
import torchaudio.functional as AF

from .base import BaseAudioEncoder


class DACEncoder(BaseAudioEncoder):
    """
    descript-audio-codec 44kHz — pre-RVQ continuous acoustic latent.

    - 입력: (B, T) @ 16kHz
    - 내부에서 44kHz로 리샘플 후 dac.encoder() 직접 호출 (RVQ 없음)
    - 출력: (B, T_enc, 1024) @ ~86fps, 항상 frozen fp32

    torch.load 패치: PyTorch >= 2.6에서 weights_only=True가 기본값으로 바뀌어
    audiotools 내부의 torch.load(path, "cpu") 호출이 실패함 → 임시 패치 적용.
    """

    def __init__(self, cfg: dict, cache_dir: str):
        super().__init__()
        import dac

        print("Loading DAC encoder (44kHz)...")
        # torch.load weights_only 패치
        _orig_load = torch.load
        torch.load = lambda f, *a, **kw: _orig_load(f, *a, **{**kw, "weights_only": False})
        model_path = dac.utils.download(model_type=cfg["model_type"])
        self.dac = dac.DAC.load(model_path)
        torch.load = _orig_load

        self.out_dim = cfg["out_dim"]
        self.tgt_sr  = cfg["tgt_sr"]
        self.hop     = cfg["hop"]

        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.dac.eval()
        return self

    def forward(
        self,
        audio_waveform: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = audio_waveform.device
        B = audio_waveform.shape[0]

        # 16kHz → 44kHz
        audio_tgt = AF.resample(audio_waveform.float(), self.src_sr, self.tgt_sr)

        if audio_lengths is not None:
            lengths_tgt = (audio_lengths.float() * self.tgt_sr / self.src_sr).long()
        else:
            lengths_tgt = torch.full((B,), audio_tgt.shape[1], dtype=torch.long, device=device)

        # (B, 1, T_44k) → dac.encoder() → (B, 1024, T_enc)
        audio_in = audio_tgt.unsqueeze(1).float().contiguous()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                feats = self.dac.encoder(audio_in)  # (B, 1024, T_enc)
        feats = feats.transpose(1, 2)  # (B, T_enc, 1024)

        T_enc = feats.shape[1]
        lengths_enc = (lengths_tgt.float() / self.hop).ceil().long().clamp(max=T_enc)
        enc_mask = torch.arange(T_enc, device=device).unsqueeze(0) < lengths_enc.unsqueeze(1)

        return feats, enc_mask
