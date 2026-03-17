import sys
from typing import Optional, Tuple

import torch
import torchaudio.functional as AF

from .base import BaseAudioEncoder

# dacvae 패키지는 AudioEnc/dacvae/ 로컬 클론에서 임포트
_DACVAE_REPO = __file__.replace("/encoders/fb_dacvae.py", "/dacvae")
if _DACVAE_REPO not in sys.path:
    sys.path.insert(0, _DACVAE_REPO)


class FbDACVAEEncoder(BaseAudioEncoder):
    """
    Facebook DACVAE (facebookresearch/dacvae) 기반 encoder.

    - 입력: (B, T) @ 16kHz
    - 내부에서 44.1kHz로 리샘플 후 DACVAE.encode() 호출
    - VAE continuous latent 반환: (B, T_enc, codebook_dim)

    전체 모델(encoder + VAEBottleneck) frozen.
    out_dim = model.quantizer.codebook_dim (pretrained 모델에서 자동 결정)

    cfg 키:
        "model_id"  : HuggingFace repo 또는 로컬 path  (default: "facebook/dacvae-watermarked")
        "cache_dir" : 모델 캐시 경로  (선택)
    """

    def __init__(self, cfg: dict, cache_dir: str):
        super().__init__()
        from dacvae import DACVAE

        model_id = cfg.get("model_id", "facebook/dacvae-watermarked")
        print(f"Loading Facebook DACVAE: {model_id} ...")
        self.dacvae = DACVAE.load(model_id)
        self.dacvae.eval()
        for p in self.dacvae.parameters():
            p.requires_grad = False

        self.tgt_sr  = self.dacvae.sample_rate         # 44100
        self.hop     = self.dacvae.hop_length           # 512
        self.out_dim = self.dacvae.quantizer.codebook_dim  # e.g. 8

        print(
            f"  sample_rate={self.tgt_sr}, hop={self.hop}, "
            f"latent_dim(out_dim)={self.out_dim}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.dacvae.eval()  # 항상 eval 유지
        return self

    def forward(
        self,
        audio_waveform: torch.Tensor,           # (B, T_16k)
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = audio_waveform.device
        B = audio_waveform.shape[0]

        # 16kHz → 44.1kHz
        audio_tgt = AF.resample(audio_waveform.float(), self.src_sr, self.tgt_sr)

        if audio_lengths is not None:
            lengths_tgt = (audio_lengths.float() * self.tgt_sr / self.src_sr).long()
        else:
            lengths_tgt = torch.full((B,), audio_tgt.shape[1], dtype=torch.long, device=device)

        # DACVAE expects (B, 1, T)
        audio_in = audio_tgt.unsqueeze(1).contiguous()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                z = self.dacvae.encode(audio_in.float())  # (B, codebook_dim, T_enc)

        feats = z.transpose(1, 2).float()  # (B, T_enc, codebook_dim)

        T_enc = feats.shape[1]
        lengths_enc = (lengths_tgt.float() / self.hop).ceil().long().clamp(max=T_enc)
        enc_mask = torch.arange(T_enc, device=device).unsqueeze(0) < lengths_enc.unsqueeze(1)

        return feats, enc_mask
