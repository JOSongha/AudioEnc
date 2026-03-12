from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.functional as AF

from .base import BaseAudioEncoder


class DACVAEEncoder(BaseAudioEncoder):
    """
    descript-audio-codec 44kHz + trainable VAE bottleneck.

    - 입력: (B, T) @ 16kHz
    - 내부에서 44kHz로 리샘플 후 dac.encoder() 직접 호출 (RVQ 없음)
    - 1024-dim 피처 → Linear → (mu, logvar) → reparameterize → latent_dim

    DAC 파라미터:         frozen (requires_grad=False)
    VAE head (to_mu_logvar): trainable (requires_grad=True)

    학습 중: z = mu + eps * exp(0.5 * logvar)  (reparameterization)
    평가 중: z = mu                             (deterministic)

    self.last_kl_loss 에 KL divergence 저장:
        KL = -0.5 * mean(1 + logvar - mu² - exp(logvar))
    학습 루프에서 선택적으로 loss += kl_weight * encoder.last_kl_loss 추가 가능.
    """

    def __init__(self, cfg: dict, cache_dir: str):
        super().__init__()
        import dac

        latent_dim = cfg.get("latent_dim", 256)

        print("Loading DAC encoder for DAC-VAE (44kHz)...")
        _orig_load = torch.load
        torch.load = lambda f, *a, **kw: _orig_load(f, *a, **{**kw, "weights_only": False})
        model_path = dac.utils.download(model_type=cfg["model_type"])
        self.dac = dac.DAC.load(model_path)
        torch.load = _orig_load

        self.out_dim = latent_dim
        self.tgt_sr  = cfg["tgt_sr"]
        self.hop     = cfg["hop"]

        for p in self.dac.parameters():
            p.requires_grad = False

        # Trainable VAE head: 1024 → latent_dim * 2 (mu + logvar)
        self.to_mu_logvar = nn.Linear(1024, latent_dim * 2)
        nn.init.xavier_uniform_(self.to_mu_logvar.weight)
        nn.init.zeros_(self.to_mu_logvar.bias)

        self.last_kl_loss: torch.Tensor = torch.tensor(0.0)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dac.eval()  # DAC는 항상 eval
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

        # DAC encoder (frozen, no_grad)
        audio_in = audio_tgt.unsqueeze(1).float().contiguous()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                feats = self.dac.encoder(audio_in)  # (B, 1024, T_enc)
        feats = feats.transpose(1, 2).float()        # (B, T_enc, 1024)

        # VAE bottleneck (trainable)
        mu_logvar = self.to_mu_logvar(feats)          # (B, T_enc, latent_dim*2)
        mu, logvar = mu_logvar.chunk(2, dim=-1)        # each: (B, T_enc, latent_dim)

        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * (0.5 * logvar).exp()
        else:
            z = mu  # deterministic at eval

        # KL: -0.5 * E[1 + logvar - mu² - exp(logvar)]
        self.last_kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

        T_enc = z.shape[1]
        lengths_enc = (lengths_tgt.float() / self.hop).ceil().long().clamp(max=T_enc)
        enc_mask = torch.arange(T_enc, device=device).unsqueeze(0) < lengths_enc.unsqueeze(1)

        return z, enc_mask
