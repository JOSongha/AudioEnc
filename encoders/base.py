from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseAudioEncoder(nn.Module, ABC):
    """
    모든 audio encoder가 지켜야 할 인터페이스.

    계약:
      - 입력 audio는 항상 16kHz (dataset 고정). 내부 리샘플링은 encoder 책임.
      - 출력: (feats, enc_mask)
          feats:    (B, T_enc, out_dim)  — fp32
          enc_mask: (B, T_enc) bool      — True=실제 프레임, False=패딩
      - 항상 frozen (requires_grad=False).
      - 항상 eval 유지 (train() override 필수).
    """

    out_dim: int      # encoder 출력 채널 수 — 서브클래스에서 __init__에서 설정
    src_sr: int = 16000  # 입력 샘플레이트 (고정)

    @abstractmethod
    def forward(
        self,
        audio_waveform: torch.Tensor,           # (B, T_16k)
        audio_lengths: Optional[torch.Tensor],  # (B,) 실제 샘플 수, None이면 전체
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feats:    (B, T_enc, out_dim)
            enc_mask: (B, T_enc) bool
        """
        ...
