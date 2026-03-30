from .base import BaseAudioEncoder
from .encodec import EnCodecEncoder
from .dac import DACEncoder
from .fb_dacvae import FbDACVAEEncoder
from .mimi_acoustic import MimiAcousticEncoder
from .mimi_semantic import MimiSemanticEncoder

ENCODER_CLASSES = {
    "encodec":        EnCodecEncoder,
    "dac":            DACEncoder,
    "fb_dacvae":      FbDACVAEEncoder,       # Facebook DACVAE — pretrained VAE continuous latent
    "mimi_acoustic":  MimiAcousticEncoder,   # encoder만 (저수준 acoustic)
    "mimi_semantic":  MimiSemanticEncoder,   # encoder + encoder_transformer (고수준 semantic)
}


def build_encoder(name: str, cfg: dict, cache_dir: str) -> BaseAudioEncoder:
    if name not in ENCODER_CLASSES:
        raise ValueError(
            f"Unknown encoder: '{name}'. Available: {list(ENCODER_CLASSES)}"
        )
    return ENCODER_CLASSES[name](cfg, cache_dir)
