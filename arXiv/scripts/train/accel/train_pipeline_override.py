"""
스트리밍 ASR 학습 파이프라인 (독립 실행형).

주요 설계 원칙:
  - torchaudio.load()로 오디오 로드 (soundfile은 OGG-Opus 미지원 → MLS/VoxPopuli 실패)
  - 데이터 파이프라인 순서: 로드 → 샤딩 → 셔플 → 처리 → 패킹
    (샤딩을 무거운 처리 이전에 수행하여 각 GPU가 자신의 슬라이스만 처리)
  - StreamingShardedTrainer: HF Trainer의 자동 DistributedSampler 삽입 우회
    (파이프라인 시작에서 이미 샤딩 완료)
  - 시퀀스 패킹: block-diagonal attention mask (eager/sdpa) 또는
    position_id 리셋 (Flash Attention 2)

사용 예시:
  # 테스트 1: SDPA + FSDP 없음 + Liger 없음 
  accelerate launch --num_processes=8 --mixed_precision=bf16 \
      train_pipeline_override.py \
      --encoder fb_dacvae --attn-impl sdpa --no-fsdp --no-liger \
      --wandb-mode disabled --max-steps 2 --datasets ls100 --stage 1

  accelerate launch --num_processes=8 --mixed_precision=bf16 \
      train_pipeline_override.py \
      --encoder fb_dacvae --attn-impl sdpa --no-fsdp --no-liger \
      --wandb-mode disabled --max-steps 2 --datasets ls100 --stage 2

  accelerate launch --num_processes=8 --mixed_precision=bf16 \
      train_pipeline_override.py \
      --encoder fb_dacvae --attn-impl sdpa --no-fsdp --no-liger \
      --wandb-mode disabled --max-steps 2 --datasets ls100 --stage all

  # 테스트 2: FA2 + Liger + FSDP 없음
  accelerate launch --num_processes=8 --mixed_precision=bf16 \
      train_pipeline_override.py \
      --encoder fb_dacvae --attn-impl flash_attention_2 --liger --no-fsdp \
      --wandb-mode disabled --max-steps 2 --datasets ls100 --stage all

  # 테스트 3: FA2 + Liger + FSDP (stage1=DDP, stage2=FSDP)
  # 주의: Stage 1: LLM frozen, projector만 학습 → DDP로 충분 (FSDP 불필요)
  # 주의: accelerate launch에 --use_fsdp를 전달하지 말 것.
  #   Stage 1은 DDP 사용 (TrainingArguments.fsdp="" 기본값).
  #   Stage 2는 앱 레벨 --fsdp 플래그로 TrainingArguments(fsdp="full_shard auto_wrap", ...)를 통해 FSDP 사용.
  #   accelerate launch에 --use_fsdp를 전달하면 FSDP가 stage 1에도 누출됨.
  accelerate launch --num_processes=8 --mixed_precision=bf16 \
      train_pipeline_override.py \
      --encoder fb_dacvae --attn-impl flash_attention_2 --liger --fsdp \
      --wandb-mode disabled --max-steps 2 --datasets ls100 --stage all

  # 본격 학습: FA2 + Liger + FSDP, 전체 데이터셋 (ls100+ls360+ls500+mls+gs+vp)
  # 주의:  max_steps가 None이면 calculate_max_steps()가 stage1_epochs, stage2_epochs, estimated_hours 기반으로 자동 계산
  accelerate launch --num_processes=8 --mixed_precision=bf16 \
      train_pipeline_override.py \
      --encoder fb_dacvae \
      --attn-impl flash_attention_2 \
      --liger --fsdp \
      --datasets ls100,ls360,ls500,mls,gs,vp \
      --cutoff-len 2048 \
      --eval-steps 500 \
      --save-steps 1000 \
      --wandb-mode online



구조:
  Imports
  logger                         ← 다른 코드보다 먼저 이동

  ── Config ──────────────────────
  TRAIN_CONFIG
  IGNORE_INDEX / DATASET_ESTIMATED_HOURS / estimate_total_hours

  ── Model ───────────────────────
  AudioQwen                      ← 핵심 클래스 우선
  build_model

  ── Data loading ────────────────
  _load_audio
  StaticEvalDataset
  build_static_eval_datasets

  ── Data pipeline ───────────────
  create_processor
  _search_for_fit / create_packer
  prepare_4d_attention_mask / batch_group_counter
  OmniCollator
  build_multi_dataset_streaming_pipeline
  StreamingShardedTrainer

  ── Evaluation ──────────────────
  _edit_distance / _compute_wer
  _greedy_batch              ← evaluate_wer보다 먼저 이동
  evaluate_wer
  WerCallback

  ── Training ────────────────────
  calculate_max_steps        ← run_stage1보다 먼저 이동
  _check_packing_efficiency
  run_stage1
  setup_stage2_model
  build_stage2_trainer       ← main() 이 직접 호출, split 루프는 main() 담당
  count_total_precomputed_bins_per_rank

  ── Entry point ─────────────────
  main
"""
import argparse
import os
import gc
from typing import Any, Literal
from dataclasses import dataclass

# MLS/VoxPopuli는 OGG-Opus 포맷 → soundfile 미지원 → HF datasets가 torchcodec을 fallback으로 시도.
# 근본 해결책: Audio(decode=False) + torchaudio.load() 직접 호출 (_load_audio 참조).
# 아래 환경변수는 decode=False로 HF 디코딩을 우회하므로 실제로는 사용되지 않음.
# (soundfile이 아닌 torchaudio를 지정: 혹시 decode=False가 누락된 경우의 보조 안전장치)
os.environ.setdefault("HF_DATASETS_AUDIO_BACKEND", "torchaudio")
import io
import bisect
import math
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator
from accelerate.logging import get_logger
from huggingface_hub import try_to_load_from_cache
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets, Audio
from transformers.trainer_pt_utils import IterableDatasetShard

from config import get_config
from encoders import build_encoder
from encoders.base import BaseAudioEncoder

def _parse_datasets(s: str) -> list:
    valid = {"ls100", "ls360", "ls500", "mls", "gs", "vp"}
    items = [x.strip() for x in s.split(",") if x.strip()]
    unknown = set(items) - valid
    if unknown:
        raise argparse.ArgumentTypeError(
            f"알 수 없는 dataset: {unknown}. 선택 가능: {valid}"
        )
    return items

# ══════════════════════════════════════════════════════════
# Logger
# ══════════════════════════════════════════════════════════

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# HTTP 요청 관련 로그 레벨을 WARNING으로 높임 (INFO 로그 차단)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


# 학습 설정은 config.py의 TRAIN_CONFIG / ENCODER_REGISTRY / get_config() 참조.

IGNORE_INDEX = -100

DATASET_ESTIMATED_HOURS = {
    "ls100": 100.0,
    "ls360": 360.0,
    "ls500": 500.0,
    "mls": 10000.0,
    "gs": 10000.0,
    "vp": 500.0,
}


def estimate_total_hours(selected_datasets: list[str]) -> float:
    """Estimate total audio hours from selected dataset keys."""
    unknown = [d for d in selected_datasets if d not in DATASET_ESTIMATED_HOURS]
    if unknown:
        raise ValueError(f"Unknown dataset keys for estimated hours: {unknown}")
    return float(sum(DATASET_ESTIMATED_HOURS[d] for d in selected_datasets))


# ══════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════

class AudioQwen(nn.Module):
    """
    인코더 독립형 Audio-LLM.

    BaseAudioEncoder를 자유롭게 주입 가능. Projector 구조는 cfg["encoder"]["proj_strides"]로 결정:
      [2, 2] → Conv1d(stride-2) × 2, 총 ×4 다운샘플 (encodec, dac, mimi_acoustic)
      [2]    → Conv1d(stride-2) × 1, 총 ×2 다운샘플 (mimi_semantic)

    Forward 시퀀스 (패킹, §41 legacy p1/p2 포맷):
      audio_features → encoder → projector → audio_embeds
      input_ids: p1 + [audio_pad]*t_audio + p2 + text_ids + EOS
        where p1 = "Audio:\n", p2 = "\nTranscript:\n"
      input_ids의 audio_pad 플레이스홀더를 audio_embeds로 in-place 교체
    """

    def __init__(self, encoder: BaseAudioEncoder, cfg: dict):
        super().__init__()
        self.encoder = encoder
        self._cfg    = cfg
        self._stage  = 1   # set by freeze_llm() / apply_lora()

        # Qwen3.5: bf16 필수 (fp16은 SSM 아키텍처 특성상 NaN 발생)
        torch_dtype = torch.bfloat16
        cache_dir   = cfg["model_cache_dir"]
        llm_name    = cfg["llm_model"]

        cached = try_to_load_from_cache(llm_name, "config.json", cache_dir=cache_dir)
        attn_impl = cfg.get("attn_implementation", "eager")
        logger.info(f"Loading LLM: {llm_name} (dtype={torch_dtype}, attn={attn_impl}, {'캐시' if cached else '다운로드'})...",) # By default, the log is called on main processes only. main_process_only=True

        # Liger kernel은 모델 로드 전에 적용해야 함 (import 시점에 모듈을 패치하기 때문)
        if cfg.get("use_liger_kernel", False):
            try:
                try:
                    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5 as apply_liger_qwen
                    apply_liger_qwen(rope=True, rms_norm=True, swiglu=True, fused_linear_cross_entropy=True)
                    logger.info("Liger kernel applied via qwen3_5 (rope, rms_norm, swiglu, fused_linear_ce).",
                                main_process_only=True)
                except ImportError:
                    # liger-kernel 0.7.0은 qwen3_5 미지원 → 직접 패치
                    # qwen3 fallback은 Qwen3_5 클래스를 건드리지 않아 사실상 no-op이므로 사용 금지
                    import copy
                    # fla를 먼저 import해야 modeling_qwen3_5 import 시 fla.modules를 찾을 수 있음
                    try:
                        import fla.modules  # noqa: F401
                        import fla.ops  # noqa: F401
                    except ImportError:
                        pass
                    import transformers.models.qwen3_5.modeling_qwen3_5 as _q35
                    from liger_kernel.transformers.monkey_patch import (
                        liger_rotary_pos_emb, LigerRMSNorm, LigerSwiGLUMLP,
                    )
                    from liger_kernel.transformers.model.qwen3 import lce_forward as _qwen3_lce_forward

                    # 1) RoPE: Qwen3.5는 partial_rotary_factor=0.25 (head_dim=256 중 64만 RoPE)
                    #    liger Triton kernel은 full head_dim RoPE를 가정하므로 직접 사용 불가.
                    #    해결: 앞쪽 rope_dim=64만 liger에 넘기고 나머지 192는 pass-through.
                    from liger_kernel.transformers.rope import LigerRopeFunction as _LigerRope

                    def _partial_liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                        # cos shape: (bsz, seq_len, rope_dim) where rope_dim = head_dim * partial_rotary_factor
                        rope_dim = cos.shape[-1]
                        # q shape: (bsz, n_heads, seq_len, head_dim)
                        q_rope, q_pass = q[..., :rope_dim], q[..., rope_dim:]
                        k_rope, k_pass = k[..., :rope_dim], k[..., rope_dim:]
                        # liger expects cos/sin as (bsz, seq_len, rope_dim) — matches
                        q_rope, k_rope = _LigerRope.apply(
                            q_rope.contiguous(), k_rope.contiguous(),
                            cos, sin, position_ids, unsqueeze_dim,
                        )
                        return torch.cat([q_rope, q_pass], dim=-1), torch.cat([k_rope, k_pass], dim=-1)

                    _q35.apply_rotary_pos_emb = _partial_liger_rotary_pos_emb

                    # 2) RMSNorm: Qwen3.5는 weight=zeros + (1+weight) 공식 → offset=1.0 필요
                    class _Qwen3_5LigerRMSNorm(LigerRMSNorm):
                        def __init__(self, dim: int, eps: float = 1e-6):
                            super().__init__(dim, eps=eps, offset=1.0, init_fn="zeros")
                    _q35.Qwen3_5RMSNorm = _Qwen3_5LigerRMSNorm

                    # 3) SwiGLU: Qwen3.5 MLP는 (config, intermediate_size) 시그니처
                    class _Qwen3_5LigerSwiGLUMLP(LigerSwiGLUMLP):
                        def __init__(self, config, intermediate_size: int):
                            cfg_copy = copy.copy(config)
                            cfg_copy.intermediate_size = intermediate_size
                            super().__init__(cfg_copy)
                    _q35.Qwen3_5MLP = _Qwen3_5LigerSwiGLUMLP

                    # 4) Fused Linear CE: Qwen3_5ForCausalLM.forward 교체
                    _q35.Qwen3_5ForCausalLM.forward = _qwen3_lce_forward

                    logger.info("Liger kernel manually patched for qwen3_5 "
                                "(rms_norm+offset, swiglu, fused_linear_ce). RoPE SKIPPED (torch 2.6.0 compat).",
                                main_process_only=True)
            except Exception as e:
                logger.warning(f"Liger kernel 적용 실패: {e}. 계속 진행합니다...",)
                raise RuntimeError("Liger kernel 적용 실패. pip install liger-kernel 필요.")

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            cache_dir=cache_dir,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_dim = self.llm.config.hidden_size

        # Projector: encoder.out_dim → llm_dim, stride Conv1d 스택으로 다운샘플
        proj_strides = cfg["encoder"].get("proj_strides", [2, 2])
        layers = []
        in_dim = encoder.out_dim
        for stride in proj_strides:
            layers.append(nn.Conv1d(in_dim, llm_dim, kernel_size=5, stride=stride, padding=2))
            layers.append(nn.GELU())
            in_dim = llm_dim
        layers.append(nn.Conv1d(llm_dim, llm_dim, kernel_size=1))
        self.projector = nn.Sequential(*layers)

        # §42+ (A/B) proj_norm_mode: collapse 대응
        #   "ln"              — 기본 LayerNorm (γ=1, β=0)
        #   "ln_small_gamma"  — LN + γ init = qwen_embed_norm/√D (스케일 매칭)
        #   "none"            — nn.Identity (LN 제거)
        _pn_mode = cfg.get("proj_norm_mode", "ln")
        if _pn_mode == "none":
            self.proj_norm = nn.Identity()
        else:
            self.proj_norm = nn.LayerNorm(llm_dim)
        self._proj_norm_mode = _pn_mode

        # Qwen2.5 <|image_pad|> (id=151655)를 audio 플레이스홀더로 재사용
        self.audio_pad_token_id = cfg.get("audio_pad_token_id", 151655)
        self._use_liger_kernel  = cfg.get("use_liger_kernel", False)
        self.projector.to(dtype=torch_dtype)
        self.proj_norm.to(dtype=torch_dtype)

        for m in self.projector.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 마지막 1×1 conv: 초기 audio embed 스케일 억제를 위해 작은 초기화
        nn.init.normal_(self.projector[-1].weight, std=0.02)
        nn.init.zeros_(self.projector[-1].bias)

        # §42+ (B) ln_small_gamma: γ init 로 첫 step 부터 Qwen embed scale 매칭.
        #   audio_embeds L2 norm ≈ γ·√D, Qwen embed norm 매칭 → collapse basin 이탈.
        if _pn_mode == "ln_small_gamma":
            with torch.no_grad():
                emb_w = self.llm.get_input_embeddings().weight
                emb_norm = emb_w.norm(dim=-1).mean().item()
                target_g = emb_norm / (llm_dim ** 0.5)
                self.proj_norm.weight.fill_(target_g)
                logger.info(
                    f"proj_norm γ init = {target_g:.4f}  (qwen emb norm={emb_norm:.4f})",
                )

        self._proj_stride = 1
        for m in self.projector.modules():
            if isinstance(m, nn.Conv1d):
                self._proj_stride *= m.stride[0]

    # ------------------------------------------------------------------
    # Audio embedding helpers
    # ------------------------------------------------------------------

    def _get_audio_embeds_batched(self, audio, audio_lengths=None):
        """DDP용: 배치 단위 encoder forward — 각 GPU가 이미 1/8 데이터를 보유."""
        enc_dtype = next(self.encoder.parameters()).dtype
        feats, _ = self.encoder(audio.to(enc_dtype), audio_lengths)     # (B, T_enc, C)
        feats = feats.to(self.projector[0].weight.dtype)
        proj = self.projector(feats.transpose(1, 2))                    # (B, llm_dim, T_proj)
        return self.proj_norm(proj.transpose(1, 2))                     # (B, T_proj, llm_dim)

    def _get_audio_embeds_sequential(self, audio, audio_lengths=None):
        """FSDP용: 청크 단위 encoder forward — encoder는 샤딩되지 않아 각 GPU가 전체 배치를 보므로,
        Conv1d OOM 방지를 위해 클립 하나씩 처리."""
        enc_dtype = next(self.encoder.parameters()).dtype
        proj_dtype = self.projector[0].weight.dtype
        all_embeds = []
        for i in range(audio.size(0)):
            a = audio[i:i+1].to(enc_dtype)
            l = audio_lengths[i:i+1] if audio_lengths is not None else None
            feats_i, _ = self.encoder(a, l)                             # (1, T_enc, C)
            proj_i = self.projector(feats_i.to(proj_dtype).transpose(1, 2))
            all_embeds.append(self.proj_norm(proj_i.transpose(1, 2)))   # (1, T_proj, llm_dim)
        return torch.cat(all_embeds, dim=0)                             # (N, T_proj, llm_dim)

    def _get_audio_embeds(self, audio, audio_lengths=None):
        if self._cfg.get("use_fsdp", False):
            return self._get_audio_embeds_sequential(audio, audio_lengths)
        return self._get_audio_embeds_batched(audio, audio_lengths)

    def _project_precomputed(self, enc_feats, enc_feat_lengths):
        """사전 계산된 인코더 피처를 projector에 통과. 인코더 호출 없음.

        enc_feats:        (N, T_enc_max, out_dim)  — Arrow에서 로드한 float32, 짧은 클립은 zero-pad
        enc_feat_lengths: (N,) long               — 각 클립의 유효 인코더 프레임 수 (T_enc)

        Returns: (audio_embeds, diversity_loss)
            audio_embeds: (1, total_T_proj, llm_dim)
            diversity_loss: scalar tensor (fp32) or None
                            §42+ (C): clip-mean cos-sim 페널티 (collapse 방지).
                            config.stage1_diversity_reg > 0 + training 일 때만 계산.

        클립별 loop 로 projector+LN 통과 — pad-stack 제거로 word-aug bin 의 메모리 폭발 방지.
        """
        proj_dtype = self.projector[0].weight.dtype
        llm_dtype  = self.llm.get_input_embeddings().weight.dtype
        clips = []
        clip_means = []  # §42+ (C) diversity reg 용: 각 clip 의 평균 벡터
        div_reg = self._cfg.get("stage1_diversity_reg", 0.0)
        compute_div = self.training and div_reg > 0
        for i in range(enc_feats.shape[0]):
            T_enc_valid = int(enc_feat_lengths[i].item())
            if T_enc_valid == 0:
                continue
            feats_i = enc_feats[i:i+1, :T_enc_valid, :].to(proj_dtype)     # (1, T_enc_valid, C)
            proj_i  = self.projector(feats_i.transpose(1, 2))              # (1, llm_dim, T_proj_valid)
            emb_i   = self.proj_norm(proj_i.transpose(1, 2)).squeeze(0)    # (T_proj_valid, llm_dim)
            clips.append(emb_i.to(llm_dtype))
            if compute_div:
                clip_means.append(emb_i.float().mean(dim=0))                # (llm_dim,) fp32

        diversity_loss = None
        if compute_div and len(clip_means) >= 2:
            cm = torch.stack(clip_means)                                    # (N, llm_dim) fp32
            cm = F.normalize(cm, dim=-1)
            cos_sim = cm @ cm.T                                             # (N, N)
            N = cm.shape[0]
            mask = ~torch.eye(N, dtype=torch.bool, device=cos_sim.device)
            diversity_loss = cos_sim[mask].abs().mean()                     # scalar fp32

        return torch.cat(clips, dim=0).unsqueeze(0), diversity_loss         # ((1, total_T_proj, llm_dim), scalar|None)

    # ------------------------------------------------------------------
    # Stage setup
    # ------------------------------------------------------------------

    def freeze_llm(self):
        """Stage 1: LLM frozen, projector만 학습."""
        logger.info("Freezing LLM (Stage 1)...",)
        self._stage = 1
        for p in self.llm.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.proj_norm.parameters():
            p.requires_grad = True
        self.encoder.to(dtype=torch.bfloat16)

    def apply_lora(self):
        """Stage 2: LLM에 LoRA 적용. projector + proj_norm도 trainable."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise RuntimeError("peft 미설치. pip install peft")

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self._cfg["lora_r"],
            lora_alpha=self._cfg["lora_alpha"],
            lora_dropout=self._cfg["lora_dropout"],
            target_modules=self._cfg["lora_target_modules"],
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        # PEFT LoRA 파라미터는 기본 float32 생성 → FSDP mixed-dtype 오류 방지
        self.llm = self.llm.to(torch.bfloat16)

        trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.llm.parameters())
        logger.info(f"LoRA applied. Trainable LLM: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)",)

        train_proj = self._cfg.get("stage2_train_projector", True)
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.proj_norm.parameters():
            p.requires_grad = True
        logger.info(f"Projector {'trainable' if train_proj else 'frozen'} in Stage 2.",)

        self._stage = 2
        self.encoder.to(dtype=torch.bfloat16)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False}
        )

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, labels, audio_features=None, audio_lengths=None,
                attention_mask=None, position_ids=None,
                num_items_in_batch=None,
                precomputed_enc_feats=None,
                **kwargs):
        """
        시퀀스 패킹 forward pass.

        audio_features:         (N_audio, 1, S_max)  raw waveform, zero-padded.
                                precomputed_enc_feats 사용 시 None이어도 됨.
        audio_lengths:          (N_audio,)
                                - raw 모드: LLM 토큰 수 (projector 출력 기준)
                                - precomputed 모드: 인코더 프레임 수 (T_enc)
        precomputed_enc_feats:  (N_audio, T_enc_max, out_dim) float32, 또는 None.
                                None이면 audio_features에서 인코더를 직접 실행한다.
        input_ids:              (B, T) [eager/sdpa] 또는 (1, sum_nonpad) [FA2]
        labels:                 input_ids와 동일한 shape
        attention_mask:         (B, 1, T, T) 4D block-diagonal [eager/sdpa] 또는 None [FA2]
        position_ids:           (1, sum_nonpad) 샘플별 리셋 [FA2] 또는 None [eager/sdpa]
        """
        # 1. Encode audio or use precomputed features
        diversity_loss = None   # §42+ (C) auxiliary reg (precomputed 모드만 지원)
        if precomputed_enc_feats is not None:
            # precomputed 모드: 인코더 생략, audio_lengths = T_enc
            audio_embeds, diversity_loss = self._project_precomputed(
                precomputed_enc_feats, audio_lengths
            )                                                            # (1, total_T_proj, llm_dim)
        else:
            # raw 모드: waveform → encoder → projector
            wavs            = audio_features.squeeze(1)                 # (N, S_max)
            spt             = self._cfg.get("samples_per_token", 1.0)
            lengths_samples = (audio_lengths.float() * spt).long()      # (N,) in samples
            audio_embeds    = self._get_audio_embeds(wavs, lengths_samples)  # (N, T_proj, llm_dim)

        # 2. Embed all tokens
        embed         = self.llm.get_input_embeddings()
        inputs_embeds = embed(input_ids).clone()                        # (B/1, T, llm_dim)

        # 3. Replace audio placeholder positions with audio embeddings
        audio_pad_mask = (input_ids == self.audio_pad_token_id)         # (B, T) bool
        audio_flat     = audio_embeds.reshape(-1, audio_embeds.shape[-1])
        n_ph           = int(audio_pad_mask.sum().item())

        if audio_flat.shape[0] != n_ph:
            # ±1 프레임 불일치 방어: 짧은 쪽에 맞춰 자르기
            n          = min(audio_flat.shape[0], n_ph)
            audio_flat = audio_flat[:n]
            flat_mask  = audio_pad_mask.reshape(-1)
            positions  = flat_mask.nonzero(as_tuple=False).squeeze(1)
            fix_mask   = torch.zeros_like(flat_mask)
            fix_mask[positions[:n]] = True
            audio_pad_mask = fix_mask.reshape(audio_pad_mask.shape)

        inputs_embeds[audio_pad_mask] = audio_flat.to(inputs_embeds.dtype)

        # 4. LLM forward
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=kwargs.get("output_hidden_states"),
            return_dict=kwargs.get("return_dict", True),
        )
        # 랭크 간 올바른 gradient 평균을 위해 로컬 유효 토큰 수로 loss 스케일링
        if num_items_in_batch is not None and out.loss is not None:
            n_valid = (labels != -100).sum()
            out.loss = out.loss * n_valid / num_items_in_batch

        # §42+ (C) diversity reg: Stage 1 collapse 페널티 (precomputed 모드만)
        if diversity_loss is not None and out.loss is not None:
            lam = self._cfg.get("stage1_diversity_reg", 0.0)
            out.loss = out.loss + lam * diversity_loss.to(out.loss.dtype)
        return out


def build_model(cfg, accelerator):
    """Rank-0이 먼저 로드(HF 캐시 채움) → barrier → 나머지 랭크는 캐시에서 로드."""
    enc_cfg  = cfg["encoder"]
    cache_dir = cfg["model_cache_dir"]
    enc_name  = cfg["encoder_name"]

    with accelerator.main_process_first():
        encoder = build_encoder(enc_name, enc_cfg, cache_dir)
        model   = AudioQwen(encoder, cfg)

    return model


# ══════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════

def _load_audio(audio_obj) -> tuple[torch.Tensor, int]:
    """torchaudio로 오디오 bytes/path 로드. FLAC, WAV, OGG, OPUS, MP3 지원.

    soundfile은 OGG-Opus를 지원하지 않으므로 torchaudio 사용.
    (MLS, VoxPopuli는 OPUS 포맷 → soundfile 실패)
    torchaudio는 이미 (channels, frames) float32 텐서 반환.
    """
    if "bytes" in audio_obj and audio_obj["bytes"] is not None:
        waveform, sr = torchaudio.load(io.BytesIO(audio_obj["bytes"]))
    elif "path" in audio_obj and audio_obj["path"] is not None:
        waveform, sr = torchaudio.load(audio_obj["path"])
    else:
        raise ValueError("audio_obj has neither 'bytes' nor 'path'")
    return waveform, sr   # (channels, frames)


class StaticEvalDataset(torch.utils.data.Dataset):
    """WER 평가용 HF 데이터셋 래퍼.

    evaluate_wer가 기대하는 (1D waveform tensor, transcript string) 튜플 반환.
    """
    def __init__(self, hf_dataset, sample_rate=16000):
        self.dataset = hf_dataset
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        audio_obj = item["audio"]
        if ("bytes" in audio_obj and audio_obj["bytes"] is not None) or \
                ("path" in audio_obj and audio_obj["path"] is not None):
            waveform, sr = _load_audio(audio_obj)
        elif "array" in audio_obj:
            waveform = torch.tensor(audio_obj["array"]).unsqueeze(0)
            sr = audio_obj["sampling_rate"]
        else:
            raise ValueError(f"Unknown audio format at index {idx}")

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        text = item.get("text") or item.get("transcript") or item.get("normalized_text") or ""
        return waveform.squeeze(0), text


def build_static_eval_datasets(cfg, tokenizer=None):
    """주기적 WER 평가용 고정 eval 스플릿 (val + train_eval) 로드.

    num_eval_samples만큼만 사용하여 eval이 학습 시간을 잡아먹지 않도록 함.
    """
    sample_rate = cfg.get("sample_rate", 16000)
    num_eval_samples = cfg.get("num_eval_samples", 300)
    root = cfg.get("data_path", "/mnt/tmp/cache")
    mls_root    = cfg.get("mls_data_path", root)

    val_hf = load_dataset(
        "openslr/librispeech_asr",
        "all",
        split=f"validation.clean[:{num_eval_samples}]",
        streaming=False,
        cache_dir=mls_root,
    )
    val_hf = val_hf.cast_column("audio", Audio(decode=False))

    train_eval_hf = load_dataset(
        "openslr/librispeech_asr",
        "all",
        split=f"train.clean.100[:{num_eval_samples}]",
        streaming=False,
        cache_dir=mls_root,
    )
    train_eval_hf = train_eval_hf.cast_column("audio", Audio(decode=False))

    val_dataset = StaticEvalDataset(val_hf, sample_rate=sample_rate)
    train_eval_dataset = StaticEvalDataset(train_eval_hf, sample_rate=sample_rate)

    return val_dataset, train_eval_dataset


# ══════════════════════════════════════════════════════════
# Word-level alignment augmentation
# ══════════════════════════════════════════════════════════

_ALIGNMENT_BASE = "/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/word_aligned_data"

# ── Processor 통계 카운터 (worker-local, debug 확인용) ────────────────────────
_PROC_STATS: dict[str, int] = {"sentence": 0, "word": 0}
_PROC_LOG_EVERY = 2000  # N 문장 샘플마다 한 번씩 stderr로 출력

_ALIGNMENT_ARROW_PATHS = {
    "ls100": f"{_ALIGNMENT_BASE}/librispeech/train-clean-100.arrow",
    "ls360": f"{_ALIGNMENT_BASE}/librispeech/train-clean-360.arrow",
    "ls500": f"{_ALIGNMENT_BASE}/librispeech/train-other-500.arrow",
    "mls":   f"{_ALIGNMENT_BASE}/mls/train.arrow",
    "gs":    f"{_ALIGNMENT_BASE}/gigaspeech/train.arrow",
    "vp":    f"{_ALIGNMENT_BASE}/voxpopuli/train.arrow",
}

# 데이터셋별 utterance_id 추출 방법: (HF 필드명, id 변환 함수 or None)
_DATASET_ID_CONFIG = {
    "ls100": ("id",            None),
    "ls360": ("id",            None),
    "ls500": ("id",            None),
    "mls":   ("original_path", lambda s: s.removesuffix(".opus")),
    "gs":    ("segment_id",    None),
    "vp":    ("audio_id",      None),
}


class AlignmentLookup:
    """Arrow 파일로부터 utterance_id → words 목록 조회.

    index (utterance_id → row_idx) 만 Python dict에 보유.
    words 실데이터는 memory-mapped pyarrow table에서 on-demand 접근 (zero-copy).
    """

    def __init__(self, arrow_path: str, id_transform=None):
        import pyarrow as pa  # 지연 import (main process에서만 생성)
        mmap = pa.memory_map(arrow_path, "r")
        self._table = pa.ipc.open_file(mmap).read_all()
        utt_ids = self._table.column("utterance_id").to_pylist()
        self._index: dict[str, int] = {uid: i for i, uid in enumerate(utt_ids)}
        self._id_transform = id_transform
        # logger.info 는 accelerate state 필요 → pack_arrow.py (plain python) 에서 crash.
        # 단순 print 로 대체 (학습 시 main_process_only 효과는 포기, 큰 영향 없음).
        print(f"AlignmentLookup: {len(self._index):,} utterances ← {arrow_path}", flush=True)

    def get(self, utterance_id: str):
        """utterance_id에 해당하는 [{word, start, end, ...}, ...] 반환. 없으면 None."""
        if self._id_transform:
            utterance_id = self._id_transform(utterance_id)
        idx = self._index.get(utterance_id)
        if idx is None:
            return None
        return self._table.column("words")[idx].as_py()  # list[dict]


class MergedAlignmentLookup:
    """여러 AlignmentLookup을 묶어 utterance_id로 통합 조회."""

    def __init__(self, lookups: "list[AlignmentLookup | None]"):
        self._lookups = [l for l in lookups if l is not None]

    def get(self, utterance_id: str):
        for lookup in self._lookups:
            words = lookup.get(utterance_id)
            if words is not None:
                return words
        return None


def build_alignment_lookups(selected_datasets: list[str]) -> dict[str, AlignmentLookup | None]:
    """선택된 데이터셋에 대한 AlignmentLookup 딕셔너리 생성.

    Arrow 파일이 없는 데이터셋은 None으로 설정.
    """
    lookups: dict[str, AlignmentLookup | None] = {}
    for key in selected_datasets:
        arrow_path = _ALIGNMENT_ARROW_PATHS.get(key)
        if arrow_path and os.path.exists(arrow_path):
            id_field, id_transform = _DATASET_ID_CONFIG.get(key, ("id", None))
            lookups[key] = AlignmentLookup(arrow_path, id_transform=id_transform)
        else:
            lookups[key] = None
            print(f"[warn] AlignmentLookup: no arrow for '{key}' at {arrow_path}", flush=True)
    return lookups


# ══════════════════════════════════════════════════════════
# Data pipeline: processor → packer → collator → streaming loader
# ══════════════════════════════════════════════════════════

def create_processor(
    tokenizer: PreTrainedTokenizer,
    audio_pad_token_id: int = 151655,
    sample_rate: int = 16000,
    hop_length: int = 1920,
    max_audio_samples: int = 5760000,
    alignment_lookup: "AlignmentLookup | None" = None,
    word_aug: bool = False,
    min_word_samples: int = 1600,   # ≥ 100ms (@ 16kHz)
):
    """(audio, text) 원시 쌍을 모델 입력 형식으로 변환하는 batched map 함수 반환.

    시퀀스 형식 (§41 legacy p1/p2):
      p1 + [audio_pad]*t_audio + p2 + text_ids + EOS
    Labels:
      [IGNORE]*(|p1|+t_audio+|p2|) + text_ids + EOS
      (p1 = "Audio:\\n", p2 = "\\nTranscript:\\n")

    word_aug=True이고 alignment_lookup이 제공된 경우,
    각 발화에 대해 원본 샘플 + 단어 단위 서브샘플을 추가 생성.
    ("utterance_id" 컬럼이 batch에 있어야 함)
    """
    p1_ids = tokenizer.encode("Audio:\n",        add_special_tokens=False)
    p2_ids = tokenizer.encode("\nTranscript:\n", add_special_tokens=False)

    def _build_one(waveform_1d: torch.Tensor, text: str):
        """(1D waveform tensor, text string) → (input_ids, labels, t_audio) 또는 None."""
        t_audio = waveform_1d.shape[0] // hop_length
        if t_audio == 0:
            return None
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        if not text_ids:
            return None
        input_ids = (
            p1_ids
            + [audio_pad_token_id] * t_audio
            + p2_ids
            + text_ids
            + [tokenizer.eos_token_id]
        )
        labels = (
            [IGNORE_INDEX] * (len(p1_ids) + t_audio + len(p2_ids))
            + text_ids
            + [tokenizer.eos_token_id]
        )
        return input_ids, labels, t_audio

    def _emit(all_input_ids, all_labels, all_audio_features, all_audio_lengths,
              waveform_1d, text):
        """단일 (waveform, text) 샘플을 출력 리스트에 추가."""
        result = _build_one(waveform_1d, text)
        if result is None:
            return
        ids, lbls, t_audio = result
        all_input_ids.append(ids)
        all_labels.append(lbls)
        all_audio_features.append(waveform_1d)
        all_audio_lengths.append(t_audio)

    def process_samples(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        all_input_ids = []
        all_labels = []
        all_audio_features = []
        all_audio_lengths = []

        audios        = examples["audio"]
        texts         = examples["text"]
        utt_ids       = examples.get("utterance_id", [None] * len(audios))

        for audio_obj, text, utt_id in zip(audios, texts, utt_ids):
            # 1. Load audio
            try:
                if ("bytes" in audio_obj and audio_obj["bytes"] is not None) or \
                        ("path" in audio_obj and audio_obj["path"] is not None):
                    waveform, sr = _load_audio(audio_obj)
                else:
                    continue
            except Exception as e:
                print(f"[omni] Audio load error: {e}")
                continue

            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if max_audio_samples is not None and waveform.shape[-1] > max_audio_samples:
                continue

            waveform_1d = waveform.squeeze(0)

            # 2. 원본 샘플 추가
            _emit(all_input_ids, all_labels, all_audio_features, all_audio_lengths,
                  waveform_1d, text)
            _PROC_STATS["sentence"] += 1

            # 3. Word-level 서브샘플 추가 (word_aug=True이고 alignment 존재 시)
            if word_aug and alignment_lookup is not None and utt_id is not None:
                words = alignment_lookup.get(utt_id)
                if words:
                    sr_ratio = sample_rate  # waveform은 이미 sample_rate로 리샘플됨
                    n_word_added = 0
                    for w in words:
                        word_text = w.get("word", "").strip()
                        if not word_text:
                            continue
                        start_s = int(w.get("start", 0.0) * sr_ratio)
                        end_s   = int(w.get("end",   0.0) * sr_ratio)
                        if end_s - start_s < min_word_samples:
                            continue  # 너무 짧은 단어 건너뜀 (< 100ms)
                        end_s = min(end_s, waveform_1d.shape[0])
                        if end_s <= start_s:
                            continue
                        word_wav = waveform_1d[start_s:end_s]
                        _emit(all_input_ids, all_labels, all_audio_features, all_audio_lengths,
                              word_wav, word_text.lower())
                        n_word_added += 1
                    _PROC_STATS["word"] += n_word_added

            # 주기적 통계 출력 (worker-local)
            if _PROC_STATS["sentence"] % _PROC_LOG_EVERY == 0 and _PROC_STATS["sentence"] > 0:
                import os as _os
                print(
                    f"[Processor pid={_os.getpid()}] "
                    f"sentence={_PROC_STATS['sentence']:,} "
                    f"word={_PROC_STATS['word']:,} "
                    f"(ratio={_PROC_STATS['word']/_PROC_STATS['sentence']:.1f}x)",
                    flush=True,
                )

        return {
            "input_ids":       all_input_ids,
            "labels":          all_labels,
            "audio_features":  all_audio_features,
            "audio_lengths":   all_audio_lengths,
        }

    return process_samples


def _search_for_fit(numbers: list[int], capacity: int) -> int:
    """정렬된 `numbers`에서 `capacity` 이하인 가장 큰 값의 인덱스 반환."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def create_packer(
    cutoff_len: int,
    pad_token_id: int,
    neat_packing: bool = True,
):
    """greedy knapsack 알고리즘으로 가변 길이 시퀀스를 `cutoff_len` 크기 고정 bin에 패킹하는
    batched map 함수 반환.

    attention_mask 값: 서브시퀀스별 1,2,3... (block-diagonal masking용), 패딩은 0.
    """

    def pack_samples(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        n = len(examples["input_ids"])

        # 길이 오름차순 정렬 (bisect 기반 탐색에 필요)
        indexed_lengths = sorted(
            [
                (len(examples["input_ids"][i]), i)
                for i in range(n)
                if len(examples["input_ids"][i]) <= cutoff_len
            ],
            key=lambda x: x[0],
        )
        sorted_lengths = [l for l, _ in indexed_lengths]
        sorted_indices = [i for _, i in indexed_lengths]

        # Greedy knapsack: 각 bin을 가장 큰 fitting 시퀀스부터 채움
        knapsacks: list[list[int]] = []
        while sorted_lengths:
            current_bin: list[int] = []
            remaining = cutoff_len

            while True:
                idx = _search_for_fit(sorted_lengths, remaining)
                if idx == -1:
                    break
                remaining -= sorted_lengths.pop(idx)
                current_bin.append(sorted_indices.pop(idx))

            knapsacks.append(current_bin)

        model_inputs: dict[str, list[Any]] = {
            "input_ids":      [],
            "labels":         [],
            "attention_mask": [],
            "audio_features": [],
            "audio_lengths":  [],
        }

        for knapsack in knapsacks:
            packed_input_ids      = []
            packed_labels         = []
            packed_attention_mask = []
            packed_audio_features = []
            packed_audio_lengths  = []

            for seq_idx, orig_idx in enumerate(knapsack):
                ids = examples["input_ids"][orig_idx]
                lbl = examples["labels"][orig_idx]
                packed_input_ids.extend(ids)
                packed_labels.extend(lbl)

                # block-diagonal masking을 위해 서브시퀀스마다 고유 ID 부여
                if neat_packing:
                    packed_attention_mask.extend([seq_idx + 1] * len(ids))
                else:
                    packed_attention_mask.extend([1] * len(ids))

                packed_audio_features.append(examples["audio_features"][orig_idx])
                packed_audio_lengths.append(examples["audio_lengths"][orig_idx])

            # 남은 공간 패딩
            pad_len = cutoff_len - len(packed_input_ids)
            if pad_len > 0:
                packed_input_ids.extend([pad_token_id] * pad_len)
                packed_labels.extend([IGNORE_INDEX] * pad_len)
                packed_attention_mask.extend([0] * pad_len)

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["attention_mask"].append(packed_attention_mask)
            model_inputs["audio_features"].append(packed_audio_features)
            model_inputs["audio_lengths"].append(packed_audio_lengths)

        return model_inputs

    return pack_samples


def prepare_4d_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """패킹된 시퀀스 ID로부터 4D block-diagonal causal mask 생성.

    attention_mask: (B, S) 정수 텐서, 예: [1, 1, 2, 2, 0, 0]
      - 같은 ID → 같은 서브시퀀스 (서로 attend 가능)
      - 0 → 패딩 (attend 불가)
    반환: (B, 1, S, S) float mask (0.0 = attend, finfo.min = 무시)
    """
    B, S = attention_mask.shape

    seq_q = attention_mask.unsqueeze(2)  # (B, S, 1)
    seq_k = attention_mask.unsqueeze(1)  # (B, 1, S)
    valid_mask = (seq_q == seq_k) & (seq_q != 0)

    causal_mask = torch.tril(torch.ones((S, S), dtype=torch.bool, device=attention_mask.device))
    full_mask = valid_mask & causal_mask.unsqueeze(0)

    min_val = torch.finfo(dtype).min
    float_mask = torch.full((B, 1, S, S), min_val, dtype=dtype, device=attention_mask.device)
    float_mask.masked_fill_(full_mask.unsqueeze(1), 0.0)

    return float_mask


def batch_group_counter(attention_mask: torch.Tensor) -> torch.Tensor:
    """서브시퀀스가 바뀔 때마다 0으로 리셋되는 position_ids 계산.

    Flash Attention 2는 시퀀스별 position_ids 리셋을 요구하므로 이 함수를 사용.
    """
    B, S = attention_mask.shape
    position_ids = torch.zeros((B, S), dtype=torch.long, device=attention_mask.device)

    for i in range(B):
        pos = 0
        curr_id = -1
        for j in range(S):
            val = attention_mask[i, j].item()
            if val == 0:
                position_ids[i, j] = 0
            else:
                if val != curr_id:
                    pos = 0
                    curr_id = val
                position_ids[i, j] = pos
                pos += 1

    return position_ids


@dataclass
class OmniCollator:
    """패킹 omni 파이프라인용 Collator.

    패킹된 샘플 리스트를 모델 입력 텐서로 변환:
      - input_ids, labels 스택 (packer에서 이미 cutoff_len으로 맞춤)
      - raw 모드: 배치 전체 waveform 평탄화 → audio_features (N_audio, 1, S_max)
      - precomputed 모드: 인코더 피처 평탄화 → precomputed_enc_feats (N_audio, T_enc_max, out_dim)
      - 4D block-diagonal mask 생성 [eager/sdpa] 또는 평탄화 + position_ids 리셋 [FA2]

    enc_out_dim > 0: precomputed 모드. audio_features는 flat float32 (T_enc × out_dim,)로
    저장되어 있으며, audio_lengths = T_enc (인코더 프레임 수).
    enc_out_dim == 0: raw waveform 모드 (기본).
    """
    pad_token_id: int
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    compute_dtype: torch.dtype = torch.bfloat16
    block_diag_attn: bool = True
    enc_out_dim: int = 0   # 0 = raw waveform 모드; >0 = precomputed encoder feature 모드
    # §42+ Runtime tag mask: GigaSpeech 의 <comma>, <period>, <noise> 등 literal 태그
    # 토큰을 labels 에서 -100 으로 마스킹. sentence_pack_arrow.py 에서 태그 제거 누락된
    # 기존 pack 재활용 시 필수 (re-pack 없이 loss 오염 제거).
    # open: ' <'(mid-sentence, Qwen id 366) 과 '<'(sentence-start edge, id 27) 둘 다 포함.
    # close: '>' (id 29). open 뒤 tag_span_max 토큰 내 close 탐색, 그 구간 전부 -100.
    open_tag_ids:  tuple[int, ...] | None = None
    close_tag_id:  int | None             = None
    # span_max=6 → <exclamationmark> (5 tokens incl. ' <' '>') 도 커버.
    tag_span_max:  int                    = 6

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        import numpy as np

        input_ids      = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        labels         = torch.tensor([f["labels"]    for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

        # Runtime tag mask: labels 에서 '<...>' 패턴 위치 전부 -100 설정.
        # (§42+ precomputed gs 태그 오염 hotfix)
        if self.open_tag_ids and self.close_tag_id is not None:
            lbl_np   = labels.numpy()
            open_set = set(self.open_tag_ids)
            close_id = self.close_tag_id
            span_max = self.tag_span_max
            B, S = lbl_np.shape
            for b in range(B):
                row = lbl_np[b]
                opens = np.flatnonzero(np.isin(row, list(open_set)))
                for oi in opens:
                    end = min(int(oi) + span_max + 1, S)
                    for ci in range(int(oi) + 1, end):
                        if row[ci] == close_id:
                            row[int(oi):ci + 1] = -100
                            break
            labels = torch.from_numpy(lbl_np)

        all_audio_lengths = []
        for f in features:
            all_audio_lengths.extend(f["audio_lengths"])
        audio_lengths = torch.tensor(all_audio_lengths, dtype=torch.long)

        if self.enc_out_dim > 0:
            # ── precomputed 모드 ───────────────────────────────────────────
            # audio_features per clip: flat numpy array (T_enc × out_dim,)
            # audio_lengths per clip: T_enc
            out_dim = self.enc_out_dim
            all_enc_feats = []
            for f in features:
                for flat_feat, t_enc in zip(f["audio_features"], f["audio_lengths"]):
                    arr = np.asarray(flat_feat, dtype=np.float32).reshape(t_enc, out_dim)
                    all_enc_feats.append(torch.from_numpy(arr))   # (T_enc, out_dim)

            if all_enc_feats:
                max_T_enc = max(e.shape[0] for e in all_enc_feats)
                enc_tensor = torch.zeros(
                    len(all_enc_feats), max_T_enc, out_dim, dtype=torch.float32
                )
                for i, e in enumerate(all_enc_feats):
                    enc_tensor[i, :e.shape[0], :] = e
            else:
                enc_tensor = torch.zeros((0, 0, out_dim), dtype=torch.float32)

            result = {
                "input_ids":             input_ids,    # (B, cutoff_len)
                "labels":                labels,        # (B, cutoff_len)
                "precomputed_enc_feats": enc_tensor,   # (N_audio, T_enc_max, out_dim)
                "audio_lengths":         audio_lengths, # (N_audio,) = T_enc per clip
            }
        else:
            # ── raw waveform 모드 (기존) ──────────────────────────────────
            all_waveforms = []
            for f in features:
                for wav in f["audio_features"]:
                    all_waveforms.append(
                        wav if isinstance(wav, torch.Tensor)
                        else torch.tensor(wav, dtype=torch.float32)
                    )

            if all_waveforms:
                max_audio_len = max(w.size(0) for w in all_waveforms)
                audio_features = torch.stack(
                    [torch.nn.functional.pad(w, (0, max_audio_len - w.size(0)))
                     for w in all_waveforms]
                ).unsqueeze(1)
            else:
                audio_features = torch.zeros((0, 1, 0), dtype=torch.float32)

            result = {
                "input_ids":      input_ids,        # (B, cutoff_len)
                "labels":         labels,            # (B, cutoff_len)
                "audio_features": audio_features,   # (N_audio, 1, max_audio_len)
                "audio_lengths":  audio_lengths,     # (N_audio,)
            }

        if self.block_diag_attn:
            if self.attn_implementation != "flash_attention_2":
                # Eager/SDPA: 4D block-diagonal mask 주입
                result["attention_mask"] = prepare_4d_attention_mask(
                    attention_mask, self.compute_dtype
                )
            else:
                # Flash Attention 2: 패딩 제거 후 (1, N)으로 평탄화, position_ids 리셋
                non_pad = attention_mask != 0
                result["input_ids"] = input_ids[non_pad].unsqueeze(0)
                result["labels"]    = labels[non_pad].unsqueeze(0)

                position_ids = batch_group_counter(attention_mask)
                position_ids = position_ids[non_pad].unsqueeze(0)

                result["labels"][position_ids == 0] = IGNORE_INDEX
                result["position_ids"] = position_ids

        return result


def build_multi_dataset_streaming_pipeline(cfg, accelerator, processor_fn, packer_fn,
                                           shuffle=True, selected_datasets=None,
                                           word_aug: bool = False):
    """스트리밍 데이터셋 파이프라인 구성: 로드 → 샤딩 → Interleave → 셔플 → 처리 → 패킹.

    로드 직후 샤딩하여 각 GPU가 1/N 슬라이스만 처리.
    Interleave 전에 컬럼명을 ["audio", "text"] (또는 word_aug=True 시 + "utterance_id")로 통일.

    word_aug=True: 각 데이터셋에서 utterance_id 컬럼을 보존.
    processor_fn에 alignment_lookup이 설정된 경우 단어 단위 서브샘플 생성.
    """
    root = cfg.get("data_path", "/mnt/tmp/cache")
    mls_root = cfg.get("mls_data_path", root)

    base_cols = ["audio", "text", "utterance_id"] if word_aug else ["audio", "text"]

    dataset_list = []
    selected = set(selected_datasets or ["ls100", "ls360", "ls500", "mls", "gs", "vp"])

    # LibriSpeech (설정별로 로드하여 'all' 메타데이터 전체 fetch 방지)
    ls_split_map = {
        "ls100": "train.clean.100",
        "ls360": "train.clean.360",
        "ls500": "train.other.500",
    }
    for key, split_name in ls_split_map.items():
        if key not in selected:
            continue
        ds_ls = load_dataset("openslr/librispeech_asr", split=split_name, streaming=True,
                             cache_dir=mls_root)
        if accelerator.num_processes > 1:
            ds_ls = ds_ls.shard(num_shards=accelerator.num_processes,
                                index=accelerator.process_index, contiguous=False)
        if word_aug:
            ds_ls = ds_ls.rename_column("id", "utterance_id")
        ds_ls = ds_ls.select_columns(base_cols)
        ds_ls = ds_ls.cast_column("audio", Audio(decode=False))
        dataset_list.append(ds_ls)

    # MLS
    if "mls" in selected:
        ds_mls = load_dataset("parler-tts/mls_eng_10k", split="train", streaming=True,
                              cache_dir=mls_root)
        if accelerator.num_processes > 1:
            ds_mls = ds_mls.shard(num_shards=accelerator.num_processes,
                                  index=accelerator.process_index, contiguous=False)
        ds_mls = ds_mls.rename_column("transcript", "text")
        if word_aug:
            # original_path: "12345_678_000001.opus" → AlignmentLookup에서 .opus 제거
            ds_mls = ds_mls.rename_column("original_path", "utterance_id")
        ds_mls = ds_mls.select_columns(base_cols)
        ds_mls = ds_mls.cast_column("audio", Audio(decode=False))
        dataset_list.append(ds_mls)

    # GigaSpeech
    if "gs" in selected:
        ds_gs = load_dataset("speechcolab/gigaspeech", "xl", split="train", streaming=True,
                             cache_dir=mls_root)
        if accelerator.num_processes > 1:
            ds_gs = ds_gs.shard(num_shards=accelerator.num_processes,
                                index=accelerator.process_index, contiguous=False)
        if "text" not in ds_gs.column_names and "transcript" in ds_gs.column_names:
            ds_gs = ds_gs.rename_column("transcript", "text")
        if word_aug:
            ds_gs = ds_gs.rename_column("segment_id", "utterance_id")
        ds_gs = ds_gs.select_columns(base_cols)
        ds_gs = ds_gs.cast_column("audio", Audio(decode=False))
        dataset_list.append(ds_gs)

    # VoxPopuli
    if "vp" in selected:
        ds_vox = load_dataset("facebook/voxpopuli", "en", split="train", streaming=True,
                              cache_dir=mls_root, trust_remote_code=True)
        if accelerator.num_processes > 1:
            ds_vox = ds_vox.shard(num_shards=accelerator.num_processes,
                                  index=accelerator.process_index, contiguous=False)
        ds_vox = ds_vox.rename_column("normalized_text", "text")
        if word_aug:
            ds_vox = ds_vox.rename_column("audio_id", "utterance_id")
        ds_vox = ds_vox.select_columns(base_cols)
        ds_vox = ds_vox.cast_column("audio", Audio(decode=False))
        dataset_list.append(ds_vox)

    if not dataset_list:
        raise ValueError("No datasets selected. Use --datasets with values like ls100,ls360,ls500,mls,gs,vp")

    ds = interleave_datasets(dataset_list, seed=cfg.get("seed", 42))

    if shuffle:
        ds = ds.shuffle(seed=cfg.get("seed", 42),
                        buffer_size=cfg.get("shuffle_buffer_size", 10000))

    remove_cols = ["audio", "text"] + (["utterance_id"] if word_aug else [])
    ds = ds.map(
        processor_fn,
        batched=True,
        batch_size=cfg.get("process_batch_size", 4),
        remove_columns=remove_cols,
    )

    if cfg.get("packing", True):
        ds = ds.map(
            packer_fn,
            batched=True,
            batch_size=cfg.get("packing_bucket_size", 1000),
        )

    if accelerator.is_main_process:
        from accelerate.logging import get_logger as _get_logger
        _logger = _get_logger(__name__)
        _logger.info(f"Pipeline initialized. Total interleaved shards: {getattr(ds._ex_iterable, 'num_shards', 1)}")

    return ds


# ══════════════════════════════════════════════════════════════════════════════
# Precomputed encoder feature 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def make_precomputed_processor_fn(cfg, tokenizer, alignment_lookup=None):
    """Pre-computed encoder feature Arrow 파일용 processor_fn.

    Arrow 스키마: utterance_id(str), text(str), features(list<float32>), feat_len(int32)
    features = flattened (T_enc × out_dim,) float32

    word-aug (alignment_lookup 제공 시):
        각 utterance row 마다 원본 (whole) + word 단위 sub-clip rows 추가 emit.
        word 슬라이싱은 features 텐서를 frame index 로 자르는 방식 (no re-encoding).
        fps = encoder.tgt_sr / encoder.hop  (예: fb_dacvae 44100/512 ≈ 86)

    출력 키:
        input_ids       : list[int]
        labels          : list[int]
        audio_features  : list[float32]  — flattened (T_enc × out_dim,)
        audio_lengths   : list[int]      — T_enc (인코더 프레임 수, collator에서 reshape에 사용)
    """
    audio_pad_id   = cfg.get("audio_pad_token_id", 151655)
    max_text_len   = cfg.get("max_text_len", 256)
    proj_stride    = cfg["encoder"].get("proj_strides", [2, 2])
    total_stride   = 1
    for s in proj_stride:
        total_stride *= s

    # word-aug fps (encoder 출력 frame 수 / 초)
    enc_cfg   = cfg["encoder"]
    fps       = enc_cfg["tgt_sr"] / enc_cfg["hop"]   # ex: 44100/512 ≈ 86.13
    min_word_frames = max(2, int(0.1 * fps))         # ≥ 100ms

    # §41 prompt format regression fix — legacy model.py:AudioQwen 과 동일한 text prompt
    # 포맷 사용. 단일 특수 토큰 `<|audio_correspond|>` 은 학습이 수렴하지 않는 주 원인
    # ([docs/prompt_format_regression.md](../docs/prompt_format_regression.md)).
    # legacy 포맷:
    #   input_ids = p1 + [audio_pad]*T_proj + p2 + text_ids + [eos]
    # 이 포맷은 Qwen 이 이미 학습한 "Audio:", "Transcript:" 등의 토큰을 그대로 써서
    # LLM 의 language prior 를 anchor 로 활용.
    p1_ids = tokenizer.encode("Audio:\n",        add_special_tokens=False)
    p2_ids = tokenizer.encode("\nTranscript:\n", add_special_tokens=False)

    def _emit(all_input_ids, all_labels, all_audio_features, all_audio_lengths,
              flat_feats_or_list, T_enc, text):
        """단일 (features, T_enc, text) 샘플을 출력 리스트에 추가."""
        if T_enc == 0:
            return
        text_ids = tokenizer.encode(text.lower().strip(), add_special_tokens=False)[:max_text_len]
        if not text_ids:
            return
        T_proj = math.ceil(T_enc / total_stride)
        input_ids = (
            p1_ids
            + [audio_pad_id] * T_proj
            + p2_ids
            + text_ids
            + [tokenizer.eos_token_id]
        )
        labels = (
            [-100] * len(p1_ids)
            + [-100] * T_proj
            + [-100] * len(p2_ids)
            + text_ids
            + [tokenizer.eos_token_id]
        )
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_audio_features.append(flat_feats_or_list)
        all_audio_lengths.append(int(T_enc))

    def process_samples(examples):
        import numpy as np
        all_input_ids, all_labels = [], []
        all_audio_features, all_audio_lengths = [], []

        utt_ids = examples.get("utterance_id", [None] * len(examples["features"]))

        for utt_id, flat_feats, feat_len, text in zip(
            utt_ids, examples["features"], examples["feat_len"], examples["text"]
        ):
            if feat_len == 0 or not flat_feats:
                continue
            T_enc = int(feat_len)

            # 1. 원본 utterance (문장 레벨)
            whole_feats = flat_feats if isinstance(flat_feats, list) else list(flat_feats)
            _emit(all_input_ids, all_labels, all_audio_features, all_audio_lengths,
                  whole_feats, T_enc, text)

            # 2. word-aug: alignment 가 있으면 word 단위 sub-clip 추가
            if alignment_lookup is None or utt_id is None:
                continue
            words = alignment_lookup.get(utt_id)
            if not words:
                continue

            # numpy 로 한 번만 reshape (zero-copy view)
            arr = np.asarray(flat_feats, dtype=np.float32)
            out_dim = arr.size // T_enc
            if arr.size != T_enc * out_dim:
                continue   # 잘못된 shape 방어
            feats_2d = arr.reshape(T_enc, out_dim)

            for w in words:
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue
                start_frame = int(round(w.get("start", 0.0) * fps))
                end_frame   = int(round(w.get("end",   0.0) * fps))
                if end_frame - start_frame < min_word_frames:
                    continue
                start_frame = max(0, start_frame)
                end_frame   = min(T_enc, end_frame)
                if end_frame <= start_frame:
                    continue
                word_T_enc = end_frame - start_frame
                word_flat  = feats_2d[start_frame:end_frame].flatten().tolist()
                _emit(all_input_ids, all_labels, all_audio_features, all_audio_lengths,
                      word_flat, word_T_enc, word_text)

        return {
            "input_ids":      all_input_ids,
            "labels":         all_labels,
            "audio_features": all_audio_features,
            "audio_lengths":  [int(x) for x in all_audio_lengths],
        }

    return process_samples


def count_total_precomputed_bins_per_rank(cfg, precomputed_dir, selected_datasets=None) -> int:
    """rank 0 의 mixed packed shard 전체 / dataset별 shard 전체 bin 수 합산.

    Stage 2 cross-split LR scheduler 가 warmup 을 맨 처음에만 수행하고 cosine decay 를
    전체 학습에 걸쳐 매끄럽게 뽑으려면 사전에 **총 step 수** 를 알아야 함.
    `pa.memory_map` + `ipc.open_file` 의 metadata-only 스캔으로 shard 당 < 1 s.

    rank-balance (§26 all_reduce MIN) 후에는 모든 rank 의 bin 수가 동일하므로 rank 0 만
    조사해도 충분. mixed packed shard 우선, 없으면 selected_datasets 의 per-dataset
    packed shard 폴백.
    """
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.ipc as ipc

    enc_name   = cfg["encoder_name"]
    cutoff_len = cfg["packing_cutoff_len"]
    selected   = selected_datasets or ["ls100", "ls360", "ls500", "mls", "gs", "vp"]
    base_dir   = Path(precomputed_dir) / enc_name

    def _count_file(path: Path) -> int:
        with pa.memory_map(str(path), "r") as src:
            rdr = ipc.open_file(src)
            return sum(rdr.get_batch(i).num_rows for i in range(rdr.num_record_batches))

    # mixed packed shard 우선. §42: half_inlv / sentence_only 패킹 출력도 동일 schema 로 수용.
    for mixed_subdir in (f"packed_{cutoff_len}",
                          f"packed_half_inlv_{cutoff_len}",
                          f"packed_sentence_only_{cutoff_len}",
                          f"packed_sentence_{cutoff_len}"):
        mixed_dir    = base_dir / "mixed" / mixed_subdir
        mixed_shards = sorted(mixed_dir.glob("rank0_s*.arrow"))
        mixed_single = mixed_dir / "rank0.arrow"
        if mixed_shards:
            return sum(_count_file(s) for s in mixed_shards)
        if mixed_single.exists():
            return _count_file(mixed_single)

    total = 0
    for ds_key in selected:
        packed = base_dir / ds_key / f"packed_{cutoff_len}" / "rank0.arrow"
        if packed.exists():
            total += _count_file(packed)
    return total


def build_precomputed_pipeline(cfg, accelerator, precomputed_dir, processor_fn, packer_fn,
                               selected_datasets=None, shuffle: bool = True,
                               word_aug: bool = False,
                               data_split: int = 0, num_data_splits: int = 1):
    """Pre-computed Arrow 파일 기반 데이터 파이프라인.

    각 rank는 자신의 rank{process_index}.arrow 파일만 로드한다.
    word_aug=True: processor_fn이 word-level sub-clip 생성 (AlignmentLookup 필요).

    Pre-packed 모드 (권장):
        {precomputed_dir}/{encoder}/{dataset}/packed_{cutoff_len}/rank{N}.arrow 가 존재하면
        processor_fn / packer_fn map을 건너뛰고 Arrow를 직접 로드한다.
        → DataLoader 학습 중 CPU packing 부하 제거.
        생성: bash precompute/run_pack.sh --encoder {encoder}

    data_split / num_data_splits:
        메모리 절약을 위해 데이터를 N등분하여 한 번에 1/N만 로드.
        예: num_data_splits=2 → epoch 0은 각 rank 파일의 전반부,
            epoch 1은 후반부만 로드. 학습 루프에서 split마다 반복 호출.
    """
    from pathlib import Path
    rank       = accelerator.process_index
    enc_name   = cfg["encoder_name"]
    cutoff_len = cfg["packing_cutoff_len"]
    selected   = selected_datasets or ["ls100", "ls360", "ls500", "mls", "gs", "vp"]
    base_dir   = Path(precomputed_dir) / enc_name

    # interleave_datasets는 모든 데이터셋의 schema가 동일해야 함.
    # pack_arrow.py는 int32+float32로 저장하지만, on-the-fly packing은
    # Python int/float → HF datasets가 int64/float64로 추론함.
    # → 모든 데이터셋을 동일 schema로 cast.
    from datasets import Features, Sequence, Value as DValue
    _PACKED_CANONICAL = Features({
        "input_ids":      Sequence(DValue("int32")),
        "labels":         Sequence(DValue("int32")),
        "attention_mask": Sequence(DValue("int32")),
        "audio_features": Sequence(Sequence(DValue("float32"))),
        "audio_lengths":  Sequence(DValue("int32")),
    })
    _PROC_FEATURES = Features({
        "input_ids":      Sequence(DValue("int32")),
        "labels":         Sequence(DValue("int32")),
        "audio_features": Sequence(DValue("float32")),
        "audio_lengths":  DValue("int32"),
    })

    dataset_list    = []
    any_needs_pack  = False   # pre-packed 아닌 데이터셋이 하나라도 있으면 True

    # ── Mixed packed Arrow 우선 (cross-dataset packing) ────────────────
    # §42: 탐색 우선순위: sentence_only → half_inlv → plain.
    # 여러 버전 공존 시 가장 최근 실험 포맷 (sentence_only) 우선.
    # §42: 탐색 우선순위 — packed_sentence > packed_sentence_only > packed_half_inlv > packed_plain
    # §42+: skip_mixed_pack=True 면 mixed 조회 전부 건너뛰고 per-dataset 으로 직행.
    skip_mixed = cfg.get("skip_mixed_pack", False)
    mixed_dir_sent = base_dir / "mixed" / f"packed_sentence_{cutoff_len}"
    mixed_dir_so   = base_dir / "mixed" / f"packed_sentence_only_{cutoff_len}"
    mixed_dir_hi   = base_dir / "mixed" / f"packed_half_inlv_{cutoff_len}"
    mixed_dir_plain = base_dir / "mixed" / f"packed_{cutoff_len}"
    if (sorted(mixed_dir_sent.glob(f"rank{rank}_s*.arrow"))
            or (mixed_dir_sent / f"rank{rank}.arrow").exists()):
        mixed_dir = mixed_dir_sent
    elif (sorted(mixed_dir_so.glob(f"rank{rank}_s*.arrow"))
            or (mixed_dir_so / f"rank{rank}.arrow").exists()):
        mixed_dir = mixed_dir_so
    elif (sorted(mixed_dir_hi.glob(f"rank{rank}_s*.arrow"))
            or (mixed_dir_hi / f"rank{rank}.arrow").exists()):
        mixed_dir = mixed_dir_hi
    else:
        mixed_dir = mixed_dir_plain
    mixed_shards = sorted(mixed_dir.glob(f"rank{rank}_s*.arrow"))
    mixed_single = mixed_dir / f"rank{rank}.arrow"
    if not skip_mixed and (mixed_shards or mixed_single.exists()):
        import math
        import pyarrow.ipc as _pa_ipc
        from datasets import Dataset as _HFDataset
        import pyarrow as _pa
        if mixed_shards:
            # shard-level split: peak load = 1/N (§13). bins-level slice 대비
            # 전체 shard를 read_all() 하지 않으므로 mixed packing 노드 OOM 방지.
            if num_data_splits > 1:
                n_sh  = len(mixed_shards)
                chunk = math.ceil(n_sh / num_data_splits)
                start = data_split * chunk
                end   = min(start + chunk, n_sh)
                sel   = mixed_shards[start:end]
            else:
                sel = mixed_shards
            _tables = [_pa_ipc.open_file(str(f)).read_all() for f in sel]
            _table  = _pa.concat_tables(_tables) if len(_tables) > 1 else _tables[0]
            file_info = f"{len(sel)}/{len(mixed_shards)} shard(s)"
        else:
            _table = _pa_ipc.open_file(str(mixed_single)).read_all()
            file_info = mixed_single.name
            if num_data_splits > 1:
                n = _table.num_rows
                chunk = n // num_data_splits
                start = data_split * chunk
                end = n if data_split == num_data_splits - 1 else start + chunk
                _table = _table.slice(start, end - start)

        # rank-balance: shard 크기가 rank 별로 다르면 NCCL collective 불일치 → deadlock.
        # 이전 로직 (MIN truncate) 은 데이터 drop 발생 → MAX pad 로 변경.
        # 부족 rank 는 자신의 앞쪽 bin 을 cycle 로 append → 모든 rank 동일 step,
        # 글로벌 drop 0, 해당 rank 에서 일부 bin 이 같은 epoch 에 2× 관찰될 뿐
        # (바이어스는 max-min ≤ num_ranks-1 bin 수준으로 작음).
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                local_n = torch.tensor([_table.num_rows], device=accelerator.device)
                _dist.all_reduce(local_n, op=_dist.ReduceOp.MAX)
                max_n = int(local_n.item())
                if _table.num_rows < max_n:
                    orig_n = _table.num_rows
                    pad = max_n - orig_n
                    if pad <= orig_n:
                        _pad_tbl = _table.slice(0, pad)
                    else:
                        reps = pad // orig_n
                        rem  = pad %  orig_n
                        _parts = [_table] * reps
                        if rem > 0:
                            _parts.append(_table.slice(0, rem))
                        _pad_tbl = _pa.concat_tables(_parts)
                    logger.info(
                        f"[rank {rank}] rank-balance pad {orig_n} → {max_n} bins (+{pad} cycled, no drop)",
                        main_process_only=False,
                    )
                    _table = _pa.concat_tables([_table, _pad_tbl])
        except Exception as _e:
            logger.warning(f"[rank {rank}] rank-balance skipped: {_e}", main_process_only=False)
        # numpy format: nested list<float32> → ndarray(object) of float32 ndarrays (§15).
        # map-style Dataset 그대로 반환 (no .to_iterable_dataset()): __len__ 가 있어
        # HF Trainer 가 num_train_epochs 만으로 step 수 자동 계산 (§19, max_steps 박지 않음).
        ds = _HFDataset(_table).with_format("numpy")
        split_info = f" [split {data_split+1}/{num_data_splits}]" if num_data_splits > 1 else ""
        logger.info(
            f"[rank {rank}] Mixed pre-packed (in-memory): {file_info}"
            f" ({_table.num_rows} bins){split_info}",
            main_process_only=False,
        )
        dataset_list.append(ds)
        # mixed 분기 (§17): interleave + shuffle 모두 스킵.
        #   - dataset_list 길이 1 → interleave 불필요
        #   - pack_arrow.py mixed 모드가 packing 전 Fisher-Yates 셔플 적용 → 이미 random
        #   - HF IterableDataset.shuffle(buffer_size=N) 가 cutoff_len=16384에서
        #     unbounded RAM 누수 (§17). buffer 1000 → 100 GB+ / proc 가 자라나며 첫 step 도달 못함.
        # epoch마다 동일 순서지만 num_data_splits=5 로 split 단위 다양성 확보.
        return ds

    for ds_key in selected:
        # ── Pre-packed Arrow 우선 ──────────────────────────────────────────
        packed_path = base_dir / ds_key / f"packed_{cutoff_len}" / f"rank{rank}.arrow"
        if packed_path.exists():
            import pyarrow.ipc as _pa_ipc
            from datasets import Dataset as _HFDataset
            _table = _pa_ipc.open_file(str(packed_path)).read_all()
            # data_split: 테이블을 N등분하여 해당 split만 유지 (메모리 절약)
            if num_data_splits > 1:
                n = _table.num_rows
                chunk = n // num_data_splits
                start = data_split * chunk
                end = n if data_split == num_data_splits - 1 else start + chunk
                _table = _table.slice(start, end - start)
            # numpy format: §15 — list<float32> → float32 ndarray, 6x smaller decode
            ds = _HFDataset(_table).with_format("numpy").to_iterable_dataset()
            split_info = f" [split {data_split+1}/{num_data_splits}]" if num_data_splits > 1 else ""
            logger.info(
                f"[rank {rank}] Pre-packed {ds_key} (in-memory): {packed_path.name}"
                f" ({_table.num_rows} bins){split_info}",
                main_process_only=False,
            )
            dataset_list.append(ds)
            continue

        # ── Per-sample Arrow (sharded 우선) ───────────────────────────────
        shard_files = sorted(
            (base_dir / ds_key).glob(f"rank{rank}_s*.arrow")
        )
        if shard_files:
            data_files = [str(p) for p in shard_files]
        else:
            arrow_path = base_dir / ds_key / f"rank{rank}.arrow"
            if not arrow_path.exists():
                logger.warning(
                    f"[rank {rank}] Precomputed file missing: {arrow_path} — skipping {ds_key}",
                    main_process_only=False,
                )
                continue
            data_files = [str(arrow_path)]

        logger.info(
            f"[rank {rank}] {'Streaming' if cfg.get('skip_mixed_pack', False) else 'Memory-mapped'} "
            f"{ds_key}: {[Path(f).name for f in data_files]}",
            main_process_only=False,
        )
        import pyarrow.ipc as _pa_ipc
        from datasets import Dataset as _HFDataset
        import pyarrow as _pa
        # §42+ skip_mixed_pack=True 모드: HF load_dataset("arrow", streaming=True) 사용.
        #   → iter 시점에 한 batch 씩만 열어 처리. 초기 RSS 폭발 없음 (vs read_all+mmap 은 페이지 전부 스캔).
        if cfg.get("skip_mixed_pack", False):
            from datasets import load_dataset as _hf_load
            ds = _hf_load("arrow", data_files=data_files, split="train", streaming=True)
            # Iterable 경로: lazy .map() — GPU forward 중 CPU worker 가 arrow 읽고 tokenize+pack.
            ds = ds.map(
                processor_fn, batched=True,
                batch_size=cfg.get("process_batch_size", 32),
                remove_columns=["utterance_id", "text", "features", "feat_len"],
                features=_PROC_FEATURES,
            )
            if cfg.get("packing", True):
                ds = ds.map(
                    packer_fn, batched=True,
                    batch_size=cfg.get("packing_bucket_size", 200),
                    features=_PACKED_CANONICAL,
                )
            any_needs_pack = True
            dataset_list.append(ds)
        else:
            # 기존 경로: 전체 read_all (RAM OOM 위험), HF map 캐시 pre-materialize.
            _tables = [_pa_ipc.open_file(f).read_all() for f in data_files]
            _table  = _pa.concat_tables(_tables) if len(_tables) > 1 else _tables[0]
            ds = _HFDataset(_table)

            # per-sample 데이터셋: processor + packer를 여기서 바로 적용
            # num_proc: 128코어 / 8 rank = 16 per rank, 캐시는 /dev/shm (RAM tmpfs)
            import os as _os
            _num_proc = max(1, _os.cpu_count() // 8)
            _shm_cache = f"/dev/shm/hf_map_cache/rank{rank}"
            _os.makedirs(_shm_cache, exist_ok=True)
            ds = ds.map(
                processor_fn,
                batched=True,
                batch_size=cfg.get("process_batch_size", 32),
                remove_columns=["utterance_id", "text", "features", "feat_len"],
                features=_PROC_FEATURES,
                num_proc=_num_proc,
                cache_file_name=f"{_shm_cache}/{ds_key}_proc.arrow",
            )
            if cfg.get("packing", True):
                ds = ds.map(
                    packer_fn,
                    batched=True,
                    batch_size=cfg.get("packing_bucket_size", 200),
                    features=_PACKED_CANONICAL,
                    num_proc=_num_proc,
                    cache_file_name=f"{_shm_cache}/{ds_key}_pack.arrow",
                )
            any_needs_pack = True
            ds = ds.to_iterable_dataset()
            dataset_list.append(ds)

    if not dataset_list:
        raise ValueError(
            f"No precomputed Arrow files found under {base_dir}. "
            f"Run: bash precompute/run_precompute.sh --encoder {enc_name}"
        )

    if any_needs_pack:
        logger.info(
            "Some datasets use on-the-fly packing. "
            f"Run `bash precompute/run_pack.sh --encoder {enc_name}` to pre-pack.",
            main_process_only=True,
        )

    ds = interleave_datasets(dataset_list, seed=cfg.get("seed", 42))
    if shuffle:
        # buffer_size: §14 — cutoff_len=16384에서 10000은 OOM. 1000으로 축소.
        ds = ds.shuffle(buffer_size=1000, seed=cfg.get("seed", 42))

    return ds


class StreamingShardedTrainer(Trainer):
    """자동 DistributedSampler 삽입을 우회하는 HF Trainer 서브클래스.

    파이프라인 시작에서 GPU별로 이미 샤딩이 완료되었으므로,
    DistributedSampler를 추가하면 이중 샤딩이 발생함. 이 서브클래스는
    sampler 래핑 없이 순수 DataLoader를 반환.

    추가로 `_save` override: Qwen3.5 tied embedding (lm_head.weight ↔ embed_tokens.weight)
    이 safetensors 공유 메모리 거부에 걸림. state_dict 에서 data_ptr 중복 탐지 후 clone
    으로 공유 해제 → safetensors 저장 성공. `save_safetensors=False` 는 transformers 5.5+
    에서 인자 자체가 제거되어 사용 불가 (§6.5), 이 override 가 유일한 우회로 (§6.4).
    """
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        # Trainer가 이미 IterableDatasetShard로 래핑했으면 언래핑
        if isinstance(train_dataset, IterableDatasetShard):
            train_dataset = train_dataset.dataset

        dataloader_params = {
            "batch_size":      self._train_batch_size,
            "collate_fn":      self.data_collator,
            "num_workers":     self.args.dataloader_num_workers,
            "pin_memory":      self.args.dataloader_pin_memory,
            "prefetch_factor": self.args.dataloader_prefetch_factor if self.args.dataloader_num_workers > 0 else None,
            "shuffle":         False,
        }

        return DataLoader(train_dataset, **dataloader_params)

    def _save(self, output_dir=None, state_dict=None):
        """Qwen3.5 tied embedding safetensors shared-memory crash 우회.

        state_dict 의 data_ptr 중복 탐지 → 나중에 나온 쪽을 clone 해서 공유 해제.
        LoRA 래핑 여부 무관 — 경로가 `llm.base_model.model.lm_head.weight` 같이 바뀌어도
        data_ptr 비교로 탐지하므로 동작. 로드 시점에 `tie_word_embeddings=True` 가 다시
        tie 복원해주므로 값 자체는 보존됨.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        seen_ptrs: dict = {}
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not hasattr(v, "data_ptr"):
                continue
            ptr = v.data_ptr()
            if ptr in seen_ptrs:
                state_dict[k] = v.clone()
            else:
                seen_ptrs[ptr] = k

        super()._save(output_dir, state_dict=state_dict)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """num_data_splits > 1 환경에서 **cross-split 단일 LR curve** 강제.

        기본 HF Trainer 는 `num_training_steps = stage2_epochs × len(current_split)` 로
        scheduler 를 만들어 split 마다 warmup + cosine 이 **리셋** 됨 → LR 톱니 패턴,
        실효 수렴 안 함.

        `_total_max_steps_override` 가 설정되어 있으면 해당 값으로 scheduler 를 구성.
        모든 split 의 bin 을 사전 스캔해서 계산한 총 step 수를 쓰므로 warmup 은 run 시작 때
        1 회, cosine 은 전체 학습 구간에 걸쳐 단조 decay.

        이 override 는 Trainer 가 **첫 split** 에서 scheduler 가 None 일 때만 작동. 두 번째
        split 이후 Trainer 는 `optimizers=(persistent_optim, persistent_sched)` 로 주입된
        scheduler 를 그대로 사용 (last_epoch 가 누적됨).
        """
        from transformers import get_scheduler

        if self.lr_scheduler is not None:
            return self.lr_scheduler

        effective = getattr(self, "_total_max_steps_override", None) or num_training_steps
        num_warmup = int(self.args.get_warmup_steps(effective))
        if optimizer is None:
            optimizer = self.optimizer
        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=effective,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )
        # HF Trainer._inner_training_loop 은 train() 시작 시 `_created_lr_scheduler`가
        # True 면 self.lr_scheduler 를 None 으로 리셋함. split 2+ 에서 scheduler 재생성
        # → warmup/LR 0 부터 재시작되는 문제 회피를 위해 False 로 남겨 "외부 주입" 처럼
        # 보이게 함. (첫 split 에서 이미 existing-scheduler 가드가 있으므로 중복 생성 없음)
        self._created_lr_scheduler = False
        return self.lr_scheduler


# ══════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════

def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _compute_wer(hyps, refs):
    total_err, total_words = 0, 0
    for hyp, ref in zip(hyps, refs):
        ref_w = ref.lower().split()
        hyp_w = hyp.lower().split()
        total_words += max(len(ref_w), 1)
        total_err   += _edit_distance(ref_w, hyp_w)
    return total_err / max(total_words, 1)


@torch.no_grad()
def _greedy_batch(model, waveforms, cfg):
    """패킹 시퀀스 모델 형식으로 waveform 배치를 greedy 디코딩.

    시퀀스 (§41 legacy p1/p2 포맷):
      [p1_embeds] + [audio_embeds] + [p2_embeds] → transcript 토큰 생성.
      p1 = "Audio:\\n", p2 = "\\nTranscript:\\n"
    """
    device = next(model.parameters()).device
    B      = len(waveforms)
    spt    = cfg["samples_per_token"]

    lengths   = [w.shape[0] for w in waveforms]
    max_len   = max(lengths)
    audio_pad = torch.zeros(B, max_len, device=device)
    for i, w in enumerate(waveforms):
        audio_pad[i, :lengths[i]] = w.to(device)
    audio_lengths = torch.tensor(lengths, device=device)

    audio_embeds = model._get_audio_embeds(audio_pad, audio_lengths)  # (B, T_proj, D)

    tokenizer = model.tokenizer
    embed     = model.llm.get_input_embeddings()

    # Stage 2 FSDP+LoRA eval 경로 방어: Qwen3.5 linear_attn 내부 in_proj_qkv 가 bf16
    # weight 를 쓰므로 입력 activation 도 bf16 이어야 함. audio_embeds / p1/p2 embeds /
    # concat 결과 모두 명시 캐스트. (proj_dtype 이 FSDP FlatParameter 하에서 fp32 로
    # 보이는 케이스 방어 — §6.19)
    audio_embeds = audio_embeds.to(torch.bfloat16)

    p1_ids = tokenizer.encode("Audio:\n",        add_special_tokens=False)
    p2_ids = tokenizer.encode("\nTranscript:\n", add_special_tokens=False)
    p1_tensor = torch.tensor([p1_ids], device=device)                     # (1, L1)
    p2_tensor = torch.tensor([p2_ids], device=device)                     # (1, L2)
    p1_embeds = embed(p1_tensor).expand(B, -1, -1).to(torch.bfloat16)     # (B, L1, D)
    p2_embeds = embed(p2_tensor).expand(B, -1, -1).to(torch.bfloat16)     # (B, L2, D)

    inputs_embeds = torch.cat([p1_embeds, audio_embeds, p2_embeds], dim=1).to(torch.bfloat16)
    attn_mask     = torch.ones(B, inputs_embeds.shape[1], device=device, dtype=torch.long)

    max_new = max(32, min(300, int(max(lengths) / max(spt, 1) / cfg["sample_rate"] * 7)))
    out = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


@torch.no_grad()
def evaluate_wer(raw_model, val_dataset, train_eval_dataset, cfg, print_n=3):
    """val (dev-clean)과 train_eval 스플릿의 WER 계산. rank-0 전용."""
    raw_model.eval()
    spt = cfg["samples_per_token"]
    max_eval_tokens = cfg["max_batch_tokens"]

    def run_split(dataset, label):
        n     = len(dataset)
        items = [dataset[i] for i in range(n)]
        order = sorted(range(n), key=lambda i: items[i][0].shape[0])

        # max_eval_tokens 초과 방지를 위해 길이 기준 greedy 배치 패킹
        batches, cur, cur_max = [], [], 0
        for idx in order:
            tok     = max(1, int(items[idx][0].shape[0] / max(spt, 1)))
            new_max = max(cur_max, tok)
            if cur and (len(cur) + 1) * new_max > max_eval_tokens:
                batches.append(cur); cur, cur_max = [idx], tok
            else:
                cur.append(idx); cur_max = new_max
        if cur:
            batches.append(cur)

        hyps = [None] * n
        for idxs in tqdm(batches, desc=f"WER[{label}]", leave=False):
            wavs = [items[i][0] for i in idxs]
            for orig_i, hyp in zip(idxs, _greedy_batch(raw_model, wavs, cfg)):
                hyps[orig_i] = hyp

        refs = [items[i][1] for i in range(n)]

        if print_n > 0:
            step = max(1, n // print_n)
            for i in range(0, n, step)[:print_n]:
                logger.info(f"  [{label}][{i:3d}] REF: {refs[i]}")
                logger.info(f"             HYP: {hyps[i]}")
        return _compute_wer(hyps, refs)

    wer_val   = run_split(val_dataset,        "val")
    wer_train = run_split(train_eval_dataset, "train")
    raw_model.train()
    return wer_val, wer_train


@torch.no_grad()
def evaluate_val_loss(raw_model, val_dataset, cfg, collator, device, max_samples=200):
    """val split의 평균 cross-entropy loss 계산. rank-0 전용.

    max_samples: 평가할 최대 샘플 수 (속도/정확도 트레이드오프).
    """
    raw_model.eval()
    tokenizer = raw_model.tokenizer
    # §41 legacy p1/p2 포맷
    p1_ids = tokenizer.encode("Audio:\n",        add_special_tokens=False)
    p2_ids = tokenizer.encode("\nTranscript:\n", add_special_tokens=False)
    hop_length = cfg["encoder"].get("hop", 1920)
    pad_id = tokenizer.pad_token_id

    # StaticEvalDataset → (waveform_1d, text) 를 모델 입력 형식으로 변환
    samples = []
    n = min(len(val_dataset), max_samples)
    for i in range(n):
        waveform, text = val_dataset[i]
        t_audio = waveform.shape[0] // hop_length
        if t_audio == 0:
            continue
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        if not text_ids:
            continue
        seq = (
            p1_ids
            + [cfg.get("audio_pad_token_id", 151655)] * t_audio
            + p2_ids
            + text_ids
            + [tokenizer.eos_token_id]
        )
        lbl = (
            [IGNORE_INDEX] * (len(p1_ids) + t_audio + len(p2_ids))
            + text_ids
            + [tokenizer.eos_token_id]
        )
        samples.append({
            "input_ids":      seq,
            "labels":         lbl,
            "audio_features": [waveform],
            "audio_lengths":  [t_audio],
            "attention_mask": [1] * len(seq),
        })

    if not samples:
        return float("nan")

    # 길이 순 정렬로 padding 최소화
    samples.sort(key=lambda s: len(s["input_ids"]))

    total_loss   = 0.0
    total_tokens = 0
    batch_size   = 4

    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        max_len = max(len(s["input_ids"]) for s in chunk)
        padded = []
        for s in chunk:
            pad_len = max_len - len(s["input_ids"])
            padded.append({
                "input_ids":      s["input_ids"]      + [pad_id]       * pad_len,
                "labels":         s["labels"]         + [IGNORE_INDEX] * pad_len,
                "audio_features": s["audio_features"],
                "audio_lengths":  s["audio_lengths"],
                "attention_mask": s["attention_mask"] + [0]            * pad_len,
            })

        batch = collator(padded)
        batch = {k: v.to(device) for k, v in batch.items()}

        out = raw_model(**batch)
        if out.loss is not None:
            n_valid      = (batch["labels"] != IGNORE_INDEX).sum().item()
            total_loss  += out.loss.item() * n_valid
            total_tokens += n_valid

    raw_model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


@torch.no_grad()
def evaluate_val_loss_precomputed(raw_model, val_dataset, cfg, device, max_samples=100):
    """precomputed 모드 전용 val loss — encoder 를 eval 시점에 돌려 features 생성.

    training forward path (encoder → projector → proj_norm → LLM) 과 동일 경로를 통과하므로
    training loss 와 직접 비교 가능. batch=1 으로 단순 루프.
    """
    raw_model.eval()
    tokenizer    = raw_model.tokenizer
    audio_pad_id = cfg.get("audio_pad_token_id", 151655)
    enc_dtype    = next(raw_model.encoder.parameters()).dtype

    # §41 legacy p1/p2 포맷
    p1_ids = tokenizer.encode("Audio:\n",        add_special_tokens=False)
    p2_ids = tokenizer.encode("\nTranscript:\n", add_special_tokens=False)

    total_loss   = 0.0
    total_tokens = 0

    n = min(len(val_dataset), max_samples)
    for i in range(n):
        waveform, text = val_dataset[i]
        text_ids = tokenizer.encode(text.lower().strip(), add_special_tokens=False)
        if not text_ids:
            continue

        wav     = waveform.unsqueeze(0).to(device)
        lengths = torch.tensor([wav.shape[-1]], device=device)
        feats, _ = raw_model.encoder(wav.to(enc_dtype), lengths)   # (1, T_enc, out_dim)
        T_enc = feats.shape[1]
        if T_enc == 0:
            continue

        T_proj = math.ceil(T_enc / raw_model._proj_stride)
        input_ids = (
            p1_ids
            + [audio_pad_id] * T_proj
            + p2_ids
            + text_ids
            + [tokenizer.eos_token_id]
        )
        labels = (
            [IGNORE_INDEX] * (len(p1_ids) + T_proj + len(p2_ids))
            + text_ids
            + [tokenizer.eos_token_id]
        )

        batch = {
            "input_ids":             torch.tensor([input_ids], dtype=torch.long, device=device),
            "labels":                torch.tensor([labels],    dtype=torch.long, device=device),
            "precomputed_enc_feats": feats.float(),                                   # (1, T_enc, out_dim)
            "audio_lengths":         torch.tensor([T_enc], dtype=torch.long, device=device),
        }
        out = raw_model(**batch)
        if out.loss is not None and torch.isfinite(out.loss):
            n_valid       = (batch["labels"] != IGNORE_INDEX).sum().item()
            total_loss   += out.loss.item() * n_valid
            total_tokens += n_valid

    raw_model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


class CumulativeWandbCallback(TrainerCallback):
    """HF Trainer 내장 WandbCallback 의 대체 + cross-split 누적 메트릭.

    main() 의 Stage 2 루프가 split 마다 `trainer.train()` 을 재호출하면 HF Trainer 의
    `state.epoch` / `state.global_step` 이 0 으로 **리셋** 됨 → wandb 에 epoch 이
    톱니 패턴으로 찍혀 시각적으로 학습 진행 감지 어려움.

    이 callback 은:
    - `epoch_offset` / `step_offset` 을 누적해 `train/epoch` / `train/global_step`
      를 **단조 증가** 로 변환
    - 각 `trainer.train()` 호출이 `1 split × num_train_epochs=1` 이라는 가정 하에,
      wandb 에 찍히는 `train/epoch` 는 `offset + state.epoch / num_data_splits` 로
      정규화 → 0 → stage2_epochs (= 전체 데이터 기준 epoch).
    - wandb.log 에 `step=` 인자 생략 → wandb 내부 auto-increment

    wandb.finish() 는 절대 호출 안 함 (main() 끝에서만).
    """
    def __init__(self, num_data_splits: int = 1,
                 initial_step_offset: int = 0,
                 initial_epoch_offset: float = 0.0,
                 cfg: dict | None = None):
        self.num_data_splits = max(1, num_data_splits)
        self.step_offset     = initial_step_offset
        self.epoch_offset    = initial_epoch_offset
        # split 경계에서 run_stage1 이 새 Trainer/Callback 를 만들어도 offset 이 이어지도록
        # cfg 에 back-reference. on_train_end 에서 cfg 에 써서 다음 split 이 preload.
        self._cfg = cfg

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or wandb.run is None:
            return
        payload = {}
        for k, v in logs.items():
            if not isinstance(v, (int, float)):
                continue
            if k == "epoch":
                # state.epoch ∈ [0, num_train_epochs=1] 가정 → split 하나가
                # 전체 데이터의 1/N 에 해당한다고 간주하여 누적 epoch 로 변환
                v = self.epoch_offset + (v / self.num_data_splits)
            key = f"train/{k}" if not k.startswith(("train/", "eval/", "val/")) else k
            payload[key] = v
        wandb_step = state.global_step + self.step_offset
        payload["train/global_step"] = wandb_step
        if payload:
            # §42: wandb 내부 step 카운터도 명시적으로 offset+global_step 으로 설정.
            # 기본 auto-increment 면 새 run 은 0 부터 시작 → resume 의 연속성 깨짐.
            # step 은 단조증가 필수 (wandb 요구사항). split 경계에서 state.global_step 이
            # 0 으로 리셋되지만 step_offset 이 이전 split 누적값이라 합은 계속 증가.
            wandb.log(payload, step=wandb_step)

    def on_train_end(self, args, state, control, **kwargs):
        # 이번 split 학습 종료 → 다음 split 을 위해 offset 이관
        self.step_offset  += state.global_step
        self.epoch_offset += (state.epoch or 0.0) / self.num_data_splits
        # cfg 에 이관 — 다음 run_stage1 이 preload 해서 이어받음
        if self._cfg is not None:
            self._cfg["_wandb_step_offset"]  = self.step_offset
            self._cfg["_wandb_epoch_offset"] = self.epoch_offset


class Stage1SplitEndSaveCallback(TrainerCallback):
    """§42: Stage 1 전용 — 매 split (tqdm 한 바퀴 = 1 `trainer.train()`) 끝에
    projector+proj_norm 저장.

    WerCallback 은 `cfg["use_fsdp"]` guard (§6.20) 로 Stage 1 에서도 skip
    (run.sh 가 `--fsdp` 를 항상 넘기고 cfg 값은 True 로 박힘). 이 callback 은
    Stage 1 trainer 에만 붙어 DDP 경로에서 작동 — projector 는 rank 간
    replicated 이므로 gather 불필요, rank0 에서 `state_dict()` 바로 필터.

    파일명: `s1_proj_split{N}.pt` (누적 split 카운터, 1-based).
    split 경계에서 callback 이 매번 재생성되므로 카운터를 cfg 에 캐리오버.
    """

    def __init__(self, accelerator, output_dir, cfg):
        self.accelerator = accelerator
        self.output_dir  = output_dir
        self.cfg         = cfg

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not self.accelerator.is_main_process or model is None:
            return
        split_idx = self.cfg.get("_s1_split_counter", 0) + 1
        self.cfg["_s1_split_counter"] = split_idx
        raw = self.accelerator.unwrap_model(model)
        proj_state = {
            k: v.detach().cpu() for k, v in raw.state_dict().items()
            if "projector" in k or "proj_norm" in k
        }
        torch.save(proj_state, os.path.join(self.output_dir, f"s1_proj_split{split_idx}.pt"))


class Stage1OptimizerSaveCallback(TrainerCallback):
    """§42 extension: Stage 1 각 split 종료 시 optimizer + scheduler state 를
    `s1_opt.pt` 로 저장 (매번 overwrite, 최종 split 상태만 보존).

    목적: 다음 run 에서 `--stage1-resume-step` 과 함께 optimizer 의 Adam m/v
    momentum 을 이어받아 full resume 근사. LR constant 환경에선 효과 작지만
    warmup/schedule 쓸 경우 필수.

    rank0 에서만 저장. FSDP 상태면 unwrap 후 `state_dict()`.
    """

    def __init__(self, accelerator, output_dir, cfg=None):
        self.accelerator = accelerator
        self.output_dir  = output_dir
        self.cfg         = cfg   # cumulative wandb step 조회용

    def on_train_end(self, args, state, control, model=None, optimizer=None,
                     lr_scheduler=None, **kwargs):
        if not self.accelerator.is_main_process:
            return
        opt_path = os.path.join(self.output_dir, "s1_opt.pt")
        # CumulativeWandbCallback.on_train_end 이 먼저 실행돼서 cfg 에 이 split 포함한
        # 누적 step 을 써 둠. auto-resume 시 이 값을 읽어서 --stage1-resume-step 으로 사용.
        cumulative_step = 0
        if self.cfg is not None:
            cumulative_step = int(self.cfg.get("_wandb_step_offset", 0))
        payload = {
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "lr_scheduler": (lr_scheduler.state_dict()
                             if lr_scheduler is not None else None),
            "split_global_step": state.global_step,
            "cumulative_step": cumulative_step,
        }
        torch.save(payload, opt_path)


class Stage1OptimizerResumeCallback(TrainerCallback):
    """§42 extension: 첫 trainer.train() 의 on_train_begin 시점에 저장된
    optimizer state 를 주입. 한 번만 발화 (self.consumed 로 guard).

    HF Trainer 는 on_train_begin 시점에 optimizer/lr_scheduler 생성 완료 +
    accelerator.prepare 된 상태 — 바로 load_state_dict 호출 안전.
    """

    def __init__(self, accelerator, opt_state_path: str):
        self.accelerator = accelerator
        self.opt_state_path = opt_state_path
        self.consumed = False

    def on_train_begin(self, args, state, control, model=None, optimizer=None,
                       lr_scheduler=None, **kwargs):
        if self.consumed:
            return
        if not os.path.exists(self.opt_state_path):
            if self.accelerator.is_main_process:
                logger.warning(f"§42 opt resume: file not found, skip ({self.opt_state_path})")
            self.consumed = True
            return
        try:
            saved = torch.load(self.opt_state_path, map_location="cpu", weights_only=False)
        except Exception as e:
            if self.accelerator.is_main_process:
                logger.warning(f"§42 opt resume: load failed ({e}), skip")
            self.consumed = True
            return
        if optimizer is not None and saved.get("optimizer"):
            try:
                optimizer.load_state_dict(saved["optimizer"])
                if self.accelerator.is_main_process:
                    logger.info(f"§42 opt resume: optimizer state loaded from {self.opt_state_path}")
            except Exception as e:
                if self.accelerator.is_main_process:
                    logger.warning(f"§42 opt resume: optimizer load_state_dict failed ({e})")
        if lr_scheduler is not None and saved.get("lr_scheduler"):
            try:
                lr_scheduler.load_state_dict(saved["lr_scheduler"])
                if self.accelerator.is_main_process:
                    logger.info(f"§42 opt resume: lr_scheduler state loaded")
            except Exception as e:
                if self.accelerator.is_main_process:
                    logger.warning(f"§42 opt resume: lr_scheduler load_state_dict failed ({e})")
        self.consumed = True


class WerCallback(TrainerCallback):
    def __init__(self, accelerator, val_dataset, train_eval_dataset, cfg,
                 output_dir, collator, eval_every_n_steps=500):
        self.accelerator        = accelerator
        self.val_dataset        = val_dataset
        self.train_eval_dataset = train_eval_dataset
        self.cfg                = cfg
        self.output_dir         = output_dir
        self.collator           = collator
        self.eval_every         = eval_every_n_steps
        # num_data_splits>1 이면 run_stage1 이 split 마다 새 Trainer/callback 생성 →
        # best_wer_val 리셋으로 "현재 split 내 best" 만 저장됨. cfg 에 캐리오버하여
        # 전체 학습 범위의 best WER 를 추적.
        self.best_wer_val       = cfg.get("_best_wer_val", float("inf"))

    def _save_proj(self, raw_model, filename):
        proj_state = {
            k: v.cpu() for k, v in raw_model.state_dict().items()
            if "projector" in k or "proj_norm" in k
        }
        torch.save(proj_state, os.path.join(self.output_dir, filename))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_every != 0:
            return

        # Stage 2 FSDP + LoRA + 커스텀 in-loop eval 조합에서 NCCL deadlock 반복 (§6.20).
        # rank0 만 summon_full_params 에 진입하고 나머지는 barrier 에 남는 경로 문제.
        # WER/val_loss 는 HF Trainer 의 save_steps 체크포인트에서 offline 계산하도록 우회.
        if self.cfg.get("use_fsdp", False):
            return

        step = state.global_step
        self.accelerator.wait_for_everyone()
        # 모든 랭크가 full params 보유해야 rank-0 eval 중 FSDP hook 이 추가 allgather
        # 를 시도하지 않음. rank0_only=True 로는 rank-0 의 generate() 가 재-allgather
        # 를 시도하다 다른 rank 들과 데드락 → NCCL timeout (§6.20).
        with FSDP.summon_full_params(model, recurse=True, writeback=False, rank0_only=False):
            if self.accelerator.is_main_process:
                raw_model = self.accelerator.unwrap_model(model)
                wer_val, wer_train = evaluate_wer(
                    raw_model, self.val_dataset, self.train_eval_dataset, self.cfg,
                )
                # precomputed 모드: encoder 직접 호출하는 전용 경로 사용.
                # raw audio 모드: 기존 collator 기반 경로.
                if self.cfg.get("precomputed_dir"):
                    val_loss = evaluate_val_loss_precomputed(
                        raw_model, self.val_dataset, self.cfg,
                        device=self.accelerator.device,
                    )
                else:
                    val_loss = evaluate_val_loss(
                        raw_model, self.val_dataset, self.cfg, self.collator,
                        device=self.accelerator.device,
                    )
                logger.info(
                    f"  [Step {step}] WER val={wer_val*100:.1f}%  train={wer_train*100:.1f}%"
                    f"  val_loss={val_loss:.4f}"
                )

                self._save_proj(raw_model, f"s1_proj_step{step}.pt")

                if wer_val < self.best_wer_val:
                    self.best_wer_val = wer_val
                    self.cfg["_best_wer_val"] = wer_val   # 다음 split callback 이 이어받음
                    self._save_proj(raw_model, "best_s1_proj.pt")
                    logger.info(f"  [Best WER={wer_val*100:.1f}%] best_s1_proj.pt 저장")

                if wandb.run is not None:
                    # step 인자 생략 → wandb auto-increment (CumulativeWandbCallback 과 동일 정책)
                    wandb.log({"val/wer": wer_val, "train/wer": wer_train, "val/loss": val_loss})
        self.accelerator.wait_for_everyone()


# ══════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════

def calculate_max_steps(
    total_estimated_hours: float,
    cutoff_len: int,
    per_device_batch_size: int,
    num_processes: int,
    grad_accum_steps: int,
    hop_length: int = 1920,
    sample_rate: int = 16000,
    target_epochs: float = 1.0,
) -> int:
    """raw audio (streaming) 모드 전용 max_steps 추정.

    precomputed (mixed) 모드에서는 build_precomputed_pipeline 가 _table.num_rows
    기반으로 cfg["max_steps"] 를 직접 채워 넣으므로 이 함수는 호출되지 않음 (§19).
    """
    tokens_per_second    = sample_rate / hop_length
    audio_tokens_per_pack = cutoff_len * 0.8
    seconds_per_pack     = audio_tokens_per_pack / tokens_per_second

    total_seconds        = total_estimated_hours * 3600
    estimated_total_packs = total_seconds / seconds_per_pack

    global_batch_size    = per_device_batch_size * num_processes * grad_accum_steps
    steps_per_epoch      = math.ceil(estimated_total_packs / global_batch_size)

    return int(steps_per_epoch * target_epochs)


def run_stage1(cfg, accelerator, model, train_dataset, val_dataset, train_eval_dataset,
               collator, run_id: str, resume=None,
               persistent_opt_sched=None, total_s1_steps: int = 0):
    """Stage 1: LLM frozen 상태에서 Projector alignment. (output_dir, total_steps, opt_sched) 반환.

    §42 ext: cross-split 단일 cosine + warmup curve 를 위해 persistent optimizer/scheduler 를
    외부에서 주입 가능. 첫 split 호출은 persistent_opt_sched=None 으로 들어오고, 반환된
    (opt, sched) 튜플을 이후 split 호출 시 다시 넘겨주면 HF Trainer 가 create_optimizer/
    create_scheduler 를 skip 하고 누적된 상태 그대로 재사용.

    total_s1_steps > 0 이면 `StreamingShardedTrainer._total_max_steps_override` 로 전달 →
    scheduler 가 per-split 이 아닌 전체 Stage 1 구간 기준으로 warmup+cosine 을 구성.
    """

    if cfg.get("max_steps") is None:
        # §19: precomputed 모드는 build_precomputed_pipeline 가 _table.num_rows 로 미리 채움.
        #      여기 도달했다는 건 raw audio 모드 또는 cfg["max_steps"] 미설정 케이스.
        cfg["max_steps"] = calculate_max_steps(
            total_estimated_hours=cfg["estimated_hours"],
            cutoff_len=cfg["packing_cutoff_len"],
            per_device_batch_size=cfg.get("per_device_train_batch_size", 10),
            num_processes=accelerator.num_processes,
            grad_accum_steps=cfg.get("gradient_accumulation_steps", 4),
            hop_length=cfg["encoder"].get("hop", 512),
            sample_rate=cfg.get("sample_rate", 16000),
            target_epochs=cfg["stage1_epochs"],
        )
        logger.info(
            f"Stage 1: Computed max_steps={cfg['max_steps']} (raw audio fallback)",
            main_process_only=True,
        )

    logger.info(f"{'='*55}",)
    logger.info(f" Stage 1: Projector Alignment  LR={cfg['stage1_lr']} Max Steps={cfg['max_steps']}",)
    logger.info(f"{'='*55}",)
    
    model.freeze_llm()
    output_dir = os.path.join(cfg["model_cache_dir"], cfg["encoder_name"], f"s1_outputs_{run_id}")

    # IterableDataset (no __len__) 은 TrainingArguments 에 per-call max_steps 필수.
    # 해당 모드:
    #   - raw audio (precomputed_dir 없음)
    #   - §42+ skip_mixed_pack=True (precomputed_dir 있지만 lazy arrow iter)
    # precomputed mixed-pack map-style 모드는 num_train_epochs 로 자동 계산 → max_steps=-1.
    _is_iter_dataset = not cfg.get("precomputed_dir") or cfg.get("skip_mixed_pack", False)
    _per_call_max = -1
    if cfg.get("max_steps") and _is_iter_dataset:
        _n_splits = max(1, cfg.get("num_data_splits", 1))
        _n_epochs = max(1, cfg.get("stage1_epochs", 1))
        _per_call_max = max(1, int(cfg["max_steps"]) // (_n_splits * _n_epochs))

    training_args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        max_steps=_per_call_max,
        num_train_epochs=cfg["stage1_epochs"],
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 10),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg["stage1_lr"],
        # §42 ext: constant → cosine + warmup. Projector random init 초기 step 보호 (warmup),
        # 전체 학습 후반 LR decay (cosine) 로 fine-grained convergence.
        lr_scheduler_type=cfg.get("stage1_lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        optim="adamw_torch_fused",

        # 주의: TrainingArguments의 gradient_checkpointing은 FSDP backward에서 불필요한
        # AllGather를 추가함; 대신 fsdp_config의 activation_checkpointing 사용 권장.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_steps=cfg.get("log_every", 10),
        # Stage 1: LLM frozen, projector 만 학습 → full-model checkpoint 불필요.
        # 또한 Qwen3.5 의 tied embedding (lm_head.weight = embed_tokens.weight) 이
        # safetensors 공유 메모리 거부에 걸려 save 시 RuntimeError 발생.
        # WerCallback 이 eval 시점에 projector state 만 직접 저장하므로 HF Trainer save 는 disable.
        save_strategy="no",
        save_total_limit=5,
        # HF Trainer 내장 WandbCallback 은 split 경계마다 wandb.finish() 호출 + step 리셋 →
        # num_data_splits>1 환경에서 split 2+ 로그가 사라짐. CumulativeWandbCallback 이 대체.
        report_to="none",

        remove_unused_columns=False,
        # Stage 1: LLM frozen, projector만 학습 → 기본 DDP.
        # --fsdp-stage1 활성화 시 FSDP full_shard: LLM 메모리 1/8 절감 대신 forward all-gather 추가.
        **({
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "fsdp_transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"],
                "fsdp_use_orig_params": True,
                "fsdp_backward_prefetch": "backward_pre",
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                "fsdp_ignored_modules": ["encoder"],
            },
        } if cfg.get("use_fsdp_stage1") else {"ddp_find_unused_parameters": False}),

        # precomputed 모드: rank당 파일 1개 → num_shards=1 → worker 1개면 충분.
        # raw audio 모드: 여러 shard 파일 → worker 8개로 병렬 로딩.
        # §42+ skip_mixed_pack 모드: mmap iterable + lazy tokenize+pack → CPU worker 4개 로 GPU 오버랩.
        dataloader_num_workers=(4 if cfg.get("skip_mixed_pack") else
                                (0 if cfg.get("precomputed_dir") else 8)),
        dataloader_prefetch_factor=(2 if cfg.get("skip_mixed_pack") or not cfg.get("precomputed_dir") else None),
    )

    wer_callback = WerCallback(
        accelerator=accelerator,
        val_dataset=val_dataset,
        train_eval_dataset=train_eval_dataset,
        cfg=cfg,
        output_dir=output_dir,
        collator=collator,
        eval_every_n_steps=cfg["eval_steps"],
    )

    # §42: Stage 1 매 split 종료 시 projector 저장 (WerCallback 은 use_fsdp guard 로 skip).
    split_save_callback = Stage1SplitEndSaveCallback(
        accelerator=accelerator,
        output_dir=output_dir,
        cfg=cfg,
    )

    # §42 Stage 1 resume: 이전 run 의 마지막 step 에 이어서 wandb 에 기록하려면
    # CumulativeWandbCallback 의 step/epoch offset 을 cfg 에서 받아 초기화.
    # split 경계에서 run_stage1() 이 매번 재호출되므로 cfg 에 캐리오버.
    wandb_cb = CumulativeWandbCallback(
        num_data_splits=cfg.get("num_data_splits", 1),
        initial_step_offset=cfg.get("_wandb_step_offset", 0),
        initial_epoch_offset=cfg.get("_wandb_epoch_offset", 0.0),
        cfg=cfg,
    )

    # §42 ext: optimizer state save callback — 매 split 끝에 s1_opt.pt overwrite.
    opt_save_callback = Stage1OptimizerSaveCallback(
        accelerator=accelerator,
        output_dir=output_dir,
        cfg=cfg,
    )
    callbacks = [wer_callback, wandb_cb, split_save_callback, opt_save_callback]

    # Resume 경로: cfg["_s1_resume_opt_path"] 가 있으면 첫 split 의 on_train_begin
    # 에서 optimizer state 로드. 소비 후 cfg key 제거 (다음 split 은 reset 유지).
    # 구조상 split 경계에서 Trainer 가 새로 만들어지므로 optimizer 는 매번 reset —
    # 따라서 **resume 첫 split 에서만** 이어받는 효과. LR constant 환경에선 충분.
    if cfg.get("_s1_resume_opt_path"):
        opt_resume_callback = Stage1OptimizerResumeCallback(
            accelerator=accelerator,
            opt_state_path=cfg["_s1_resume_opt_path"],
        )
        callbacks.append(opt_resume_callback)
        cfg.pop("_s1_resume_opt_path", None)

    # §42 ext: persistent optimizer/scheduler 주입 (이후 split 에서 같은 opt/sched 재사용).
    # 첫 split 에선 None → Trainer 가 내부 생성. 이후 튜플 주입 → HF Trainer 가 skip create.
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    if persistent_opt_sched is not None:
        trainer_kwargs["optimizers"] = persistent_opt_sched

    trainer = StreamingShardedTrainer(**trainer_kwargs)

    # cross-split 단일 cosine curve 를 위해 _total_max_steps_override 설정.
    # StreamingShardedTrainer.create_scheduler 가 이 값을 읽어 scheduler 를 구성.
    if total_s1_steps > 0:
        trainer._total_max_steps_override = total_s1_steps

    resume_ckpt = resume if resume != "latest" else True
    trainer.train(resume_from_checkpoint=resume_ckpt)
    s1_total_steps = trainer.state.global_step

    s1_proj_path = os.path.join(output_dir, "s1_proj.pt")
    accelerator.wait_for_everyone()
    # accelerator.get_state_dict: FSDP/DDP 무관하게 full params gather (모든 rank 참여 필요)
    full_state = accelerator.get_state_dict(trainer.model)
    if accelerator.is_main_process:
        proj_state = {
            k: v.cpu() for k, v in full_state.items()
            if "projector" in k or "proj_norm" in k
        }
        torch.save(proj_state, s1_proj_path)
        logger.info(f"Stage 1 projector saved: {s1_proj_path}")

    # wandb.finish() 는 split loop 전체가 끝난 뒤 main() 에서만 호출.
    accelerator.wait_for_everyone()

    # §42 ext: optimizer/scheduler 를 다음 split 에 전달. HF Trainer 가 accelerator.prepare
    # 결과를 self.optimizer, self.lr_scheduler 에 유지하고 있으므로 그대로 추출 가능.
    opt_sched_out = (trainer.optimizer, trainer.lr_scheduler)

    return output_dir, s1_total_steps, opt_sched_out


def setup_stage2_model(cfg, accelerator, s1_output_dir):
    """Stage 2 모델 빌드 + Stage 1 projector 로드 + bf16 cast.

    split 루프에 걸쳐 **한 번만** 호출. 이후 main() 의 루프가 이 모델을 재사용하여
    optimizer/scheduler state 를 유지하며 LR 을 cross-split 연속으로 뽑는다.
    """
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"{'='*55}",)
    logger.info(
        f" Stage 2: LoRA Fine-tuning  LR={cfg['stage2_lr']}  "
        f"warmup={cfg.get('warmup_ratio', 0.03)} epochs={cfg['stage2_epochs']}",
    )
    logger.info(f"{'='*55}",)

    model = build_model(cfg, accelerator)
    model.apply_lora()

    # Stage 1 projector 가중치 로드 — **현 run 의 것만** 사용.
    # §42: 이전 run 자동 복구 로직 제거. "기존 s1 은 무시하고 새로 s1 부터 학습" 보장.
    # 현 run 의 s1_proj.pt 가 없으면 즉시 실패 (Stage 1 을 먼저 돌려야 함).
    s1_proj_path = os.path.join(s1_output_dir, "s1_proj.pt")
    if not os.path.exists(s1_proj_path):
        raise RuntimeError(
            f"Stage 1 projector not found at {s1_proj_path}. "
            f"Stage 2 를 독립 실행하려면 먼저 `--stage 1` 로 projector 를 학습해야 합니다. "
            f"(이전 run 자동 복구는 §42 에서 비활성화됨.)"
        )

    logger.info(f"Stage 1 projector loaded: {s1_proj_path}")
    proj_state = torch.load(s1_proj_path, map_location="cpu", weights_only=True)
    proj_state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                  for k, v in proj_state.items()}
    model.load_state_dict(proj_state, strict=False)

    # FSDP 는 shard 내 dtype 균일성 요구. LoRA init fp32 + projector 로드 잔존 fp32 → bf16.
    _fp32_params = [(n, p.dtype) for n, p in model.named_parameters()
                    if p.is_floating_point() and p.dtype != torch.bfloat16]
    if _fp32_params:
        logger.info(
            f"Casting {len(_fp32_params)} non-bf16 param(s) to bf16 for FSDP: "
            + ", ".join(n for n, _ in _fp32_params[:5])
            + ("..." if len(_fp32_params) > 5 else "")
        )
        for param in model.parameters():
            if param.is_floating_point() and param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)

    return model


def build_stage2_trainer(cfg, accelerator, model, first_train_dataset,
                         val_dataset, train_eval_dataset, collator,
                         s2_output_dir: str, total_max_steps: int,
                         num_data_splits: int = 1):
    """Stage 2 용 `StreamingShardedTrainer` 인스턴스를 **단 한 번** 생성.

    main() 의 split 루프는 이 Trainer 를 재사용 — 매 split 마다 `trainer.train_dataset`
    를 교체하고 `trainer.train()` 재호출. Trainer 가 한 번만 FSDP 래핑하므로 optimizer 가
    일관된 param 참조를 유지하고, `_total_max_steps_override` 기반 scheduler 가
    cross-split 단일 LR curve 를 뽑는다.

    `first_train_dataset` 은 split 0 의 dataset (placeholder) — Trainer 초기화에만 사용.
    """
    wer_callback = WerCallback(
        accelerator=accelerator,
        val_dataset=val_dataset,
        train_eval_dataset=train_eval_dataset,
        cfg=cfg,
        output_dir=s2_output_dir,
        collator=collator,
        eval_every_n_steps=cfg["eval_steps"],
    )

    training_args = TrainingArguments(
        output_dir=s2_output_dir,
        bf16=True,
        # 각 trainer.train() = 1 split × 1 local epoch. 외부 main() 루프가
        # stage2_epochs 번 전체 split 을 shuffled 순서로 순회.
        num_train_epochs=1,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 10),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg["stage2_lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=0.01,
        optim="adamw_torch_fused",

        # FSDP full_shard 하에서 HF 의 gradient_checkpointing=True 는 backward 에
        # redundant AllGather 삽입 → fsdp_config.activation_checkpointing 으로 이관.
        # (ref HF #30404) FSDP off 시에는 기존대로 HF grad checkpoint 사용.
        gradient_checkpointing=not cfg.get("use_fsdp", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_steps=cfg.get("log_every", 10),
        # HF Trainer 내장 WandbCallback 은 split 경계마다 wandb.finish() 호출 + step 리셋 →
        # num_data_splits>1 환경에서 split 2+ 로그가 사라짐. CumulativeWandbCallback 이 대체.
        report_to="none",
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        # Qwen3.5 tied embedding 공유 메모리 거부는 `StreamingShardedTrainer._save` override
        # (data_ptr 중복 탐지 + clone) 에서 처리. transformers 5.5+ 는 `save_safetensors`
        # 인자 제거돼서 사용 불가 (§6.5).

        remove_unused_columns=False,
        dataloader_num_workers=0 if cfg.get("precomputed_dir") else 8,
        dataloader_prefetch_factor=2 if not cfg.get("precomputed_dir") else None,

        **({
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "fsdp_transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"],
                "fsdp_use_orig_params": True,
                "fsdp_backward_prefetch": "backward_pre",
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                "limit_all_gathers": True,
                "fsdp_ignored_modules": ["encoder"],
                "activation_checkpointing": True,
            },
        } if cfg.get("use_fsdp", True) else {}),
    )

    trainer = StreamingShardedTrainer(
        model=model,
        args=training_args,
        train_dataset=first_train_dataset,
        data_collator=collator,
        callbacks=[
            wer_callback,
            CumulativeWandbCallback(num_data_splits=num_data_splits),
        ],
    )
    # cross-split 단일 LR curve: `StreamingShardedTrainer.create_scheduler` 가
    # 첫 train() 호출 시 이 값을 기준으로 warmup+cosine 을 구성. 이후 split 은 동일
    # scheduler 가 유지되어 last_epoch 누적 → LR 단조 decay.
    if total_max_steps > 0:
        trainer._total_max_steps_override = total_max_steps

    return trainer


# ══════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True,
                        choices=["encodec", "dac", "fb_dacvae", "mimi_acoustic", "mimi_semantic"],
                        help="사용할 audio encoder")
    parser.add_argument("--llm", default=None, help="LLM 모델 이름")
    parser.add_argument("--cache-dir", default=None, help="모델 캐시 경로 (기본: TRAIN_CONFIG의 model_cache_dir)")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--datasets", default=None, type=_parse_datasets,
                        metavar="ls100,ls360,ls500,mls,gs,vp",
                        help="사용할 데이터셋 (쉼표 구분). 기본: ls100,ls360,ls500,mls,gs,vp")
    parser.add_argument("--estimated-hours", default=None, type=float,
                        help="총 데이터셋 시간(시간 단위) 수동 지정. 미지정 시 --datasets 기반 자동 계산")
    parser.add_argument("--max-steps", default=None, type=int,
                        help="학습을 종료할 최대 Step 수 (스트리밍 전용). 미지정 시 estimated_hours 기반 계산")    
    parser.add_argument("--cutoff-len", default=None, type=int, help="Packing 시퀀스 최대 길이 (기본: config의 packing_cutoff_len=2048)")
    parser.add_argument("--eval-steps", default=None, type=int, help="WER 평가 주기 (기본: config의 eval_steps=500)")
    parser.add_argument("--save-steps", default=None, type=int, help="Trainer 체크포인트 저장 주기 (기본: config의 save_steps=5000)")
    parser.add_argument("--resume", default=None, help="체크포인트에서 재개")
    parser.add_argument("--stage1-start-split", default=0, type=int,
                        help="Stage 1 split loop 시작 idx (0-based). 기본 0. "
                             "이전 run 의 s1_proj.pt 가 있으면 projector 를 preload 한 뒤 "
                             "지정 split 부터 이어서 학습 (Adam/scheduler 는 리셋)")
    parser.add_argument("--stage1-resume-step", default=0, type=int,
                        metavar="N",
                        help="§42: Stage 1 resume 전용 — 이전 run 의 최신 s1_proj.pt 를 preload "
                             "하고 wandb step_offset 을 N 으로 시작 (첫 새 optimizer step → step N+1 로 기록). "
                             "추가 epoch 은 --stage1-epochs 로 지정. "
                             "0 = 비활성. -1 = auto (s1_opt.pt 의 cumulative_step 자동 감지)")
    parser.add_argument("--stage", choices=["all", "1", "2"], default="all",
                        help="실행할 스테이지: all (1→2, 기본값), 1, or 2")
    parser.add_argument("--stage1-epochs", type=int, default=None,
                        metavar="N", help="Stage 1 epoch 수 (기본: config 값)")
    parser.add_argument("--stage2-epochs", type=int, default=None,
                        metavar="N", help="Stage 2 epoch 수 (기본: config 값)")
    # 기능 플래그
    parser.add_argument("--attn-impl", default=None,
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Attention 구현체 (기본: config의 flash_attention_2)")
    parser.add_argument("--liger", default=True, action=argparse.BooleanOptionalAction,
                        help="Liger fused kernel 활성화/비활성화 (기본: --liger)")
    parser.add_argument("--fsdp", default=True, action=argparse.BooleanOptionalAction,
                        help="Stage 2 FSDP 활성화/비활성화 (기본: --fsdp)")
    parser.add_argument("--fsdp-stage1", default=False, action=argparse.BooleanOptionalAction,
                        help="Stage 1 FSDP 활성화 (기본: --no-fsdp-stage1 = DDP)")
    parser.add_argument("--word-aug", action="store_true", default=False,
                        help="단어 단위 ASR 서브샘플 생성 활성화 (word alignment Arrow 필요)")
    parser.add_argument("--precomputed-dir", default=None,
                        help="Pre-computed encoder feature 디렉토리 "
                             "(기본: None = raw audio 모드). "
                             "예: /mnt/fr20tb/wbl_residency/jos/ddn/precomputed")
    parser.add_argument("--skip-mixed-pack", action="store_true", default=False,
                        help="precomputed dir 내 mixed/packed_* 디렉토리를 무시하고 "
                             "per-dataset rank*.arrow (encoder features) 를 memory-map 으로 "
                             "직접 읽어 runtime pack. RAM OOM 회피 + offline pack 우회.")
    args = parser.parse_args()

    cfg = get_config(args.encoder)
    if args.llm:       cfg["llm_model"]          = args.llm
    if args.cache_dir: cfg["model_cache_dir"]    = args.cache_dir
    if args.attn_impl: cfg["attn_implementation"] = args.attn_impl
    cfg["use_liger_kernel"] = args.liger
    cfg["use_fsdp"]         = args.fsdp
    cfg["use_fsdp_stage1"]  = args.fsdp_stage1
    cfg["wandb_mode"]         = args.wandb_mode
    cfg["packing_cutoff_len"] = args.cutoff_len if args.cutoff_len is not None else cfg["packing_cutoff_len"]
    cfg["eval_steps"]         = args.eval_steps  if args.eval_steps  is not None else cfg.get("eval_steps",  500)
    cfg["save_steps"]         = args.save_steps if args.save_steps is not None else cfg.get("save_steps", 5000)
    cfg["max_steps"]          = args.max_steps
    if args.stage1_epochs is not None: cfg["stage1_epochs"] = args.stage1_epochs
    if args.stage2_epochs is not None: cfg["stage2_epochs"] = args.stage2_epochs
    cfg["precomputed_dir"]  = args.precomputed_dir   # None = raw audio 모드
    cfg["skip_mixed_pack"]  = args.skip_mixed_pack   # True = mixed/ 건너뛰고 per-sample memory-map

    selected_datasets = args.datasets or ["ls100", "ls360", "ls500", "mls", "gs", "vp"]
    cfg["estimated_hours"] = (
        args.estimated_hours if args.estimated_hours is not None
        else estimate_total_hours(selected_datasets)
    )

    os.makedirs(cfg["model_cache_dir"], exist_ok=True)
    os.environ.setdefault("HF_HOME", cfg["model_cache_dir"])

    if cfg.get("wandb_mode") == "disabled":
        os.environ["WANDB_MODE"] = "disabled"

    # num_data_splits>1: split 마다 새 Trainer → split 1 끝에 HF Trainer WandbCallback 이
    # wandb.finish() 호출 → split 2~N 의 Trainer 가 새 wandb run 을 기본 project "huggingface"
    # 로 떨어뜨림. WANDB_PROJECT 환경변수를 박아 HF Trainer 의 auto-init 가 우리 project 를
    # 사용하게 하고, WANDB_RUN_ID + WANDB_RESUME=allow 로 모든 split 을 동일 run 에 이어붙임.
    import datetime
    run_id = datetime.datetime.now().strftime("%m%d_%H%M")
    llm_tag  = "2b" if "2B" in cfg.get("llm_model", "") else "4b"
    stage_tag = {"1": "S1", "2": "S2", "all": "S1S2"}.get(args.stage, "S1")
    run_name = f"{cfg.get('encoder_name', 'encoder')}_{llm_tag}_{stage_tag}_{run_id}"
    if cfg.get("wandb_mode") != "disabled":
        os.environ["WANDB_PROJECT"] = cfg.get("project_name", "audio-qwen")
        os.environ.setdefault("WANDB_RUN_ID", run_name)
        os.environ["WANDB_RESUME"]  = "allow"

    accelerator = Accelerator(log_with="wandb")

    if accelerator.is_main_process and cfg.get("wandb_mode") != "disabled":
        wandb.init(project=cfg.get("project_name", "audio-qwen"), config=cfg,
                   name=run_name, id=run_name, resume="allow",
                   mode=cfg.get("wandb_mode", "online"))

    # ── Startup configuration summary ────────────────────────────────────────
    if accelerator.is_main_process:
        _w = 56
        logger.info("=" * _w)
        logger.info(f"  PIPELINE CONFIG")
        logger.info("=" * _w)
        logger.info(f"  Encoder       : {cfg['encoder_name']}")
        logger.info(f"  LLM           : {cfg['llm_model']}")
        logger.info(f"  Stage(s)      : {args.stage}")
        logger.info(f"  Datasets      : {', '.join(selected_datasets)}")
        logger.info(f"  Est. hours    : {cfg['estimated_hours']:.0f}h")
        logger.info(f"  Cutoff len    : {cfg['packing_cutoff_len']} tokens")
        logger.info("  ── Optimizations ──────────────────────────────")
        logger.info(f"  Seq Packing   : ✓ (always on, cutoff={cfg['packing_cutoff_len']})")
        logger.info(f"  Flash Attn 2  : {'✓' if cfg.get('attn_implementation') == 'flash_attention_2' else '✗ (' + cfg.get('attn_implementation', '?') + ')'}")
        logger.info(f"  Liger Kernel  : {'✓' if cfg.get('use_liger_kernel') else '✗'}")
        logger.info(f"  FSDP (Stage1) : {'✓' if cfg.get('use_fsdp_stage1') else '✗ (DDP)'}")
        logger.info(f"  FSDP (Stage2) : {'✓' if cfg.get('use_fsdp') else '✗'}")
        logger.info("  ── Word Augmentation ──────────────────────────")
        logger.info(f"  Word-aug      : {'✓' if args.word_aug else '✗ (disabled)'}")
        if args.word_aug:
            for ds_key in selected_datasets:
                arrow_path = _ALIGNMENT_ARROW_PATHS.get(ds_key, "")
                exists = os.path.exists(arrow_path) if arrow_path else False
                status = f"✓ {arrow_path.split('/')[-2]}/{arrow_path.split('/')[-1]}" if exists else "✗ arrow not found"
                logger.info(f"    {ds_key:<8}: {status}")
        logger.info("  ── Training Schedule ──────────────────────────")
        logger.info(f"  Stage1 epochs : {cfg.get('stage1_epochs', 2)}")
        logger.info(f"  Stage2 epochs : {cfg.get('stage2_epochs', 2)}")
        logger.info(f"  Eval steps    : {cfg['eval_steps']}")
        logger.info(f"  Save steps    : {cfg['save_steps']}")
        logger.info("=" * _w)
    # ─────────────────────────────────────────────────────────────────────────

    # 1. Build model (AudioQwen with encoder + projector + LLM)
    model     = build_model(cfg, accelerator)
    tokenizer = model.tokenizer

    # 2. Build data pipeline components
    logger.info("Initializing Omni Data Pipeline Components...",)

    # Word-aug: AlignmentLookup 로드 (word_aug=True 이고 rank-0에서만 먼저 로드 후 fork)
    merged_alignment: "MergedAlignmentLookup | None" = None
    if args.word_aug:
        lookups_dict = build_alignment_lookups(selected_datasets)
        merged_alignment = MergedAlignmentLookup(list(lookups_dict.values()))
        logger.info(f"Word-aug enabled. Loaded {sum(1 for v in lookups_dict.values() if v)} alignment lookups.")

    packer_fn = create_packer(
        cutoff_len=cfg["packing_cutoff_len"],
        pad_token_id=tokenizer.pad_token_id,
        neat_packing=True,
    )

    precomputed_dir = cfg.get("precomputed_dir")   # None = raw audio 모드

    num_data_splits = cfg.get("num_data_splits", 1)
    # Raw audio / skip_mixed_pack 모드: lazy iter 데이터셋 → "split" 개념 없음. 1 로 고정.
    _is_iter_mode = not precomputed_dir or cfg.get("skip_mixed_pack", False)
    if _is_iter_mode and num_data_splits > 1:
        logger.info(f"Iter 모드 감지 — num_data_splits {num_data_splits} → 1 로 강제 override",
                    main_process_only=True)
        num_data_splits = 1
        cfg["num_data_splits"] = 1

    # §42+ Runtime gs tag mask: sentence_pack_arrow.py 가 최신 (태그 제거됨) 이지만
    # 기존 pack 의 경우 labels 에 literal <TAG> 토큰 포함. ' <' (mid-sentence) / '<'
    # (sentence-start) / '>' 모두 single token. 예기치 않게 multi-token 되면 mask 비활성.
    _open_mid   = tokenizer.encode(" <", add_special_tokens=False)
    _open_start = tokenizer.encode("<",  add_special_tokens=False)
    _close_ids  = tokenizer.encode(">",  add_special_tokens=False)
    _open_ids: list[int] = []
    if len(_open_mid)   == 1: _open_ids.append(_open_mid[0])
    if len(_open_start) == 1 and _open_start[0] not in _open_ids:
        _open_ids.append(_open_start[0])
    _close_tag_id = _close_ids[0] if len(_close_ids) == 1 else None
    if not _open_ids or _close_tag_id is None:
        logger.warning(f"tag-mask disabled: mid={_open_mid}, start={_open_start}, close={_close_ids}")
        _open_tag_ids = None
    else:
        _open_tag_ids = tuple(_open_ids)
        logger.info(f"tag-mask enabled: open={_open_tag_ids}, close={_close_tag_id}")

    if precomputed_dir:
        # ── Precomputed 모드: Arrow 파일에서 인코더 피처 직접 로드 ──────────
        logger.info(f"Precomputed mode: loading encoder features from {precomputed_dir}")
        processor_fn = make_precomputed_processor_fn(cfg, tokenizer)
        collator = OmniCollator(
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
            block_diag_attn=True,
            enc_out_dim=model.encoder.out_dim,
            open_tag_ids=_open_tag_ids,
            close_tag_id=_close_tag_id,
        )
    else:
        # ── Raw audio 모드 (기존): HF streaming + on-the-fly encoding ───────
        processor_fn = create_processor(
            tokenizer=tokenizer,
            audio_pad_token_id=cfg.get("audio_pad_token_id", 151655),
            sample_rate=cfg.get("sample_rate", 16000),
            alignment_lookup=merged_alignment,
            word_aug=args.word_aug,
        )
        collator = OmniCollator(
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
            block_diag_attn=True,
            enc_out_dim=0,   # raw waveform 모드
            open_tag_ids=_open_tag_ids,
            close_tag_id=_close_tag_id,
        )

    # 5. Static eval datasets for WER callback
    val_dataset, train_eval_dataset = build_static_eval_datasets(cfg, tokenizer)

    # 7. Stage 1: projector alignment
    s1_output_dir = os.path.join(cfg["model_cache_dir"], cfg["encoder_name"], f"s1_outputs_{run_id}")
    if args.stage in ("all", "1"):
        # §42 split shuffle RNG — Stage 2 와 동일 방식 (run_id 기반 deterministic seed,
        # 모든 rank 동일 순서). Stage 1 에서도 shard 레벨 셔플을 하고 싶다는 요구 (extra epoch).
        import random as _s1_random
        _s1_split_rng = _s1_random.Random(int(run_id.replace("_", "")))

        def _s1_shuffled_split_order() -> list[int]:
            order = list(range(num_data_splits))
            _s1_split_rng.shuffle(order)
            return order

        # §42 Stage 1 resume:
        #   --stage1-resume-step N (N>0): 이전 run 의 최신 s1_proj.pt 를 preload 하고
        #     wandb step_offset 을 N 으로 시작. 추가 epoch 을 shuffled 순서로 돌림.
        #   --stage1-start-split N (legacy): 같은 run 의 split 0..N-1 을 건너뛰고 N 부터 순차.
        resume_step = args.stage1_resume_step
        # §42 ext: resume_step == -1 이면 최신 s1_opt.pt 의 cumulative_step 자동 감지.
        if resume_step == -1:
            enc_dir_auto = os.path.join(cfg["model_cache_dir"], cfg["encoder_name"])
            opt_candidates = sorted(
                [d for d in os.listdir(enc_dir_auto)
                 if d.startswith("s1_outputs_")
                 and os.path.exists(os.path.join(enc_dir_auto, d, "s1_opt.pt"))]
            ) if os.path.isdir(enc_dir_auto) else []
            if not opt_candidates:
                raise RuntimeError(
                    f"--stage1-resume-step=-1 (auto) 요청됐으나 "
                    f"{enc_dir_auto}/s1_outputs_*/s1_opt.pt 없음. "
                    f"--stage1-resume-step N 으로 명시하거나 이전 run 이 §42 ext 이상이어야 함."
                )
            latest_opt_dir = os.path.join(enc_dir_auto, opt_candidates[-1])
            opt_payload = torch.load(os.path.join(latest_opt_dir, "s1_opt.pt"),
                                     map_location="cpu", weights_only=False)
            resume_step = int(opt_payload.get("cumulative_step", 0))
            if resume_step == 0:
                raise RuntimeError(
                    f"{latest_opt_dir}/s1_opt.pt 에 cumulative_step 이 0 이거나 없음. "
                    f"이전 run 이 §42 ext post-fix 이후여야 auto 가능."
                )
            if accelerator.is_main_process:
                logger.info(
                    f"§42 auto-resume: detected cumulative_step={resume_step} "
                    f"from {latest_opt_dir}/s1_opt.pt"
                )

        if resume_step > 0:
            enc_dir = os.path.join(cfg["model_cache_dir"], cfg["encoder_name"])
            candidates = sorted(
                [d for d in os.listdir(enc_dir)
                 if d.startswith("s1_outputs_")
                 and os.path.exists(os.path.join(enc_dir, d, "s1_proj.pt"))]
            ) if os.path.isdir(enc_dir) else []
            if not candidates:
                raise RuntimeError(
                    f"--stage1-resume-step={resume_step} 요청됐으나 "
                    f"{enc_dir}/s1_outputs_*/s1_proj.pt 를 찾을 수 없음."
                )
            recovered_dir = os.path.join(enc_dir, candidates[-1])
            resume_proj_path = os.path.join(recovered_dir, "s1_proj.pt")
            logger.info(
                f"§42 Stage 1 resume: loading projector from {resume_proj_path}, "
                f"wandb step_offset={resume_step}"
            )
            proj_state = torch.load(resume_proj_path, map_location="cpu", weights_only=True)
            proj_state = {k: v.float() if v.is_floating_point() else v
                          for k, v in proj_state.items()}
            missing, unexpected = model.load_state_dict(proj_state, strict=False)
            if accelerator.is_main_process:
                loaded_keys = [k for k in proj_state.keys()
                                if k.startswith(("projector", "proj_norm"))]
                logger.info(f"  → loaded {len(loaded_keys)} projector keys")
            cfg["_wandb_step_offset"]  = resume_step
            cfg["_wandb_epoch_offset"] = 0.0   # 새 epoch 계산 시작 (기존 wandb 는 별개 run)
            # §42 ext: optimizer state 도 함께 load (있으면). s1_opt.pt 가 같은 디렉토리에
            # 있으면 첫 split 의 Adam m/v + lr_scheduler state 이어받음.
            resume_opt_path = os.path.join(recovered_dir, "s1_opt.pt")
            if os.path.exists(resume_opt_path):
                cfg["_s1_resume_opt_path"] = resume_opt_path
                if accelerator.is_main_process:
                    logger.info(f"§42 opt resume: queued {resume_opt_path}")
            else:
                if accelerator.is_main_process:
                    logger.warning(
                        f"§42 opt resume: no s1_opt.pt found in {recovered_dir} "
                        f"(previous run 이 §42 ext 전이면 없음). projector warm-start only."
                    )

        elif args.stage1_start_split > 0:
            enc_dir = os.path.join(cfg["model_cache_dir"], cfg["encoder_name"])
            candidates = sorted(
                [d for d in os.listdir(enc_dir)
                 if d.startswith("s1_outputs_")
                 and os.path.exists(os.path.join(enc_dir, d, "s1_proj.pt"))]
            ) if os.path.isdir(enc_dir) else []
            if not candidates:
                logger.warning(
                    f"--stage1-start-split={args.stage1_start_split} 요청됐으나 "
                    f"{enc_dir}/s1_outputs_*/s1_proj.pt 없음. split 0 부터 처음부터 시작."
                )
            else:
                recovered_dir = os.path.join(enc_dir, candidates[-1])
                resume_proj_path = os.path.join(recovered_dir, "s1_proj.pt")
                logger.info(f"Resuming Stage 1: loading projector from {resume_proj_path}")
                proj_state = torch.load(resume_proj_path, map_location="cpu", weights_only=True)
                # Stage 1 projector 는 fp32 (freeze_llm 에서 .float()), 여기서도 맞춤
                proj_state = {k: v.float() if v.is_floating_point() else v
                              for k, v in proj_state.items()}
                missing, unexpected = model.load_state_dict(proj_state, strict=False)
                if accelerator.is_main_process:
                    loaded_keys = [k for k in proj_state.keys()
                                    if k.startswith(("projector", "proj_norm"))]
                    logger.info(f"  → loaded {len(loaded_keys)} projector keys")

        # Split 순회 순서 결정 — Stage 2 와 동일 방식으로 매 epoch 마다 shuffle.
        # stage1-start-split 은 deprecated 경로 (sequential) 와 공존 불가 → 결합 금지.
        stage1_epochs_total = cfg.get("stage1_epochs", 1)

        # §42 ext: Stage 1 total_max_steps 사전 계산 (cross-split 단일 cosine+warmup).
        # Stage 2 와 동일 공식.
        total_s1_steps = 0
        if precomputed_dir:
            try:
                total_bins_per_rank_s1 = count_total_precomputed_bins_per_rank(
                    cfg, precomputed_dir, selected_datasets
                )
                bs_s1    = cfg.get("per_device_train_batch_size", 10)
                ga_s1    = cfg.get("gradient_accumulation_steps", 4)
                total_s1_steps = math.ceil(
                    total_bins_per_rank_s1 * stage1_epochs_total / (bs_s1 * ga_s1)
                )
                if accelerator.is_main_process:
                    logger.info(
                        f"Stage 1 total_s1_steps (cross-split) = {total_s1_steps} "
                        f"(bins/rank={total_bins_per_rank_s1}, bs={bs_s1}, "
                        f"grad_accum={ga_s1}, epochs={stage1_epochs_total}, warmup 10%)"
                    )
            except Exception as _e:
                if accelerator.is_main_process:
                    logger.warning(
                        f"Stage 1 total_s1_steps 계산 실패 → per-split scheduler: {_e}"
                    )
                total_s1_steps = 0

        # cross-split 단일 optimizer/scheduler 보존
        persistent_opt_sched = None

        for stage1_epoch in range(stage1_epochs_total):
            order = _s1_shuffled_split_order()
            logger.info(
                f" Stage 1 epoch {stage1_epoch+1}/{stage1_epochs_total} — "
                f"split order: {[s+1 for s in order]}",
                main_process_only=True,
            )

            # legacy --stage1-start-split 호환: epoch 0 만 skip 적용
            if stage1_epoch == 0 and resume_step == 0 and args.stage1_start_split > 0:
                order = [s for s in order if s >= args.stage1_start_split]

            for data_split in order:
                if num_data_splits > 1:
                    logger.info(f"{'='*55}")
                    logger.info(
                        f" Stage 1 epoch {stage1_epoch+1}/{stage1_epochs_total} — "
                        f"Data split {data_split+1}/{num_data_splits}"
                    )
                    logger.info(f"{'='*55}")

                # Build dataset for this split
                if precomputed_dir:
                    logger.info("Building Precomputed Pipeline...",)
                    train_streaming_dataset = build_precomputed_pipeline(
                        cfg=cfg,
                        accelerator=accelerator,
                        precomputed_dir=precomputed_dir,
                        processor_fn=processor_fn,
                        packer_fn=packer_fn,
                        selected_datasets=selected_datasets,
                        word_aug=args.word_aug,
                        data_split=data_split,
                        num_data_splits=num_data_splits,
                    )
                else:
                    logger.info("Building Streaming Pipeline...",)
                    train_streaming_dataset = build_multi_dataset_streaming_pipeline(
                        cfg=cfg,
                        accelerator=accelerator,
                        processor_fn=processor_fn,
                        packer_fn=packer_fn,
                        shuffle=True,
                        selected_datasets=selected_datasets,
                        word_aug=args.word_aug,
                    )

                s1_output_dir, _, persistent_opt_sched = run_stage1(
                    cfg=cfg,
                    accelerator=accelerator,
                    model=model,
                    train_dataset=train_streaming_dataset,
                    val_dataset=val_dataset,
                    train_eval_dataset=train_eval_dataset,
                    collator=collator,
                    run_id=run_id,
                    resume=args.resume if (stage1_epoch == 0 and data_split == order[0]) else None,
                    persistent_opt_sched=persistent_opt_sched,
                    total_s1_steps=total_s1_steps,
                )
                del train_streaming_dataset
                import gc; gc.collect()

    accelerator.wait_for_everyone()

    # 8. Stage 2: LoRA fine-tuning — single Trainer + dataset swap per split
    if args.stage in ("all", "2"):
        # ── 모델 한 번만 빌드 (split 전체에서 재사용) ──────────────────────
        s2_model = setup_stage2_model(cfg, accelerator, s1_output_dir)

        # ── 총 step 수 사전 계산 (cross-split 단일 LR curve) ──────────────
        total_max_steps = 0
        if precomputed_dir:
            try:
                total_bins_per_rank = count_total_precomputed_bins_per_rank(
                    cfg, precomputed_dir, selected_datasets
                )
                bs          = cfg.get("per_device_train_batch_size", 10)
                grad_accum  = cfg.get("gradient_accumulation_steps", 4)
                n_epochs    = cfg["stage2_epochs"]
                total_max_steps = math.ceil(
                    total_bins_per_rank * n_epochs / (bs * grad_accum)
                )
                logger.info(
                    f"Stage 2 total_max_steps (cross-split) = "
                    f"{total_max_steps}  (bins/rank={total_bins_per_rank}, "
                    f"bs={bs}, grad_accum={grad_accum}, epochs={n_epochs})",
                    main_process_only=True,
                )
            except Exception as _e:
                logger.warning(
                    f"Stage 2 total step 사전 계산 실패 → per-split scheduler 로 fallback: {_e}",
                    main_process_only=True,
                )
                total_max_steps = 0

        # ── 공통 s2_output_dir ────────────────────────────────────────────
        s2_output_dir = os.path.join(
            cfg["model_cache_dir"], cfg["encoder_name"], f"s2_outputs_{run_id}"
        )
        os.makedirs(s2_output_dir, exist_ok=True)

        # ── Split 0 dataset 빌드 → Trainer 초기화 (한 번만) ───────────────
        def _build_stage2_dataset(split_idx: int):
            if precomputed_dir:
                return build_precomputed_pipeline(
                    cfg=cfg,
                    accelerator=accelerator,
                    precomputed_dir=precomputed_dir,
                    processor_fn=processor_fn,
                    packer_fn=packer_fn,
                    selected_datasets=selected_datasets,
                    word_aug=args.word_aug,
                    data_split=split_idx,
                    num_data_splits=num_data_splits,
                )
            return build_multi_dataset_streaming_pipeline(
                cfg=cfg,
                accelerator=accelerator,
                processor_fn=processor_fn,
                packer_fn=packer_fn,
                shuffle=True,
                selected_datasets=selected_datasets,
                word_aug=args.word_aug,
            )

        # ── Split 순서 shuffle 용 RNG ───────────────────────────────────
        # Python 3 의 `hash(str)` 는 PYTHONHASHSEED 로 프로세스마다 salt 돼서
        # rank 마다 다른 값을 내놓음 → 기존 `hash(run_id)` seed 는 각 rank 가 서로 다른
        # shuffle 순서로 돌아 1 full-epoch 안에 shard 중복 방문/미방문 발생 (§33).
        # run_id 는 "MMDD_HHMM" 포맷이라 int 변환이 결정적 → 모든 rank 동일 seed.
        import random as _random
        split_rng = _random.Random(int(run_id.replace("_", "")))

        def _shuffled_split_order() -> list:
            order = list(range(num_data_splits))
            split_rng.shuffle(order)
            return order

        # ── Full-epoch 1 의 첫 split 로 Trainer 초기화 ───────────────────
        full_epoch_count = cfg["stage2_epochs"]
        first_order = _shuffled_split_order()
        logger.info(
            f" Stage 2 full-epoch 1/{full_epoch_count} — split order: "
            f"{[s+1 for s in first_order]}",
            main_process_only=True,
        )
        first_split_idx = first_order[0]
        logger.info(
            f"{'='*55}\n Stage 2 full-epoch 1/{full_epoch_count} — "
            f"split {first_split_idx+1}/{num_data_splits}\n{'='*55}",
            main_process_only=True,
        )
        first_ds = _build_stage2_dataset(first_split_idx)

        s2_trainer = build_stage2_trainer(
            cfg=cfg,
            accelerator=accelerator,
            model=s2_model,
            first_train_dataset=first_ds,
            val_dataset=val_dataset,
            train_eval_dataset=train_eval_dataset,
            collator=collator,
            s2_output_dir=s2_output_dir,
            total_max_steps=total_max_steps,
            num_data_splits=num_data_splits,
        )

        # ── Outer loop: full data epoch × stage2_epochs ─────────────────
        #   - Inner: num_data_splits 를 shuffled 순서로 1 회씩 (num_train_epochs=1)
        #   - 동일 Trainer 재사용: optimizer / scheduler / FSDP wrapper 유지
        #   - scheduler.last_epoch 누적 → cross-split 단일 LR curve
        #   - 매 split 종료 후 `trainer.save_model` 수동 호출 (§34): HF 의
        #     `save_strategy="steps" save_steps=60` 은 split 당 step (~58) < save_steps 라
        #     자연 발화 불가. + 매 split 마다 state.global_step 이 0 으로 리셋되어 HF 기본
        #     `checkpoint-{step}` 명명이 매번 동일 경로로 덮어써짐 → 명시적 rename 경로로 우회.
        def _save_split(fe_idx: int, split_idx_1based: int):
            ckpt_dir = os.path.join(
                s2_output_dir, f"checkpoint_fe{fe_idx}_split{split_idx_1based}"
            )
            logger.info(f"  → saving split checkpoint: {os.path.basename(ckpt_dir)}",
                        main_process_only=True)
            s2_trainer.save_model(ckpt_dir)   # 모든 rank 호출 필수 (FSDP state_dict gather)

        def _print_assignment(fe_idx: int, split_idx_0based: int):
            """매 split 시작 시 rank/GPU/shard 배정 표를 rank-0 에서 출력 (§35).
            §33 이후 모든 rank 는 동일한 split_idx 를 로드하므로 표의 shard 열은 전부 동일."""
            if not accelerator.is_main_process:
                return
            ws = accelerator.num_processes
            lines = [
                f"┌──────┬──────┬──────────────────────┐",
                f"│ GPU  │ rank │ shard file           │",
                f"├──────┼──────┼──────────────────────┤",
            ]
            for r in range(ws):
                shard_file = f"rank{r}_s{split_idx_0based}.arrow"
                lines.append(f"│ {r:>4} │ {r:>4} │ {shard_file:<20} │")
            lines.append(f"└──────┴──────┴──────────────────────┘")
            header = f" fe{fe_idx}/split{split_idx_0based+1} — rank/GPU/shard 배정"
            logger.info("\n" + header + "\n" + "\n".join(lines), main_process_only=True)

        _print_assignment(1, first_split_idx)
        s2_trainer.train()     # full-epoch 1 의 첫 split
        _save_split(1, first_split_idx + 1)
        for split_idx in first_order[1:]:
            logger.info(
                f"{'='*55}\n Stage 2 full-epoch 1/{full_epoch_count} — "
                f"split {split_idx+1}/{num_data_splits}\n{'='*55}",
                main_process_only=True,
            )
            _print_assignment(1, split_idx)
            new_ds = _build_stage2_dataset(split_idx)
            s2_trainer.train_dataset = new_ds
            s2_trainer.train()
            _save_split(1, split_idx + 1)
            del new_ds
            import gc; gc.collect()

        # ── 남은 full-epoch 들 ────────────────────────────────────────────
        for full_epoch in range(1, full_epoch_count):
            order = _shuffled_split_order()
            logger.info(
                f" Stage 2 full-epoch {full_epoch+1}/{full_epoch_count} — "
                f"split order: {[s+1 for s in order]}",
                main_process_only=True,
            )
            for split_idx in order:
                logger.info(
                    f"{'='*55}\n Stage 2 full-epoch {full_epoch+1}/{full_epoch_count} — "
                    f"split {split_idx+1}/{num_data_splits}\n{'='*55}",
                    main_process_only=True,
                )
                _print_assignment(full_epoch + 1, split_idx)
                new_ds = _build_stage2_dataset(split_idx)
                s2_trainer.train_dataset = new_ds
                s2_trainer.train()
                _save_split(full_epoch + 1, split_idx + 1)
                del new_ds
                import gc; gc.collect()

        # ── 최종 save ────────────────────────────────────────────────────
        s2_trainer.save_model(s2_output_dir)
        logger.info(f"Stage 2 complete. Saved: {s2_output_dir}", main_process_only=True)
        accelerator.wait_for_everyone()

    # 모든 split 루프 완료 후에만 wandb run 종료.
    if accelerator.is_main_process and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()