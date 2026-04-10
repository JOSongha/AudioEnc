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
  run_stage2

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

    Forward 시퀀스 (패킹):
      audio_features → encoder → projector → audio_embeds
      input_ids: [audio_pad]*t_audio + <|audio_correspond|> + text_ids + EOS
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
                    import transformers.models.qwen3_5.modeling_qwen3_5 as _q35
                    from liger_kernel.transformers.monkey_patch import (
                        liger_rotary_pos_emb, LigerRMSNorm, LigerSwiGLUMLP,
                    )
                    from liger_kernel.transformers.model.qwen3 import lce_forward as _qwen3_lce_forward

                    # 1) RoPE: 동일 함수 시그니처, 직접 교체
                    _q35.apply_rotary_pos_emb = liger_rotary_pos_emb

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
                                "(rope, rms_norm+offset, swiglu, fused_linear_ce).",
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

        new_tokens = ["<|audio_correspond|>"]
        num_added_toks = self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if num_added_toks > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

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
        self.proj_norm = nn.LayerNorm(llm_dim)

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

        Returns: (1, total_T_proj, llm_dim)
            각 클립의 유효 projector 프레임만 연결. total_T_proj = sum ceil(T_enc_i / stride)
        """
        feats  = enc_feats.to(self.projector[0].weight.dtype)   # (N, T_enc_max, out_dim)
        proj   = self.projector(feats.transpose(1, 2))          # (N, llm_dim, T_proj_max)
        embeds = self.proj_norm(proj.transpose(1, 2))           # (N, T_proj_max, llm_dim)
        embeds = embeds.to(self.llm.get_input_embeddings().weight.dtype)

        clips = []
        for i in range(enc_feats.shape[0]):
            T_enc_valid  = int(enc_feat_lengths[i].item())
            T_proj_valid = math.ceil(T_enc_valid / self._proj_stride)
            clips.append(embeds[i, :T_proj_valid, :])

        return torch.cat(clips, dim=0).unsqueeze(0)             # (1, total_T_proj, llm_dim)

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
        if precomputed_enc_feats is not None:
            # precomputed 모드: 인코더 생략, audio_lengths = T_enc
            audio_embeds = self._project_precomputed(
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

_ALIGNMENT_BASE = "/mnt/tmp/cache/word_alignments_merged"

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
        logger.info(f"AlignmentLookup: {len(self._index):,} utterances ← {arrow_path}")

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
            logger.warning(f"AlignmentLookup: no arrow for '{key}' at {arrow_path}")
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

    시퀀스 형식: [audio_pad]*t_audio + <|audio_correspond|> + text_ids + EOS
    Labels:     [IGNORE]*t_audio    + [IGNORE]              + text_ids + EOS

    word_aug=True이고 alignment_lookup이 제공된 경우,
    각 발화에 대해 원본 샘플 + 단어 단위 서브샘플을 추가 생성.
    ("utterance_id" 컬럼이 batch에 있어야 함)
    """
    audio_correspond_id = tokenizer.convert_tokens_to_ids("<|audio_correspond|>")

    def _build_one(waveform_1d: torch.Tensor, text: str):
        """(1D waveform tensor, text string) → (input_ids, labels, t_audio) 또는 None."""
        t_audio = waveform_1d.shape[0] // hop_length
        if t_audio == 0:
            return None
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        if not text_ids:
            return None
        input_ids = (
            [audio_pad_token_id] * t_audio
            + [audio_correspond_id]
            + text_ids
            + [tokenizer.eos_token_id]
        )
        labels = (
            [IGNORE_INDEX] * (t_audio + 1)
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

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        import numpy as np

        input_ids      = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        labels         = torch.tensor([f["labels"]    for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

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

def make_precomputed_processor_fn(cfg, tokenizer):
    """Pre-computed encoder feature Arrow 파일용 processor_fn.

    Arrow 스키마: utterance_id(str), text(str), features(list<float32>), feat_len(int32)
    features = flattened (T_enc × out_dim,) float32

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

    def process_samples(examples):
        all_input_ids, all_labels = [], []
        all_audio_features, all_audio_lengths = [], []

        for flat_feats, feat_len, text in zip(
            examples["features"], examples["feat_len"], examples["text"]
        ):
            if feat_len == 0 or not flat_feats:
                continue
            T_enc = int(feat_len)
            T_proj = math.ceil(T_enc / total_stride)

            text_ids = tokenizer.encode(
                text.lower().strip(), add_special_tokens=False
            )[:max_text_len]
            if not text_ids:
                continue

            # [audio_pad × T_proj] + [text_ids] + [EOS]
            input_ids = ([audio_pad_id] * T_proj) + text_ids + [tokenizer.eos_token_id]
            labels    = ([-100] * T_proj) + text_ids + [tokenizer.eos_token_id]

            # flat float list 그대로 보관 (collator에서 reshape)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_audio_features.append(flat_feats if isinstance(flat_feats, list)
                                      else list(flat_feats))
            all_audio_lengths.append(T_enc)

        return {
            "input_ids":      all_input_ids,
            "labels":         all_labels,
            "audio_features": all_audio_features,
            "audio_lengths":  [int(x) for x in all_audio_lengths],
        }

    return process_samples


def build_precomputed_pipeline(cfg, accelerator, precomputed_dir, processor_fn, packer_fn,
                               selected_datasets=None, shuffle: bool = True,
                               word_aug: bool = False):
    """Pre-computed Arrow 파일 기반 데이터 파이프라인.

    각 rank는 자신의 rank{process_index}.arrow 파일만 로드한다.
    word_aug=True: processor_fn이 word-level sub-clip 생성 (AlignmentLookup 필요).

    Pre-packed 모드 (권장):
        {precomputed_dir}/{encoder}/{dataset}/packed_{cutoff_len}/rank{N}.arrow 가 존재하면
        processor_fn / packer_fn map을 건너뛰고 Arrow를 직접 로드한다.
        → DataLoader 학습 중 CPU packing 부하 제거.
        생성: bash precompute/run_pack.sh --encoder {encoder}
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

    for ds_key in selected:
        # ── Pre-packed Arrow 우선 ──────────────────────────────────────────
        packed_path = base_dir / ds_key / f"packed_{cutoff_len}" / f"rank{rank}.arrow"
        if packed_path.exists():
            import pyarrow.ipc as _pa_ipc
            from datasets import Dataset as _HFDataset
            _table = _pa_ipc.open_file(str(packed_path)).read_all()
            ds = _HFDataset(_table).to_iterable_dataset()
            logger.info(
                f"[rank {rank}] Pre-packed {ds_key} (in-memory): {packed_path.name}",
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
            f"[rank {rank}] Loading precomputed {ds_key} into memory: {[Path(f).name for f in data_files]}",
            main_process_only=False,
        )
        import pyarrow.ipc as _pa_ipc
        from datasets import Dataset as _HFDataset
        import pyarrow as _pa
        _tables = [_pa_ipc.open_file(f).read_all() for f in data_files]
        _table = _pa.concat_tables(_tables) if len(_tables) > 1 else _tables[0]
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
        ds = ds.shuffle(buffer_size=10000, seed=cfg.get("seed", 42))

    return ds


class StreamingShardedTrainer(Trainer):
    """자동 DistributedSampler 삽입을 우회하는 HF Trainer 서브클래스.

    파이프라인 시작에서 GPU별로 이미 샤딩이 완료되었으므로,
    DistributedSampler를 추가하면 이중 샤딩이 발생함. 이 서브클래스는
    sampler 래핑 없이 순수 DataLoader를 반환.
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

    시퀀스: [audio_embeds] + [<|audio_correspond|>] → transcript 토큰 생성.
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

    corr_id     = tokenizer.convert_tokens_to_ids("<|audio_correspond|>")
    corr_tensor = torch.tensor([[corr_id]], device=device)          # (1, 1)
    corr_embeds = embed(corr_tensor).expand(B, -1, -1)              # (B, 1, D)

    inputs_embeds = torch.cat([audio_embeds, corr_embeds], dim=1)
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
    audio_correspond_id = tokenizer.convert_tokens_to_ids("<|audio_correspond|>")
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
            [cfg.get("audio_pad_token_id", 151655)] * t_audio
            + [audio_correspond_id]
            + text_ids
            + [tokenizer.eos_token_id]
        )
        lbl = [IGNORE_INDEX] * (t_audio + 1) + text_ids + [tokenizer.eos_token_id]
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
        self.best_wer_val       = float("inf")

    def _save_proj(self, raw_model, filename):
        proj_state = {
            k: v.cpu() for k, v in raw_model.state_dict().items()
            if "projector" in k or "proj_norm" in k
        }
        torch.save(proj_state, os.path.join(self.output_dir, filename))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_every != 0:
            return

        step = state.global_step
        self.accelerator.wait_for_everyone()
        # 모든 랭크가 collective에 참여해야 함; 전체 파라미터는 rank-0만 보유
        with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
            if self.accelerator.is_main_process:
                raw_model = self.accelerator.unwrap_model(model)
                wer_val, wer_train = evaluate_wer(
                    raw_model, self.val_dataset, self.train_eval_dataset, self.cfg,
                )
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
                    self._save_proj(raw_model, "best_s1_proj.pt")
                    logger.info(f"  [Best WER={wer_val*100:.1f}%] best_s1_proj.pt 저장")

                if wandb.run is not None:
                    wandb.log(
                        {"val/wer": wer_val, "train/wer": wer_train, "val/loss": val_loss},
                        step=step,
                    )
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
    """스트리밍 학습용 max_steps 추정 (데이터셋 시간 기반).

    스트리밍 데이터셋은 len()이 없으므로 오디오 시간으로 스텝 수 추정.
    각 패킹 시퀀스의 약 80%가 오디오 토큰이라고 가정 (보수적).
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
               collator, run_id: str, resume=None):
    """Stage 1: LLM frozen 상태에서 Projector alignment. (output_dir, total_steps) 반환."""

    if cfg.get("max_steps") is None:
        cfg["max_steps"] = calculate_max_steps(
            total_estimated_hours=cfg["estimated_hours"],
            cutoff_len=cfg["packing_cutoff_len"],
            per_device_batch_size=2,
            num_processes=accelerator.num_processes,
            grad_accum_steps=cfg.get("gradient_accumulation_steps", 4),
            hop_length=cfg["encoder"].get("hop", 512),
            sample_rate=cfg.get("sample_rate", 16000),
            target_epochs=cfg["stage1_epochs"],
        )
        logger.info(
            f"Stage 1: Computed max_steps={cfg['max_steps']} from estimated_hours={cfg['estimated_hours']}",
            main_process_only=True,
        )

    logger.info(f"{'='*55}",)
    logger.info(f" Stage 1: Projector Alignment  LR={cfg['stage1_lr']} Max Steps={cfg['max_steps']}",)
    logger.info(f"{'='*55}",)
    
    model.freeze_llm()
    output_dir = os.path.join(cfg["model_cache_dir"], cfg["encoder_name"], f"s1_outputs_{run_id}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=1,   # packed bin 1개 = 1 batch (기본값 8이면 8 bin 묶음 → 8× 느림)
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg["stage1_lr"],
        lr_scheduler_type="constant",
        optim="adamw_torch_fused",

        # 주의: TrainingArguments의 gradient_checkpointing은 FSDP backward에서 불필요한
        # AllGather를 추가함; 대신 fsdp_config의 activation_checkpointing 사용 권장.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_steps=cfg.get("log_every", 10),
        # Disable auto-save only for very short test runs (< 10 steps)
        save_strategy="no" if (cfg.get("max_steps") and cfg["max_steps"] < 10) else "steps",
        save_steps=cfg.get("save_steps", 500) if cfg.get("max_steps", 1000) >= 10 else None,
        save_total_limit=5,  # Keep best and latest 5 checkpoints
        report_to="wandb",

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
        dataloader_num_workers=1 if cfg.get("precomputed_dir") else 8,
        dataloader_prefetch_factor=2,
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

    trainer = StreamingShardedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[wer_callback],
    )

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

        if wandb.run is not None:
            wandb.finish()

    accelerator.wait_for_everyone()

    return output_dir, s1_total_steps


def run_stage2(cfg, train_packed, val_dataset, train_eval_dataset, collator,
               accelerator, s1_output_dir, run_id: str, args):
    """Stage 2: FSDP를 사용한 LLM + projector LoRA 파인튜닝.

    새 모델을 빌드하여 Stage 1 DDP 모델에 FSDP를 이중 래핑하는 문제 방지.
    """
    gc.collect()
    torch.cuda.empty_cache()

    if cfg.get("max_steps") is None:
        cfg["max_steps"] = calculate_max_steps(
            total_estimated_hours=cfg["estimated_hours"],
            cutoff_len=cfg["packing_cutoff_len"],
            per_device_batch_size=2,
            num_processes=accelerator.num_processes,
            grad_accum_steps=cfg.get("gradient_accumulation_steps", 4),
            hop_length=cfg["encoder"].get("hop", 512),
            sample_rate=cfg.get("sample_rate", 16000),
            target_epochs=cfg["stage2_epochs"],
        )
        logger.info(
            f"Stage 2: Computed max_steps={cfg['max_steps']} from estimated_hours={cfg['estimated_hours']}",
            main_process_only=True,
        )

    logger.info(f"{'='*55}",)
    logger.info(f" Stage 2: LoRA Fine-tuning  LR={cfg['stage2_lr']}  warmup={cfg.get('warmup_ratio', 0.03)} max_steps={cfg['max_steps']}",)
    logger.info(f"{'='*55}",)

    model = build_model(cfg, accelerator)
    model.apply_lora()

    # Stage 1 projector 가중치 로드
    s1_proj_path = os.path.join(s1_output_dir, "s1_proj.pt")
    if not os.path.exists(s1_proj_path):
        logger.warning(f"Stage 1 projector not found at {s1_proj_path}. 최근 run에서 복구 시도...")
        # encoder 디렉터리 아래 s1_outputs_* 중 s1_proj.pt가 있는 가장 최근 디렉터리 탐색
        enc_dir = os.path.join(cfg["model_cache_dir"], cfg["encoder_name"])
        candidates = sorted(
            [
                d for d in os.listdir(enc_dir)
                if d.startswith("s1_outputs_")
                and os.path.exists(os.path.join(enc_dir, d, "s1_proj.pt"))
            ]
        ) if os.path.isdir(enc_dir) else []
        if candidates:
            recovered_dir  = os.path.join(enc_dir, candidates[-1])
            s1_proj_path   = os.path.join(recovered_dir, "s1_proj.pt")
            logger.warning(f"  → 복구됨: {s1_proj_path}  (원래 경로: {s1_output_dir})")
        else:
            s1_proj_path = None
            logger.warning(f"  → 복구 실패: {enc_dir} 아래 s1_outputs_*/s1_proj.pt 없음. 초기화 가중치로 Stage 2 진행.")

    if s1_proj_path is not None:
        logger.info(f"Stage 1 projector loaded: {s1_proj_path}")
        proj_state = torch.load(s1_proj_path, map_location="cpu", weights_only=True)
        # 항상 명시적으로 bf16 변환 (model.llm.dtype이 PeftModel 래핑 후 불안정할 수 있음)
        proj_state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                      for k, v in proj_state.items()}
        model.load_state_dict(proj_state, strict=False)

    # FSDP는 각 shard 내 dtype 균일성을 요구함.
    # load_state_dict 이후에 명시적으로 캐스트하여 LoRA init fp32 + projector 로드 등 잔존 fp32 처리.
    # (model.to() 대신 per-param 직접 캐스트: PeftModel.to()가 adapter weight를 건너뛸 가능성 방어)
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

    s2_output_dir = os.path.join(cfg["model_cache_dir"], cfg["encoder_name"], f"s2_outputs_{run_id}")
    os.makedirs(s2_output_dir, exist_ok=True)

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
        max_steps=cfg.get("max_steps", 10000),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg["stage2_lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=0.01,
        optim="adamw_torch_fused",

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_steps=cfg.get("log_every", 10),
        report_to="wandb",
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=3,

        remove_unused_columns=False,
        dataloader_num_workers=1 if cfg.get("precomputed_dir") else 8,
        dataloader_prefetch_factor=2,

        **({
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "fsdp_transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"],
                "fsdp_use_orig_params": True,
                "fsdp_backward_prefetch": "backward_pre",
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                "limit_all_gathers": True,
                "fsdp_ignored_modules": ["encoder"],
            },
        } if cfg.get("use_fsdp", True) else {}),
    )

    trainer = StreamingShardedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_packed,
        data_collator=collator,
        callbacks=[wer_callback],
    )

    trainer.train()
    # save_model doesn't take safe_serialization in newer transformers
    trainer.save_model(s2_output_dir)
    logger.info(f"Stage 2 complete. Saved: {s2_output_dir}",)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and wandb.run is not None:
        wandb.finish()


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
    cfg["precomputed_dir"] = args.precomputed_dir   # None = raw audio 모드

    selected_datasets = args.datasets or ["ls100", "ls360", "ls500", "mls", "gs", "vp"]
    cfg["estimated_hours"] = (
        args.estimated_hours if args.estimated_hours is not None
        else estimate_total_hours(selected_datasets)
    )

    os.makedirs(cfg["model_cache_dir"], exist_ok=True)
    os.environ.setdefault("HF_HOME", cfg["model_cache_dir"])

    if cfg.get("wandb_mode") == "disabled":
        os.environ["WANDB_MODE"] = "disabled"

    accelerator = Accelerator(log_with="wandb")

    import datetime
    run_id = datetime.datetime.now().strftime("%m%d_%H%M")

    if accelerator.is_main_process and cfg.get("wandb_mode") != "disabled":
        llm_tag  = "2b" if "2B" in cfg.get("llm_model", "") else "4b"
        run_name = f"{cfg.get('encoder_name', 'encoder')}_{llm_tag}_S1_{run_id}"
        wandb.init(project=cfg.get("project_name", "audio-qwen"), config=cfg,
                   name=run_name, mode=cfg.get("wandb_mode", "online"))

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

    if precomputed_dir:
        # ── Precomputed 모드: Arrow 파일에서 인코더 피처 직접 로드 ──────────
        logger.info(f"Precomputed mode: loading encoder features from {precomputed_dir}")
        processor_fn = make_precomputed_processor_fn(cfg, tokenizer)
        collator = OmniCollator(
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
            block_diag_attn=True,
            enc_out_dim=model.encoder.out_dim,
        )
        # 3. Build precomputed pipeline
        logger.info("Building Precomputed Pipeline...",)
        train_streaming_dataset = build_precomputed_pipeline(
            cfg=cfg,
            accelerator=accelerator,
            precomputed_dir=precomputed_dir,
            processor_fn=processor_fn,
            packer_fn=packer_fn,
            selected_datasets=selected_datasets,
            word_aug=args.word_aug,
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
        )
        # 3. Build streaming dataset (Load → Shard → Shuffle → Process → Pack)
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

    # 5. Static eval datasets for WER callback
    val_dataset, train_eval_dataset = build_static_eval_datasets(cfg, tokenizer)

    # 7. Stage 1: projector alignment
    s1_output_dir = os.path.join(cfg["model_cache_dir"], cfg["encoder_name"], f"s1_outputs_{run_id}")
    if args.stage in ("all", "1"):
        s1_output_dir, _ = run_stage1(
            cfg=cfg,
            accelerator=accelerator,
            model=model,
            train_dataset=train_streaming_dataset,
            val_dataset=val_dataset,
            train_eval_dataset=train_eval_dataset,
            collator=collator,
            run_id=run_id,
            resume=args.resume,
        )

    accelerator.wait_for_everyone()

    # 8. Stage 2: LoRA fine-tuning
    if args.stage in ("all", "2"):
        run_stage2(
            cfg=cfg,
            train_packed=train_streaming_dataset,
            val_dataset=val_dataset,
            train_eval_dataset=train_eval_dataset,
            collator=collator,
            accelerator=accelerator,
            s1_output_dir=s1_output_dir,
            run_id=run_id,
            args=args,
        )


if __name__ == "__main__":
    main()