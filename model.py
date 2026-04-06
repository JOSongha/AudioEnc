import os

import torch
import torch.nn as nn
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from encoders.base import BaseAudioEncoder


class AudioQwen(nn.Module):
    """
    Encoder-agnostic Audio-LLM.

    어떤 BaseAudioEncoder 구현체든 주입 가능.
    projector 구조는 cfg["encoder"]["proj_strides"]로 결정:
      [2, 2] → 2×Conv1d(stride-2), total ×4 다운샘플 (encodec, dac, mimi_acoustic)
      [2]    → 1×Conv1d(stride-2), total ×2 다운샘플 (mimi_semantic — q_ming.py 원본)

    cfg["llm_type"] 으로 프롬프트 포맷 결정:
      "instruct" → ChatML (<|im_start|>system ... <|im_end|>)
      "base"     → 단순 prefix ("Audio:\n" ... "\nTranscript:\n")

    입력 sequence 구조:
      instruct: [ChatML system+user] + [audio embeds] + [ChatML suffix] + [transcript]
      base:     ["Audio:\n"] + [audio embeds] + ["\nTranscript:\n"] + [transcript]
    Loss: transcript 토큰에 대한 cross-entropy만 계산.
    """

    def __init__(self, encoder: BaseAudioEncoder, cfg: dict):
        super().__init__()
        self.encoder = encoder
        self._cfg    = cfg  # apply_lora에서 LoRA 설정 참조

        # Qwen3.5-4B는 SSM 아키텍처로 fp16에서 NaN 발생 → bf16 사용
        torch_dtype = torch.bfloat16
        cache_dir   = cfg["model_cache_dir"]
        llm_name    = cfg["llm_model"]

        cached = try_to_load_from_cache(llm_name, "config.json", cache_dir=cache_dir)
        attn_impl = cfg.get("attn_implementation", "eager")
        print(f"Loading LLM: {llm_name} (dtype={torch_dtype}, attn={attn_impl}, {'캐시' if cached else '다운로드'})...")

        # Liger kernel: LLM 로드 직전 적용 (모듈 교체 방식이므로 로드 전 등록)
        if cfg.get("use_liger_kernel", False):
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_qwen2
                apply_liger_kernel_to_qwen2(
                    rope=True, rms_norm=True, swiglu=True,
                    fused_linear_cross_entropy=True,
                )
                print("Liger kernel applied (rope, rms_norm, swiglu, fused_linear_ce).")
            except ImportError:
                raise RuntimeError(
                    "liger-kernel 미설치. pip install liger-kernel"
                )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
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

        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        llm_dim = self.llm.config.hidden_size

        # Projector: encoder.out_dim → llm_dim
        # proj_strides: encoder별로 다름 (config.py ENCODER_REGISTRY 참조)
        #   [2, 2] → 2×stride-2, total ×4 (encodec/dac/mimi_acoustic)
        #   [2]    → 1×stride-2, total ×2 (mimi_semantic — q_ming.py 원본과 동일)
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
        self.ctc_head  = None  # init_ctc_head()로 활성화 (--debug c)

        # audio placeholder token id (Qwen2.5 <|image_pad|>=151655, packed forward에서 사용)
        self.audio_pad_token_id = cfg.get("audio_pad_token_id", 151655)
        # Liger fused_linear_ce 활성 여부 (packed forward에서만 적용)
        self._use_liger_kernel  = cfg.get("use_liger_kernel", False)
        self.projector.to(dtype=torch_dtype)
        self.proj_norm.to(dtype=torch_dtype)

        # projector 초기화
        for m in self.projector.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 마지막 1×1 conv: small-scale init — 초기 audio embed 스케일 억제 (fp16 안정성)
        nn.init.normal_(self.projector[-1].weight, std=0.02)
        nn.init.zeros_(self.projector[-1].bias)

        # projector 총 stride 자동 계산 (mask downsampling에 사용)
        self._proj_stride = 1
        for m in self.projector.modules():
            if isinstance(m, nn.Conv1d):
                self._proj_stride *= m.stride[0]

        # 프롬프트 토큰 버퍼 (forward마다 tokenize 반복 방지)
        llm_type = cfg.get("llm_type", "instruct")
        if llm_type == "instruct":
            p1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            p2 = "\nTranscribe the audio to text.<|im_end|>\n<|im_start|>assistant\n"
        else:  # base
            p1 = "Audio:\n"
            p2 = "\nTranscript:\n"
        self.register_buffer(
            "prompt_p1_ids",
            self.tokenizer.encode(p1, add_special_tokens=False, return_tensors="pt"),
        )
        self.register_buffer(
            "prompt_p2_ids",
            self.tokenizer.encode(p2, add_special_tokens=False, return_tensors="pt"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_audio_embeds(self, audio, audio_lengths=None):
        # cast audio to encoder's dtype (fp32 normally; bf16 under FSDP)
        enc_dtype = next(self.encoder.parameters()).dtype
        feats, enc_mask = self.encoder(audio.to(enc_dtype), audio_lengths)  # (B, T_enc, C)

        feats = feats.to(self.projector[0].weight.dtype)

        proj = self.projector(feats.transpose(1, 2))          # (B, llm_dim, T_proj)
        audio_embeds = self.proj_norm(proj.transpose(1, 2))   # (B, T_proj, llm_dim)
        audio_embeds = audio_embeds.to(self.llm.get_input_embeddings().weight.dtype)

        T_proj = audio_embeds.shape[1]
        proj_mask = enc_mask[:, :: self._proj_stride][:, :T_proj]
        if proj_mask.shape[1] < T_proj:
            pad = torch.zeros(
                proj_mask.shape[0], T_proj - proj_mask.shape[1],
                dtype=torch.bool, device=proj_mask.device,
            )
            proj_mask = torch.cat([proj_mask, pad], dim=1)

        return audio_embeds, proj_mask

    # ------------------------------------------------------------------
    # Stage 1 / Stage 2 설정
    # ------------------------------------------------------------------

    def init_ctc_head(self, n_chars: int = 28):
        """Stage 1 CTC head 초기화 (blank=0, a-z=1-26, space=27).
        freeze_llm() 후, optimizer 생성 전에 호출해야 optimizer에 포함됨."""
        llm_dim = self.llm.config.hidden_size
        self.ctc_head = nn.Linear(llm_dim, n_chars)
        nn.init.xavier_uniform_(self.ctc_head.weight)
        nn.init.zeros_(self.ctc_head.bias)
        self.ctc_head.float()
        self.ctc_head.requires_grad_(True)
        print(f"CTC head initialized: Linear({llm_dim}, {n_chars})")

    def freeze_llm(self):
        """Stage 1: LLM frozen, projector만 학습."""
        print("Freezing LLM (Stage 1)...")
        for p in self.llm.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.proj_norm.parameters():
            p.requires_grad = True

        self.llm.gradient_checkpointing_disable()
        self.projector.float()
        self.proj_norm.float()

    def apply_lora(self):
        """Stage 2: LLM에 LoRA 적용. LoRA 설정은 cfg에서 읽음."""
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

        trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.llm.parameters())
        print(f"LoRA applied. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        train_proj = self._cfg.get("stage2_train_projector", True)
        for p in self.projector.parameters():
            p.requires_grad = train_proj
        for p in self.proj_norm.parameters():
            p.requires_grad = train_proj
        print(f"Projector {'trainable' if train_proj else 'frozen'} in Stage 2.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, audio=None, audio_lengths=None, transcript_input_ids=None,
                ctc_targets=None, ctc_target_lengths=None, eos_weight: float = 1.0,
                eos_first: bool = False,
                # ── packed 경로 (--packing 시 사용) ──────────────────────
                input_ids=None, labels=None,
                audio_features=None, audio_feat_lengths=None,
                attention_mask=None, position_ids=None):
        if input_ids is not None:
            # ── packed 경로 ───────────────────────────────────────────────
            return self._forward_packed(
                input_ids, labels, audio_features, audio_feat_lengths,
                attention_mask, position_ids,
            )

        # ── 기존(legacy) 경로 ─────────────────────────────────────────────
        audio_embeds, audio_mask = self._get_audio_embeds(audio, audio_lengths)

        if transcript_input_ids is None:
            return audio_embeds, audio_mask

        # CTC loss (--debug c, projector 출력에서 직접 계산 — LLM 우회)
        ctc_loss = None
        if self.ctc_head is not None and ctc_targets is not None:
            ctc_logits   = self.ctc_head(audio_embeds.float())          # (B, T_proj, n_chars)
            log_probs    = ctc_logits.log_softmax(-1).transpose(0, 1)   # (T_proj, B, n_chars)
            input_lengths = audio_mask.long().sum(dim=1).cpu()
            ctc_loss = torch.nn.functional.ctc_loss(
                log_probs, ctc_targets.cpu(), input_lengths, ctc_target_lengths.cpu(),
                blank=0, reduction="mean", zero_infinity=True,
            )

        device = audio_embeds.device
        B      = audio_embeds.shape[0]
        embed  = self.llm.get_input_embeddings()

        p1_embeds         = embed(self.prompt_p1_ids).expand(B, -1, -1)
        p2_embeds         = embed(self.prompt_p2_ids).expand(B, -1, -1)
        transcript_embeds = embed(transcript_input_ids)

        inputs_embeds = torch.cat([p1_embeds, audio_embeds, p2_embeds, transcript_embeds], dim=1)

        p1_mask         = torch.ones(B, p1_embeds.shape[1], device=device, dtype=torch.long)
        p2_mask         = torch.ones(B, p2_embeds.shape[1], device=device, dtype=torch.long)
        transcript_mask = (transcript_input_ids != self.tokenizer.pad_token_id).long()
        attention_mask  = torch.cat([p1_mask, audio_mask.long(), p2_mask, transcript_mask], dim=1)

        len_ctx    = p1_embeds.shape[1] + audio_embeds.shape[1] + p2_embeds.shape[1]
        ctx_labels = torch.full((B, len_ctx), -100, dtype=torch.long, device=device)
        tgt_labels = transcript_input_ids.clone()
        if eos_first:
            # EOS-first: [tok, EOS, PAD, PAD] — PAD는 모두 -100, EOS는 자동 보존 (EOS≠PAD)
            tgt_labels[tgt_labels == self.tokenizer.pad_token_id] = -100
        else:
            # EOS-last: [tok, PAD, PAD, EOS] — 마지막 열(EOS)은 보존, 앞쪽 PAD만 -100
            tgt_labels[:, :-1][tgt_labels[:, :-1] == self.tokenizer.pad_token_id] = -100
        labels     = torch.cat([ctx_labels, tgt_labels], dim=1)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        if eos_weight != 1.0:
            # EOS 위치 loss에 추가 가중치 적용 (over-generation 억제)
            shift_logits = out.logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_tok = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100, reduction="none",
            ).view(B, -1)
            is_eos = (shift_labels == self.tokenizer.eos_token_id).float()
            weights = 1.0 + (eos_weight - 1.0) * is_eos
            valid   = (shift_labels != -100).float()
            out.loss = (per_tok * weights * valid).sum() / (valid * weights).sum().clamp(min=1)
        if ctc_loss is not None:
            out.loss = out.loss + ctc_loss
        return out

    def _forward_packed(self, input_ids, labels, audio_features, audio_feat_lengths,
                        attention_mask, position_ids):
        """
        Sequence packing 경로 (--packing 플래그 사용 시).

        audio_features:     (N_audio, 1, S_max)  배치 내 모든 오디오 flatten + zero-pad
        audio_feat_lengths: (N_audio,)            각 오디오 token 수 (t_audio, projector 출력 기준)
        input_ids:          (B, T) [eager] 또는 (1, sum_nonpad) [FA2]
        labels:             동일 shape
        attention_mask:     (B, 1, T, T) 4D block-diagonal mask [eager] 또는 None [FA2]
        position_ids:       (1, sum_nonpad) 샘플별 리셋 [FA2] 또는 None [eager]
        """
        # 1. 오디오 인코딩
        #    audio_features: (N, 1, S_max) → squeeze → (N, S_max)
        #    audio_feat_lengths: t_audio (LLM token 기준) → 실제 audio sample 수로 역산
        wavs            = audio_features.squeeze(1)                     # (N, S_max)
        spt             = self._cfg.get("samples_per_token", 1.0)
        lengths_samples = (audio_feat_lengths.float() * spt).long()     # (N,) audio sample 수
        audio_embeds, _ = self._get_audio_embeds(wavs, lengths_samples) # (N, T_proj, llm_dim)

        # 2. 전체 토큰 임베딩
        embed         = self.llm.get_input_embeddings()
        inputs_embeds = embed(input_ids).clone()                        # (B/1, T, llm_dim)

        # 3. audio placeholder 위치를 audio embedding으로 교체
        audio_pad_mask = (input_ids == self.audio_pad_token_id)         # (B, T) bool
        audio_flat     = audio_embeds.reshape(-1, audio_embeds.shape[-1])
        n_ph           = int(audio_pad_mask.sum().item())

        if audio_flat.shape[0] != n_ph:
            # ±1 frame 불일치 방어: 짧은 쪽에 맞춰 trim
            n          = min(audio_flat.shape[0], n_ph)
            audio_flat = audio_flat[:n]
            flat_mask  = audio_pad_mask.reshape(-1)
            positions  = flat_mask.nonzero(as_tuple=False).squeeze(1)
            fix_mask   = torch.zeros_like(flat_mask)
            fix_mask[positions[:n]] = True
            audio_pad_mask = fix_mask.reshape(audio_pad_mask.shape)

        inputs_embeds[audio_pad_mask] = audio_flat.to(inputs_embeds.dtype)

        # 4. LLM forward
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
        )
