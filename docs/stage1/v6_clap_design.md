# CLAP encoder 통합 설계 메모 (v6 후보)

Stage-1 v6 배치에 CLAP 변종을 추가하기 전 정리. yaml 초안만 placeholder 로 두고
(`configs/ASR/stage1_clap_v6.yaml`), 본 문서의 open decisions 가 닫힌 후에야
projL config / audio_encoder.py / 학습 launch 가능.

## 1. 왜 까다로운가

기존 5개 encoder (DAC-VAE / Whisper-{small,tiny} / EnCodec / WavTokenizer) 는
모두 **per-frame token 시퀀스** 를 출력. AudioProjector 가 [B, T, D] 를 받아
LLM token stream 에 끼워넣는 구조 ([encodec audio_encoder.py](../../external/models/Qwen3.5AE-4B-encodec-projL/audio_encoder.py) 참고).

CLAP 은 contrastive language-audio pretraining 모델이라 기본 사용은
**clip-level pooled embedding** ([B, D] single vector) 이고, omni stage1 의
ASR / sound caption / emotion 같은 frame-level supervision 과 직접 호환 안 됨.

## 2. CLAP 변종 정리 (2026-05 기준)

| variant | repo / weight | backbone | sample rate | clip embed dim | 비고 |
|---|---|---|---|---|---|
| **LAION-CLAP** | `LAION-AI/CLAP` (laion_clap pkg) | HTSAT (audio) + RoBERTa (text) | 48 kHz | 512 | sound retrieval/captioning 용으로 가장 많이 쓰임. PANNs 변종도 있음. |
| **Microsoft CLAP (MS-CLAP)** | `microsoft/CLAP` (msclap pkg) | CNN14 / HTSAT 둘 다 | 44.1 / 48 kHz | 1024 | speech/music/general 도메인 strong. msclap.CLAP 클래스로 wrap. |
| **Pengi-CLAP** | Pengi 페이퍼 | HTSAT-22 + GPT2 | 32 kHz | 768 | 캡션/QA 생성용 prefix tuning. retrieval 용 아님. |

`/mnt/tmp/datasets/laion_epidemic_clapv2/data` 의 의미상 user 는 LAION-CLAP
v2 를 한 번 돌려본 흔적 있음. 따라서 **LAION-CLAP-HTSAT-base** 를 베이스로
가정하고 진행.

## 3. Frame-level feature 추출 후보

HTSAT 는 4-stage Swin Transformer. 입력 mel-spectrogram (10ms hop, 64 mel bins)
이 patch=4 로 split 되어 들어가고 stage 마다 spatial 2x downsample.

| 추출 위치 | 시간 stride | 30 s 입력 fps | 30 s frame 수 | adapter cutoff |
|---|---|---|---|---|
| stage 1 끝 (post Swin block 1) | 10 ms × 4 = 40 ms | 25 fps | 750 | 3584 OK |
| stage 2 끝 | 80 ms | 12.5 fps | 375 | 3584 OK |
| stage 3 끝 | 160 ms | 6.25 fps | 188 | 3584 매우 여유 |
| stage 4 끝 (pre-pool) | 320 ms | 3.13 fps | 94 | 3584 매우 여유 |
| pooled (default CLAP) | n/a | 1 token / clip | 1 | 의미 없음 |

ASR 정확도 vs latency tradeoff: stage 1-2 추출이 합리적. 단, HTSAT 는 보통
**stage 4 + global pool** 출력만 sound classification 으로 학습됐으므로
앞 stage feature 의 phonetic 정보 quality 는 **검증 필요** (Whisper 만큼 ASR 친화적 X).

## 4. 권고 audio_config 스키마 (LAION-CLAP-HTSAT-base 기준)

```json
{
  "model_type": "audio_adapter",
  "adapter_hidden_size": 1024,
  "intermediate_size": 4096,
  "num_attention_heads": 16,
  "num_key_value_heads": 16,
  "num_adapter_layers": 4,
  "head_dim": 64,
  "audio_hidden_size": 768,
  "llm_embed_size": 2560,
  "max_position_embeddings": 8192,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000.0,

  "clap_backbone": "htsat_base",
  "clap_ckpt_path": "/mnt/tmp/cache/laion_clap/630k-audioset-best.pt",
  "clap_sample_rate": 48000,
  "clap_extract_stage": 2,
  "clap_extract_fps": 12.5,
  "clap_mel_bins": 64,
  "clap_window_size": 1024,
  "clap_hop_length": 480
}
```

`audio_hidden_size`: stage 2 출력 dim (HTSAT-base = 256), stage 4 = 768.
숫자는 backbone 코드 확인 후 fix 필요. 위 표는 stage 4 가정.

## 5. audio_encoder.py 작성 시 체크리스트

기존 [encodec audio_encoder.py](../../external/models/Qwen3.5AE-4B-encodec-projL/audio_encoder.py) 패턴 참고:

- [ ] `laion_clap` (또는 `msclap`) 패키지 audio_lmf env 에 설치
- [ ] `from laion_clap import CLAP_Module` 로 backbone load, `model.audio_branch` 가 HTSAT
- [ ] forward hook 으로 stage N 의 pre-pool feature 캡처 (stage 4 = `model.audio_branch.norm` 이후)
- [ ] 시간 차원 정렬: HTSAT 출력은 [B, T_freq, D] 이므로 freq 축 mean / max pool 해서 [B, T, D] 만들기
- [ ] omni dataloader 와 동일하게 raw waveform 입력 (mel 변환은 encoder 내부에서)

## 6. yaml 초안 (stage1_clap_v6.yaml)

placeholder 가 들어간 v6 yaml 을 함께 커밋 — 학습 자체는 4-5 절 결정 후에 가능:

- `model_name_or_path`: `external/models/Qwen3.5AE-4B-clap-projL` (아직 없음)
- `omni_sample_rate`: 48000
- `omni_hop_length`: 3840 (= 48000 / 12.5 fps, stage-2 가정)
- `omni_max_audio_samples`: 1440000 (30 s @ 48 k)
- `cutoff_len`: 3584
- `disable_gradient_checkpointing`: false (HTSAT 무거움)
- 나머지 (per-modality interleave / lr / steps) 는 다른 v6 yaml 과 동일

## 7. Open decisions

A. **Backbone**: LAION-CLAP-HTSAT-base 확정? 아니면 MS-CLAP 도 검토?
B. **Extraction stage**: stage 4 (3 fps) vs stage 2 (12.5 fps). ASR 정확도 떨어지면 stage 2 가 나을 수 있으나 cutoff_len / VRAM 압박.
C. **Freeze 정책**: HTSAT 전체 freeze + projector 만 학습 (다른 5개와 동일) 가 default. unfreeze 옵션은 실험 후.
D. **FSD50K / AudioSet 평가**: stage1 v4 README 의 caption 평가 (sentence-mode wrapper) 가 CLAP feature 와 호환되는지 확인. mel 차이 클 수 있음.

## 8. 후속 작업 순서

1. § 7 A-C 확정
2. backbone 코드 읽고 `clap_*` config 키 / extract layer 정확히 fix
3. `external/models/Qwen3.5AE-4B-clap-projL/{config.json, audio_encoder.py}` 작성
4. setup_v6_projL_models.sh 에 CLAP 라인 추가 (또는 별도 setup_clap.sh)
5. `configs/ASR/stage1_clap_v6.yaml` placeholder 완성
6. `scripts/ASR/run_stage1_clap_v6.sh` 작성

— 2026-05-07 작성, jos
