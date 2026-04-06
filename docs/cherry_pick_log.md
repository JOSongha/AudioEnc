# cherry-pick 로그: sehyun/fast_train_merged → main

브랜치: `origin/sehyun/fast_train_merged`
작업 시작: 2026-04-06

---

## 대상 커밋 목록

| 해시 | 날짜 | 커밋 메시지 | 상태 |
|------|------|-------------|------|
| `cdeeb0d` | 2026-04-03 | feat: add stage1 training scripts, EOS-first collate, inference, CLAUDE.md | ✅ 일부 반영 |
| `d599ae6` | - | Merge branch 'main' into sehyun | ⏭ 스킵 (merge 커밋) |
| `2cf37e0` | - | add Sequence Packing + Flash Attention 2 + Liger Kernel + FSDP | 🔲 미정 |
| `e67a916` | - | Merge remote-tracking branch 'origin/main' into sehyun/fast_train_merged | ⏭ 스킵 (merge 커밋) |

---

## cdeeb0d — 2026-04-03 (일부 반영)

**반영 커밋**: `f9acfa9` (2026-04-06)

### 반영된 파일

| 파일 | 변경 | 내용 |
|------|------|------|
| `CLAUDE.md` | +102줄 (신규) | 리포 가이드 (아키텍처, 실행법, 인코더 레지스트리) |
| `inference.py` | +445줄 (신규) | WER 평가 스크립트 (LibriSpeech split별, 멀티GPU 지원) |
| `train_stage1_eos_first.py` | +505줄 (신규) | EOS-first collate 변형 (`[tok, EOS, PAD]`) |

### 제외된 파일

| 파일 | 이유 |
|------|------|
| `config.py` | `stage1_epochs` 2→20 변경 — 별도 검토 후 적용 |
| `train_stage1.py` | 보류 |

---

## 2cf37e0 — 2026-04-06 (일부 반영)

**반영 커밋**: `7a6d2f7` (2026-04-06)

### 반영된 파일

| 파일 | 변경 | 내용 |
|------|------|------|
| `docs/packing_fa2_liger_fsdp.md` | +240줄 (신규) | Sequence Packing / FA2 / Liger / FSDP 설계 문서 |

### 로컬 전용 (미커밋, 실험용)

| 파일 | 내용 |
|------|------|
| `config.py` | 최적화 플래그 (`use_packing`, `attn_implementation`, `use_liger_kernel`, `use_fsdp`, `log_every`), `audio_pad_token_id` |
| `dataset.py` | `_gigaspeech_lengths()`, Sequence Packing 파이프라인 전체 (`PackedDataset`, `PackedCollator` 등) |
| `model.py` | Liger kernel, FA2, FSDP dtype 대응, `_forward_packed()`, `eos_first` 파라미터 |
