# Plan: debug `o` — EOS-before-PAD vs PAD-before-EOS 비교

## 목표

| 모드 | 시퀀스 형태 | 설명 |
|------|------------|------|
| 기본 (현재) | `[tok1, tok2, tok3, PAD, PAD, EOS]` | tokenizer padding 후 EOS append |
| `--debug o` | `[tok1, tok2, tok3, EOS, PAD, PAD]` | 각 시퀀스에 EOS 먼저 붙이고 padding |

---

## 변경 파일

### 1. `dataset.py` — `collate_fn_factory`

- 파라미터 추가: `eos_before_pad: bool = False`
- `eos_before_pad=True`일 때:
  1. `padding=False`로 tokenize → 가변 길이 list of tensors
  2. 각 시퀀스 뒤에 EOS 개별 append
  3. `pad_sequence` 또는 수동 padding으로 배치 정렬 (right-pad)
- `eos_before_pad=False`일 때: 기존 동작 유지

### 2. `train_debug.py` — `run_stage1` / `run_stage2`

- `collate_fn_factory` 호출 시 `eos_before_pad=("o" in debug)` 전달
- `debug_dir` 이름 태그에 `"eos_after"` / `"eos_before"` 포함 (run 구분용)
- `--debug` help string에 `o` 옵션 문서화

### 3. `train_debug.py` — `main()`

- `debug_dir` 자동 생성 이름에 `o` 여부 반영
- wandb config에 `eos_order: "pad_then_eos" | "eos_then_pad"` 키 추가

---

## 구현 순서

1. `dataset.py`: `collate_fn_factory` 시그니처 및 내부 분기 추가
2. `train_debug.py` `run_stage1`: collate 호출부 수정
3. `train_debug.py` `run_stage2`: collate 호출부 수정
4. `train_debug.py` `main()`: debug_dir 태그, wandb config, argparse help 수정

---

## 검증 포인트

- `eos_before_pad=True`일 때, 짧은 시퀀스의 EOS 위치가 `seq_len` 번째인지 확인
- `attention_mask`가 PAD를 0으로 마스킹하는지 확인 (현재 코드에서 `attention_mask` 미사용 여부도 재확인)
- loss 계산 시 PAD 토큰이 `-100`으로 ignore되는지 확인 (label shift 로직 점검)
