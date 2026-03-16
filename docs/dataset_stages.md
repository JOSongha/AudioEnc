# Stage별 데이터셋 구성

## Stage 1 vs Stage 2 데이터 규모

| | LibriSpeech | MLS | 합계 |
|---|---|---|---|
| Stage 1 | ~200h (58k utterances) | ~400h (160k samples) | **~600h** |
| Stage 2 | ~960h (전체) | ~10,000h (전체) | **~10,960h** |

## 왜 Stage 1은 데이터를 줄이는가

Stage 1은 projector(Conv1d)만 학습하고 LLM은 frozen이다.
projector는 파라미터 수가 적고 빠르게 수렴하므로 전체 데이터를 쓸 필요가 없다.
전체 ~11,000h를 쓰면 에폭 1회에 수일이 걸려 반복 실험이 어렵다.

Stage 2는 LoRA로 LLM도 함께 학습하므로 더 많은 데이터로 충분히 파인튜닝한다.

## 설정

```python
# config.py TRAIN_CONFIG
"stage1_librispeech_num_samples": 58_000,   # ~200h (전체 ~281k utterances, ~960h)
"stage1_mls_num_samples":        160_000,   # ~400h (전체 ~4,050k samples, ~10,000h)
```

값을 변경하려면 `config.py`의 `TRAIN_CONFIG`만 수정하면 된다.

## 구현

`train.py`의 `main()`에서 Stage 1/2용 데이터셋을 각각 빌드한다.

```python
stage1_cfg["mls_num_samples"]         = cfg["stage1_mls_num_samples"]
stage1_cfg["librispeech_num_samples"] = cfg["stage1_librispeech_num_samples"]
stage1_train_dataset, val_dataset = build_datasets(stage1_cfg)

stage2_train_dataset, _ = build_datasets(cfg)   # 전체 데이터
```

`build_datasets(cfg)`에서 `librispeech_num_samples` 키가 있으면
전체 LibriSpeech ConcatDataset에서 `random.Random(42).sample()`로 서브샘플한 뒤
`torch.utils.data.Subset`으로 래핑한다. seed=42로 고정되어 재현성이 보장된다.

```python
librispeech = ConcatDataset([train-clean-100, train-clean-360, train-other-500])
if ls_num and ls_num < len(librispeech):
    indices = random.Random(42).sample(range(len(librispeech)), ls_num)
    librispeech = Subset(librispeech, sorted(indices))
```

`_collect_lengths`는 `Subset`도 처리한다.

```python
elif isinstance(dataset, Subset):
    parent_lengths = _collect_lengths(dataset.dataset)
    return [parent_lengths[i] for i in dataset.indices]
```
