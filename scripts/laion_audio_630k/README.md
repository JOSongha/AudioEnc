# LAION-Audio-630K download scripts

자세한 가이드/카탈로그/우회 경로는 [`docs/laion_audio_630k_download.md`](../../docs/laion_audio_630k_download.md).

| Sub-source | 스크립트 | 사이즈 | 시간(@~100 MB/s) | 메모 |
|---|---|---:|---:|---|
| Freesound (no-overlap) | [`freesound_dl.py`](freesound_dl.py) | ~650 GB | ~2 h | HF Webdataset, `Meranti/CLAP_freesound`, `freesound_no_overlap/*` 만 받음 |
| BBC Sound Effects | [`bbc_dl.sh`](bbc_dl.sh) | 166 GB | ~28 min | HF mirror `marianna13/BBCSoundEffects` (LAION csv는 죽음) |
| Epidemic Sound | [`epidemic_dl.sh`](epidemic_dl.sh) | 72 GB | ~12 min | HF mirror `CLAPv2/epidemic_sound_effects_t5_debiased` (오디오 임베드 parquet) |
| Audiostock | [`audiostock_dl.py`](audiostock_dl.py) | ~250 MB | ~20 min | sample preview만 (`audiostock.net` → `.jp` URL 패치) |
| AudioCaps | [`audiocaps_dl.sh`](audiocaps_dl.sh) | ~44 GB | ~8 min | LAION 외 보너스 — `OpenSound/AudioCaps` HF mirror |

기본 경로는 모두 `/mnt/tmp/datasets/laion_*` (env `OUT=...` 로 override 가능). HF 캐시는 `/mnt/tmp/cache/huggingface`. `/mnt/ddn`은 Lustre quota 때문에 데이터 write 금지.

```bash
# 일반적인 백그라운드 실행 패턴
nohup bash scripts/laion_audio_630k/bbc_dl.sh \
    > /mnt/tmp/datasets/laion_bbc_hf/download.log 2>&1 &
```

비공개 sub-source 4개 (Free To Use Sounds / Sonniss / WeSoundEffects / Paramount, 합계 290 h) 는 LAION이 비공개로 구매한 거라 회수 불가 — docs §1 참조.
