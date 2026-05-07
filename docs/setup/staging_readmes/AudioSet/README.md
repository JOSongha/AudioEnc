---
language:
- en
license: cc-by-4.0
size_categories:
- 10K<n<100K
- 1M<n<10M
source_datasets:
- original
task_categories:
- audio-classification
paperswithcode_id: audioset
pretty_name: AudioSet
config_names:
- balanced
- unbalanced
tags:
- audio
configs:
- config_name: balanced
  default: true
  data_files:
  - split: train
    path: data/bal_train/*.parquet
  - split: test
    path: data/eval/*.parquet
- config_name: unbalanced
  data_files:
  - split: train
    path: data/unbal_train/*.parquet
  - split: test
    path: data/eval/*.parquet
- config_name: full
  data_files:
  - split: bal_train
    path: data/bal_train/*.parquet
  - split: unbal_train
    path: data/unbal_train/*.parquet
  - split: eval
    path: data/eval/*.parquet
---

# Dataset Card for AudioSet

## Dataset Description

- **Homepage**: https://research.google.com/audioset/index.html
- **Paper**: https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/45857.pdf
- **Leaderboard**: https://paperswithcode.com/sota/audio-classification-on-audioset

### Dataset Summary

[AudioSet](https://research.google.com/audioset/dataset/index.html) is a dataset of 10-second clips from YouTube, annotated into one or more sound categories, following the AudioSet ontology.

### Supported Tasks and Leaderboards

- `audio-classification`: Classify audio clips into categories. The leaderboard is available [here](https://paperswithcode.com/sota/audio-classification-on-audioset)

### Languages

The class labels in the dataset are in English.

## Dataset Structure

### Data Instances

Example instance from the dataset:
```python
{
 'video_id': '--PJHxphWEs',
 'audio': {
  'path': 'audio/bal_train/--PJHxphWEs.flac',
  'array': array([-0.04364824, -0.05268681, -0.0568949 , ...,  0.11446512,
          0.14912748,  0.13409865]),
  'sampling_rate': 48000
 },
 'labels': ['/m/09x0r', '/t/dd00088'],
 'human_labels': ['Speech', 'Gush']
}
```

### Data Fields

Instances have the following fields:
- `video_id`: a `string` feature containing the original YouTube ID.
- `audio`: an `Audio` feature containing the audio data and sample rate.
- `labels`: a sequence of `string` features containing the labels
  associated with the audio clip.
- `human_labels`: a sequence of `string` features containing the
  human-readable forms of the same labels as in `labels`.

### Data Splits

The distribuion of audio clips is as follows:

#### `balanced` configuration
|           |train|test |
|-----------|----:|----:|
|# instances|18683|17141|

#### `unbalanced` configuration
|           |train  |test |
|-----------|------:|----:|
|# instances|1738657|17141|


## Dataset Creation

### Curation Rationale

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the source language producers?

The labels are from the AudioSet ontology. Audio clips are from YouTube.

### Annotations

#### Annotation process

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the annotators?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Personal and Sensitive Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Discussion of Biases

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Other Known Limitations

1. The YouTube videos in this copy of AudioSet were downloaded in March 2023, so not all of the original audios are available. The number of clips able to be downloaded is as follows:
   - Balanced train: 18683 audio clips out of 22160 originally.
   - Unbalanced train: 1738788 clips out of 2041789 originally.
   - Evaluation: 17141 audio clips out of 20371 originally.
2. Most audio is sampled at 48 kHz 24 bit, but about 10% is sampled at 44.1 kHz 24 bit. Audio files are stored in the FLAC format.

## Additional Information

### Dataset Curators

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Licensing Information

The AudioSet data is licensed under CC-BY-4.0

## Citation

```bibtex
@inproceedings{jort_audioset_2017,
	title	= {Audio Set: An ontology and human-labeled dataset for audio events},
	author	= {Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter},
	year	= {2017},
	booktitle	= {Proc. IEEE ICASSP 2017},
	address	= {New Orleans, LA}
}
```
