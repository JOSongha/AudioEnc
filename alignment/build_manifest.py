"""
LibriSpeech 전체 발화 목록을 manifest.jsonl + shard_N.jsonl 로 생성.

Usage:
    python3 build_manifest.py \
        --librispeech-root /mnt/tmp/cache/LibriSpeech \
        --output-dir /mnt/tmp/cache/word_alignments \
        --num-shards 8
"""

import argparse
import json
import os
import glob


def iter_librispeech_split(root: str, split: str):
    """
    LibriSpeech 디렉토리를 순회하여 (flac_path, transcript, speaker_id, chapter_id, utt_id) 반환.
    .trans.txt 파일에서 transcript를 직접 읽음 — torchaudio 의존성 없음.
    """
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        return

    for trans_path in sorted(glob.glob(os.path.join(split_dir, "*", "*", "*.trans.txt"))):
        chapter_dir = os.path.dirname(trans_path)
        parts = chapter_dir.replace(split_dir + "/", "").split("/")
        speaker_id, chapter_id = parts[0], parts[1]

        with open(trans_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, *rest = line.split(" ")
                transcript = " ".join(rest).lower()
                flac_path = os.path.join(chapter_dir, f"{utt_id}.flac")
                if not os.path.exists(flac_path):
                    continue
                yield {
                    "flac_path":   flac_path,
                    "transcript":  transcript,
                    "split":       split,
                    "speaker_id":  speaker_id,
                    "chapter_id":  chapter_id,
                    "utterance_id": utt_id,
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", default="/mnt/tmp/cache/LibriSpeech")
    parser.add_argument("--output-dir",       default="/mnt/tmp/cache/word_alignments")
    parser.add_argument("--num-shards",       default=8, type=int)
    parser.add_argument("--splits",           default="train-clean-100,train-clean-360,train-other-500,dev-clean")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",")]
    all_entries = []
    for split in splits:
        before = len(all_entries)
        for entry in iter_librispeech_split(args.librispeech_root, split):
            all_entries.append(entry)
        print(f"  {split}: {len(all_entries) - before} utterances")

    print(f"Total: {len(all_entries)} utterances")

    # manifest.jsonl
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {manifest_path}")

    # shard_N.jsonl
    n = args.num_shards
    for i in range(n):
        shard = all_entries[i::n]
        shard_path = os.path.join(args.output_dir, f"shard_{i}.jsonl")
        with open(shard_path, "w") as f:
            for entry in shard:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  shard_{i}: {len(shard)} utterances → {shard_path}")


if __name__ == "__main__":
    main()
