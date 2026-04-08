"""
word alignment JSON 파일들을 split별 JSONL로 병합 + 검증.

Usage:
    python3 merge_alignments.py \
        --alignment-dir /mnt/tmp/cache/word_alignments \
        --manifest     /mnt/tmp/cache/word_alignments/manifest.jsonl \
        --output-dir   /mnt/tmp/cache/word_alignments

출력:
    {output-dir}/train-clean-100.jsonl
    {output-dir}/train-clean-360.jsonl
    {output-dir}/train-other-500.jsonl
    {output-dir}/dev-clean.jsonl
    {output-dir}/merge_report.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

VENV = "/mnt/fr20tb/wbl_residency/jos/.venv310/lib/python3.10/site-packages"
if VENV not in sys.path:
    sys.path.insert(0, VENV)

REQUIRED_FIELDS = {"utterance_id", "split", "speaker_id", "chapter_id",
                   "audio_duration", "transcript", "words", "alignment_model", "processed_at"}
REQUIRED_WORD_FIELDS = {"word", "start", "end", "score"}
LOW_SCORE_THRESHOLD = 0.5  # 평균 score 이 미만이면 품질 경고


def validate_entry(data: dict) -> list[str]:
    """검증 실패 이유 목록 반환. 빈 리스트면 정상."""
    issues = []

    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        issues.append(f"missing fields: {missing}")
        return issues  # 이후 검증 불가

    if not isinstance(data["words"], list):
        issues.append("words is not a list")
        return issues

    if len(data["words"]) == 0:
        issues.append("words is empty")
        return issues

    scores = []
    for i, w in enumerate(data["words"]):
        missing_w = REQUIRED_WORD_FIELDS - set(w.keys())
        if missing_w:
            issues.append(f"word[{i}] missing: {missing_w}")
            continue
        if w["end"] < w["start"]:
            issues.append(f"word[{i}] end < start")
        s = float(w.get("score", 0))
        scores.append(s)

    if scores:
        avg_score = sum(scores) / len(scores)
        if avg_score < LOW_SCORE_THRESHOLD:
            issues.append(f"low avg_score={avg_score:.3f} < {LOW_SCORE_THRESHOLD}")

    if data["audio_duration"] <= 0:
        issues.append(f"invalid audio_duration={data['audio_duration']}")

    return issues


def output_json_path(alignment_dir: str, split: str, speaker_id: str,
                     chapter_id: str, utterance_id: str) -> str:
    return os.path.join(alignment_dir, split, speaker_id, chapter_id,
                        utterance_id + ".json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-dir", default="/mnt/tmp/cache/word_alignments")
    parser.add_argument("--manifest",      default="/mnt/tmp/cache/word_alignments/manifest.jsonl")
    parser.add_argument("--output-dir",    default="/mnt/tmp/cache/word_alignments")
    parser.add_argument("--splits",        default="train-clean-100,train-clean-360,train-other-500,dev-clean")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]

    # manifest 로드
    print(f"Loading manifest: {args.manifest}")
    manifest: dict[str, dict] = {}
    with open(args.manifest) as f:
        for line in f:
            e = json.loads(line)
            manifest[e["utterance_id"]] = e
    print(f"  {len(manifest):,} utterances in manifest")

    # split별 utterance_id 목록
    split_utt_ids: dict[str, list] = defaultdict(list)
    for uid, e in manifest.items():
        split_utt_ids[e["split"]].append(uid)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }

    total_ok = total_missing = total_invalid = total_low_score = 0

    for split in splits:
        utt_ids = split_utt_ids.get(split, [])
        if not utt_ids:
            print(f"  [{split}] 0 utterances in manifest — skipping")
            continue

        out_path = os.path.join(args.output_dir, f"{split}.jsonl")
        print(f"\n[{split}] {len(utt_ids):,} utterances → {out_path}")

        ok = missing = invalid = low_score = 0
        invalid_examples = []

        with open(out_path, "w") as out_f:
            for uid in sorted(utt_ids):
                e = manifest[uid]
                json_path = output_json_path(
                    args.alignment_dir,
                    e["split"], e["speaker_id"], e["chapter_id"], uid,
                )
                if not os.path.exists(json_path):
                    missing += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(f"MISSING: {uid}")
                    continue

                try:
                    with open(json_path) as jf:
                        data = json.load(jf)
                except Exception as ex:
                    invalid += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(f"BAD_JSON {uid}: {ex}")
                    continue

                issues = validate_entry(data)
                if issues:
                    low_score_only = all("low avg_score" in i for i in issues)
                    if low_score_only:
                        low_score += 1
                        # 낮은 score 항목도 JSONL에 포함 (필터링은 학습 시)
                        out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        ok += 1
                    else:
                        invalid += 1
                        if len(invalid_examples) < 5:
                            invalid_examples.append(f"INVALID {uid}: {issues}")
                        continue
                else:
                    out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    ok += 1

        pct_ok = 100 * ok / len(utt_ids) if utt_ids else 0
        print(f"  ok={ok:,} ({pct_ok:.1f}%)  missing={missing:,}  "
              f"invalid={invalid:,}  low_score={low_score:,}")
        if invalid_examples:
            print("  예시 문제:")
            for ex in invalid_examples:
                print(f"    {ex}")

        report["splits"][split] = {
            "total_manifest": len(utt_ids),
            "ok": ok,
            "missing": missing,
            "invalid": invalid,
            "low_score_flagged": low_score,
            "coverage_pct": round(pct_ok, 2),
        }
        total_ok += ok
        total_missing += missing
        total_invalid += invalid
        total_low_score += low_score

    total = total_ok + total_missing + total_invalid
    report["total"] = {
        "manifest": len(manifest),
        "ok": total_ok,
        "missing": total_missing,
        "invalid": total_invalid,
        "low_score_flagged": total_low_score,
        "coverage_pct": round(100 * total_ok / len(manifest), 2) if manifest else 0,
    }

    report_path = os.path.join(args.output_dir, "merge_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== 전체 결과 ===")
    print(f"  ok={total_ok:,}  missing={total_missing:,}  invalid={total_invalid:,}  "
          f"low_score={total_low_score:,}")
    print(f"  커버리지: {report['total']['coverage_pct']:.1f}%")
    print(f"  리포트: {report_path}")

    if total_missing > 0:
        print(f"\n  WARNING: {total_missing:,}개 발화 누락 — 워커를 재실행하거나 확인 필요")
        sys.exit(1)
    else:
        print("\n  OK: 모든 발화 처리 완료")


if __name__ == "__main__":
    main()
