from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a resume list for autorubric outputs with missing/invalid scores."
    )
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tag", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos_dir = args.videos_dir.resolve()
    summary_path = args.summary.resolve()
    output_path = args.output.resolve()
    suffix = f"_{args.output_tag}_autorubric_eval.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed_names = {
        Path(item["video_file"]).name
        for item in summary.get("failures", [])
        if isinstance(item, dict) and item.get("video_file")
    }

    pending: list[str] = []
    for index in range(223):
        video_name = f"{index:04d}.mp4"
        eval_name = f"{index:04d}{suffix}"
        eval_path = videos_dir / eval_name
        if video_name in failed_names or not eval_path.exists():
            pending.append(video_name)
            continue

        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        score_by_id = {
            row.get("id"): row.get("score")
            for row in results
            if isinstance(row, dict)
        }
        if not all(isinstance(score_by_id.get(key), int) for key in ("Cin", "Pur", "Mot", "Phy")):
            pending.append(video_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(pending) + ("\n" if pending else ""), encoding="utf-8")
    print(f"Pending autorubric videos: {len(pending)}")
    print(f"Resume list written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
