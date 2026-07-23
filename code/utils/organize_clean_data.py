"""
================================================================================
 VGIF 数据整理脚本 — 只保留每个模型的最新权威QA数据
================================================================================
 目标: 为14个模型提取223个标准视频的最新依赖轮次QA JSON，
       去重(同一视频多evaluator版本)、去掉旧数据，放入统一文件夹。
================================================================================
"""
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DIR = BASE_DIR / "data" / "clean_qa"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ── 模型名与目录映射 ──
MODEL_DIRS = [
    "Kling-V3", "Seedance-2.0", "Wan-2.7", "ViduQ3-Turbo",
    "PixVerse-V6", "Wan2.2-A14B", "HyVideo-1.5", "LongCat-Video",
    "Mochi-1", "CogVideoX-1.5", "MAGI-1", "URSA", "InfinityStar",
]
LTX_JSON_DIR = BASE_DIR / "LTX-2.0" / "home" / "raochongling" / ".jupyter" / \
               "xusongyu" / "LTX" / "outputs" / "ltx2_videos"

# ── 有权威CSV的模型 ──
_MODEL_CSV_PATHS = {
    "Kling-V3": "Kling-V3/outputs/kling_v3_720p_5s/20260419_172612_all-entries-223/vgif_score_dependency_rounds_per_video.csv",
    "Seedance-2.0": "Seedance-2.0/outputs/seedance_2_0_720p_5s/20260419_160031_all-entries-223/vgif_score_dependency_rounds_per_video.csv",
    "Wan-2.7": "Wan-2.7/outputs/wan_2_7_720p_5s_all_entries/20260420_eval_all_at_once_merged_221/vgif_score_dependency_rounds_per_video.csv",
    "ViduQ3-Turbo": "ViduQ3-Turbo/outputs/viduq3_turbo_720p_5s_all_entries/20260419_173826_all-entries-223/vgif_score_dependency_rounds_per_video_gemini-3-pro-preview.csv",
    "PixVerse-V6": "PixVerse-V6/outputs/pixverse_v6_720p_5s_full/20260419_222116/vgif_score_dependency_rounds_per_video.csv",
}
_OLD_DIR_TO_NEW = {"kling_t2v": "Kling-V3", "seedance2.0": "Seedance-2.0"}


def _fix_path(csv_path_str):
    for old, new in _OLD_DIR_TO_NEW.items():
        csv_path_str = csv_path_str.replace(
            f"{BASE_DIR.as_posix()}/{old}/".replace("\\", "/"),
            f"{BASE_DIR / new}".replace("\\", "/"),
        )
        csv_path_str = csv_path_str.replace(
            f"{BASE_DIR}\\{old}\\",
            f"{BASE_DIR / new}\\",
        )
    return csv_path_str


def collect_authoritative_jsons():
    """收集每个模型的权威JSON路径列表。"""
    model_jsons = {}  # model_name -> [Path, ...]

    # ── 1. 有CSV的模型: 直接用CSV中的qa_eval_file列 ──
    for model_name, csv_rel in _MODEL_CSV_PATHS.items():
        csv_path = BASE_DIR / csv_rel
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception:
            continue
        if "qa_eval_file" not in df.columns:
            continue

        jsons = []
        for _, row in df.iterrows():
            p = _fix_path(row["qa_eval_file"])
            jp = Path(p)
            if not jp.exists():
                found = list(BASE_DIR.rglob(jp.name))
                jp = found[0] if found else None
            if jp and jp.exists():
                jsons.append(jp)
        model_jsons[model_name] = jsons
        print(f"  [CSV] {model_name}: {len(jsons)} JSONs")

    # ── 2. 无CSV的模型: 目录扫描 + 去重 ──
    for display_name in MODEL_DIRS:
        if display_name in model_jsons:
            continue
        model_dir = BASE_DIR / display_name
        if not model_dir.exists():
            continue

        all_files = list(model_dir.rglob("*_dependency_rounds.json"))
        video_groups = {}
        for jp in all_files:
            base_name = re.sub(
                r'_(?:gemini.*?_|g\d+.*?_|latestcfg_)?qa_eval_dependency_rounds\.json$',
                '', jp.name,
            )
            is_standard = ("_qa_eval_dependency_rounds.json" in jp.name and
                           not re.search(r'_(?:gemini|g\d+[a-z]*|latestcfg)_', jp.name))
            video_groups.setdefault(base_name, []).append((jp, is_standard))

        jsons = []
        for candidates in video_groups.values():
            candidates.sort(key=lambda x: (0 if x[1] else 1))
            jp = candidates[0][0]
            if jp.exists():
                jsons.append(jp)
        model_jsons[display_name] = jsons
        dupes = sum(1 for v in video_groups.values() if len(v) > 1)
        print(f"  [扫描] {display_name}: {len(jsons)} JSONs (去重{dupes}个)")

    # ── 3. LTX-2.0 ──
    if LTX_JSON_DIR.exists():
        jsons = list(LTX_JSON_DIR.glob("*_dependency_rounds.json"))
        model_jsons["LTX-2.0"] = jsons
        print(f"  [LTX] LTX-2.0: {len(jsons)} JSONs")

    return model_jsons


def copy_to_clean_dir(model_jsons):
    """将权威JSON复制到干净的目录结构中。"""
    stats = {}
    for model_name, jsons in model_jsons.items():
        model_out = CLEAN_DIR / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        count = 0
        for jp in jsons:
            dest = model_out / jp.name
            if not dest.exists():
                shutil.copy2(jp, dest)
            count += 1
        stats[model_name] = count

    print(f"\n{'='*60}")
    print(f"整理完成！文件已复制到: {CLEAN_DIR}")
    print(f"\n{'模型':<18s} {'JSON数':>6s}")
    print(f"{'-'*26}")
    total = 0
    for m in sorted(stats.keys(), key=lambda x: stats[x], reverse=True):
        print(f"  {m:<18s} {stats[m]:>6d}")
        total += stats[m]
    print(f"{'-'*26}")
    print(f"  {'总计':<18s} {total:>6d}")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print("VGIF 数据整理 — 仅保留最新权威QA JSON")
    print("=" * 60)
    print()

    model_jsons = collect_authoritative_jsons()
    copy_to_clean_dir(model_jsons)


if __name__ == "__main__":
    main()
