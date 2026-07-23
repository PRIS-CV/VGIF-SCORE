from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import sys


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

from dataset_io import load_entries_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "vgif_bench" / "vgif_bench.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "vgif_bench"

EXPECTED_COUNTS = {
    "samples": 223,
    "nodes": 3656,
    "edges": 3940,
    "qa_pairs": 3445,
    "rubric_dimensions": 892,
}
NODE_TYPES = {"entity", "attribute", "location", "action", "state", "causal"}
EDGE_TYPES = {"solid_dependency", "causal_dependency"}
RUBRIC_KEYS = {"cinematography", "purity", "motion_smoothness", "physics_adherence"}
DEPENDENCY_TOKEN_RE = re.compile(r"q\d+|AND|OR|\(|\)", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and export the canonical VGIF-Bench dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-count-drift", action="store_true")
    return parser.parse_args()


def load_entries(path: Path) -> list[dict[str, Any]]:
    entries = load_entries_file(path)
    if entries is None:
        raise ValueError("The benchmark source must be a JSON array or JSON Lines file.")
    for entry in entries:
        dimensions = ((entry.get("autorubric") or {}).get("dimensions") or {})
        if isinstance(dimensions, dict) and "visual_purity" in dimensions and "purity" not in dimensions:
            dimensions["purity"] = dimensions.pop("visual_purity")
    return entries


def dependency_ids(expression: Any) -> list[str]:
    if not isinstance(expression, str) or expression.strip().lower() in {"", "none"}:
        return []
    return re.findall(r"q\d+", expression, flags=re.IGNORECASE)


def validate_dependency(expression: Any, known_ids: set[str]) -> str | None:
    if not isinstance(expression, str) or expression.strip().lower() in {"", "none"}:
        return None
    compact = re.sub(r"\s+", "", expression)
    tokens = DEPENDENCY_TOKEN_RE.findall(expression)
    if "".join(tokens).lower() != compact.lower():
        return f"invalid dependency syntax: {expression!r}"
    balance = 0
    previous = "operator"
    for token in tokens:
        upper = token.upper()
        if upper == "(":
            if previous == "operand":
                return f"missing operator in dependency: {expression!r}"
            balance += 1
            previous = "operator"
        elif upper == ")":
            balance -= 1
            if balance < 0 or previous != "operand":
                return f"invalid parentheses in dependency: {expression!r}"
            previous = "operand"
        elif upper in {"AND", "OR"}:
            if previous != "operand":
                return f"misplaced operator in dependency: {expression!r}"
            previous = "operator"
        else:
            if previous == "operand":
                return f"missing operator in dependency: {expression!r}"
            if token not in known_ids:
                return f"unknown QA id {token!r} in dependency"
            previous = "operand"
    if balance != 0 or previous != "operand":
        return f"incomplete dependency: {expression!r}"
    return None


def graph_depths(nodes: list[dict[str, Any]]) -> dict[str, int]:
    parents = {
        str(node.get("node_id")): [str(item) for item in node.get("parent_nodes", [])]
        for node in nodes
    }
    visiting: set[str] = set()
    memo: dict[str, int] = {}

    def visit(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            raise ValueError(f"cycle detected at {node_id}")
        visiting.add(node_id)
        parent_ids = parents.get(node_id, [])
        depth = 0 if not parent_ids else 1 + max(visit(parent_id) for parent_id in parent_ids)
        visiting.remove(node_id)
        memo[node_id] = depth
        return depth

    for key in parents:
        visit(key)
    return memo


def qa_dependency_depths(qa_pairs: list[dict[str, Any]]) -> dict[str, int]:
    """Validate the QA dependency DAG independently of JSON list order."""
    parents = {
        str(qa.get("id")): dependency_ids(qa.get("dependency"))
        for qa in qa_pairs
    }
    visiting: set[str] = set()
    memo: dict[str, int] = {}

    def visit(qa_id: str) -> int:
        if qa_id in memo:
            return memo[qa_id]
        if qa_id in visiting:
            raise ValueError(f"QA dependency cycle detected at {qa_id}")
        visiting.add(qa_id)
        parent_ids = parents.get(qa_id, [])
        depth = 0 if not parent_ids else 1 + max(visit(parent_id) for parent_id in parent_ids)
        visiting.remove(qa_id)
        memo[qa_id] = depth
        return depth

    for key in parents:
        visit(key)
    return memo


def validate_entries(entries: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    sample_ids: set[str] = set()
    indices: set[int] = set()
    totals = Counter()
    node_types = Counter()
    edge_types = Counter()
    categories = Counter()
    graph_depth_distribution = Counter()
    multi_parent_distribution = Counter()

    for position, entry in enumerate(entries):
        prefix = f"entry[{position}]"
        sample_id = entry.get("sample_id")
        index = entry.get("index")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{prefix}: missing sample_id")
        elif sample_id in sample_ids:
            errors.append(f"{prefix}: duplicate sample_id {sample_id}")
        else:
            sample_ids.add(sample_id)
        if not isinstance(index, int) or index in indices:
            errors.append(f"{prefix}: invalid or duplicate index {index!r}")
        else:
            indices.add(index)

        graph = entry.get("st_dag") if isinstance(entry.get("st_dag"), dict) else {}
        nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
        edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
        node_ids = {str(node.get("node_id")) for node in nodes if node.get("node_id")}
        if len(node_ids) != len(nodes):
            errors.append(f"{prefix}: missing or duplicate node ids")
        for node in nodes:
            node_id = str(node.get("node_id"))
            node_type = str(node.get("type"))
            node_types[node_type] += 1
            if node_type not in NODE_TYPES:
                errors.append(f"{prefix}/{node_id}: invalid node type {node_type!r}")
            parents = node.get("parent_nodes", [])
            if not isinstance(parents, list) or any(str(parent) not in node_ids for parent in parents):
                errors.append(f"{prefix}/{node_id}: invalid parent_nodes")
            else:
                multi_parent_distribution[len(parents)] += 1
        for edge in edges:
            source, target, relation = edge.get("source"), edge.get("target"), edge.get("relation")
            edge_types[str(relation)] += 1
            if str(source) not in node_ids or str(target) not in node_ids:
                errors.append(f"{prefix}: edge references an unknown node")
            if relation not in EDGE_TYPES:
                errors.append(f"{prefix}: invalid edge relation {relation!r}")
        try:
            depths = graph_depths(nodes)
            graph_depth_distribution[max(depths.values(), default=0)] += 1
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")

        qa_pairs = entry.get("original_qa_pairs")
        if not isinstance(qa_pairs, list):
            qa_pairs = []
            errors.append(f"{prefix}: original_qa_pairs must be a list")
        qa_ids = {str(qa.get("id")) for qa in qa_pairs if qa.get("id")}
        if len(qa_ids) != len(qa_pairs):
            errors.append(f"{prefix}: missing or duplicate QA ids")
        for qa in qa_pairs:
            qa_id = str(qa.get("id"))
            if str(qa.get("node_id")) not in node_ids:
                errors.append(f"{prefix}/{qa_id}: unknown node_id")
            if qa.get("type") not in NODE_TYPES:
                errors.append(f"{prefix}/{qa_id}: invalid QA type")
            error = validate_dependency(qa.get("dependency"), qa_ids)
            if error:
                errors.append(f"{prefix}/{qa_id}: {error}")
        try:
            qa_dependency_depths(qa_pairs)
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")

        dimensions = ((entry.get("autorubric") or {}).get("dimensions") or {})
        if not isinstance(dimensions, dict) or set(dimensions) != RUBRIC_KEYS:
            errors.append(f"{prefix}: rubric dimensions must be exactly {sorted(RUBRIC_KEYS)}")

        macro = str((entry.get("domain_info") or {}).get("macro_domain", "UNKNOWN")).split(" (", 1)[0]
        categories[macro] += 1
        totals.update(
            samples=1,
            nodes=len(nodes),
            edges=len(edges),
            qa_pairs=len(qa_pairs),
            rubric_dimensions=len(dimensions) if isinstance(dimensions, dict) else 0,
        )

    summary = {
        **dict(totals),
        "average_nodes_per_prompt": round(totals["nodes"] / max(totals["samples"], 1), 4),
        "average_edges_per_prompt": round(totals["edges"] / max(totals["samples"], 1), 4),
        "categories": dict(sorted(categories.items())),
        "node_types": dict(sorted(node_types.items())),
        "edge_types": dict(sorted(edge_types.items())),
        "graph_depth_distribution": {str(k): v for k, v in sorted(graph_depth_distribution.items())},
        "multi_parent_distribution": {str(k): v for k, v in sorted(multi_parent_distribution.items())},
    }
    return summary, errors


def write_outputs(entries: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "vgif_bench.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for entry in entries:
            file_obj.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    (output_dir / "dataset_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    entries = load_entries(source)
    summary, errors = validate_entries(entries)
    if not args.allow_count_drift:
        for key, expected in EXPECTED_COUNTS.items():
            if summary.get(key) != expected:
                errors.append(f"paper count mismatch for {key}: expected {expected}, got {summary.get(key)}")
    if errors:
        print(f"VGIF-Bench validation failed with {len(errors)} error(s):")
        for error in errors[:50]:
            print(f"- {error}")
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.validate_only:
        write_outputs(entries, summary, args.output_dir.resolve())
        print(f"Exported VGIF-Bench to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
