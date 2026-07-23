---
pretty_name: VGIF-Bench
license: cc-by-nc-4.0
task_categories:
  - text-to-video
tags:
  - video-generation
  - instruction-following
  - evaluation
  - spatio-temporal-reasoning
configs:
  - config_name: default
    data_files:
      - split: test
        path: vgif_bench.jsonl
---

# VGIF-Bench

VGIF-Bench evaluates long, structurally entangled text-to-video instructions.
Each sample contains the prompt, its ST-DAG decomposition, dependency-aware QA
pairs, and four instruction-conditioned AutoRubric dimensions.

## Dataset Summary

| Field | Count |
| --- | ---: |
| Prompts | 223 |
| ST-DAG nodes | 3,656 |
| ST-DAG edges | 3,940 |
| Dependency-aware QA pairs | 3,445 |
| AutoRubric dimensions | 892 |

The benchmark covers eight macro categories and 38 micro categories. The test
split is the complete benchmark; there are no training labels.

## Sample Structure

Each JSONL row includes:

- `sample_id`, `index`, and `domain_info`
- `prompt` and prompt complexity metadata
- `st_dag.nodes` and `st_dag.edges`
- `original_qa_pairs` with node type and Boolean dependency expression
- `autorubric.dimensions` for cinematography, purity, motion smoothness, and
  physics adherence
- `autorubric.overall_assessment` diagnostic guidance

All expected QA answers are `Yes`; the evaluator determines whether the
generated video visibly satisfies each item. A QA node only receives credit if
its answer matches and its recursive dependency expression passes.

## Subjective Score

The four 1-5 ratings are averaged and divided by the maximum rating of 5. This
is equivalent to averaging `rating / 5` over the four dimensions and reproduces
the finalized paper tables. The historical fifth `Rub` output is not part of
VGIF-Score.

Objective, subjective, and VGIF scores are first computed for each
prompt-video pair and then macro-averaged over the benchmark. The six semantic
node-type diagnostics are micro-averaged instead: all correct dependency-aware
QA nodes of a type are divided by all evaluated QA nodes of that type.

## Validation

From the repository root:

```text
python code/benchmark/build_vgif_bench.py --validate-only
```

The exporter validates sample identity, node and edge references, DAG
acyclicity, QA dependency syntax, rubric schema, and camera-ready totals.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("Notyourkev/VGIF-Bench", split="test")
```

The dataset repository is public; authentication is not required for loading.

## License

VGIF-Bench is released under the Creative Commons Attribution-NonCommercial
4.0 International License (`CC BY-NC 4.0`). See `LICENSE` for the dataset
license notice and attribution requirements.

## Citation

```bibtex
@misc{xu2026vgifscoreinterpretablediagnosticevaluation,
  title={VGIF-Score: Interpretable and Diagnostic Evaluation of Spatio-Temporal Instruction Following in Video Generation},
  author={Songyu Xu and Xin Wang and Qiang Chen and Xinran Wang and Muxi Diao and Yuxuan Zhang and Kongming Liang and Rui Lin and Zhanyu Ma},
  year={2026},
  eprint={2607.13527},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2607.13527},
}
```
