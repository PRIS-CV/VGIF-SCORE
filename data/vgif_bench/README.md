---
pretty_name: VGIF-Bench
license: cc-by-nc-4.0
homepage: https://pris-cv.github.io/VGIF-SCORE/
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

<div align="center">
  <h1>VGIF-Bench</h1>
  <p><strong>A diagnostic benchmark for spatio-temporal instruction following<br>in text-to-video generation</strong></p>
  <p>
    <a href="https://pris-cv.github.io/VGIF-SCORE/"><img src="https://img.shields.io/badge/Project-Page-2f75b5?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project page"></a>
    <a href="https://arxiv.org/abs/2607.13527"><img src="https://img.shields.io/badge/arXiv-2607.13527-c43b3b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv paper"></a>
    <a href="https://github.com/PRIS-CV/VGIF-SCORE"><img src="https://img.shields.io/badge/GitHub-Source-7659a8?style=for-the-badge&logo=github&logoColor=white" alt="GitHub source"></a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Split-Test%20only-2f75b5?style=flat-square" alt="Test split">
    <img src="https://img.shields.io/badge/Prompts-223-3a8b67?style=flat-square" alt="223 prompts">
    <img src="https://img.shields.io/badge/Domains-8%20macro%20%7C%2038%20micro-7659a8?style=flat-square" alt="8 macro and 38 micro domains">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-b97914?style=flat-square" alt="CC BY-NC 4.0">
  </p>
</div>

<p align="center">
  <a href="https://pris-cv.github.io/VGIF-SCORE/"><img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/hero.jpg" width="100%" alt="Generated video first frames spanning eight VGIF-Bench domains with an ST-DAG overlay"></a>
</p>
<p align="center"><sub>Eight macro domains, from product showcase and cinematic narrative to physical interaction, embodied performance, and the living world.</sub></p>

A video can look convincing and still miss the instruction that matters. It may
drop a trigger action, transform the wrong object, or break the requested
temporal order. VGIF-Bench makes those failures measurable by representing each
prompt as an explicit spatio-temporal dependency graph.

The [project page](https://pris-cv.github.io/VGIF-SCORE/#case-explorer) presents
one curated diagnostic case per macro domain. The
[benchmark comparison](https://pris-cv.github.io/VGIF-SCORE/#benchmark-comparison)
places VGIF-Bench within the camera-ready evaluation landscape.

## At a Glance

| Field | Count |
| --- | ---: |
| Prompts | 223 |
| Macro / micro domains | 8 / 38 |
| ST-DAG nodes | 3,656 |
| ST-DAG edges | 3,940 |
| Dependency-aware QA pairs | 3,445 |
| AutoRubric dimensions | 892 |

<p align="center">
  <img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/benchmark.png" width="100%" alt="VGIF-Bench prompt, graph, and node statistics">
</p>

## What Each Sample Provides

Each JSONL row contains:

- the long-form generation `prompt` and domain metadata;
- typed `st_dag.nodes` and dependency `st_dag.edges`;
- atomic `original_qa_pairs` with Boolean dependency expressions;
- prompt-specific `autorubric.dimensions` for cinematography, visual purity,
  motion smoothness, and physics adherence;
- diagnostic guidance for interpreting the generated video.

The test split is the complete benchmark. It contains evaluation specifications
and no training labels.

## Evaluation Story

<p align="center">
  <img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/pipeline.png" width="100%" alt="VGIF-Score evaluation pipeline">
</p>

1. The prompt is decomposed into atomic, typed ST-DAG nodes.
2. Dependency-aware QA checks visible completion and preserves prerequisite
   relations between events.
3. The instruction-conditioned AutoRubric evaluates four perceptual dimensions
   with prompt-specific 1-5 anchors.
4. Objective and subjective scores are combined for each prompt-video pair
   before benchmark-level averaging.

```text
S_objective  = completed dependency-aware nodes / all nodes
S_subjective = mean(Cin, Pur, Mot, Phy) / 5
VGIF-Score   = 0.5 * S_objective + 0.5 * S_subjective
```

## Load the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("Notyourkev/VGIF-Bench", split="test")
print(dataset[0]["prompt"])
```

The repository is public; authentication is not required for loading.

## Validate the Export

From the source repository root:

```text
python code/benchmark/build_vgif_bench.py --validate-only
```

The validator checks sample identity, node and edge references, DAG acyclicity,
QA dependency syntax, rubric schema, and camera-ready totals.

## License

VGIF-Bench is released under the Creative Commons Attribution-NonCommercial
4.0 International License (`CC BY-NC 4.0`). See `LICENSE` for attribution and
usage requirements.

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
