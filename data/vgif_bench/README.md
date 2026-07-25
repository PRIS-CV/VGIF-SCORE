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
  <p><strong>When a video looks right but follows the instruction wrong.</strong></p>
  <p>A diagnostic benchmark for spatio-temporal instruction following<br>in text-to-video generation</p>
  <p>
    <a href="https://pris-cv.github.io/VGIF-SCORE/"><img src="https://img.shields.io/badge/Project-Interactive%20Demo-2f75b5?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Explore the project page"></a>
    <a href="https://arxiv.org/abs/2607.13527"><img src="https://img.shields.io/badge/arXiv-2607.13527-c43b3b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Read the arXiv paper"></a>
    <a href="https://github.com/PRIS-CV/VGIF-SCORE"><img src="https://img.shields.io/badge/GitHub-Code-7659a8?style=for-the-badge&logo=github&logoColor=white" alt="Open the source code"></a>
  </p>
  <p>
    <a href="#why-vgif-bench"><img src="https://img.shields.io/badge/01-Why-3a8b67?style=flat-square" alt="Jump to benchmark motivation"></a>
    <a href="#inside-one-sample"><img src="https://img.shields.io/badge/02-Schema-7659a8?style=flat-square" alt="Jump to sample schema"></a>
    <a href="#evaluation-story"><img src="https://img.shields.io/badge/03-Evaluation-c65b66?style=flat-square" alt="Jump to evaluation method"></a>
    <a href="#load-the-dataset"><img src="https://img.shields.io/badge/04-Load-b97914?style=flat-square" alt="Jump to loading instructions"></a>
  </p>
</div>

<p align="center">
  <a href="https://pris-cv.github.io/VGIF-SCORE/"><img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/hero.jpg" width="100%" alt="Thirty-two generated video frames from five commercial models across all eight VGIF-Bench domains, arranged as an asymmetric collage with an ST-DAG overlay"></a>
</p>
<p align="center"><sub>32 generated frames &middot; 5 commercial models &middot; 8 macro domains &middot; 38 micro domains</sub></p>

<div align="center">
  <img src="https://img.shields.io/badge/Split-Test%20only-2f75b5?style=flat-square" alt="Test split">
  <img src="https://img.shields.io/badge/Prompts-223-3a8b67?style=flat-square" alt="223 prompts">
  <img src="https://img.shields.io/badge/ST--DAG%20Nodes-3%2C656-c65b66?style=flat-square" alt="3656 ST-DAG nodes">
  <img src="https://img.shields.io/badge/Domains-8%20macro%20%7C%2038%20micro-7659a8?style=flat-square" alt="8 macro and 38 micro domains">
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-b97914?style=flat-square" alt="CC BY-NC 4.0">
</div>

## Why VGIF-Bench

> A video may look polished while quietly dropping the event that should
> trigger a transformation, changing the wrong object, or breaking the
> requested temporal order.

VGIF-Bench represents every prompt as an explicit **Spatio-Temporal Directed
Acyclic Graph (ST-DAG)**. The benchmark therefore measures more than visual
similarity: it exposes which instruction units were completed, which
prerequisites failed, and which downstream events became impossible to credit.

<div align="center">
  <img src="https://img.shields.io/badge/Product-Showcase-3a8b67?style=flat-square" alt="Product showcase">
  <img src="https://img.shields.io/badge/Cinematic-Narrative-2f75b5?style=flat-square" alt="Cinematic narrative">
  <img src="https://img.shields.io/badge/Creative-Surreal-7659a8?style=flat-square" alt="Creative surreal">
  <img src="https://img.shields.io/badge/Physical-Interaction-c65b66?style=flat-square" alt="Physical interaction">
  <img src="https://img.shields.io/badge/Emotion-Atmosphere-b97914?style=flat-square" alt="Emotion and atmosphere">
  <img src="https://img.shields.io/badge/Spatial-Orchestration-317d87?style=flat-square" alt="Spatial orchestration">
  <img src="https://img.shields.io/badge/Embodied-Performance-a25574?style=flat-square" alt="Embodied performance">
  <img src="https://img.shields.io/badge/Living-World-568b3f?style=flat-square" alt="Living world">
</div>

| Prompts | Macro / micro domains | Nodes | Edges | QA pairs | AutoRubric dimensions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **223** | **8 / 38** | **3,656** | **3,940** | **3,445** | **892** |

<p align="center">
  <img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/benchmark.png" width="100%" alt="VGIF-Bench prompt, graph, and node statistics">
</p>

## Inside One Sample

<table>
  <tr>
    <td width="33%" valign="top">
      <img src="https://img.shields.io/badge/1-PROMPT-2f75b5?style=flat-square" alt="Prompt"><br><br>
      A long-form generation instruction plus macro and micro domain metadata.
    </td>
    <td width="33%" valign="top">
      <img src="https://img.shields.io/badge/2-ST--DAG-3a8b67?style=flat-square" alt="ST-DAG"><br><br>
      Typed atomic nodes, directed edges, and Boolean prerequisite expressions.
    </td>
    <td width="33%" valign="top">
      <img src="https://img.shields.io/badge/3-AUTORUBRIC-7659a8?style=flat-square" alt="AutoRubric"><br><br>
      Prompt-specific criteria and 1-5 anchors for four perceptual dimensions.
    </td>
  </tr>
</table>

Each JSONL row contains `prompt`, `domain_info`, `complexity`, `st_dag`,
`original_qa_pairs`, and `autorubric`. The test split is the complete benchmark;
it contains evaluation specifications and no training labels.

## Evaluation Story

<p align="center">
  <img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/pipeline.png" width="100%" alt="VGIF-Score evaluation pipeline">
</p>

| Objective branch | Subjective branch | Final sample score |
| :---: | :---: | :---: |
| Dependency-aware atomic QA | Cin + Pur + Mot + Phy | `0.5 x S_obj + 0.5 x S_sub` |

The four rubric ratings use a 1-5 scale and are normalized as
`S_sub = mean(Cin, Pur, Mot, Phy) / 5`. Objective, subjective, and VGIF scores
are computed per prompt-video pair before benchmark-level averaging.

<p align="center">
  <a href="https://pris-cv.github.io/VGIF-SCORE/#case-explorer"><img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/diagnosis.png" width="100%" alt="Dependency-aware causal chain diagnosis"></a>
</p>
<p align="center"><sub>Follow a missed atomic requirement into its dependency-blocked downstream events.</sub></p>

## Load the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("Notyourkev/VGIF-Bench", split="test")
sample = dataset[0]

print(sample["prompt"])
print(sample["st_dag"]["nodes"][:2])
print(sample["original_qa_pairs"][:2])
```

The repository is public; authentication is not required for loading.

<details>
<summary><strong>Validate the canonical export</strong></summary>

From the source repository root:

```text
python code/benchmark/build_vgif_bench.py --validate-only
```

The validator checks sample identity, node and edge references, DAG acyclicity,
QA dependency syntax, rubric schema, and camera-ready totals.
</details>

<details>
<summary><strong>Dataset license</strong></summary>

VGIF-Bench is released under the Creative Commons Attribution-NonCommercial
4.0 International License (`CC BY-NC 4.0`). See `LICENSE` for attribution and
usage requirements.
</details>

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
