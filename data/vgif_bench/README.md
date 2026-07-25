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
</div>

<p align="center">
  <a href="https://pris-cv.github.io/VGIF-SCORE/"><img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/hero.jpg" width="100%" alt="Thirty-two generated video frames from five commercial models across all eight VGIF-Bench domains, arranged as an asymmetric collage with an ST-DAG overlay"></a>
</p>
<p align="center"><sub>32 generated frames &middot; 5 commercial models &middot; 8 macro domains &middot; 38 micro domains</sub></p>

<table align="center" style="margin-left: auto; margin-right: auto;">
  <thead>
    <tr>
      <th align="center" style="text-align: center;">Split</th>
      <th align="center" style="text-align: center;">Prompts</th>
      <th align="center" style="text-align: center;">ST-DAG nodes</th>
      <th align="center" style="text-align: center;">Macro / micro domains</th>
      <th align="center" style="text-align: center;">License</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" style="text-align: center;"><strong>Test</strong></td>
      <td align="center" style="text-align: center;"><strong>223</strong></td>
      <td align="center" style="text-align: center;"><strong>3,656</strong></td>
      <td align="center" style="text-align: center;"><strong>8 / 38</strong></td>
      <td align="center" style="text-align: center;"><strong>CC BY-NC 4.0</strong></td>
    </tr>
  </tbody>
</table>

## Why VGIF-Bench

<p align="center"><em>A polished video can still drop the trigger action, change the wrong object, or break the requested temporal order.</em></p>

VGIF-Bench represents every prompt as an explicit **Spatio-Temporal Directed
Acyclic Graph (ST-DAG)**. The benchmark therefore measures more than visual
similarity: it exposes which instruction units were completed, which
prerequisites failed, and which downstream events became impossible to credit.

<p align="center">
  <strong>Product Showcase</strong> &middot; <strong>Cinematic Narrative</strong> &middot; <strong>Creative &amp; Surreal</strong> &middot; <strong>Physical Interaction</strong><br>
  <strong>Emotion &amp; Atmosphere</strong> &middot; <strong>Spatial Orchestration</strong> &middot; <strong>Embodied Performance</strong> &middot; <strong>Living World</strong>
</p>

<table align="center" style="margin-left: auto; margin-right: auto;">
  <thead>
    <tr>
      <th align="center" style="text-align: center;">Prompts</th>
      <th align="center" style="text-align: center;">Macro / micro domains</th>
      <th align="center" style="text-align: center;">Nodes</th>
      <th align="center" style="text-align: center;">Edges</th>
      <th align="center" style="text-align: center;">QA pairs</th>
      <th align="center" style="text-align: center;">AutoRubric dimensions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" style="text-align: center;"><strong>223</strong></td>
      <td align="center" style="text-align: center;"><strong>8 / 38</strong></td>
      <td align="center" style="text-align: center;"><strong>3,656</strong></td>
      <td align="center" style="text-align: center;"><strong>3,940</strong></td>
      <td align="center" style="text-align: center;"><strong>3,445</strong></td>
      <td align="center" style="text-align: center;"><strong>892</strong></td>
    </tr>
  </tbody>
</table>

<p align="center">
  <img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/benchmark.png" width="100%" alt="VGIF-Bench prompt, graph, and node statistics">
</p>

## Inside One Sample

<table align="center" style="margin-left: auto; margin-right: auto;">
  <thead>
    <tr>
      <th align="center" style="text-align: center;">Component</th>
      <th align="center" style="text-align: center;">Included content</th>
      <th align="center" style="text-align: center;">Evaluation role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" style="text-align: center;"><strong>Prompt</strong></td>
      <td align="center" style="text-align: center;">Long-form instruction and domain metadata</td>
      <td align="center" style="text-align: center;">Defines the requested video</td>
    </tr>
    <tr>
      <td align="center" style="text-align: center;"><strong>ST-DAG</strong></td>
      <td align="center" style="text-align: center;">Typed nodes, edges, and Boolean dependencies</td>
      <td align="center" style="text-align: center;">Makes instruction structure explicit</td>
    </tr>
    <tr>
      <td align="center" style="text-align: center;"><strong>Atomic QA</strong></td>
      <td align="center" style="text-align: center;">Node-level visual questions</td>
      <td align="center" style="text-align: center;">Measures dependency-aware completion</td>
    </tr>
    <tr>
      <td align="center" style="text-align: center;"><strong>AutoRubric</strong></td>
      <td align="center" style="text-align: center;">Prompt-specific criteria and 1-5 anchors</td>
      <td align="center" style="text-align: center;">Measures visual satisfaction</td>
    </tr>
  </tbody>
</table>

Each JSONL row contains `prompt`, `domain_info`, `complexity`, `st_dag`,
`original_qa_pairs`, and `autorubric`. The test split is the complete benchmark;
it contains evaluation specifications and no training labels.

## Evaluation Story

<p align="center">
  <img src="https://raw.githubusercontent.com/PRIS-CV/VGIF-SCORE/main/docs/assets/readme/pipeline.png" width="100%" alt="VGIF-Score evaluation pipeline">
</p>

<table align="center" style="margin-left: auto; margin-right: auto;">
  <thead>
    <tr>
      <th align="center" style="text-align: center;">Objective branch</th>
      <th align="center" style="text-align: center;">Subjective branch</th>
      <th align="center" style="text-align: center;">Final sample score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" style="text-align: center;">Dependency-aware atomic QA</td>
      <td align="center" style="text-align: center;">Cin + Pur + Mot + Phy</td>
      <td align="center" style="text-align: center;"><code>0.5 x S_obj + 0.5 x S_sub</code></td>
    </tr>
  </tbody>
</table>

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
