<div align="center">
  <h1>VGIF-Score</h1>
  <p><strong>When a video looks right but follows the instruction wrong.</strong></p>
  <p>Interpretable and Diagnostic Evaluation of Spatio-Temporal<br>Instruction Following in Video Generation</p>
  <p>Songyu Xu, Xin Wang, Qiang Chen, Xinran Wang, Muxi Diao,<br>Yuxuan Zhang, Kongming Liang, Rui Lin, and Zhanyu Ma</p>
  <p>
    <a href="https://pris-cv.github.io/VGIF-SCORE/"><img src="https://img.shields.io/badge/Project-Explore%20the%20Demo-2f75b5?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Explore the project page"></a>
    <a href="https://arxiv.org/abs/2607.13527"><img src="https://img.shields.io/badge/arXiv-2607.13527-c43b3b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Read the arXiv paper"></a>
    <a href="https://huggingface.co/datasets/Notyourkev/VGIF-Bench"><img src="https://img.shields.io/badge/Hugging%20Face-VGIF--Bench-f6c344?style=for-the-badge&logo=huggingface&logoColor=111111" alt="Open VGIF-Bench on Hugging Face"></a>
  </p>
</div>

<p align="center">
  <a href="https://pris-cv.github.io/VGIF-SCORE/"><img src="docs/assets/readme/hero.jpg" width="100%" alt="Thirty-two generated video frames from five commercial models across all eight VGIF-Bench domains, arranged as an asymmetric collage with an ST-DAG overlay"></a>
</p>
<p align="center"><sub>32 generated frames &middot; 5 commercial models &middot; 8 macro domains &middot; one dependency-aware evaluation framework</sub></p>

| Prompts | ST-DAG nodes | Dependencies | Models | Macro / micro domains |
| :---: | :---: | :---: | :---: | :---: |
| **223** | **3,656** | **3,940** | **14** | **8 / 38** |

## The Problem

> **A beautiful video can still miss the instruction.** It may preserve the
> subject and style while dropping the trigger action, changing the wrong
> object, or reversing the requested temporal order.

VGIF-Score turns that opaque failure into a traceable evaluation. Instead of
asking only whether a video is visually plausible, it asks which visible
commitments were completed, how they depend on one another, and whether the
result remains perceptually convincing.

## The Method

<p align="center">
  <img src="docs/assets/readme/pipeline.png" width="100%" alt="VGIF-Score evaluation pipeline">
</p>

| Evaluation branch | Objective completion | Subjective satisfaction |
| :---: | :---: | :---: |
| **Flow** | Prompt &rarr; ST-DAG &rarr; atomic QA | Prompt &rarr; AutoRubric &rarr; four judgments |
| **Evaluates** | Dependency-aware instruction completion | Cinematography, purity, motion, and physics |
| **Normalized score** | `S_obj = completed nodes / all nodes` | `S_sub = mean(Cin, Pur, Mot, Phy) / 5` |

<p align="center"><strong>VGIF(sample) = 0.5 x S_obj + 0.5 x S_sub</strong></p>

Objective, subjective, and VGIF scores are computed **for each prompt-video
pair first** and then averaged over the benchmark. This preserves the sample as
the unit of evaluation and enables node-type, depth, domain, and causal-chain
diagnostics.

## VGIF-Bench

| Prompts | Macro domains | Micro domains | ST-DAG nodes | Dependency edges | QA pairs | Models |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **223** | **8** | **38** | **3,656** | **3,940** | **3,445** | **14** |

<p align="center">
  <img src="docs/assets/readme/benchmark.png" width="100%" alt="VGIF-Bench prompt, graph, and node statistics">
</p>

VGIF-Bench is built around long, dependency-rich prompts rather than isolated
visual attributes. Its eight macro domains cover product showcase, cinematic
narrative, surreal expression, physical interaction, emotion, spatial
orchestration, embodied performance, and the living world. Explore the full
[benchmark landscape](https://pris-cv.github.io/VGIF-SCORE/#benchmark-comparison)
or compare scores across all [8 macro and 38 micro domains](https://pris-cv.github.io/VGIF-SCORE/#leaderboard).

## Read a Failure Trace

<p align="center">
  <a href="https://pris-cv.github.io/VGIF-SCORE/#case-explorer"><img src="docs/assets/readme/diagnosis.png" width="100%" alt="Dependency-aware causal chain diagnosis comparing two generated videos"></a>
</p>

The evaluator isolates the first missed atomic requirement and marks dependent
events as blocked, rather than folding every downstream symptom into one
number. The interactive explorer provides one curated case for every macro
domain, with 16 videos from eight model families.

<div align="center">
  <a href="https://pris-cv.github.io/VGIF-SCORE/#case-explorer"><img src="https://img.shields.io/badge/Open-Case%20Explorer-2f75b5?style=for-the-badge" alt="Open case explorer"></a>
  <a href="https://pris-cv.github.io/VGIF-SCORE/#leaderboard"><img src="https://img.shields.io/badge/Explore-Interactive%20Leaderboard-3a8b67?style=for-the-badge" alt="Explore interactive leaderboard"></a>
</div>

Bulk per-sample benchmark outputs are not published.

## Use the Code

<details open>
<summary><strong>1. Install the evaluation toolkit</strong></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```powershell
$env:VGIF_API_KEY = "..."
$env:VGIF_BASE_URL = "https://your-gemini-compatible-gateway"
```
</details>

<details open>
<summary><strong>2. Load VGIF-Bench from Hugging Face</strong></summary>

```python
from datasets import load_dataset

dataset = load_dataset("Notyourkev/VGIF-Bench", split="test")
print(dataset[0]["prompt"])
```
</details>

<details>
<summary><strong>3. Validate and rebuild public assets</strong></summary>

```powershell
python code/benchmark/build_vgif_bench.py --validate-only
python code/benchmark/build_vgif_bench.py
python code/benchmark/build_results_manifest.py
python code/benchmark/build_project_page_data.py
python code/benchmark/export_hero_collage.py
python code/benchmark/export_readme_media.py
```
</details>

<details>
<summary><strong>4. Evaluate one generated video</strong></summary>

```powershell
python code/evaluation/evaluate_video_qa_accuracy.py `
  --video path\to\video.mp4 `
  --entries data\vgif_bench\vgif_bench.jsonl `
  --metadata-root path\to\metadata `
  --question-mode dependency-rounds

python code/evaluation/evaluate_video_autorubric_scores.py `
  --video path\to\video.mp4 `
  --entries data\vgif_bench\vgif_bench.jsonl `
  --metadata-root path\to\metadata
```
</details>

<details>
<summary><strong>5. Run the test suite</strong></summary>

```powershell
python -m unittest discover -s tests -v
```
</details>

## Repository

| Path | Purpose |
| :---: | :---: |
| `code/benchmark/` | Dataset validation, result manifests, and public visual exports |
| `code/evaluation/` | VLM QA, AutoRubric, dependency propagation, and VGIF scoring |
| `code/generation/` | Provider-specific and open-model generation scripts |
| `code/plotting/` | Camera-ready analysis and figure scripts |
| `data/vgif_bench/` | Hugging Face-ready JSONL export and statistics |
| `docs/` | Interactive project page and curated public media |
| `camera_ready_paper_id_499/` | Camera-ready PDF and LaTeX sources |

The code is released under Apache-2.0. VGIF-Bench is released separately under
CC BY-NC 4.0. See `LICENSE` and `data/vgif_bench/LICENSE`.

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
