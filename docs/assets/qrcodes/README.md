# VGIF-Score QR assets

Print-ready QR codes for the public VGIF-Score resources. Each destination is
available as a high-resolution PNG and a scalable SVG.

| Asset | Destination |
| :---: | :---: |
| `paper-qr` | <https://arxiv.org/abs/2607.13527> |
| `dataset-qr` | <https://huggingface.co/datasets/Notyourkev/VGIF-Bench> |
| `code-qr` | <https://github.com/PRIS-CV/VGIF-SCORE> |
| `project-qr` | <https://pris-cv.github.io/VGIF-SCORE/> |

`vgif-score-links.png` and `vgif-score-links.svg` combine all four destinations
into one presentation-ready card.

Regenerate the complete asset set from the repository root:

```bash
python code/benchmark/generate_qr_assets.py
```
