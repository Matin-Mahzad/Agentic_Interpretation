# Agentic LLM vs. Deep Segmentation Networks for Seismic Fault Delineation

Companion code and study summary for the manuscript:

> **Agentic Large Language Model Interpretation versus Deep Segmentation Networks for Seismic Fault Delineation: A Blind-Test Comparison of Claude Opus 5.5, U-Net, and Swin UNETR**
> Matin Mahzad · [Journal / preprint link: TODO]

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![PyTorch >= 2.1](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.1-ee4c2c.svg)

---

## TL;DR

Supervised segmentation networks dominate automated seismic fault interpretation, but their outputs are typically **fragmented along continuous faults** and **contaminated by isolated false-positive blobs**. This study asks whether a general-purpose **agentic LLM**, with no task-specific training, can delineate faults at a comparable quality.

**Claude Opus 5.5** operated the manual-interpretation module of [SeismicFlow](https://doi.org/10.1190/GEO-2025-1020) through computer use (screenshots, mouse and keyboard). It was compared with two networks of matched capacity (about 25 M parameters), a **U-Net** and a **Swin UNETR**, both trained on the [Thebe](https://doi.org/10.1016/j.dib.2021.107219) dataset, on seven unseen 2D seismic sections.

- On three labeled sections, **Claude and the Swin UNETR did not differ significantly** in F1, and **both outperformed the U-Net** (two-way ANOVA + Tukey HSD, 45 evaluations).
- Qualitatively, the networks produced fragmented faults, truncated tips and isolated blobs on **all seven** sections. Claude produced continuous traces with **no isolated blobs**, partly because its output is a polyline, but it stopped traces short in low signal-to-noise zones and once joined two faults that the reference keeps separate.
- The two approaches **fail in different ways**.

## Contents

- [Repository contents](#repository-contents)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Self-test and benchmark](#self-test-and-benchmark)
- [Model details](#model-details)
- [Training recipe](#training-recipe)
- [Inference and evaluation protocol](#inference-and-evaluation-protocol)
- [Study design](#study-design)
- [Results](#results)
- [Failure modes](#failure-modes)
- [Limitations](#limitations)
- [Citation](#citation)
- [License and attribution](#license-and-attribution)

## Repository contents

```
.
├── fault_seg_models.py   # U-Net and Swin UNETR (PyTorch), self-test, MAC counter, GPU benchmark
├── LICENSE               # MIT
└── README.md
```

<!-- TODO: list any additional files you add (training script, evaluation/metrics, per-run F1 scores, agent prompt, etc.) -->

`fault_seg_models.py` is self-contained and exposes:

| Name | Description |
|---|---|
| `UNet` | Convolutional baseline (Ronneberger et al., 2015), six-level pyramid, default `base_width=58` |
| `SwinUNETR` | Swin Transformer encoder with a convolutional decoder, default `feature_size=48` |
| `build_model(name, **kwargs)` | Factory: `"unet"` or `"swin_unetr"` |
| `count_params(model)` | Number of trainable parameters |
| `count_macs(model, size)` | Multiply-accumulates of one forward pass (FLOPs / 2) |

## Installation

```bash
git clone https://github.com/Matin-Mahzad/<repo-name>.git
cd <repo-name>
pip install "torch>=2.1"
```

PyTorch >= 2.1 is the only dependency. A CUDA GPU is needed only for `--benchmark`.

## Quick start

Both networks map a standardized single-channel section `(B, 1, H, W)` to a fault-probability map of the same size. Inputs are reflect-padded internally to a multiple of 32 and the output is cropped back, so **any section size is accepted**.

```python
import torch
from fault_seg_models import build_model

model = build_model("swin_unetr")        # or "unet"
# model.load_state_dict(torch.load("your_weights.pt", map_location="cpu"))
model.eval()

# section: 2D amplitude array (H x W), time along the vertical axis
x = torch.as_tensor(section, dtype=torch.float32)[None, None]
x = (x - x.mean()) / x.std()             # per-section standardization (blind-test protocol)

with torch.no_grad():
    prob = model(x)                      # (1, 1, H, W), fault probability, fp32

mask = (prob >= 0.85).squeeze().numpy()  # threshold tau = 0.85, as in the study
```

Custom configurations go through the constructors, for example:

```python
build_model("unet", base_width=58)
build_model("swin_unetr", feature_size=48, window_size=8, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24))
```

> **Tip:** for a Swin UNETR, input sizes that are multiples of 128 need no window padding in the deepest Swin stage. Training used 256 x 256 crops.

## Self-test and benchmark

```bash
python fault_seg_models.py                    # shape checks, gradient checks, parameter and MAC table
python fault_seg_models.py --benchmark        # AdamW step time and peak GPU memory (bf16 autocast)
python fault_seg_models.py --crop 256 --batch 16 --benchmark
```

The self-test verifies that both networks return the input shape for several section sizes, that every parameter receives a gradient, and reports parameters and GMACs (about 25.7 M / 18.6 GMACs for the U-Net and about 25.2 M / 17.6 GMACs for the Swin UNETR at 256 x 256, parameter ratio about 1.02).

## Model details

Both networks are encoder-decoders with a **six-level pyramid** (strides 1 to 32), **five skip connections** and the channel pattern `(w, w, 2w, 4w, 8w, 16w)`. They share the decoder topology (2 x 2 transposed convolution, concatenation with the skip, convolution block, 1 x 1 convolution, sigmoid) and use BatchNorm and LeakyReLU(0.01) in every convolutional block.

| Property | U-Net (baseline) | Swin UNETR |
|---|---|---|
| Encoder | Stacked 3 x 3 convolutions with max pooling | Swin Transformer: patch size 2, depths 2-2-2-2, heads 3-6-12-24, window 8, MLP ratio 4 |
| Decoder blocks | Double convolution | Residual double convolution, with residual adapters on encoder features (all except stride 16) |
| Channel widths | 58, 58, 116, 232, 464, 928 | 48, 48, 96, 192, 384, 768 |
| Spatial reach | 284 px theoretical receptive field | 16 to 128 px attention windows (stages at strides 2, 4, 8, 16) |
| Parameters | 25.7 M | 25.2 M |
| Compute at 256 x 256 | 18.6 GMACs | 17.6 GMACs |
| Peak training memory (batch 16) | 2.5 GiB | 5.5 GiB |
| Training step time (batch 16) | 1,245 ms | 1,684 ms |

Memory and step time were measured for one AdamW step in bf16 autocast at batch size 16 on an NVIDIA RTX 4090 (24 GB). Timing depends on hardware and implementation; re-run `--benchmark` on your own machine.

The networks are matched in **capacity but not in cost**: the Swin UNETR needs about 2.2x the training memory and 1.35x the step time of the U-Net.

## Training recipe

The networks were trained identically except for the peak learning rate, which was chosen per model from a three-point grid at equal compute. (`fault_seg_models.py` contains the loss in `_loss` and the peak step sizes in `PEAK_LR`.)

| Item | Setting |
|---|---|
| Training data | Thebe, 256 x 256 non-overlapping crops of crossline sections |
| Data split (along crossline axis) | 900 training / 200 validation / 703 test crosslines (1,803 total) |
| Preprocessing | Standardization to zero mean, unit variance (training-partition statistics during training) |
| Model selection | Thebe validation crosslines only |
| Loss | Binary cross-entropy + Dice, equal weights |
| Optimizer | AdamW, betas (0.9, 0.999), weight decay 0.01 |
| Schedule | Linear warmup over 5% of iterations (1,500), cosine decay to 1e-6 |
| Peak step size | U-Net: 1e-3, Swin UNETR: 3e-4 |
| Iterations / batch size | 30,000 / 16 |
| Precision | bf16 autocast, loss computed in fp32 |
| Hardware | NVIDIA RTX 4090 (24 GB) |

Each of the five evaluations per network in the study used an **independently trained model with a different random seed**, so reproducing the statistics requires training five models per network. The blind-test sections played no role in training, validation or model selection.

## Inference and evaluation protocol

- **Networks:** one forward pass per full section, then thresholding at **tau = 0.85**. Each test section is standardized with its own statistics.
- **Agent:** instructed to mark only faults it judged with **at least 85% confidence**, so that the operating points are comparable.
- **Scoring:** all three systems output binary masks. A **4-pixel boundary tolerance** is applied identically to every system, because the Thebe reference labels are bands about 8 pixels wide (the agent's pen was 4 pixels thick).
- **Metrics:** precision, recall, F1, IoU, accuracy, specificity, balanced accuracy and Matthews correlation coefficient. **F1 is the summary metric** used for the statistics, because accuracy and specificity are dominated by the 92% to 96% background pixels.
- **Isolated blob** (qualitative analysis): a connected component of a predicted mask that touches no reference fault (labeled sections) or lies on no fault trace of the same system (unlabeled sections), counted by inspection of the figures.

## Study design

### Blind-test sections (seven 2D sections, none used for training, validation or model selection)

| Test | Section | Labels | Fault pixels |
|---|---|---|---|
| 1 | Thebe test set, crossline 230 | Yes | 7.83% |
| 2 | Thebe test set, crossline 470 | Yes | 4.07% |
| 3 | Thebe test set, crossline 381 | Yes | 4.88% |
| 4 | Offshore Middle East, crossline 880 | No | n/a |
| 5 | Opunake 3D (Taranaki Basin, NZ), crossline 3454 | No | n/a |
| 6 | Opunake 3D, inline 2294 | No | n/a |
| 7 | Parihaka 3D (NZ), crossline 4200 | No | n/a |

Sections 4 to 7 are used only to compare behavior under domain shift (raw amplitude ranges differ by almost five orders of magnitude between surveys).

### The agent

- **Model:** Claude Opus 5.5 (`claude-opus-5-5`), run at maximum effort.
- **Interface:** the computer-use toolset (screenshots, clicks, drags, typing, scrolling). The agent loads each section in SeismicFlow and traces faults with its manual-interpretation module, which stores each trace as an ordered list of vertices and draws it with a **4-pixel pen** directly into a label tensor. No rasterization step is applied.
- **Instructions:** follow the Thebe labeling criteria and mark only faults judged with at least 85% confidence.
- **Isolation:** no access to the reference labels, the network outputs, or its own earlier traces (including those of its other runs).
- **Repetitions:** five independent runs on each labeled section, one run on each unlabeled section.

### Statistics

Each system was evaluated **5 times on each of the 3 labeled sections (45 evaluations)**. F1 was analyzed with a two-way ANOVA with interaction (system and section as fixed factors, repeated evaluations as the error term), followed by Tukey's HSD at alpha = 0.05.

## Results

### Mean F1 over the five evaluations (mean ± SD per section)

| System | Test 1 | Test 2 | Test 3 | **Overall mean F1** |
|---|---|---|---|---|
| Claude Opus 5.5 | 0.848 ± 0.003 | 0.845 ± 0.019 | 0.865 ± 0.004 | **0.853** |
| Swin UNETR | 0.839 ± 0.005 | 0.862 ± 0.019 | 0.874 ± 0.009 | **0.858** |
| U-Net | 0.822 ± 0.009 | 0.831 ± 0.022 | 0.840 ± 0.013 | **0.831** |

The agent, whose decision process is not deterministic, varied no more than the networks across evaluations.

### Two-way ANOVA (F1)

| Source | F | p |
|---|---|---|
| System | F(2, 36) = 17.64 | < 0.0001 |
| Section | F(2, 36) = 11.23 | 0.0002 |
| System x Section | F(4, 36) = 1.38 | 0.261 |

The system explains about 36% of the total variance in F1 (eta-squared = 0.36). The system-by-section interaction is not significant.

### Tukey HSD pairwise comparisons (mean over 15 evaluations per system)

| Comparison | Mean ΔF1 | 95% CI | Adjusted p | Significant? |
|---|---|---|---|---|
| Claude − Swin UNETR | −0.006 | (−0.018, 0.006) | 0.474 | No |
| Claude − U-Net | +0.022 | (0.010, 0.034) | 0.001 | Yes |
| Swin UNETR − U-Net | +0.028 | (0.016, 0.039) | 0.001 | Yes |

> **How to read this:** Claude and the Swin UNETR cannot be statistically distinguished on these sections. This is a *no significant difference detected* result, **not** evidence that the two systems are equivalent (the analysis tests for a difference, not for equivalence).

### Isolated blobs (counted by inspection)

| Section | Claude (traces / blobs) | Swin UNETR (blobs) | U-Net (blobs) |
|---|---|---|---|
| Test 1 | 4 / 0 | 3 | 4 |
| Test 2 | 3 / 0 | 3 | 5 to 7 |
| Test 3 | 11 / 0 | 7 to 10 | 7 to 10 |
| Offshore Middle East | 1 / 0 | about 6 | about 6 |
| Opunake crossline | 6 / 0 | about 6 | about 4 |
| Opunake inline | 11 / 0 | about 6 | about 8 |
| Parihaka | 1 / 0 | about 5 | about 6 |

For the U-Net on the Opunake inline section, five of the blobs lie in the water column and at the seabed reflection, where no fault can exist.

## Failure modes

| Failure mode | Networks | Agent | Probable cause |
|---|---|---|---|
| Fragmentation along a fault | Frequent on every section, worse for the U-Net | Not observed | Local evidence (bounded receptive fields / attention windows); continuity is built into polyline traces |
| Isolated blobs | On every section | Not observed | Noise, chaotic reflections and stratigraphic edges mimic faults; domain shift; polylines cannot form blobs |
| Fault tips and deep extents cut short | Frequent | Occasional (Test 3 central fault, Test 2 right fault) | Offset below the wavelet period; low signal-to-noise ratio at depth |
| Separate faults joined | Not observed | Once (Test 2) | The agent connects tips across a subtle gap |

"Not observed" refers only to these seven sections. All three systems lose fault extent where reflection offset vanishes or the signal-to-noise ratio falls, which suggests that part of the residual error belongs to the data and to the (interpretive) labels rather than to the systems.

Because the approaches fail differently, a natural next step is **combining them**: a network proposes, and an agent verifies and connects.

## Limitations

- Only the three Thebe sections carry reference labels; the comparison on the other surveys is **qualitative**. Establishing equivalence between Claude and the Swin UNETR would require labeled data from multiple surveys.
- The Thebe labels are conservative (only faults the interpreters were certain of, with more than 20 m of vertical displacement, within a defined area of interest), so a real fault outside this definition is scored as a false positive for every system.
- The study uses 2D sections; extension to 3D data, additional labeled datasets and benchmarks interpreted independently by multiple interpreters is left for future work.
- The architecture and parameters of Claude Opus 5.5 are not publicly documented, so the agent is specified at the level of its interface rather than its internal computation.

## Citation

If you use this code or build on the study, please cite:

```bibtex
@article{TODO_citation_key,
  title   = {Agentic Large Language Model Interpretation versus Deep Segmentation Networks for Seismic Fault Delineation: A Blind-Test Comparison of Claude Opus 5.5, U-Net, and Swin UNETR},
  author  = {[Author name(s)]},
  journal = {[Journal]},
  year    = {2026},
  note    = {[DOI / URL]}
}
```

Related work this study builds on:

- Mahzad, M. (2026). *SeismicFlow: The first production-grade Python-native GUI platform for integrated geoscience workflows.* Geophysics, 91(6), F13-F21. https://doi.org/10.1190/GEO-2025-1020
- An, Y., Guo, J., Ye, Q., Childs, C., Walsh, J., & Dong, R. (2021). *A gigabyte interpreted seismic dataset for automatic fault recognition.* Data in Brief, 37, 107219. https://doi.org/10.1016/j.dib.2021.107219
- Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. R., & Xu, D. (2022). *Swin UNETR.* arXiv:2201.01266.
- Liu, Z., et al. (2021). *Swin Transformer: Hierarchical vision transformer using shifted windows.* ICCV 2021.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI 2015.

## License and attribution

This repository is released under the **MIT License** (see `LICENSE`).

The Swin UNETR structure follows the [MONAI](https://github.com/Project-MONAI/MONAI) reference implementation (Apache-2.0) of Hatamizadeh et al. (2022), with these adaptations: window size 8 instead of 7, BatchNorm instead of InstanceNorm in the convolution blocks, and truncated-normal initialization of the Linear layers.

The Thebe dataset, and the Opunake and Parihaka surveys, are **not** included in this repository and remain subject to their own terms.

## Disclosure

The author has no relevant financial or non-financial interests to disclose, no affiliation with, and received no funding from, Anthropic or any other developer of the evaluated systems. This study was not supported by any sponsor or funder.
