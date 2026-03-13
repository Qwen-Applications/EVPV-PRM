<div align="center">
<h1>EVPV-PRM: Explicit Visual Premise Verification for Multimodal Process Reward Models</h1>

<!-- Badges -->
<a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?style=for-the-badge" alt="arXiv"></a>
<a href="https://github.com/Qwen-Applications/EVPV-PRM"><img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License"></a>

<p align="center">
  <i><b>Multimodal Reasoning Team</b></i>
</p>

EVPV-PRM grounds step-level reward estimation in *verifiable visual facts*. Before judging each reasoning step the model (i) extracts a structured JSON description of the image, (ii) extracts the step's *visual-dependency checklist*, and (iii) scores the checklist against the structured description to produce a *reliability gate* that modulates the raw step score. This makes the reward signal robust to visual hallucinations and improves Best-of-N reranking across diverse multimodal benchmarks.

<!-- 若有框架图，取消注释并放入 assets/framework.png -->
<!--
<p align="center">
  <img src="assets/framework.png" alt="EVPV-PRM framework" width="90%"/>
</p>
<p align="center">
  <b>Figure 1.</b> Overview of the EVPV-PRM framework. Structured image description, visual-dependency checklist, and reliability gate modulate step-level rewards.
</p>
-->

</div>

## ⚙️ 1. Setup and Installation

From the repository root:

```bash
pip install -r requirements.txt
```

**Models required** (download separately):

| Role | Recommended model |
|------|-------------------|
| EVPV-PRM verifier | Your fine-tuned InternVL checkpoint |
| Policy model | InternVL2.5-8B / 26B / 76B |
| API verifier | Qwen2.5-VL-72B (remote) |

## 📥 2. Data and Model Preparation

Data layout:

- **VisualProcessBench**: `data/visualprocessbench/visualprocessbench.jsonl` and `data/visualprocessbench/images/`
- **General benchmarks** (MathVista, etc.): `data/benchmarks/benchmarks.jsonl`, `data/benchmarks/mmmu.jsonl` and `data/benchmarks/images/`

Place image files in the corresponding `images/` directories. See **Data Format** below for JSONL schemas.

---

## 🔄 3. Training Pipeline (ms-swift)

We use **[ms-swift](https://github.com/modelscope/swift)** for SFT and DPO training. The pipeline below prepares data from [VisualPRM400K](https://modelscope.cn/datasets/OpenGVLab/VisualPRM400K) and runs vision SFT, optional DPO, step-discrimination SFT, and LoRA merge.

**Config:** Copy `scripts/config.example.env` to `config.env`, set your paths and (if using API-based data construction) `LLM_API_URL` and `LLM_API_TOKEN`. Source it before running:

```bash
cp scripts/config.example.env config.env
# Edit config.env, then:
set -a && source config.env && set +a
```

### Step 1 — Download training data

```bash
pip install modelscope
modelscope download --dataset OpenGVLab/VisualPRM400K
```

Set `VISUAL_PRM_ROOT` to the directory where the dataset is extracted (contains `images/` and `annotations/`).

### Step 2 — Extract preference pairs

From each subset (e.g. GeomVerse), extract (+, −) pairs and write `swiftdata.jsonl` and images under `OUTPUT_BASE_DIR/<subset>/`:

```bash
export VISUAL_PRM_ROOT=/path/to/VisualPRM400K
export IMAGE_SUBSET=GeomVerse
export OUTPUT_BASE_DIR=/path/to/evpv_data
python scripts/data/extract_preference_pairs.py
```

Repeat for other subsets by changing `IMAGE_SUBSET`, or extend the script to loop over subsets.

### Step 3 — Build vision-understanding SFT data (optional, API)

If you have an LLM API for image description, generate `sft_vision_data.jsonl`:

```bash
export LLM_API_URL=https://your-llm-api/v1/api/chat
export LLM_API_TOKEN=your_token
export OUTPUT_BASE_DIR=/path/to/evpv_data
export DATA_SUB_DIRS=Geo170K,GeometryData,TabMWP,UniGeo,GeomVerse,GEOS,MAVIS-Geometry
python scripts/data/build_vision_sft_data.py
```

Output: `$OUTPUT_BASE_DIR/sft_vision_data.jsonl`.

### Step 4 — Merge image descriptions into per-subset JSONL

Attach descriptions from Step 3 to each subset’s swiftdata so step-judge scripts can read them:

```bash
python scripts/data/merge_vision_descriptions.py
```

This creates `swiftdata_image_describe.jsonl` in each subset dir under `OUTPUT_BASE_DIR`.

### Step 5 — Build step-discrimination SFT data (optional, API)

Run the step-judge data builder (reads `swiftdata_image_describe.jsonl` from each dir, calls LLM API):

```bash
export LLM_API_URL=... LLM_API_TOKEN=...
export OUTPUT_BASE_DIR=/path/to/evpv_data
python scripts/data/build_step_judge_sft_data.py
```

Output: `$OUTPUT_BASE_DIR/sft_processed_data_multithread.jsonl` (and a progress file for resume).

### Step 6 — Vision SFT training (ms-swift)

```bash
export OUTPUT_BASE_DIR=/path/to/evpv_data
export WORKSPACE_DIR=/path/to/workspace
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train/01_sft_vision.sh
```

Checkpoints are saved under `$WORKSPACE_DIR/Qwen2.5VL-VisionOutput` (or `VISION_SFT_OUTPUT_DIR`).

### Step 7 — Vision DPO training (optional)

Prepare `dpo_mm_data.jsonl` (preference pairs in ms-swift DPO format), then:

```bash
export DPO_DATASET=/path/to/evpv_data/dpo_mm_data.jsonl
export WORKSPACE_DIR=/path/to/workspace
bash scripts/train/02_dpo_vision.sh
```

### Step 8 — Merge LoRA into base model

After vision SFT, merge a checkpoint into the base model so you can run step-judge SFT (Step 9) or deploy:

```bash
export WORKSPACE_DIR=/path/to/workspace
export BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export TRAIN_OUTPUT_DIR=$WORKSPACE_DIR/Qwen2.5VL-VisionOutput
export CKPT_NAME=checkpoint-564
bash scripts/train/04_merge_lora.sh
```

Merged model is written to `$WORKSPACE_DIR/Qwen2.5VL-VisionOutput-merged` (or `MERGED_MODEL_DIR`).

### Step 9 — Step-discrimination SFT training (ms-swift)

Use the **merged** vision model from Step 8 as base:

```bash
export MERGED_VISION_MODEL=/path/to/workspace/Qwen2.5VL-VisionOutput-merged
export STEP_JUDGE_DATASET=/path/to/evpv_data/sft_processed_data_multithread.jsonl
bash scripts/train/03_sft_step_judge.sh
```

| Script | Purpose |
|--------|---------|
| `scripts/data/extract_preference_pairs.py` | Extract (+, −) pairs from VisualPRM400K → swiftdata.jsonl |
| `scripts/data/build_vision_sft_data.py` | Build vision SFT JSONL via optional LLM API |
| `scripts/data/merge_vision_descriptions.py` | Merge descriptions into swiftdata_image_describe.jsonl |
| `scripts/data/build_step_judge_sft_data.py` | Build step-judge SFT JSONL via optional LLM API |
| `scripts/train/01_sft_vision.sh` | Vision SFT (ms-swift) |
| `scripts/train/02_dpo_vision.sh` | Vision DPO (ms-swift) |
| `scripts/train/03_sft_step_judge.sh` | Step-judge SFT (ms-swift, from merged vision model) |
| `scripts/train/04_merge_lora.sh` | Merge LoRA checkpoint to full model |

---

## 🚀 4. Inference and Evaluation (EVPV-PRM)

All scripts are run as Python modules from the repository root so that relative imports (`from .prompts import …`) resolve correctly.

| Task | Script | Description |
|------|--------|-------------|
| **VisualProcessBench** (local) | `python -m evpv_prm.step_verifier_local` | Step verification with local vLLM |
| **VisualProcessBench** (API) | `python -m evpv_prm.step_verifier_api` | Step verification via remote API (set `API_URL` / `API_HEADERS` in script) |
| **Best-of-N** (general) | `policy_inference` → `evpv_prm_inference` → `compute_bon_metrics` | Generate 8 candidates, score with EVPV-PRM, compute Pass@k / BoN@k |
| **Best-of-N** (MMMU) | `policy_inference_mmmu` → `evpv_prm_inference_mmmu` → `compute_bon_metrics` | Same pipeline for MMMU |
| **Perception intervention** | `perception_intervention_inference` → `evpv_prm_perception_intervention_eval` | Causal study of visual-evidence quality |
| **Constraint corruption** | `constraint_corruption_ablation` | DROP/FLIP constraint noise ablation |
| **VPBench ablation** | `vpbench_ablation_runner` | 27-config ablation on VisualProcessBench |

**Example — Best-of-N reranking (general benchmarks):**

```bash
# Step 1: generate 8 candidates per question
python -m evpv_prm.policy_inference

# Step 2: score candidates with EVPV-PRM
python -m evpv_prm.evpv_prm_inference

# Step 3: compute Pass@k and BoN@k
python -m evpv_prm.compute_bon_metrics
```

**Example — VisualProcessBench (local vLLM):**

```bash
python -m evpv_prm.step_verifier_local
```

**Key results** (representative):

| Benchmark | Policy | Pass@8 | BoN@8 (EVPV-PRM) | ΔBoN |
|-----------|--------|--------|-------------------|------|
| MathVista | InternVL2.5-8B | 74.4 | **79.1** | +4.7 |
| MathVision | InternVL2.5-8B | 32.6 | **36.8** | +4.2 |
| MathVerse-VO | InternVL2.5-8B | 55.2 | **60.1** | +4.9 |
| VisualProcessBench (F1) | — | — | **68.3** | — |

## 📁 Repository Structure

```
EVPV-PRM/
├── README.md
├── requirements.txt
├── scripts/
│   ├── config.example.env                  # Example env config (copy to config.env)
│   ├── data/
│   │   ├── extract_preference_pairs.py     # Extract (+, −) pairs from VisualPRM400K
│   │   ├── build_vision_sft_data.py        # Build vision SFT data (optional API)
│   │   ├── merge_vision_descriptions.py   # Merge descriptions → swiftdata_image_describe
│   │   └── build_step_judge_sft_data.py    # Build step-judge SFT data (optional API)
│   └── train/
│       ├── 01_sft_vision.sh                 # Vision SFT (ms-swift)
│       ├── 02_dpo_vision.sh                # Vision DPO (ms-swift)
│       ├── 03_sft_step_judge.sh            # Step-judge SFT (ms-swift)
│       └── 04_merge_lora.sh                # Merge LoRA into base model
├── evpv_prm/
│   ├── __init__.py
│   ├── prompts.py                          # All prompt templates (single source of truth)
│   ├── utils.py                            # Shared JSON parsing and IO helpers
│   ├── step_verifier_local.py              # Step verification — local vLLM
│   ├── step_verifier_api.py                # Step verification — remote API
│   ├── policy_inference.py                 # Policy: generate 8 candidates (general benchmarks)
│   ├── policy_inference_mmmu.py            # Policy: generate 8 candidates (MMMU)
│   ├── evpv_prm_inference.py               # EVPV-PRM scoring pipeline (general benchmarks)
│   ├── evpv_prm_inference_mmmu.py          # EVPV-PRM scoring pipeline (MMMU)
│   ├── compute_bon_metrics.py              # Compute Pass@k / BoN@k metrics
│   ├── perception_intervention_inference.py
│   ├── evpv_prm_perception_intervention_eval.py
│   ├── constraint_corruption_ablation.py   # Constraint noise vs. accuracy
│   └── vpbench_ablation_runner.py          # Ablation suite on VisualProcessBench
└── data/
    ├── visualprocessbench/
    │   ├── visualprocessbench.jsonl
    │   └── images/
    └── benchmarks/
        ├── benchmarks.jsonl
        ├── mmmu.jsonl
        └── images/
```

**Core modules**: `prompts.py` (templates), `evpv_prm_inference.py` / `evpv_prm_inference_mmmu.py` (three-stage EVPV-PRM scoring), `compute_bon_metrics.py` (Pass@k / BoN@k).

## 📋 Data Format

**VisualProcessBench** (`data/visualprocessbench/visualprocessbench.jsonl`):

```json
{
  "question": "...",
  "image": "relative/path/to/image.png",
  "answer": "C",
  "response": {
    "steps": ["Step 1: ...", "Step 2: ..."],
    "process_correctness": [1, -1, 1]
  }
}
```

**General benchmarks** (`data/benchmarks/benchmarks.jsonl`):

```json
{
  "pid": "mathvista_001",
  "question": "...",
  "image": "relative/path/to/image.png",
  "answer": "42"
}
```

**Policy output** (from `policy_inference.py`):

```json
{
  "pid": "...",
  "vlmresponse1": {
    "reasoningprocess": [
      {"steptext": "...", "visualdependency": "...or null"}
    ],
    "finalanswer": "C"
  },
  "vlmresponse2": { "..." }
}
```

| Module | Purpose |
|--------|---------|
| `prompts.py` | Single source of truth for all prompt templates and builder functions |
| `utils.py` | Robust JSON parsing, image path resolution, thread-safe JSONL IO |
| `step_verifier_*.py` | Step-level verification on VisualProcessBench |
| `policy_inference*.py` | Generate diverse candidate solutions with InternVL |
| `evpv_prm_inference*.py` | Three-stage EVPV-PRM scoring with checkpoint support |
| `compute_bon_metrics.py` | Compute Pass@k / BoN@k from scored outputs |
| `perception_intervention_*.py` | Causal study of visual-evidence quality |
| `constraint_corruption_ablation.py` | DROP/FLIP constraint noise experiments |
| `vpbench_ablation_runner.py` | 27-configuration ablation on VisualProcessBench |

## 🙏 Acknowledgements

We build on and thank the open-source communities behind InternVL, vLLM, and the benchmark datasets (MathVista, MathVision, MathVerse, VisualProcessBench, MMMU, etc.).

## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@article{evpv_prm_2025,
  title   = {EVPV-PRM: Explicit Visual Premise Verification for Multimodal Process Reward Models},
  author  = {Anonymous},
  journal = {arXiv preprint},
  year    = {2025},
}
```
