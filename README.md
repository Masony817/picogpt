# PicoGPT

A clean, hackable, and surprisingly capable GPT implementation written in PyTorch.

This is learning code for me to learn language models and trasnformers from scratch. It's also a research testbed for me to learn and implement some modern techniques like Rotary Positional Embeddings (RoPE), the experimental Muon optimizer, and Flash Attention, along with some other updates in the future or more training runs. It's designed to be readable, fast, and easy to modify.

## Features
*   **Modern Architecture**: Standard GPT with optional RoPE and Flash Attention.
*   **Optimizers**: Supports standard AdamW and the new **Muon** optimizer (for 2D tensor updates).
*   **Training**: Distributed Data Parallel (DDP) support, gradient accumulation, and mixed precision (bfloat16).
*   **Data**: Efficient streaming of the FineWeb-Edu dataset trained on the 10B token subset.
*   **Evaluation**: Built-in integration with `lm-eval-harness` for real benchmarks (HellaSwag, PIQA, etc.).

## Setup

The dependencies sit in the standard requirements.txt and there is a node traning setup for a lambda labs 8x h100 node. 

## Usage

### 1. Prepare Data

Download and tokenize the FineWeb-Edu sample (10B tokens):

```bash
python fineweb_set.py
```

This will create a `edu_fineweb10B` directory with 99 sharded training numpy arrays and a validation array.

### 2. Train

You can run a baseline training run or enable specific features.

**Baseline (AdamW, learned pos embeddings):**
```bash
python train.py --baseline
```

**Modern Config (Muon + RoPE + Data shuffling):**
```bash
python train.py --run-2
```

**Custom flags:**
```bash
python train.py --use-muon --use-rope --data-fixes
```

Training logs are sent to Weights & Biases automatically - config with your own if you want to

### 3. Evaluate

Run benchmarks and perplexity checks against pretrained GPT-2 or your own checkpoints:

```bash
python eval.py
```

#### Note: The current eval setup is strictly aligned to my personal project and entity, change this if you run and I plan to fix this in the future. 

## Code Structure

*   `gpt.py`: The model definition. Clean, single-file implementation.
*   `train.py`: The training loop. Handles DDP, logging, and optimization.
*   `data.py`: DataLoader logic for the sharded dataset.
*   `fineweb_set.py`: Data preparation script.
*   `eval.py`: Evaluation for the models and a final report to go with it