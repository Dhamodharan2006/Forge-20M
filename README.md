# Forge-20M

A 20.75M parameter, GPT-2-style decoder-only transformer, architected and trained completely from scratch — no pretrained weights, no fine-tuning, no HuggingFace shortcuts. Built and trained end-to-end on a single free Kaggle T4 GPU.

---

## Overview

Forge-20M is a small language model built to explore, line by line, how a transformer-based LLM actually works: causal self-attention, positional embeddings, the training loop, mixed-precision optimization, and the practical debugging that goes into making all of it run correctly on constrained hardware.

The project covers the full pipeline:
- Architecture implementation (attention, MLP, transformer blocks) from scratch in PyTorch
- A custom training loop with gradient accumulation, mixed precision, and a cosine learning rate schedule
- Two training runs on real web text, totaling ~628M tokens seen
- A repetition-penalty-aware inference/sampling function
- A Gradio-based chat interface for interactive testing

This is a research/learning project. It is not intended to produce factually reliable output — see [Limitations](#limitations) below for why, and what would be required to change that.

---

## Architecture

| Component | Detail |
|---|---|
| Type | Decoder-only transformer (GPT-2 style) |
| Layers | 10 |
| Attention heads | 8 |
| Embedding dimension | 256 |
| Context length (block size) | 512 tokens |
| Vocabulary size | 50,304 (GPT-2 byte-level BPE, padded to nearest multiple of 64) |
| Positional encoding | Learned absolute |
| Normalization | LayerNorm, pre-norm placement |
| Activation | GELU |
| Attention implementation | Causal multi-head self-attention, with PyTorch's `scaled_dot_product_attention` (flash attention) when available |
| Weight tying | Token embedding and output projection (`lm_head`) share weights |
| Bias | Disabled on all Linear and LayerNorm layers |
| Dropout | 0.0 |

**Total parameters:** 20,878,592 (20.75M)
- Token embedding: 12,877,824
- Position embedding: 131,072
- Transformer blocks (×10): 7,869,440
- Final LayerNorm: 256
- `lm_head`: 0 (tied to token embedding, no additional parameters)

---

## Dataset

- **Source:** [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` split
- **Tokenizer:** `tiktoken`, GPT-2 encoding (byte-level BPE)
- **Format:** streamed from source, tokenized, and stored as `uint16` binary (`train.bin` / `val.bin`)
- **Split:** 95% train / 5% validation
- **Total unique tokens used across both runs:** ~400M (100M in Run 1, 300M fresh/non-overlapping tokens in Run 2)

---

## Training

Trained in two stages on a single NVIDIA Tesla T4 (16GB) via Kaggle.

### Run 1 — Initial training
| Setting | Value |
|---|---|
| Tokens | 100M (`sample-10BT`, first shard) |
| Steps | 5,000 |
| Batch size | 32 |
| Gradient accumulation | 4 |
| Effective batch size | 128 |
| Tokens per iteration | 65,536 |
| Optimizer | AdamW (fused), lr=5e-4, weight_decay=0.1, betas=(0.9, 0.95) |
| LR schedule | Linear warmup (100 iters) + cosine decay to 5e-5 |
| Precision | fp16, autocast + gradient scaler |
| Training time | ~94 minutes |
| Loss (train → val) | 10.87 → 4.18 |

### Run 2 — Extended training on fresh data
| Setting | Value |
|---|---|
| Tokens | 300M fresh, non-overlapping tokens (streamed past the first 100M already consumed) |
| Steps | Resumed from checkpoint, cumulative to 4,931 |
| Optimizer | AdamW (fused), lr=3e-4 (reduced peak for refinement), weight_decay=0.1 |
| LR schedule | Fresh cosine decay cycle over the new steps, min_lr=2e-5 |
| Precision | fp16, autocast + gradient scaler |
| Training time | ~92 minutes |
| Loss (train → val) | 6.29 → 4.21 |

**Both runs:** train/val loss gap stayed under 0.05 throughout — no overfitting observed at any checkpoint.

### Loss at key checkpoints (Run 1)

| Iteration | Train Loss | Val Loss |
|---|---|---|
| 1,000 | 5.13 | 5.15 |
| 2,000 | 4.55 | 4.54 |
| 3,000 | 4.35 | 4.35 |
| 4,000 | 4.23 | 4.24 |
| 4,750 (final) | 4.22 | 4.21 |

---

## Hardware & Environment

| | |
|---|---|
| GPU | NVIDIA Tesla T4, 16GB VRAM |
| Platform | Kaggle Notebooks |
| CPU / RAM | 4 cores / 13GB |
| Python | 3.12 |
| PyTorch | 2.x |
| Key libraries | `torch`, `tiktoken`, `datasets`, `numpy`, `gradio` |

No `torch.compile`, no distributed training — single-GPU, single-process throughout.

---

## Inference

Text is generated with temperature and top-k sampling, plus a repetition penalty applied at each decoding step against all previously generated tokens:

```python
response = ask_gpt_with_penalty(
    prompt="Albert Einstein was",
    max_new_tokens=150,
    temperature=0.7,
    top_k=25,
    repetition_penalty=1.3,
)
```

A Gradio interface (`app.py`) wraps this function with adjustable sliders for max tokens, temperature, top-k, and repetition penalty, plus streaming word-by-word output.

---

## What It Learned vs. What It Didn't

**After 100M tokens:**
- Learned: grammar, punctuation, basic sentence structure, common word co-occurrence patterns
- Did not learn: fell into exact-phrase repetition loops (e.g. "the law of nature is the law of nature..."), topic drift after 2–3 sentences, no reliable factual grounding

**After the additional 300M tokens:**
- Improved: repetition dropped sharply (especially combined with the inference-time penalty), picked up real document formats (Q&A, academic prose, FAQ structure), coherence extended to 4–5 sentences, noticeably richer vocabulary
- Still limited: factual content remains fabricated, multi-step reasoning is absent

This gap is expected at this parameter count — see [Limitations](#limitations).

---

## Limitations

A 20.75M parameter model has a hard capacity ceiling, independent of how much data it sees:

- **No reliable factual recall.** Storing specific facts requires dedicating model capacity to memorization; a model this size has too few parameters to hold broad factual knowledge alongside fluent language modeling. Expect plausible-sounding but frequently incorrect factual claims.
- **No multi-step reasoning.** The model predicts locally plausible next tokens; it does not perform explicit reasoning or planning.
- **Coherence degrades over longer generations.** Topic drift becomes more likely past 4–6 sentences.
- **Not instruction-tuned.** This is a base completion model, not a chat/Q&A model — it continues text in the style of its training data (FineWeb-Edu's educational web-article register) rather than directly answering questions.

More pretraining tokens improve fluency and reduce repetition, but do not meaningfully address the above — that requires either more parameters or a shift to instruction-formatted fine-tuning.

---

## Debugging Notes (Lessons Learned)

A few real issues encountered and fixed during this project, documented here since they were as instructive as the architecture itself:

1. **Repeated-epoch memorization → repetition loops.** Training too many epochs over a small (100M-token) subset caused the model to fall back on repeating memorized phrase patterns rather than generalizing. Fixed by streaming a larger, non-repeating token set for the second run.
2. **Repetition penalty defined but never wired in.** The generation config specified a repetition penalty, but the actual `generate()` function never used it — silently doing nothing. Fixed by implementing the penalty explicitly in the sampling loop.
3. **LR schedule not offset on resume.** nanoGPT-style resume logic restores `iter_num` from the checkpoint, but the cosine LR schedule needs either an offset or a re-scoped `lr_decay_iters` to avoid resuming into an already-decayed, near-flat learning rate.
4. **Checkpoint file mix-up.** A checkpoint believed to be the fully-trained 5,000-step model was later found (by directly inspecting `iter_num` and `best_val_loss` inside the file) to actually be a much earlier 330-step save. Root cause: reused filenames across separate save points, with a later, less-trained checkpoint silently overwriting an earlier, better one. Fixed by adopting unique, self-describing filenames (e.g. `ckpt_final_iter4931_valloss4.211.pt`) and verifying checkpoint contents immediately after every save.

---

## Repository Structure

```
.
├── model.py              # GPTConfig, GPT, Block, CausalSelfAttention, MLP, LayerNorm
├── prepare_data.py       # Streams FineWeb-Edu, tokenizes, writes train.bin / val.bin
├── train.py               # Training loop (scratch + resume), checkpointing, eval
├── generate.py            # ask_gpt_with_penalty — sampling with repetition penalty
├── app.py                  # Gradio chat interface
├── ckpt_final_iter4931_valloss4.211.pt   # Final trained checkpoint
└── README.md
```

---

## Usage

**Load the trained model:**
```python
import torch
from model import GPTConfig, GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('ckpt_final_iter4931_valloss4.211.pt', map_location=device)

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()
```

**Generate text:**
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

response = ask_gpt_with_penalty("The main concept of physics is", repetition_penalty=1.3)
print(response)
```

**Run the chat interface:**
```bash
python app.py
```

---

## Acknowledgements

Architecture based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. Trained on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by HuggingFace.

---

## License

MIT
