# mini-claude-server

Train a small GPT from scratch, fine-tune it with instructions and LoRA, measure it with its own eval harness, then serve it with a real LLM serving stack: continuous batching, paged KV cache, SSE streaming, a Go router with consistent hashing, a Raft-replicated config store, Kubernetes autoscaling, and hand-written GPU kernels measured with a roofline.

Every claim in this repo is backed by a test or a benchmark.

## Roadmap

The model half follows Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning, 2024). Its three stages map to milestones M1–M6; M7 onwards is serving and infrastructure.

| Book stage | Chapters | Milestone | What gets built |
|---|---|---|---|
| Stage 1: build the LLM | ch02 | M1 | Repo, tokenizer, sliding-window data pipeline |
| Stage 1: build the LLM | ch03–04 | M2 | Causal multi-head attention, GPT model |
| Stage 2: foundation model | ch05, appD | M3 | Pretraining loop with experiment tracking |
| Stage 2: foundation model | ch05 + bonus | M4 | GPT-2 weights, sampling, KV cache, GQA, RoPE |
| Stage 3: fine-tuning | ch06–07, appE | M5 | Instruction SFT and LoRA |
| Stage 3: fine-tuning | ch07 + bonus | M6 | Eval harness (and DPO) |
| Serving | | M7–M10 | Inference server, continuous batching, paged KV, Docker, Kubernetes, load tests |
| Distributed systems | | M11–M13 | Go router, Raft config store, failure and linearizability tests |
| GPU | | M14–M16 | CUDA and Triton kernels, OpenCL port, roofline benchmarks |
| Publish | | M17 | Write-up and demo |

## Status

- [ ] M1: repo skeleton and CI are in place; tokenizer and data pipeline next.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install
uv run pytest -q             # tests
uv run ruff check .          # lint
uv run pre-commit install    # git hooks
```
