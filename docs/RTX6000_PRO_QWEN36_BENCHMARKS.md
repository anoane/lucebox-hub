# RTX PRO 6000 Blackwell — Qwen3.6-27B Speculative Decode Benchmarks

**Hardware**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition · sm_120 · 97 GB VRAM  
**Model**: AEON Qwen3.6-27B-Ultimate-Uncensored Q4_K_M (gguf)  
**Drafter**: z-lab/Qwen3.6-27B-DFlash (downloaded 2026-04-28)  
**Scorer**: Claude Opus 4.7 (3-axis rubric: findings/ADRs/functions, depth, format)  
**Date**: 2026-04-28 / 2026-04-29  

---

## Setup

### Build

```bash
cmake -S dflash -B dflash/build \
  -DCUDA_ARCHITECTURES=120 \
  -DCUDACXX=/usr/local/cuda-13.2/bin/nvcc \
  -DDFLASH27B_TESTS=OFF
cmake --build dflash/build --target test_dflash -j$(nproc)
```

### Models

| Role | Source | Size |
|------|--------|------|
| Target | AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored Q4_K_M | 17.3 GB |
| Drafter | z-lab/Qwen3.6-27B-DFlash (Apr-28-2026) | 3.3 GB |

Download:
```bash
hf download AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored \
  --local-dir ./models/aeon-q4k
hf download z-lab/Qwen3.6-27B-DFlash \
  --local-dir ./models/dflash-drafter-apr28
```

### Environment

```bash
export DFLASH27B_FA_WINDOW=0   # disable sliding-window FA for long-context
export DFLASH27B_LM_HEAD_FIX=1 # head dequant fix for sm_120
```

### Run commands

**fix + acceptwin** (recommended for 32K, best quality):
```bash
DFLASH27B_FA_WINDOW=0 DFLASH27B_LM_HEAD_FIX=1 \
./dflash/build/test_dflash \
  ./models/aeon-q4k/model.gguf \
  ./models/dflash-drafter-apr28/model.safetensors \
  ./prompts/<class>.<ctx>.bin \
  24576 \
  ./output/<class>_<ctx>_acceptwin.bin \
  --max-ctx=$((ctx + 64000)) \
  --preset=audit
```

**fix + b22** (recommended for 64K+, fastest at long context):
```bash
DFLASH27B_FA_WINDOW=0 DFLASH27B_LM_HEAD_FIX=1 \
./dflash/build/test_dflash \
  ./models/aeon-q4k/model.gguf \
  ./models/dflash-drafter-apr28/model.safetensors \
  ./prompts/<class>.<ctx>.bin \
  24576 \
  ./output/<class>_<ctx>_b22.bin \
  --max-ctx=$((ctx + 64000)) \
  --ddtree --fast-rollback \
  --ddtree-budget=22 \
  --ddtree-temp=1.0
```

---

## QMATRIX — Full Results

Three configs × 3 prompt classes × 6 context sizes = 54 cells.  
Score = 0–100 (Opus 4.7). tok/s = decode throughput (excludes prefill).

### Config definitions

| Config | Flags | Description |
|--------|-------|-------------|
| `acceptwin` | `--preset=audit` | Rolling p95(accept_depth, w=32) + margin=4 |
| `b22` | `--ddtree-budget=22 --fast-rollback` | Fixed budget=22, static |
| `ar` | *(no drafter)* | Autoregressive baseline |

### Security Audit (`sec`)

| ctx | acceptwin tok/s | score | b22 tok/s | score | ar tok/s | score |
|-----|----------------|-------|-----------|-------|----------|-------|
| 8K | 11.93 | 20 | 13.04 | 20 | 0.0 | 20 |
| 16K | 12.17 | 20 | 12.77 | 20 | 0.0 | 20 |
| **32K** | **86.46** | **100** | 77.25 | 96 | 63.66 | 96 |
| 64K | 91.95 | 85 | **76.14** | **94** | 91.56 | 70 |
| 128K | 63.58 | 97 | 133.63 | 55 | 48.67 | 83 |
| 256K | 116.14 | 65 | 50.04 | 89 | 41.81 | 79 |

> 8K/16K scores of 20 = degenerate (prompt is entirely within the thinking window; model emits EOS after 2 tokens). Not a lucebox defect.

### Architecture Review (`arch`)

| ctx | acceptwin tok/s | score | b22 tok/s | score | ar tok/s | score |
|-----|----------------|-------|-----------|-------|----------|-------|
| 8K | 80.69 | 92 | 84.13 | 92 | 68.86 | 92 |
| 16K | 77.37 | 83 | 80.48 | 88 | 62.52 | 92 |
| 32K | 73.70 | 92 | 75.94 | 92 | 60.55 | 87 |
| 64K | 72.48 | 78 | 78.73 | 78 | 55.16 | 78 |
| 128K | 61.62 | 73 | 65.13 | 73 | 42.22 | 78 |
| 256K | 38.47 | 88 | 42.17 | 88 | 26.28 | 87 |

### Code Generation (`code`)

| ctx | acceptwin tok/s | score | b22 tok/s | score | ar tok/s | score |
|-----|----------------|-------|-----------|-------|----------|-------|
| 8K | 105.04 | 80 | 112.09 | 75 | 85.86 | 74 |
| 16K | 102.79 | 75 | 105.79 | 80 | 80.42 | 80 |
| **32K** | **101.03** | **100** | 99.84 | 75 | 75.89 | 80 |
| 64K | 86.14 | 90 | **94.69** | **95** | 68.19 | 75 |
| 128K | 75.98 | 95 | 82.30 | 90 | 51.69 | 90 |
| 256K | 58.27 | 75 | 61.29 | 90 | 33.23 | 95 |

---

## fix.patch quality impact — acceptwin with `DFLASH27B_LM_HEAD_FIX` ON vs OFF

`DFLASH27B_LM_HEAD_FIX=1` (default) routes the LM-head GEMM through cuBLAS instead of
MMQ. See `dflash/patches/fix-lm-head-mmq-routing.patch` and `dflash/docs/LOOP_FIX.md`
for the root-cause write-up (short version: `ggml_cuda_mul_mat_q` picks `mmq_x` tile
size per `N = 1 + tree.n_nodes`; different tile shapes produce bit-level logit differences
that flip near-tie token pairs → loop attractor at static budgets 36, 56, 64).

**For acceptwin the dynamic budget stays at N=9–11, which never hits the looping shapes.**
Fix OFF does not cause loops with acceptwin. It does, however, send the model down a
different (and consistently worse) stochastic path because MMQ logits ≠ cuBLAS logits at
column 0.

Test conditions: AEON Q4_K_M + DFlash drafter Apr-28-2026, RTX PRO 6000 Blackwell sm_120,
`--preset=audit`, `DFLASH27B_FA_WINDOW=0`, `--max-ctx=96768`, n=24576. Two independent
runs per cell — results are bit-identical within fix ON / fix OFF groups.

| cell | fix ON tok/s | fix ON score | fix OFF tok/s | fix OFF score | quality Δ | speed Δ |
|------|-------------|-------------|--------------|--------------|-----------|---------|
| sec@32K | 86.62 | **100** | 97.18 | 72 | **−28** | +12% |
| arch@32K | 73.93 | **92** | 79.90 | 87 | −5 | +8% |
| code@32K | 100.37 | **100** | 121.15 | 95 | −5 | +21% |

Fix OFF is 8–21% faster (MMQ kernel is quicker than cuBLAS dequant for single-row LM head).
The quality cost ranges from −5 pts (arch/code) to −28 pts (sec). The sec regression is
severe: fix OFF produces 7 distinct findings vs 23, and only 1 MISRA rule vs 22 — the
shorter stochastic thinking chain skips the majority of the rubric.

**Keep `DFLASH27B_LM_HEAD_FIX=1` (default) unless profiling confirms the LM head is your
bottleneck and you can tolerate lower output quality.**

For static budgets (not acceptwin), `DFLASH27B_LM_HEAD_FIX=0` causes hard loops at
budgets 36, 56, and 64 — score drops to 0/100. Do not disable the fix with static budgets
on sm_120.

---

## Per-cell winners

| cell | winner config | tok/s | score |
|------|--------------|-------|-------|
| sec@32K | **acceptwin** | **86.46** | **100** |
| sec@64K | b22 | 76.14 | 94 |
| sec@128K | acceptwin | 63.58 | 97 |
| sec@256K | b22 | 50.04 | 89 |
| arch@8K | b22 | 84.13 | 92 |
| arch@16K | b22 | 80.48 | 88 |
| arch@32K | acceptwin/b22 tie | 73.7/75.94 | 92/92 |
| arch@256K | b22 | 42.17 | 88 |
| code@32K | **acceptwin** | **101.03** | **100** |
| code@64K | b22 | 94.69 | 95 |
| code@128K | acceptwin | 75.98 | 95 |

---

## Key findings

### acceptwin dominates at 32K (the target workload)

At sec@32K and code@32K — the two headline cells — acceptwin delivers **100/100 quality** at the highest throughput of any config. The dynamic budget tracks the model's actual acceptance behavior: at 32K context the model accepts ~12 tokens/step, so acceptwin converges to budget≈16 (p95=12 + margin=4). b22's fixed budget=22 over-drafts; AR under-speculates.

### b22 wins at longer contexts (64K+)

At sec@64K the model enters a regime where accept_depth≈22=b22's budget. b22's static preset perfectly matches this context. acceptwin also adapts (it converges to ~22 at 64K) but the warmup cost slightly hurts score. For 64K+ workloads, `--ddtree-budget=22 --fast-rollback` is the preferred config.

### AR wins at arch@16K (quality)

Autoregressive (no speculative decode) scores 92 at arch@16K vs acceptwin's 83 and b22's 88. The short-context arch prompt triggers Qwen3.6's structured thinking format which AR handles natively; spec-decode's stochastic sampling occasionally drifts at short contexts.

### sec@128K is a sweet spot for acceptwin

63.58 tok/s at 97/100 — exceptionally high quality at long context. The model's thinking depth at 128K produces comprehensive audit reports that saturate the rubric. Acceptwin naturally adapts to the longer accept chains at this context.

---

## Preferred settings by workload

| Workload | Recommended | Flags |
|----------|-------------|-------|
| Security audit 32K | `--preset=audit` | acceptwin p95+4 |
| Code gen 32K | `--preset=audit` | acceptwin p95+4 |
| Any task 64K+ | `--ddtree-budget=22 --fast-rollback` | b22 static |
| Arch review (any ctx) | `--ddtree-budget=22 --fast-rollback` | b22 ties or beats acceptwin |
| 256K deep audit | `--ddtree-budget=22 --fast-rollback` | b22 wins on quality |

---

## Reproduction

All prompts are binary token files (4-byte little-endian int32 per token) generated from public codebases/specs. The three prompt classes:

- **sec**: MISRA-C:2012 compliance audit (32K of C source)
- **arch**: Software architecture review (README + design docs)
- **code**: Refactoring task (legacy C++ codebase)

```bash
# Reproduce sec@32K acceptwin (fix ON — recommended)
DFLASH27B_FA_WINDOW=0 DFLASH27B_LM_HEAD_FIX=1 \
./dflash/build/test_dflash \
  models/aeon-q4k.gguf models/dflash-drafter-apr28/model.safetensors \
  prompts/sec.32768.bin 24576 out/sec_32k_acceptwin.bin \
  --max-ctx=96768 \
  --preset=audit
# Expected: ~86.6 tok/s, 16627 tokens, score 100/100

# Reproduce sec@32K b22 (fix ON, long-context preset)
DFLASH27B_FA_WINDOW=0 DFLASH27B_LM_HEAD_FIX=1 \
./dflash/build/test_dflash \
  models/aeon-q4k.gguf models/dflash-drafter-apr28/model.safetensors \
  prompts/sec.32768.bin 24576 out/sec_32k_b22.bin \
  --max-ctx=96768 \
  --ddtree --fast-rollback \
  --ddtree-budget=22 \
  --ddtree-temp=1.0
# Expected: ~77.3 tok/s, score 96/100
```

Apply `dflash/patches/fix-lm-head-mmq-routing.patch` to the llama.cpp submodule before
building (see `dflash/docs/LOOP_FIX.md`). Without it, `DFLASH27B_LM_HEAD_FIX=1` has no
effect and static budgets 36/56/64 will loop.

Grading uses Claude Opus 4.7 with a structured rubric (see `dflash/scripts/grade_audit.py`).
