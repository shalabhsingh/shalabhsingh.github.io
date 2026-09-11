---
layout: post
title: "Local Coding-Agent Inference on Apple Silicon: What the Numbers Actually Say"
---

## Setup

**Hardware:** Apple M4, 24GB unified memory  
**OS:** macOS (Darwin 25.6.0)  
**Serving engine:** vllm-metal v0.27.1 (MLX backend)  
**Model:** Qwen/Qwen3-8B (dense, not the Qwen3.5/3.6/3.8 hybrid/MoE series)  
**Goal:** Measure what inference-layer choices — quantization level, prefix caching, concurrency — actually do to a coding-agent workload on constrained hardware.

---

## Why Qwen3-8B, Not Qwen3.6-27B

The original plan was Qwen3.6-27B. Two reasons to change:

**Memory:** At 4-bit quantization, 27B weights run ~16GB. That leaves roughly 5GB for KV cache, macOS overhead (~3–4GB), and anything else. At 8K–32K context — realistic for a coding agent resending a long system prompt and tool schema each turn — you run out. Not a theoretical problem; a practical one.

**Prefix caching support:** vllm-metal's automatic prefix caching (the most interesting benchmark target for an agentic workload) is ✅ fully supported on the dense Qwen3 family and only 🔵 experimental on the Qwen3.5/3.6/3.8 hybrid/MoE series. The whole point of this benchmark is the prefix-caching finding; running it on experimental support makes the results untrustworthy.

Qwen3-8B at 4-bit: ~4.5GB weights. That leaves ~15GB for KV cache, OS, and headroom. The benchmark can actually run.

---

## First Encounter: The BF16 Memory Wall

**What happened:** Downloaded Qwen3-8B BF16 weights (~16GB) and ran:

```
vllm serve <local-path> --max-model-len 8192 --enable-prefix-caching
```

**Error:**

```
ValueError: Paged attention: not enough Metal memory for KV cache.
metal_limit=19.07GB, fraction=0.92, usable_metal=17.54GB,
model_memory=16.38GB, overhead=1.30GB, kv_budget=-0.13GB.
```

**What this means:**

- macOS + other apps consumed ~5GB of the 24GB unified memory pool
- That left 19.07GB available to Metal (the GPU subsystem)
- At 92% fraction: 17.54GB usable
- BF16 weights: 16.38GB
- After weights + overhead (1.30GB): 17.68GB — already over budget
- KV cache budget: **negative**

There is no memory fraction you can set to fix this. `VLLM_METAL_MEMORY_FRACTION=0.98` would yield ~0.3GB for KV cache — not useful. The weights themselves are the problem.

**The real constraint:** On 24GB unified memory, BF16 Qwen3-8B is not a viable serving config. You need quantization. This is not a vllm-metal limitation; it's physics.

To get it running: quit everything including Notes, ran `sudo purge`. Notepad-era
vibes — one terminal window, nothing else open. Worked, but barely.

KV budget after weights: 1.08GB → **7,296 tokens max**. A system prompt + one
file read gets you there. Not a usable config; a data point.

**Fix:** 4-bit MLX quantization, 16.38GB → ~4.5GB weights, ~14GB free for KV cache.

**Command:**
```bash
mlx_lm.convert --hf-path ~/models/Qwen--Qwen3-8B \
  --mlx-path ~/models/Qwen3-8B-4bit-mlx -q --q-bits 4
```

Conversion took ~15 seconds. MLX quantization is a weight format rewrite —
it rounds float16 values to int4/int8 and writes them out, no forward passes.
No calibration data, no error minimization. Fast, but naive: AWQ and GPTQ spend
10–30 minutes running sample prompts through the model to minimize quantization
error at each layer. That's the quality you're trading away for the 15-second
turnaround. It's the right call for a first benchmark pass; AWQ comparison
belongs in the "what I'd do on GPU infra" section since calibration-based quants
aren't practically runnable on Metal.

---

## Apple Silicon Quantization: What's Actually Available

This is the finding that most "ran a model locally" posts skip. The quantization
landscape on Apple Silicon is different from CUDA, and being specific about it
is part of what makes this benchmark honest.

### What doesn't work on Metal (and why)

**GPTQ:** Not supported in vllm-metal. GPTQ inference relies on custom CUDA
kernels for the dequantization math. There is no Metal equivalent in vllm-metal
or mlx-lm. Checkpoints in GPTQ format will fail to load.

**W4A8 / W8A8 (activation-quantized schemes):** These are CUDA-specific in
practice. The "A8" part — quantizing activations to int8 at runtime — requires
CUDA integer dot-product instructions (dp4a, IMMA) to yield any speedup. On
Metal/MLX, even if you load a model with an activation-quantized config, the
activations stay in BF16. The "A8" label becomes decorative. llm-compressor can
produce these checkpoints, but running them on vllm-metal gives you W4A16
performance with W4A8 labeling — the comparison is invalid.

**llm-compressor as primary quantization tool:** llm-compressor is vLLM's
official quantization toolkit but is designed for CUDA vLLM. Its output formats
(GPTQ, W4A8, W8A8, FP8) mostly don't map to vllm-metal's MLX backend. The one
exception is AWQ W4A16 — vllm-metal can load HF-format AWQ checkpoints through
mlx-lm's weight repack path.

### What does work on Metal

| Scheme | Tool | What it actually does | vllm-metal support |
|--------|------|----------------------|-------------------|
| BF16 (baseline) | — | Full precision | ✅ (requires ~22GB free) |
| MLX 8-bit | `mlx_lm.convert --q-bits 8` | int8 weights, BF16 activations | ✅ |
| MLX 4-bit | `mlx_lm.convert --q-bits 4` | int4 weights, BF16 activations | ✅ |
| AWQ W4A16 | llm-compressor / AutoAWQ | int4 weights (group-wise), BF16 activations | ✅ (via MLX repack) |
| GGUF Q8_0 / Q4_0 | llama.cpp tooling | int8/int4 per-tensor | ✅ |

**mlx-lm is the right quantization tool for this hardware.** It is Apple
Silicon-native, produces MLX-format checkpoints that vllm-metal loads directly,
and is what the mlx-community checkpoints on HuggingFace use. The quantization
comparison this benchmark runs is MLX 4-bit vs MLX 8-bit vs AWQ W4A16 — same
source weights, same serving engine, different quantization algorithms. That is
a controlled experiment. Mixing in GPTQ or W4A8 checkpoints (even if they could
load) would not be.

### Installing AutoAWQ on Apple Silicon

Two installation failures, in order:

**Failure 1** — build subprocess can't see the venv's torch:
```
ModuleNotFoundError: No module named 'torch'
```
Fix: `pip install autoawq --no-build-isolation`

**Note:** AutoAWQ is officially deprecated as of 2025 — absorbed into llm-compressor.
It still works for our purposes but the deprecation warning appears at import time.

**Failure 2** — AutoAWQ depends on `triton`, which has no Mac distribution:
```
autoawq 0.2.9 depends on triton
triton: no matching distributions available for your environment
ERROR: ResolutionImpossible
```
`triton` is a CUDA-only library. All AutoAWQ versions ≥0.2.7 require it.
`pip install autoawq --no-deps` bypasses the resolver; whether calibration
actually runs without triton depends on whether the CPU path calls any triton
ops at runtime (TBD — worth trying).

If AutoAWQ is unrunnable on Mac, options in order of preference:
1. Produce the AWQ checkpoint on a free Colab T4 (~10 min), pull it back
2. Use a pre-built HF checkpoint (`Qwen/Qwen3-8B-AWQ` or `bartowski/Qwen3-8B-AWQ`)
   and note it in the results

Either way, the AWQ comparison is still valid — the serving path (vllm-metal's
MLX AWQ repack) is the same. The limitation is in the quantization tooling on
Mac, not in inference.

**Failure 3** — calibration dataset download with no internet:
```
FileNotFoundError: Couldn't find 'mit-han-lab/pile-val-backup' on the Hub
```
AutoAWQ by default fetches the Pile validation set as calibration data.
Fix: pass `calib_data=<list of strings>` directly to `model.quantize()`.
Using coding-relevant calibration samples (code snippets, debugging prompts,
system prompts from real agent sessions) is actually more principled here —
it protects the weights that matter for this specific workload rather than
the generic text distribution of the Pile.

### Producing self-quantized checkpoints

All quantization is done from the same BF16 source weights — this is what makes
the comparison controlled. Script: `quantize.py`.

```bash
# MLX naive (seconds each)
python quantize.py --method mlx4   # → ~/models/Qwen3-8B-4bit-mlx
python quantize.py --method mlx8   # → ~/models/Qwen3-8B-8bit-mlx

# AWQ calibrated (60–90 min each, CPU)
python quantize.py --method awq4   # → ~/models/Qwen3-8B-awq-4bit
# AWQ 8-bit: not supported — AutoAWQ's GEMM kernel is 4-bit only.
# At 8-bit, naive MLX rounding error is already small; calibration adds little value.
```

### AWQ 4-bit vs MLX 4-bit: the weight size surprise

AWQ 4-bit loads at **6.18GB**, not 4.61GB like MLX 4-bit — same bit width, 34%
heavier. The startup log explains why:

```
AWQ load: aligned 147 non-quantized floating params to mlx.core.bfloat16
```

AWQ deliberately keeps some layers (typically the first and last, plus
high-sensitivity attention projections) in bfloat16. That's the "activation-aware"
part working as intended — it identified those 147 parameter tensors as too
sensitive to quantize without quality loss and left them at full precision.
The tradeoff: more memory, better quality. The KV budget drops from 12.81GB
(MLX 4-bit) to 10.48GB (AWQ 4-bit), and max cached tokens from 86,880 to 71,088.

This is the comparison the quality benchmark will resolve: does the extra 1.57GB
in weights buy enough quality improvement to justify the smaller context window?

---

## AWQ 4-bit Benchmark Results — Surprise: It's Slow

**Serve command:**
```bash
VLLM_METAL_MEMORY_FRACTION=0.90 vllm serve ~/models/Qwen3-8B-awq-4bit \
  --max-model-len 32768 --enable-prefix-caching
```

Note: Had to drop fraction to 0.90 (from 0.95 used for MLX configs). At 0.95,
AWQ crashed mid-inference with a Metal OOM inside `mx.eval(logits_2d)`. Root
cause: AWQ's 147 bfloat16 non-quantized params + activation spikes push the
peak allocation over the wired_limit=17.8GB ceiling. The 0.90 fraction gives
just enough headroom.

**Numbers:**

| Metric | Value |
|--------|-------|
| Median TTFT | 477ms |
| Throughput | 7.3–7.5 tok/s |
| Prefix cache — unpaired prompt | ~0.82x speedup (slower than no cache) |
| Prefix cache — same prompt pair | 3.94x speedup (one out of four pairs) |

**Why so slow?**

MLX 4-bit at the same quantization level runs ~30 tok/s. AWQ at 7.3–7.5 tok/s
is 4× slower on the same hardware with the same bit width. Likely causes:

1. **Mixed precision dequant penalty:** AWQ's 147 bfloat16 layers don't participate
   in the fast int4 matmul path — the Metal kernel has to handle mixed-precision
   dispatch on every forward pass.
2. **AWQ kernel is a CUDA-optimized format:** The group-wise quantization scheme
   (128-element groups, interleaved zero-points) is designed for CUDA integer
   dot-product. The MLX repack path adapts it to Metal, but the layout isn't
   native to MLX's contiguous-block memory model. The MLX 4-bit format was
   designed for this hardware from the start.
3. **Prefix cache almost non-functional:** The cache hit rate was near zero for
   most prompt pairs (median 0.82x = slightly slower than no cache), suggesting
   the AWQ KV format isn't being recognized for cache reuse as reliably as the
   MLX format.

**Takeaway for Apple Silicon:** AWQ 4-bit is the "better quantization algorithm"
on paper, but MLX 4-bit is the right tool for this hardware. The quality
difference (if any) almost certainly doesn't offset a 4× throughput penalty.
This is a hardware-format match problem, not an accuracy tradeoff.

---

## Qwen3.5-9B on vllm-metal: Setup Steps and Gotchas

Qwen3.5-9B (MLX 4-bit, `mlx-community/Qwen3.5-9B-MLX-4bit`) was added as a newer-architecture comparison point. Several non-obvious steps required:

**Download:** ModelScope mirrors `mlx-community` but its CDN drops connections mid-redirect for both large and small files. Fixed in `download_model.py` with User-Agent header, retry-on-disconnect, and skip-if-exists logic. After two partial attempts the weights landed intact (~5.95GB across two shards).

**Corrupt shard detection:** The first run crashed with:
```
RuntimeError: [load_safetensors] Tensor 'language_model.model.embed_tokens.biases'
invalid data offsets (5016729984, 5048514944) exceeding the size of the file.
```
Root cause: `remote_size()` HEAD requests were also dropping, so skip-if-exists fired on incomplete shards. Fix: delete and re-download any shard that fails to load.

**Missing chat template:** First inference attempt returned HTTP 400:
```
"As of transformers v4.44, default chat template is no longer allowed"
```
The mlx-community tokenizer_config.json had no `chat_template` field — it was omitted from their upload. Fix: patch it from the local Qwen3-8B tokenizer config (same tokenizer family):
```python
dst['chat_template'] = src['chat_template']  # from Qwen--Qwen3-8B/tokenizer_config.json
```

**Serve command:**
```bash
VLLM_METAL_MEMORY_FRACTION=0.95 vllm serve ~/models/mlx-community--Qwen3.5-9B-MLX-4bit \
  --max-model-len 32768 --enable-prefix-caching
```

---

## Engine Comparison: vllm-metal vs Ollama

### Pulling Ollama models

```bash
ollama pull qwen3:8b       # Q4_K_M, ~5GB
ollama pull qwen3.5:9b     # Q4_K_M, ~6.6GB, 256K context
```

Ollama default tag = Q4_K_M (4-bit, GGUF/llama.cpp). Matched bit-width to vllm MLX 4-bit for a fair comparison. The algorithm differs (MLX naive rounding vs K-quant grouping) but the memory footprint is equivalent.

**Thinking mode:** Qwen3 and Qwen3.5 have thinking mode on by default. Ollama's OpenAI-compatible endpoint ignores `think: false` in the request body — the reasoning tokens land in a `"reasoning"` field (not `"reasoning_content"`), consuming the entire token budget before any content tokens appear. Fix: use Ollama's native `/api/chat` endpoint with `"think": false` in the payload — this actually disables thinking.

---

## Full Benchmark Results

All runs: vllm-metal v0.27.1, M4 24GB, prefix caching enabled.  
TTFT = median time-to-first-token (warm runs, excluding cold start).  
Throughput = median tok/s across coding questions.  
Cache = median speedup of second identical call vs first (>1 = cache helping).

| Model | Engine | Quant | TTFT (ms) | Throughput (tok/s) | Cache speedup | HumanEval pass@1 | HumanEval+ pass@1 |
|-------|--------|-------|-----------|-------------------|---------------|------------------|-------------------|
| Qwen3-8B | vllm | BF16 | 349 | 10.8 | 1.43x | — | — |
| Qwen3-8B | vllm | MLX 8-bit | 390 | 17.8 | 1.12x | — | — |
| Qwen3-8B | vllm | MLX 4-bit | 348 | 25.0 | 0.80x | — | — |
| Qwen3-8B | vllm | AWQ 4-bit | 463 | 7.4 | 0.79x | — | — |
| Qwen3-8B | Ollama | Q4_K_M | 95 | 21.3 | 0.67x | — | — |
| Qwen3.5-9B | Ollama | Q4_K_M | 158 | 16.5 | 1.31x | **90.9%** | **87.2%** |
| Qwen3.5-9B | vllm | MLX 4-bit | 554 | 29.6 | 1.53x | **86.6%** | **82.3%** |

### Key findings

**Throughput: vllm wins.** For the same Qwen3.5-9B model at 4-bit, vllm delivers 29.6 tok/s vs Ollama's 16.5 tok/s — nearly 2×. The MLX backend is doing real hardware-optimised matmuls; llama.cpp's Metal backend is not as tuned. For a coding agent that generates long completions, this matters most.

**TTFT: Ollama wins.** Ollama's TTFT of 95ms (Qwen3-8B) vs vllm's 348ms reflects a difference in request path overhead — Ollama's llama.cpp backend starts streaming faster. For interactive use where you want the cursor to start moving quickly, Ollama has a real latency edge. vllm's TTFT is dominated by KV allocation and chunked prefill scheduling.

**Prefix cache: vllm wins, but the signal is noisy.** vllm's 1.43x speedup on BF16 is the clearest cache hit signal. At 4-bit MLX, the speedup drops to 0.80x (slower) — possibly because the 4-bit KV blocks are small enough that the scheduling overhead outweighs the cache benefit at this context length. Qwen3.5-9B on vllm shows 1.53x, the strongest result in the set. Ollama's cache is inconsistent: 0.67x on Qwen3-8B (harming latency), 1.31x on Qwen3.5-9B.

**AWQ 4-bit: worst of both worlds on Metal.** 7.4 tok/s at 4-bit — 3.4× slower than MLX 4-bit. The AWQ format is CUDA-optimised; the MLX repack path doesn't recover the performance. Not a viable config for Apple Silicon.

**Quality: Ollama Q4_K_M beats vllm MLX 4-bit.** 90.9% vs 86.6% on HumanEval base; 87.2% vs 82.3% on HumanEval+. Both are strong results, but Q4_K_M's k-quant grouping algorithm produces measurably better weights than MLX's naive rounding at the same 4-bit width. The quality tradeoff is real and quantifiable.

**Qwen3.5-9B quality advantage:** LiveCodeBench score of 65.6 vs 22.8 for Qwen3-8B — a genuine 3× improvement. The extra inference cost (29.6 vs 25.0 tok/s, higher TTFT) is worth it for a coding workload. This is the model to run on this hardware.

---

## Quality Evaluation: HumanEval via evalplus

**Tool:** [evalplus](https://github.com/evalplus/evalplus) — the standard community harness for HumanEval and MBPP, with 80× more test cases (HumanEval+) that catch silent failures most models pass on the basic tests.

**Why not lm-eval?** lm-eval-harness runs HumanEval with execution, but evalplus is the reference implementation for the HumanEval+ extended test suite and is directly supported by both vllm and Ollama via OpenAI-compatible endpoints.

**Setup gotchas on this network:**
- `huggingface.co` is blocked; `datasets-server.huggingface.co` returns 422
- GitHub Releases CDN (`objects.githubusercontent.com`) returns 502 — `HumanEvalPlus.jsonl.gz` must be downloaded externally and placed at `~/Library/Caches/evalplus/HumanEvalPlus-v0.1.10.jsonl`
- evalplus's `reliability_guard` calls `resource.setrlimit(RLIMIT_AS, ...)` which raises `ValueError` on macOS — patched in `evalplus/eval/utils.py` to swallow the error (memory sandbox is advisory on macOS anyway)

**Commands:**
```bash
# Generate completions (skips if already cached)
python -m evalplus.evaluate \
  --model "/path/to/model" --dataset humaneval \
  --backend openai --base-url http://localhost:8000/v1 --greedy

# Evaluate pre-generated completions (no model needed)
python -m evalplus.evaluate \
  --dataset humaneval --samples <path-to-.jsonl>
```

**Results — Qwen3.5-9B, 164 problems, greedy decoding:**

| Engine | Quant | HumanEval pass@1 | HumanEval+ pass@1 | Δ (base→plus) |
|--------|-------|------------------|-------------------|---------------|
| Ollama | Q4_K_M | **90.9%** | **87.2%** | −3.7pp |
| vllm | MLX 4-bit | **86.6%** | **82.3%** | −4.3pp |

**Key finding:** Q4_K_M beats MLX 4-bit by 4.3pp on base tests and 4.9pp on HumanEval+. Both use 4-bit quantization, but the algorithms differ: Q4_K_M applies k-quant grouping (minimizing error across groups of weights), while MLX naive rounding treats each weight independently. The quality difference is real and consistent across both test sets.

The small HumanEval → HumanEval+ drop for both engines (3.7–4.3pp vs the typical 10–15pp) suggests solutions are genuinely correct rather than test-case pattern-matching.

---

## Version Log

| Component | Version |
|-----------|---------|
| vllm-metal | 0.27.1 |
| mlx-lm | 0.31.3 |
| Ollama | 0.33.3 |
| Models | Qwen/Qwen3-8B (BF16 baseline + self-quantized); mlx-community/Qwen3.5-9B-MLX-4bit; qwen3.5:9b (Ollama Q4_K_M) |
| macOS | Darwin 25.6.0 |

---

## What I'd Do Differently on Real GPU Infra

The 24GB unified-memory constraint forces choices that production infra doesn't face. Naming them explicitly makes the benchmark more useful as a production-engineering reference.

### Quantization: W4A8 and FP8 become viable

On Metal, "4-bit" means W4A16 — weights in int4, activations in BF16. This is the only 4-bit mode that works. On an H100, two additional options matter:

**W4A8 (via llm-compressor):** Activations are also quantized to int8 at runtime. The H100's dp4a integer dot-product units make this a real 2× bandwidth win over W4A16 — not decorative. On Metal, the same config degrades to W4A16 performance because Apple Silicon lacks the equivalent instruction path.

**FP8 (W8A8 float8):** H100/H200 SXM has native FP8 GEMM. A 70B model that would need aggressive 4-bit quantization on a single A100 can run at FP8 on an H100 with better quality and similar throughput. Not available on Metal.

**Practical implication:** The quality vs. throughput tradeoff curve looks different on CUDA. The 4.3pp quality gap between MLX 4-bit and Q4_K_M (86.6% vs 90.9% HumanEval) would largely disappear if both ran at FP8 — quantization error at 8-bit is small enough that algorithm choice stops mattering.

### Prefix caching: economics change with real concurrency

The benchmark measured cache speedup at concurrency 1 — a single user. Under that condition, cache hit rates are modest (1.31–1.53× speedup) because the prefix pool is only ever seeded by your own prior requests.

At real production concurrency (10–100 coding-agent users sharing a server), the math changes:
- Every user sends the same long system prompt + tool schema at the start of each session
- After the first few requests, that prefix is cached — every subsequent user gets it for free
- Cache hit rates compound: what was a 1.5× speedup per-user becomes a 5–10× effective throughput multiplier for the fleet

This is the primary argument for vllm over Ollama in production: vllm's PagedAttention prefix cache is designed for shared multi-user serving. Ollama is single-process and single-user by design — the cache only helps you, not the next user.

### TTFT inversion under load

The benchmark shows Ollama with lower TTFT at idle (95ms vs 554ms for Qwen3.5-9B). This advantage disappears or inverts under load:

- vllm's continuous batching means new requests join an in-flight batch — TTFT stays bounded even as concurrency grows
- Ollama serializes requests (one at a time); the second user's TTFT is the first user's full generation time

For a team of 5+ engineers using a shared coding agent, vllm's higher idle TTFT is irrelevant — everyone's median TTFT under real load is lower with vllm.

### What stays the same

The key finding — that quantization algorithm choice (naive rounding vs k-quant vs calibration-based) produces measurable quality differences at 4-bit — holds on any hardware. The 4.3pp HumanEval gap between MLX 4-bit and Q4_K_M reflects a real difference in how the two algorithms handle weight precision. On GPU infra, you'd express the same tradeoff as AWQ vs GPTQ vs naive int4, and you'd see similar gaps. The hardware changes; the quantization economics don't.
