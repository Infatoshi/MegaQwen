# MegaQwen Experiment Results

**Model**: Qwen3-0.6B (28 layers, 16 heads, 1024 hidden dim)
**Hardware**: NVIDIA RTX 3090 (24GB, 420W TDP)
**Kernel**: 82 blocks x 256 threads, ~225 grid.sync() per decode step

---

## 1. Framework Comparison

### Full Benchmark (Long Prompt, 200 tokens)

**Prompt**: "Write a detailed essay about lobsters, covering their biology, habitat..." (~22 input tokens)

| Framework | tok/s | Avg Power (W) | Peak Power (W) | tok/J | Speedup vs HF |
|-----------|-------|---------------|----------------|-------|---------------|
| **TensorRT-LLM** | **355** | 290 | 290 | **1.22** | **6.01x** |
| Megakernel | 158 | 205 | 233 | 0.77 | 2.68x |
| vLLM | 107 | 196 | 206 | 0.55 | 1.82x |
| SGLang | 107 | 210 | 210 | 0.51 | 1.81x |
| ExLlamaV2 | 98 | 197 | 207 | 0.50 | 1.66x |
| HuggingFace | 59 | 186 | 192 | 0.32 | 1.0x |
| llama.cpp | 50 | 195 | 201 | 0.26 | 0.85x |

### Short Prompt Benchmark (100 tokens)

**Prompt**: "Hello" (1 input token) - minimal KV cache overhead

| Framework | tok/s | Speedup |
|-----------|-------|---------|
| **Megakernel** | **530** | **3.91x** |
| HuggingFace | 136 | 1.0x |

**Note**: Decode throughput decreases with longer context due to attention reading more KV cache entries.

### Key Findings

- **TensorRT-LLM is 6x faster** and **3.8x more energy efficient** than HuggingFace
- **TensorRT-LLM is 2.24x faster** than Megakernel (but requires engine compilation)
- **Megakernel is 2.68x faster** than HuggingFace with no compilation step
- **Megakernel is 1.47x faster** than vLLM
- **llama.cpp (GGUF F16)** is slower than HuggingFace on GPU - better suited for CPU inference
- Higher peak power (233W) but significantly faster completion = better energy efficiency

---

## 2. Quality Metrics

| Framework | KL Divergence | Argmax Match | Notes |
|-----------|---------------|--------------|-------|
| HuggingFace | 0.0 (ref) | 100% | Reference implementation |
| **Megakernel** | **0.000582** | varies | Near-identical distributions |
| vLLM | - | 100% | Logits not exposed |
| llama.cpp | - | - | Token IDs not exposed |

**KL Divergence Analysis**:
- Megakernel KL = 0.000582 indicates **near-identical probability distributions**
- Small difference due to bf16 vs fp32 accumulation in matrix operations
- Argmax can differ on close calls despite near-identical distributions

---

## 3. Cooperative Kernel vs CUDA Graph

### Synchronization Overhead (Empty Kernels)

| Approach | Time | Per-Op Cost |
|----------|------|-------------|
| Cooperative + 225 grid.sync() | 167.3 us | 0.73 us/sync |
| CUDA graph (225 kernels) | 186.9 us | 0.83 us/kernel |
| 225 regular kernel launches | 347.5 us | 1.54 us/launch |

### Conclusion

**Cooperative kernel wins by 19.7 us** over CUDA graph for pure sync overhead.

But the real benefit of the megakernel isn't sync savings - it's **memory bandwidth savings**:
- Avoids ~340 intermediate global memory writes/reads
- Saves ~2.7 MB memory traffic per token
- Estimated savings: 1000+ us per decode step

Splitting at grid.sync() points would lose these memory benefits.

---

## 4. Optimization Experiments

### Redundant RMSNorm (Implemented)

**Approach**: All 82 blocks compute RMSNorm redundantly instead of only block 0.
**Syncs eliminated**: 56 (2 per layer x 28 layers)

| Position | Original | Optimized | Speedup |
|----------|----------|-----------|---------|
| 1 | 5.655ms | 3.982ms | 1.42x |
| 10 | 5.708ms | 4.006ms | 1.42x |
| 50 | 5.819ms | 4.107ms | 1.42x |
| 100 | 5.936ms | 4.227ms | 1.40x |
| 200 | 6.202ms | 6.883ms | 0.90x |

**Result**: +26.3% throughput (170 -> 215 tok/s) at short sequences. Degrades at long sequences due to L2 cache pressure from KV cache.

**Trade-off**: Best for interactive use (short contexts). Not recommended for long-context workloads.

### Head-Based Work Distribution (NOT VIABLE)

**Approach**: Assign 5 blocks per Q head so QKV + attention can proceed without grid.sync().
**Syncs eliminated**: 28 (1 per layer)

| Position | Original | Optimized | Speedup |
|----------|----------|-----------|---------|
| 1 | 4.023ms | 6.862ms | 0.59x |
| 10 | 4.051ms | 6.892ms | 0.59x |
| 50 | 4.151ms | 6.990ms | 0.59x |

**Result**: -33% throughput (213 -> 142 tok/s). **Not viable.**

**Why it failed**: QKV is memory-bound. Reducing from 82 to 16 working blocks loses parallelism. The memory bandwidth loss far exceeds sync savings.

**Lesson**: Don't sacrifice block utilization to eliminate syncs.

### Fused Phases (Not Implemented)

**Approach**: Fuse adjacent phases (QKV + QK norm + RoPE, O proj + residual + post-attn RMSNorm).
**Expected syncs eliminated**: ~56

Status: Not yet implemented.

---

## Summary Table

| Metric | TensorRT-LLM | Megakernel | vs HuggingFace |
|--------|--------------|------------|----------------|
| Decode tok/s | 355 | 158 | 6.01x / 2.68x |
| Energy (tok/J) | 1.22 | 0.77 | 3.81x / 2.41x |
| Compilation | Required | None | - |
| KL Divergence | - | 0.000582 | near-identical |

---

## Running Benchmarks

```bash
# Full benchmark (throughput + power + quality)
python experiments/framework_bench/full_benchmark.py

# Framework throughput only
python experiments/framework_bench/benchmark_suite.py

# Power consumption only
python experiments/framework_bench/power_benchmark.py

# Quality metrics (KL divergence, argmax match)
python experiments/framework_bench/quality_metrics.py

# Sync overhead analysis
python experiments/sync_overhead.py

# Optimization experiments
python experiments/optimizations/redundant_rmsnorm/benchmark.py
python experiments/optimizations/compare_all.py
```

---

## Model Conversion

```bash
# Convert to GGUF for llama.cpp
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
python /tmp/llama.cpp/convert_hf_to_gguf.py \
    ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/  \
    --outfile /tmp/qwen3_gguf/qwen3-0.6b-f16.gguf \
    --outtype f16
```

---

## TODO

- [x] Expose logits for KL divergence measurement
- [x] Add llama.cpp benchmark (GGUF F16)
- [x] Add vLLM benchmark
- [x] Add ExLlamaV2 benchmark (required flash-attn 2.8.3)
- [x] Add SGLang benchmark (server mode via OpenAI API)
- [ ] Implement fused phases optimization
- [x] Add TensorRT-LLM benchmark (355 tok/s, 6x faster than HF)
