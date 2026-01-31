# Architecture

## Project Structure

```
MegaQwen/
├── csrc/
│   ├── megakernel/
│   │   ├── fused_decode_ldg.cu    # Main decode megakernel
│   │   ├── fused_prefill.cu       # Prefill megakernel
│   │   ├── fused_prefill_megakernel.cu  # Alternative prefill implementation
│   │   ├── config.cuh             # Kernel configuration constants
│   │   ├── megakernel_decode.py   # Python interface for decode
│   │   └── benchmark_prefill.py   # Prefill benchmarking
│   └── kernels/
│       ├── rms_norm.cu            # Standalone RMSNorm kernel
│       ├── rope.cu                # Rotary Position Embeddings
│       ├── attention_decode.cu    # Attention decode with online softmax
│       └── silu_mul.cu            # Fused SiLU activation * up projection
├── chat.py                    # Interactive chat interface
├── benchmark_suite.py         # Comprehensive benchmarks
├── demo_e2e.py               # Torch/Triton/CUDA comparison
├── generate.py               # Token generation utilities
├── qwen3-0.6b.py            # Full model implementation
├── SPEC.md                   # Technical specification
└── verify_correctness.py     # Correctness validation
```

## Model Configuration (Qwen3-0.6B)

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Intermediate size | 3072 |
| Attention heads (Q) | 16 |
| KV heads | 8 |
| Head dimension | 128 |
| Layers | 28 |
| Vocab size | 151936 |
| Max sequence length | 512 |

## Kernel Configuration

### Decode Megakernel (`fused_decode_ldg.cu`)

- **Grid**: 82 thread blocks
- **Block**: 256 threads (8 warps)
- **Launch**: Cooperative kernel for grid-wide sync

Key optimizations:
- `__ldg()` cached reads for weights via texture cache
- Vectorized vec4 memory loads
- Online softmax for memory-efficient attention
- Warp-level reductions for RMSNorm and attention

### LM Head

- **Grid**: 1184 thread blocks
- **Block**: 256 threads
- Two-phase argmax reduction for vocab size 151936

## Data Flow (Single Decode Step)

```
Input Token ID
      │
      ▼
┌─────────────┐
│  Embedding  │  (lookup from 151936 x 1024)
└─────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│         28x Transformer Block        │
│  ┌─────────────────────────────────┐ │
│  │ RMSNorm → QKV Proj → QK Norm   │ │
│  │    → RoPE → KV Cache Update    │ │
│  │    → Attention → O Proj        │ │
│  │    → Residual → RMSNorm        │ │
│  │    → Gate/Up → SiLU → Down     │ │
│  │    → Residual                  │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────┐
│ Final Norm  │
└─────────────┘
      │
      ▼
┌─────────────┐
│   LM Head   │  (1024 → 151936, argmax)
└─────────────┘
      │
      ▼
Output Token ID
```

## Memory Layout

### KV Cache
- Shape: `[num_layers, num_kv_heads, max_seq_len, head_dim]`
- Type: bfloat16
- Size: `28 * 8 * 512 * 128 * 2 bytes = 28 MB`

### Intermediate Buffers (Global Memory)
- `g_activations`: `[1024]` float32
- `g_residual`: `[1024]` float32
- `g_q`: `[2048]` float32 (16 heads * 128)
- `g_k`: `[1024]` float32 (8 heads * 128)
- `g_v`: `[1024]` float32
- `g_attn_out`: `[2048]` float32
- `g_mlp_intermediate`: `[3072]` float32

## Synchronization

The megakernel uses CUDA cooperative groups for grid-wide synchronization:

```cpp
cg::grid_group grid = cg::this_grid();
// ... computation ...
grid.sync();  // All blocks synchronize
```

This requires:
- `cudaLaunchCooperativeKernel()` for launch
- Kernel occupancy check to ensure all blocks fit on GPU simultaneously
