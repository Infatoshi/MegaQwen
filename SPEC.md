# Qwen3-0.6B Decode Megakernel Specification

## Goal
Single fused CUDA kernel for one transformer block decode step (batch=1, seq_len=1).
Target: RTX 3090 (SM86, Ampere, no TMA).

## Model Dimensions
```
hidden_size:         1024
intermediate_size:   3072
num_attention_heads: 16
num_kv_heads:        8
head_dim:            128
q_proj:              (2048, 1024)  # 16 heads * 128
k_proj:              (1024, 1024)  # 8 heads * 128
v_proj:              (1024, 1024)
o_proj:              (1024, 2048)
gate_proj:           (3072, 1024)
up_proj:             (3072, 1024)
down_proj:           (1024, 3072)
```

## Tunable Parameters
```cpp
// Weight tiling
TILE_ROWS           // Output elements per tile: 16, 32, 64
TILE_COLS           // Should match hidden_size or be a divisor

// Pipelining
NUM_PIPELINE_STAGES // Weight buffer stages: 2, 3, 4

// Thread organization
BLOCK_SIZE          // Threads per block: 128, 256, 512
NUM_CONSUMER_WARPS  // Warps doing compute
NUM_LOADER_WARPS    // Warps doing async loads (0 = all warps load)

// Attention
KV_BLOCK_SIZE       // KV cache elements to process per iteration
```

## Initial Configuration (to be tuned)
```cpp
TILE_ROWS           = 16
TILE_COLS           = 1024
NUM_PIPELINE_STAGES = 3
BLOCK_SIZE          = 256  // 8 warps
NUM_CONSUMER_WARPS  = 6
NUM_LOADER_WARPS    = 2
KV_BLOCK_SIZE       = 64
```

## Memory Budget (RTX 3090)
- Shared memory per SM: 100KB configurable
- Registers per SM: 65536
- Registers per thread (256 threads): 256 max

### Shared Memory Layout
```
Weight tile 0:     TILE_ROWS * TILE_COLS * 2 bytes
Weight tile 1:     TILE_ROWS * TILE_COLS * 2 bytes
Weight tile 2:     TILE_ROWS * TILE_COLS * 2 bytes
Activation buffer: hidden_size * 4 bytes (float32 for compute)
Reduction scratch: NUM_WARPS * 32 * 4 bytes
RMS norm scratch:  NUM_WARPS * 4 bytes
---
Example: 3 * 16 * 1024 * 2 + 1024 * 4 + 256 * 4 + 32 = ~102KB
Need smaller tiles or fewer stages.

Revised: TILE_ROWS=16, TILE_COLS=512, 3 stages
= 3 * 16 * 512 * 2 + 4K + 1K = ~54KB (fits)
```

## Data Flow

### Phase 1: RMSNorm + QKV Projection
```
Input:  hidden_states[1024] from global memory (once)
Output: Q[2048], K[1024], V[1024] in registers/smem

1. Load hidden_states into shared memory
2. Compute RMSNorm:
   - Each warp handles 1024/NUM_WARPS elements
   - Warp-reduce sum of squares
   - Block-reduce to get total
   - Broadcast rsqrt, apply weight
3. Stream QKV weights in tiles:
   - Async load tile N+1 while computing tile N
   - Matvec: broadcast activation, element-wise mul, row-reduce
   - Accumulate partial outputs
```

### Phase 2: QK Norm + RoPE + Cache Update
```
Input:  Q[2048], K[1024], V[1024]
Output: Q_rope[2048], K written to cache, V written to cache

1. Reshape Q to (16, 128), K to (8, 128)
2. Per-head RMSNorm on Q and K
3. Apply RoPE rotation (pair elements)
4. Write K, V to global KV cache at position
```

### Phase 3: Attention Decode
```
Input:  Q[16, 128], K_cache[8, cache_len, 128], V_cache[8, cache_len, 128]
Output: attn_out[2048]

1. For each query head (maps to KV head via GQA):
   - Stream K_cache blocks into shared memory
   - Compute Q @ K^T scores (dot products)
   - Track running max for online softmax
2. Second pass (or fused):
   - Compute exp(score - max), accumulate sum
   - Stream V_cache, compute weighted sum
3. Output attention result
```

### Phase 4: O Projection + Residual
```
Input:  attn_out[2048], original hidden_states[1024]
Output: hidden_states[1024] (updated)

1. Stream o_proj weights
2. Matvec: (1024, 2048) @ (2048,)
3. Add residual (need to keep original hidden_states)
```

### Phase 5: RMSNorm + Gate/Up + SiLU
```
Input:  hidden_states[1024]
Output: mlp_hidden[3072]

1. RMSNorm
2. Interleaved gate/up projection:
   - Can load gate row and up row alternately
   - Or do gate fully then up
3. SiLU(gate) * up element-wise
```

### Phase 6: Down Projection + Residual
```
Input:  mlp_hidden[3072], hidden_states[1024]
Output: hidden_states[1024] (final)

1. Stream down_proj weights
2. Matvec: (1024, 3072) @ (3072,)
3. Add residual
```

## Implementation Order
1. [ ] Matvec kernel with pipelining (standalone test)
2. [ ] RMSNorm fused with matvec
3. [ ] QKV projection fused
4. [ ] QK norm + RoPE
5. [ ] Attention decode with online softmax
6. [ ] O projection + residual
7. [ ] MLP block (norm + gate/up + silu + down)
8. [ ] Full block integration
9. [ ] Multi-layer extension

## Correctness Validation
- Compare against naive CUDA kernels (already validated)
- Check at each phase before fusing more
- Tolerance: atol=1e-2, rtol=1e-2 (bf16 precision)

## Performance Metrics
- Tokens/second
- Memory bandwidth utilization
- Kernel execution time breakdown
