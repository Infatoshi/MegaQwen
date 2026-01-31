#pragma once

#include "config.cuh"

// =============================================================================
// RoPE (Rotary Position Embeddings) Implementation
// =============================================================================
//
// Applies rotary embeddings to Q and K vectors.
// Uses the "split-half" format:
//   out[..., :half] = x[..., :half] * cos - x[..., half:] * sin
//   out[..., half:] = x[..., half:] * cos + x[..., :half] * sin
//
// All computation in float32, operates on shared memory vectors.
// =============================================================================

// -----------------------------------------------------------------------------
// Apply RoPE to a single head vector in shared memory
// head_vec: [head_dim] float32 in shared memory
// cos, sin: [head_dim] precomputed for this position
// -----------------------------------------------------------------------------
__device__ __forceinline__ void rope_single_head(
    float* __restrict__ head_vec,        // [head_dim] in shared memory
    const __nv_bfloat16* __restrict__ cos,  // [head_dim] for position
    const __nv_bfloat16* __restrict__ sin,  // [head_dim] for position
    int head_dim
) {
    int half_dim = head_dim / 2;

    // Each thread processes pairs of elements
    for (int i = threadIdx.x; i < half_dim; i += BLOCK_SIZE) {
        float x0 = head_vec[i];
        float x1 = head_vec[i + half_dim];

        float c = __bfloat162float(cos[i]);
        float s = __bfloat162float(sin[i]);

        // Rotation
        head_vec[i] = x0 * c - x1 * s;
        head_vec[i + half_dim] = x1 * c + x0 * s;
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Apply RoPE to multiple heads in parallel
// Q: [num_q_heads, head_dim] in shared memory
// K: [num_kv_heads, head_dim] in shared memory
// -----------------------------------------------------------------------------
__device__ __forceinline__ void rope_qk(
    float* __restrict__ smem_q,          // [num_q_heads * head_dim]
    float* __restrict__ smem_k,          // [num_kv_heads * head_dim]
    const __nv_bfloat16* __restrict__ cos,  // [head_dim] for this position
    const __nv_bfloat16* __restrict__ sin,  // [head_dim] for this position
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int half_dim = head_dim / 2;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Load cos/sin into registers (assuming head_dim=128, half=64)
    // Each thread in warp loads 2 values (64/32=2)
    float cos_reg[2], sin_reg[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = lane_id * 2 + i;
        if (idx < half_dim) {
            cos_reg[i] = __bfloat162float(cos[idx]);
            sin_reg[i] = __bfloat162float(sin[idx]);
        }
    }

    // Process Q heads
    int total_q_heads = num_q_heads;
    int q_heads_per_warp = (total_q_heads + NUM_WARPS - 1) / NUM_WARPS;
    int q_head_start = warp_id * q_heads_per_warp;
    int q_head_end = min(q_head_start + q_heads_per_warp, total_q_heads);

    for (int h = q_head_start; h < q_head_end; h++) {
        float* head_vec = smem_q + h * head_dim;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = lane_id * 2 + i;
            if (idx < half_dim) {
                float x0 = head_vec[idx];
                float x1 = head_vec[idx + half_dim];

                head_vec[idx] = x0 * cos_reg[i] - x1 * sin_reg[i];
                head_vec[idx + half_dim] = x1 * cos_reg[i] + x0 * sin_reg[i];
            }
        }
    }

    __syncthreads();

    // Process K heads
    int total_kv_heads = num_kv_heads;
    int kv_heads_per_warp = (total_kv_heads + NUM_WARPS - 1) / NUM_WARPS;
    int kv_head_start = warp_id * kv_heads_per_warp;
    int kv_head_end = min(kv_head_start + kv_heads_per_warp, total_kv_heads);

    for (int h = kv_head_start; h < kv_head_end; h++) {
        float* head_vec = smem_k + h * head_dim;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = lane_id * 2 + i;
            if (idx < half_dim) {
                float x0 = head_vec[idx];
                float x1 = head_vec[idx + half_dim];

                head_vec[idx] = x0 * cos_reg[i] - x1 * sin_reg[i];
                head_vec[idx + half_dim] = x1 * cos_reg[i] + x0 * sin_reg[i];
            }
        }
    }

    __syncthreads();
}

// -----------------------------------------------------------------------------
// Apply RoPE with cos/sin loaded from position in sequence
// Handles the lookup into the precomputed cos/sin tables
// -----------------------------------------------------------------------------
__device__ __forceinline__ void rope_qk_at_position(
    float* __restrict__ smem_q,              // [num_q_heads * head_dim]
    float* __restrict__ smem_k,              // [num_kv_heads * head_dim]
    const __nv_bfloat16* __restrict__ cos_table,  // [max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ sin_table,  // [max_seq_len, head_dim]
    int position,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    // Get cos/sin for this position
    const __nv_bfloat16* cos_pos = cos_table + position * head_dim;
    const __nv_bfloat16* sin_pos = sin_table + position * head_dim;

    rope_qk(smem_q, smem_k, cos_pos, sin_pos, num_q_heads, num_kv_heads, head_dim);
}
