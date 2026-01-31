#pragma once

#include "config.cuh"

// =============================================================================
// RMSNorm Implementation
// =============================================================================
//
// out = (x / sqrt(mean(x^2) + eps)) * weight
//
// All computation in float32, output can stay in float32 for fusion
// or be converted to bf16 for storage.
// =============================================================================

// -----------------------------------------------------------------------------
// In-place RMSNorm on shared memory activations
// Result stays in smem_activations (float32)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void rmsnorm_inplace(
    float* __restrict__ smem_activations,  // [size] in shared memory
    const __nv_bfloat16* __restrict__ weight,  // [size] in global memory
    float* __restrict__ scratch,            // [NUM_WARPS] for reduction
    int size,
    float eps
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Step 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE) {
        float val = smem_activations[i];
        sum_sq += val * val;
    }

    // Block-level reduction
    sum_sq = warp_reduce_sum(sum_sq);
    if (lane_id == 0) {
        scratch[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < NUM_WARPS) ? scratch[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
        if (lane_id == 0) {
            scratch[0] = sum_sq;
        }
    }
    __syncthreads();

    // Step 2: Compute rsqrt
    float variance = scratch[0] / float(size);
    float rstd = rsqrtf(variance + eps);

    // Step 3: Normalize and scale
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE) {
        float val = smem_activations[i];
        float w = __bfloat162float(weight[i]);
        smem_activations[i] = val * rstd * w;
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// RMSNorm from global input to shared memory output
// -----------------------------------------------------------------------------
__device__ __forceinline__ void rmsnorm_to_smem(
    const __nv_bfloat16* __restrict__ input,   // [size] in global memory
    const __nv_bfloat16* __restrict__ weight,  // [size] in global memory
    float* __restrict__ smem_output,            // [size] in shared memory
    float* __restrict__ scratch,                // [NUM_WARPS] for reduction
    int size,
    float eps
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Step 1: Load input and compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE) {
        float val = __bfloat162float(input[i]);
        smem_output[i] = val;  // Store for later
        sum_sq += val * val;
    }
    __syncthreads();

    // Block-level reduction
    sum_sq = warp_reduce_sum(sum_sq);
    if (lane_id == 0) {
        scratch[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < NUM_WARPS) ? scratch[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
        if (lane_id == 0) {
            scratch[0] = sum_sq;
        }
    }
    __syncthreads();

    // Step 2: Compute rsqrt
    float variance = scratch[0] / float(size);
    float rstd = rsqrtf(variance + eps);

    // Step 3: Normalize and scale
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE) {
        float val = smem_output[i];
        float w = __bfloat162float(weight[i]);
        smem_output[i] = val * rstd * w;
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Per-head RMSNorm for Q/K normalization
// Processes multiple heads in parallel
// Input: [num_heads, head_dim] flattened
// Weight: [head_dim] shared across heads
// -----------------------------------------------------------------------------
__device__ __forceinline__ void rmsnorm_per_head(
    float* __restrict__ smem_vectors,      // [num_heads * head_dim] in shared memory
    const __nv_bfloat16* __restrict__ weight,  // [head_dim] in global memory
    float* __restrict__ scratch,            // [NUM_WARPS] for reduction
    int num_heads,
    int head_dim,
    float eps
) {
    // Each warp can handle one or more heads depending on configuration
    // For simplicity, process heads sequentially with all threads

    for (int h = 0; h < num_heads; h++) {
        float* head_vec = smem_vectors + h * head_dim;

        // Compute sum of squares for this head
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += BLOCK_SIZE) {
            float val = head_vec[i];
            sum_sq += val * val;
        }

        // Block reduction
        sum_sq = block_reduce_sum(sum_sq, scratch);

        // Compute rsqrt and apply
        float variance = sum_sq / float(head_dim);
        float rstd = rsqrtf(variance + eps);

        for (int i = threadIdx.x; i < head_dim; i += BLOCK_SIZE) {
            float val = head_vec[i];
            float w = __bfloat162float(weight[i]);
            head_vec[i] = val * rstd * w;
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// Optimized per-head RMSNorm - process all heads in parallel
// Each warp handles subset of heads
// -----------------------------------------------------------------------------
__device__ __forceinline__ void rmsnorm_per_head_parallel(
    float* __restrict__ smem_vectors,      // [num_heads * head_dim] in shared memory
    const __nv_bfloat16* __restrict__ weight,  // [head_dim] in global memory
    float* __restrict__ scratch,            // [num_heads] for storing rstd values
    int num_heads,
    int head_dim,
    float eps
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Load weight into registers (shared across all heads)
    // Assuming head_dim=128, WARP_SIZE=32, each thread loads 4 elements
    float w_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = lane_id * 4 + i;
        if (idx < head_dim) {
            w_reg[i] = __bfloat162float(weight[idx]);
        }
    }

    // Each warp processes multiple heads
    int heads_per_warp = (num_heads + NUM_WARPS - 1) / NUM_WARPS;
    int head_start = warp_id * heads_per_warp;
    int head_end = min(head_start + heads_per_warp, num_heads);

    for (int h = head_start; h < head_end; h++) {
        float* head_vec = smem_vectors + h * head_dim;

        // Load head values and compute sum of squares
        float sum_sq = 0.0f;
        float vals[4];

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = lane_id * 4 + i;
            if (idx < head_dim) {
                vals[i] = head_vec[idx];
                sum_sq += vals[i] * vals[i];
            }
        }

        // Warp reduction
        sum_sq = warp_reduce_sum(sum_sq);

        // Compute rsqrt (lane 0 broadcasts)
        float rstd;
        if (lane_id == 0) {
            float variance = sum_sq / float(head_dim);
            rstd = rsqrtf(variance + eps);
        }
        rstd = __shfl_sync(0xffffffff, rstd, 0);

        // Apply normalization
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = lane_id * 4 + i;
            if (idx < head_dim) {
                head_vec[idx] = vals[i] * rstd * w_reg[i];
            }
        }
    }

    __syncthreads();
}
