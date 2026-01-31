/**
 * Fused Prefill Kernel for Qwen3-0.6B
 *
 * Processes multiple tokens in parallel using batched matrix multiplications.
 * Uses cuBLAS for compute-bound GEMM operations (BF16 x BF16 with FP32 accumulation)
 * and custom kernels for:
 * - RMSNorm (batched)
 * - RoPE (all positions)
 * - Causal attention with online softmax
 * - SiLU activation
 * - KV cache population
 */

#include "config.cuh"
#include <cublas_v2.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// =============================================================================
// Configuration
// =============================================================================

constexpr int PREFILL_BLOCK_SIZE = 256;
constexpr int PREFILL_NUM_WARPS = PREFILL_BLOCK_SIZE / WARP_SIZE;
constexpr float PREFILL_RMS_EPS = 1e-6f;

struct PrefillLayerWeights {
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float prefill_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float prefill_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float prefill_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Embedding lookup kernel (outputs BF16)
// =============================================================================

__global__ void prefill_embed_kernel(
    const int* __restrict__ token_ids,            // [seq_len]
    const __nv_bfloat16* __restrict__ embed_table,  // [vocab_size, hidden_size]
    __nv_bfloat16* __restrict__ output,           // [seq_len, hidden_size]
    int seq_len,
    int hidden_size
) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= seq_len) return;

    int token_id = token_ids[seq_idx];
    const __nv_bfloat16* embed_row = embed_table + token_id * hidden_size;
    __nv_bfloat16* out_row = output + seq_idx * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = embed_row[i];
    }
}

// =============================================================================
// RMSNorm kernel (BF16 input/output)
// Each block handles one token position
// =============================================================================

__global__ void prefill_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ input,   // [seq_len, hidden_size]
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    __nv_bfloat16* __restrict__ output,        // [seq_len, hidden_size]
    __nv_bfloat16* __restrict__ residual,      // [seq_len, hidden_size] - can be nullptr
    int seq_len,
    int hidden_size
) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= seq_len) return;

    const __nv_bfloat16* in_row = input + seq_idx * hidden_size;
    __nv_bfloat16* out_row = output + seq_idx * hidden_size;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ float smem_reduce[PREFILL_NUM_WARPS];

    // Compute sum of squares
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += PREFILL_BLOCK_SIZE) {
        float v = __bfloat162float(in_row[i]);
        // Save residual if requested
        if (residual != nullptr) {
            residual[seq_idx * hidden_size + i] = in_row[i];
        }
        local_sum_sq += v * v;
    }

    local_sum_sq = prefill_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < PREFILL_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = prefill_warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(hidden_size) + PREFILL_RMS_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    // Apply normalization
    for (int i = threadIdx.x; i < hidden_size; i += PREFILL_BLOCK_SIZE) {
        float v = __bfloat162float(in_row[i]);
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16(v * rstd * w);
    }
}

// =============================================================================
// QK Norm + RoPE + KV Cache Kernel
// Operates on BF16 projected Q/K, writes BF16 to cache
// =============================================================================

__global__ void prefill_qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ q,              // [seq_len, num_q_heads, head_dim]
    __nv_bfloat16* __restrict__ k,              // [seq_len, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,        // [seq_len, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ q_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ cos_table,      // [max_seq, head_dim]
    const __nv_bfloat16* __restrict__ sin_table,      // [max_seq, head_dim]
    __nv_bfloat16* __restrict__ k_cache,  // [num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16* __restrict__ v_cache,  // [num_kv_heads, max_seq_len, head_dim]
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int position_offset  // Starting position in KV cache
) {
    // Each block handles one (position, head) pair
    // Grid: (seq_len, max(num_q_heads, num_kv_heads))
    int pos = blockIdx.x;
    int head = blockIdx.y;

    if (pos >= seq_len) return;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + pos * head_dim;
    const __nv_bfloat16* sin_pos = sin_table + pos * head_dim;

    __shared__ float smem_reduce[8];
    __shared__ float smem_normed[HEAD_DIM];

    // Process Q heads
    if (head < num_q_heads) {
        __nv_bfloat16* q_head = q + pos * num_q_heads * head_dim + head * head_dim;

        // RMSNorm for Q head
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __bfloat162float(q_head[i]);
            sum_sq += v * v;
        }

        // Warp reduction
        sum_sq = prefill_warp_reduce_sum(sum_sq);
        if (lane_id == 0 && warp_id < 8) {
            smem_reduce[warp_id] = sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < (blockDim.x / WARP_SIZE)) ? smem_reduce[lane_id] : 0.0f;
            sum = prefill_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(head_dim) + PREFILL_RMS_EPS);
            }
        }
        __syncthreads();

        float scale = smem_reduce[0];

        // Load normalized values to shared memory
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            smem_normed[i] = __bfloat162float(q_head[i]) * scale * __bfloat162float(q_norm_weight[i]);
        }
        __syncthreads();

        // Apply RoPE
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float cos_v = __bfloat162float(cos_pos[i]);
            float sin_v = __bfloat162float(sin_pos[i]);

            int pair_idx = (i < head_dim/2) ? (i + head_dim/2) : (i - head_dim/2);
            float pair_v = smem_normed[pair_idx];

            float result;
            if (i < head_dim/2) {
                result = smem_normed[i] * cos_v - pair_v * sin_v;
            } else {
                result = pair_v * sin_v + smem_normed[i] * cos_v;
            }
            q_head[i] = __float2bfloat16(result);
        }
    }
    __syncthreads();

    // Process K heads (and populate cache)
    if (head < num_kv_heads) {
        __nv_bfloat16* k_head = k + pos * num_kv_heads * head_dim + head * head_dim;
        const __nv_bfloat16* v_head = v + pos * num_kv_heads * head_dim + head * head_dim;

        int cache_pos = position_offset + pos;
        __nv_bfloat16* k_cache_head = k_cache + head * max_seq_len * head_dim + cache_pos * head_dim;
        __nv_bfloat16* v_cache_head = v_cache + head * max_seq_len * head_dim + cache_pos * head_dim;

        // RMSNorm for K head
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __bfloat162float(k_head[i]);
            sum_sq += v * v;
        }

        sum_sq = prefill_warp_reduce_sum(sum_sq);
        if (lane_id == 0 && warp_id < 8) {
            smem_reduce[warp_id] = sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < (blockDim.x / WARP_SIZE)) ? smem_reduce[lane_id] : 0.0f;
            sum = prefill_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(head_dim) + PREFILL_RMS_EPS);
            }
        }
        __syncthreads();

        float scale = smem_reduce[0];

        // Load normalized K to shared memory
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            smem_normed[i] = __bfloat162float(k_head[i]) * scale * __bfloat162float(k_norm_weight[i]);
        }
        __syncthreads();

        // Apply RoPE and write to cache
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float cos_v = __bfloat162float(cos_pos[i]);
            float sin_v = __bfloat162float(sin_pos[i]);

            int pair_idx = (i < head_dim/2) ? (i + head_dim/2) : (i - head_dim/2);
            float pair_v = smem_normed[pair_idx];

            float k_final;
            if (i < head_dim/2) {
                k_final = smem_normed[i] * cos_v - pair_v * sin_v;
            } else {
                k_final = pair_v * sin_v + smem_normed[i] * cos_v;
            }

            k_head[i] = __float2bfloat16(k_final);
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = v_head[i];  // V doesn't need RoPE
        }
    }
}

// =============================================================================
// Causal Attention Kernel (BF16 input, BF16 output)
// Each warp handles one (query_pos, q_head) pair
// =============================================================================

__global__ void prefill_causal_attention_kernel(
    const __nv_bfloat16* __restrict__ q,       // [seq_len, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k,       // [seq_len, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,       // [seq_len, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ output,        // [seq_len, num_q_heads, head_dim]
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attn_scale
) {
    // Each warp handles one (query_pos, q_head) pair
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int total_work = seq_len * num_q_heads;
    if (warp_idx >= total_work) return;

    int q_pos = warp_idx / num_q_heads;
    int q_head = warp_idx % num_q_heads;
    int kv_head = q_head / (num_q_heads / num_kv_heads);  // GQA mapping

    const __nv_bfloat16* q_vec = q + q_pos * num_q_heads * head_dim + q_head * head_dim;
    __nv_bfloat16* out_vec = output + q_pos * num_q_heads * head_dim + q_head * head_dim;

    // Each lane accumulates for its portion of head_dim
    // head_dim=128, WARP_SIZE=32, so 4 elements per lane
    constexpr int ELEMS_PER_LANE = HEAD_DIM / WARP_SIZE;  // 4

    float out_acc[ELEMS_PER_LANE] = {0.0f};
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Load Q values
    float q_local[ELEMS_PER_LANE];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        q_local[e] = __bfloat162float(q_vec[lane_id + e * WARP_SIZE]);
    }

    // Process all KV positions up to and including q_pos (causal)
    for (int kv_pos = 0; kv_pos <= q_pos; kv_pos++) {
        const __nv_bfloat16* k_vec = k + kv_pos * num_kv_heads * head_dim + kv_head * head_dim;
        const __nv_bfloat16* v_vec = v + kv_pos * num_kv_heads * head_dim + kv_head * head_dim;

        // Compute dot product Q @ K
        float score = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            score += q_local[e] * __bfloat162float(k_vec[lane_id + e * WARP_SIZE]);
        }

        // Warp reduction for dot product
        score = prefill_warp_reduce_sum(score) * attn_scale;
        // Broadcast reduced score to all lanes (only lane 0 has correct value after reduction)
        score = __shfl_sync(0xffffffff, score, 0);

        // Online softmax
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = expf(old_max - max_score);
        sum_exp = sum_exp * exp_diff + expf(score - max_score);

        // Update output accumulator
        float weight = expf(score - max_score);
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            out_acc[e] = out_acc[e] * exp_diff + weight * __bfloat162float(v_vec[lane_id + e * WARP_SIZE]);
        }
    }

    // Write output
    float sum_inv = 1.0f / sum_exp;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        out_vec[lane_id + e * WARP_SIZE] = __float2bfloat16(out_acc[e] * sum_inv);
    }
}

// =============================================================================
// Residual Add Kernel (BF16)
// =============================================================================

__global__ void prefill_residual_add_kernel(
    const __nv_bfloat16* __restrict__ input,     // [seq_len, hidden_size]
    const __nv_bfloat16* __restrict__ residual,  // [seq_len, hidden_size]
    __nv_bfloat16* __restrict__ output,          // [seq_len, hidden_size]
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float a = __bfloat162float(input[idx]);
        float b = __bfloat162float(residual[idx]);
        output[idx] = __float2bfloat16(a + b);
    }
}

// =============================================================================
// SiLU + Element-wise Multiply Kernel (BF16)
// =============================================================================

__global__ void prefill_silu_mul_kernel(
    const __nv_bfloat16* __restrict__ gate,   // [seq_len, intermediate_size]
    const __nv_bfloat16* __restrict__ up,     // [seq_len, intermediate_size]
    __nv_bfloat16* __restrict__ output,       // [seq_len, intermediate_size]
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        output[idx] = __float2bfloat16(prefill_silu(g) * u);
    }
}

// =============================================================================
// Final Norm Kernel (only process last token, output BF16)
// =============================================================================

__global__ void prefill_final_norm_kernel(
    const __nv_bfloat16* __restrict__ input,       // [seq_len, hidden_size]
    const __nv_bfloat16* __restrict__ weight,      // [hidden_size]
    __nv_bfloat16* __restrict__ output,            // [hidden_size] - only last token
    int seq_len,
    int hidden_size
) {
    // Only process last token
    const __nv_bfloat16* in_row = input + (seq_len - 1) * hidden_size;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ float smem_reduce[PREFILL_NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += PREFILL_BLOCK_SIZE) {
        float v = __bfloat162float(in_row[i]);
        local_sum_sq += v * v;
    }

    local_sum_sq = prefill_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < PREFILL_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = prefill_warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(hidden_size) + PREFILL_RMS_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < hidden_size; i += PREFILL_BLOCK_SIZE) {
        float v = __bfloat162float(in_row[i]);
        float w = __bfloat162float(weight[i]);
        output[i] = __float2bfloat16(v * rstd * w);
    }
}

// =============================================================================
// LM Head with argmax (BF16 input)
// =============================================================================

constexpr int PREFILL_LM_NUM_BLOCKS = 1184;
constexpr int PREFILL_LM_BLOCK_SIZE = 256;
constexpr int PREFILL_VOCAB_SIZE = 151936;

__global__ void prefill_lm_head_phase1(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += PREFILL_LM_BLOCK_SIZE) {
        s_hidden[i] = __bfloat162float(hidden[i]);
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (PREFILL_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, PREFILL_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += PREFILL_LM_BLOCK_SIZE / WARP_SIZE) {
        const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;

        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k+1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k+2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = prefill_warp_reduce_sum(sum);

        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[PREFILL_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[PREFILL_LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < PREFILL_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < PREFILL_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = max_val;
            block_max_idxs[blockIdx.x] = max_idx;
        }
    }
}

__global__ void prefill_lm_head_phase2(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks
) {
    __shared__ float s_max_vals[1024];
    __shared__ int s_max_idxs[1024];

    int tid = threadIdx.x;

    float local_max = -INFINITY;
    int local_idx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = block_max_vals[i];
        if (val > local_max) {
            local_max = val;
            local_idx = block_max_idxs[i];
        }
    }

    s_max_vals[tid] = local_max;
    s_max_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_vals[tid + s] > s_max_vals[tid]) {
                s_max_vals[tid] = s_max_vals[tid + s];
                s_max_idxs[tid] = s_max_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output_token = s_max_idxs[0];
    }
}

// =============================================================================
// Main prefill launch function (all BF16)
// =============================================================================

extern "C" void launch_prefill_float(
    const int* input_token_ids,      // [seq_len] on device
    int* output_token_id,            // [1] on device
    int seq_len,
    const void* embed_weight,
    const PrefillLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    // BF16 buffers (renamed from float for compatibility)
    void* hidden_bf16,       // [seq_len, hidden_size] bf16
    void* residual,          // [seq_len, hidden_size] bf16
    void* normalized,        // [seq_len, hidden_size] bf16
    void* q_proj,            // [seq_len, q_size] bf16
    void* k_proj,            // [seq_len, kv_size] bf16
    void* v_proj,            // [seq_len, kv_size] bf16
    void* attn_out,          // [seq_len, q_size] bf16
    void* o_proj_out,        // [seq_len, hidden_size] bf16
    void* mlp_norm,          // [seq_len, hidden_size] bf16
    void* gate_out,          // [seq_len, intermediate_size] bf16
    void* up_out,            // [seq_len, intermediate_size] bf16
    void* mlp_intermediate,  // [seq_len, intermediate_size] bf16
    void* down_out,          // [seq_len, hidden_size] bf16
    void* final_hidden,      // [hidden_size] bf16
    void* block_max_vals,
    void* block_max_idxs,
    void* hidden_bf16_out,   // [hidden_size] bf16 - for decode continuation
    int num_layers,
    int max_seq_len,
    float attn_scale,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    // cuBLAS parameters for BF16 x BF16 -> BF16 with FP32 accumulation
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;

    cublasSetStream(cublas_handle, stream);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    // Embedding lookup (to BF16)
    prefill_embed_kernel<<<seq_len, 256, 0, stream>>>(
        input_token_ids,
        (const __nv_bfloat16*)embed_weight,
        (__nv_bfloat16*)hidden_bf16,
        seq_len,
        HIDDEN_SIZE
    );

    for (int layer = 0; layer < num_layers; layer++) {
        const PrefillLayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = (__nv_bfloat16*)k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = (__nv_bfloat16*)v_cache + layer * kv_cache_layer_stride;

        // 1. Input LayerNorm (bf16 -> bf16, save residual)
        prefill_rmsnorm_kernel<<<seq_len, PREFILL_BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)hidden_bf16,
            (const __nv_bfloat16*)w.input_layernorm_weight,
            (__nv_bfloat16*)normalized,
            (__nv_bfloat16*)residual,
            seq_len,
            HIDDEN_SIZE
        );

        // 2. Q Projection: [seq_len, hidden] @ [q_size, hidden]^T -> [seq_len, q_size]
        // cuBLAS col-major: C = alpha * op(A) * op(B) + beta * C
        // Row-major: C[seq,q] = input[seq,hid] @ W[q,hid]^T
        // Col-major: C^T[q,seq] = W[q,hid] @ input^T[hid,seq]
        // => cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N, q_size, seq_len, hidden, W, input, C)
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Q_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.q_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            normalized, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            q_proj, CUDA_R_16BF, Q_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 3. K Projection
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            KV_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.k_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            normalized, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            k_proj, CUDA_R_16BF, KV_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 4. V Projection
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            KV_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.v_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            normalized, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            v_proj, CUDA_R_16BF, KV_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 5. QK Norm + RoPE + KV Cache
        dim3 grid_rope(seq_len, max(NUM_Q_HEADS, NUM_KV_HEADS));
        prefill_qk_norm_rope_kernel<<<grid_rope, 128, 0, stream>>>(
            (__nv_bfloat16*)q_proj,
            (__nv_bfloat16*)k_proj,
            (const __nv_bfloat16*)v_proj,
            (const __nv_bfloat16*)w.q_norm_weight,
            (const __nv_bfloat16*)w.k_norm_weight,
            (const __nv_bfloat16*)cos_table,
            (const __nv_bfloat16*)sin_table,
            layer_k_cache,
            layer_v_cache,
            seq_len,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            max_seq_len,
            0  // position_offset = 0 for prefill
        );

        // 6. Causal Attention
        int total_queries = seq_len * NUM_Q_HEADS;
        int warps_per_block = 4;
        int threads_per_block = warps_per_block * WARP_SIZE;
        int num_blocks_attn = (total_queries + warps_per_block - 1) / warps_per_block;

        prefill_causal_attention_kernel<<<num_blocks_attn, threads_per_block, 0, stream>>>(
            (const __nv_bfloat16*)q_proj,
            (const __nv_bfloat16*)k_proj,
            (const __nv_bfloat16*)v_proj,
            (__nv_bfloat16*)attn_out,
            seq_len,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            attn_scale
        );

        // 7. O Projection: [seq_len, q_size] @ [hidden, q_size]^T -> [seq_len, hidden]
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            HIDDEN_SIZE, seq_len, Q_SIZE,
            &alpha_f,
            w.o_proj_weight, CUDA_R_16BF, Q_SIZE,
            attn_out, CUDA_R_16BF, Q_SIZE,
            &beta_f,
            o_proj_out, CUDA_R_16BF, HIDDEN_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 8. Residual add
        int total_hidden = seq_len * HIDDEN_SIZE;
        int block_size_res = 256;
        int num_blocks_res = (total_hidden + block_size_res - 1) / block_size_res;
        prefill_residual_add_kernel<<<num_blocks_res, block_size_res, 0, stream>>>(
            (const __nv_bfloat16*)o_proj_out,
            (const __nv_bfloat16*)residual,
            (__nv_bfloat16*)hidden_bf16,
            total_hidden
        );

        // 9. Post-attention LayerNorm (save residual)
        prefill_rmsnorm_kernel<<<seq_len, PREFILL_BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)hidden_bf16,
            (const __nv_bfloat16*)w.post_attn_layernorm_weight,
            (__nv_bfloat16*)mlp_norm,
            (__nv_bfloat16*)residual,
            seq_len,
            HIDDEN_SIZE
        );

        // 10. Gate projection
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            INTERMEDIATE_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.gate_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            mlp_norm, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            gate_out, CUDA_R_16BF, INTERMEDIATE_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 11. Up projection
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            INTERMEDIATE_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.up_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            mlp_norm, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            up_out, CUDA_R_16BF, INTERMEDIATE_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 12. SiLU + multiply
        int total_inter = seq_len * INTERMEDIATE_SIZE;
        int num_blocks_silu = (total_inter + 255) / 256;
        prefill_silu_mul_kernel<<<num_blocks_silu, 256, 0, stream>>>(
            (const __nv_bfloat16*)gate_out,
            (const __nv_bfloat16*)up_out,
            (__nv_bfloat16*)mlp_intermediate,
            total_inter
        );

        // 13. Down projection
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            HIDDEN_SIZE, seq_len, INTERMEDIATE_SIZE,
            &alpha_f,
            w.down_proj_weight, CUDA_R_16BF, INTERMEDIATE_SIZE,
            mlp_intermediate, CUDA_R_16BF, INTERMEDIATE_SIZE,
            &beta_f,
            down_out, CUDA_R_16BF, HIDDEN_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 14. Residual add
        prefill_residual_add_kernel<<<num_blocks_res, block_size_res, 0, stream>>>(
            (const __nv_bfloat16*)down_out,
            (const __nv_bfloat16*)residual,
            (__nv_bfloat16*)hidden_bf16,
            total_hidden
        );
    }

    // Final norm (only last token)
    prefill_final_norm_kernel<<<1, PREFILL_BLOCK_SIZE, 0, stream>>>(
        (const __nv_bfloat16*)hidden_bf16,
        (const __nv_bfloat16*)final_norm_weight,
        (__nv_bfloat16*)final_hidden,
        seq_len,
        HIDDEN_SIZE
    );

    // Copy final hidden to output for decode continuation
    cudaMemcpyAsync(
        hidden_bf16_out,
        final_hidden,
        HIDDEN_SIZE * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice,
        stream
    );

    // LM Head
    prefill_lm_head_phase1<<<PREFILL_LM_NUM_BLOCKS, PREFILL_LM_BLOCK_SIZE, 0, stream>>>(
        (const __nv_bfloat16*)final_hidden,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    prefill_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        PREFILL_LM_NUM_BLOCKS
    );
}
