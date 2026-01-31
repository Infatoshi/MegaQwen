/**
 * Fused Prefill Megakernel for Qwen3-0.6B
 *
 * Strategy: Replace cuBLAS-based prefill with a fully fused megakernel that
 * processes multiple tokens in parallel using cooperative groups.
 *
 * For seq_len 4-64 tokens, treat multi-token projection as batched GEMV:
 * - Each output element is an independent dot product
 * - Parallelize across both tokens AND output rows
 * - Reuses proven decode kernel patterns (vec4 loads, warp reduction)
 *
 * Configuration:
 * - PREFILL_MK_NUM_BLOCKS = 82 (one per SM)
 * - PREFILL_MK_BLOCK_SIZE = 256 (8 warps)
 * - MAX_PREFILL_SEQ_LEN = 64
 */

#include "config.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// =============================================================================
// Configuration
// =============================================================================

constexpr int PREFILL_MK_NUM_BLOCKS = 82;
constexpr int PREFILL_MK_BLOCK_SIZE = 256;
constexpr int PREFILL_MK_NUM_WARPS = PREFILL_MK_BLOCK_SIZE / WARP_SIZE;
constexpr int MAX_PREFILL_SEQ_LEN = 64;
constexpr float PREFILL_MK_RMS_EPS = 1e-6f;

// LM head configuration
constexpr int PREFILL_MK_LM_NUM_BLOCKS = 1184;
constexpr int PREFILL_MK_LM_BLOCK_SIZE = 256;
constexpr int PREFILL_MK_VOCAB_SIZE = 151936;

// Shared memory layout for caching normalized inputs
// Cache up to 8 tokens at a time for batched operations
constexpr int PREFILL_MK_TOKEN_CACHE = 8;

struct PrefillMKLayerWeights {
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

__device__ __forceinline__ float prefill_mk_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float prefill_mk_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float prefill_mk_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Phase 1: Embedding Lookup (distributed across grid)
// =============================================================================

__device__ void prefill_mk_embedding(
    cg::grid_group& grid,
    const int* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_weight,
    float* __restrict__ hidden,  // [seq_len, HIDDEN_SIZE]
    int seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    // Total elements = seq_len * HIDDEN_SIZE
    // Distribute across grid
    int total_elements = seq_len * HIDDEN_SIZE;
    int elements_per_block = (total_elements + num_blocks - 1) / num_blocks;
    int start = block_id * elements_per_block;
    int end = min(start + elements_per_block, total_elements);

    for (int idx = start + threadIdx.x; idx < end; idx += PREFILL_MK_BLOCK_SIZE) {
        int token_idx = idx / HIDDEN_SIZE;
        int dim_idx = idx % HIDDEN_SIZE;
        int token_id = token_ids[token_idx];
        hidden[idx] = __bfloat162float(__ldg(embed_weight + token_id * HIDDEN_SIZE + dim_idx));
    }

    grid.sync();
}

// =============================================================================
// Phase 2: Batched RMSNorm (Input LayerNorm)
// Each block handles a subset of tokens
// =============================================================================

__device__ void prefill_mk_rmsnorm(
    cg::grid_group& grid,
    const float* __restrict__ input,     // [seq_len, HIDDEN_SIZE]
    const __nv_bfloat16* __restrict__ weight,  // [HIDDEN_SIZE]
    float* __restrict__ output,          // [seq_len, HIDDEN_SIZE]
    float* __restrict__ residual,        // [seq_len, HIDDEN_SIZE] - can be same as input
    int seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ float smem_reduce[PREFILL_MK_NUM_WARPS];

    // Distribute tokens across blocks
    int tokens_per_block = (seq_len + num_blocks - 1) / num_blocks;
    int token_start = block_id * tokens_per_block;
    int token_end = min(token_start + tokens_per_block, seq_len);

    for (int token_idx = token_start; token_idx < token_end; token_idx++) {
        const float* in_row = input + token_idx * HIDDEN_SIZE;
        float* out_row = output + token_idx * HIDDEN_SIZE;
        float* res_row = residual + token_idx * HIDDEN_SIZE;

        // Compute sum of squares
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += PREFILL_MK_BLOCK_SIZE) {
            float v = in_row[i];
            // Save residual
            res_row[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = prefill_mk_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < PREFILL_MK_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = prefill_mk_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + PREFILL_MK_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        // Apply normalization
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += PREFILL_MK_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(weight + i));
            out_row[i] = in_row[i] * rstd * w;
        }
        __syncthreads();
    }

    grid.sync();
}

// =============================================================================
// Phase 3: Batched QKV Projection (Batched GEMV)
// Distribute seq_len * (Q_SIZE + 2*KV_SIZE) outputs across grid
// =============================================================================

__device__ void prefill_mk_qkv_projection(
    cg::grid_group& grid,
    const float* __restrict__ normalized,  // [seq_len, HIDDEN_SIZE]
    const __nv_bfloat16* __restrict__ q_weight,  // [Q_SIZE, HIDDEN_SIZE]
    const __nv_bfloat16* __restrict__ k_weight,  // [KV_SIZE, HIDDEN_SIZE]
    const __nv_bfloat16* __restrict__ v_weight,  // [KV_SIZE, HIDDEN_SIZE]
    float* __restrict__ q_out,  // [seq_len, Q_SIZE]
    float* __restrict__ k_out,  // [seq_len, KV_SIZE]
    float* __restrict__ v_out,  // [seq_len, KV_SIZE]
    int seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Total outputs = seq_len * (Q_SIZE + KV_SIZE + KV_SIZE)
    // Each warp computes one output element
    constexpr int TOTAL_PROJ_SIZE = Q_SIZE + KV_SIZE + KV_SIZE;  // 4096
    int total_outputs = seq_len * TOTAL_PROJ_SIZE;

    // Distribute across grid: each block gets a range of outputs
    int outputs_per_block = (total_outputs + num_blocks - 1) / num_blocks;
    int output_start = block_id * outputs_per_block;
    int output_end = min(output_start + outputs_per_block, total_outputs);

    // Each warp in the block handles outputs
    for (int out_base = output_start; out_base < output_end; out_base += PREFILL_MK_NUM_WARPS) {
        int out_idx = out_base + warp_id;

        if (out_idx < output_end) {
            // Decode which token and which output row
            int token_idx = out_idx / TOTAL_PROJ_SIZE;
            int proj_idx = out_idx % TOTAL_PROJ_SIZE;

            const float* input_row = normalized + token_idx * HIDDEN_SIZE;

            const __nv_bfloat16* weight_row;
            float* output_ptr;

            if (proj_idx < Q_SIZE) {
                // Q projection
                weight_row = q_weight + proj_idx * HIDDEN_SIZE;
                output_ptr = q_out + token_idx * Q_SIZE + proj_idx;
            } else if (proj_idx < Q_SIZE + KV_SIZE) {
                // K projection
                int k_idx = proj_idx - Q_SIZE;
                weight_row = k_weight + k_idx * HIDDEN_SIZE;
                output_ptr = k_out + token_idx * KV_SIZE + k_idx;
            } else {
                // V projection
                int v_idx = proj_idx - Q_SIZE - KV_SIZE;
                weight_row = v_weight + v_idx * HIDDEN_SIZE;
                output_ptr = v_out + token_idx * KV_SIZE + v_idx;
            }

            // Compute dot product with vec4 loads
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * input_row[k] +
                       __bfloat162float(w_ptr[1]) * input_row[k+1] +
                       __bfloat162float(w_ptr[2]) * input_row[k+2] +
                       __bfloat162float(w_ptr[3]) * input_row[k+3];
            }

            sum = prefill_mk_warp_reduce_sum(sum);
            if (lane_id == 0) {
                *output_ptr = sum;
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Phase 4: QK Norm + RoPE + KV Cache
// Distribute (seq_len, head) pairs across grid
// =============================================================================

__device__ void prefill_mk_qk_norm_rope_cache(
    cg::grid_group& grid,
    float* __restrict__ q,  // [seq_len, Q_SIZE]
    float* __restrict__ k,  // [seq_len, KV_SIZE]
    const float* __restrict__ v,  // [seq_len, KV_SIZE]
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int seq_len,
    int max_seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Process Q heads: total = seq_len * NUM_Q_HEADS
    int total_q_heads = seq_len * NUM_Q_HEADS;
    int q_heads_per_block = (total_q_heads + num_blocks - 1) / num_blocks;
    int q_start = block_id * q_heads_per_block;
    int q_end = min(q_start + q_heads_per_block, total_q_heads);

    for (int qh_base = q_start; qh_base < q_end; qh_base += PREFILL_MK_NUM_WARPS) {
        int qh_idx = qh_base + warp_id;

        if (qh_idx < q_end) {
            int pos = qh_idx / NUM_Q_HEADS;
            int head = qh_idx % NUM_Q_HEADS;

            float* q_head = q + pos * Q_SIZE + head * HEAD_DIM;
            const __nv_bfloat16* cos_pos = cos_table + pos * HEAD_DIM;
            const __nv_bfloat16* sin_pos = sin_table + pos * HEAD_DIM;

            // RMSNorm for Q head
            float sum_sq = 0.0f;
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                sum_sq += q_head[i] * q_head[i];
            }
            sum_sq = prefill_mk_warp_reduce_sum(sum_sq);
            float scale = rsqrtf(sum_sq / float(HEAD_DIM) + PREFILL_MK_RMS_EPS);
            scale = __shfl_sync(0xffffffff, scale, 0);

            // Load normalized Q to registers
            float q_local[HEAD_DIM / WARP_SIZE];
            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
                q_local[j] = q_head[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));
            }

            // Apply RoPE
            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
                float cos_v = __bfloat162float(__ldg(cos_pos + i));
                float sin_v = __bfloat162float(__ldg(sin_pos + i));

                int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
                int pair_idx = i + pair_offset;
                int pair_j = pair_idx / WARP_SIZE;
                float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);

                if (i < HEAD_DIM/2) {
                    q_head[i] = q_local[j] * cos_v - pair_v * sin_v;
                } else {
                    q_head[i] = pair_v * sin_v + q_local[j] * cos_v;
                }
            }
        }
    }

    // Process K heads + write to cache: total = seq_len * NUM_KV_HEADS
    int total_k_heads = seq_len * NUM_KV_HEADS;
    int k_heads_per_block = (total_k_heads + num_blocks - 1) / num_blocks;
    int k_start = block_id * k_heads_per_block;
    int k_end = min(k_start + k_heads_per_block, total_k_heads);

    for (int kh_base = k_start; kh_base < k_end; kh_base += PREFILL_MK_NUM_WARPS) {
        int kh_idx = kh_base + warp_id;

        if (kh_idx < k_end) {
            int pos = kh_idx / NUM_KV_HEADS;
            int head = kh_idx % NUM_KV_HEADS;

            float* k_head = k + pos * KV_SIZE + head * HEAD_DIM;
            const float* v_head = v + pos * KV_SIZE + head * HEAD_DIM;
            const __nv_bfloat16* cos_pos = cos_table + pos * HEAD_DIM;
            const __nv_bfloat16* sin_pos = sin_table + pos * HEAD_DIM;

            __nv_bfloat16* k_cache_head = k_cache + head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            __nv_bfloat16* v_cache_head = v_cache + head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            // RMSNorm for K head
            float sum_sq = 0.0f;
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                sum_sq += k_head[i] * k_head[i];
            }
            sum_sq = prefill_mk_warp_reduce_sum(sum_sq);
            float scale = rsqrtf(sum_sq / float(HEAD_DIM) + PREFILL_MK_RMS_EPS);
            scale = __shfl_sync(0xffffffff, scale, 0);

            // Load normalized K to registers
            float k_local[HEAD_DIM / WARP_SIZE];
            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
                k_local[j] = k_head[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));
            }

            // Apply RoPE and write to cache
            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
                float cos_v = __bfloat162float(__ldg(cos_pos + i));
                float sin_v = __bfloat162float(__ldg(sin_pos + i));

                int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
                int pair_idx = i + pair_offset;
                int pair_j = pair_idx / WARP_SIZE;
                float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);

                float k_final;
                if (i < HEAD_DIM/2) {
                    k_final = k_local[j] * cos_v - pair_v * sin_v;
                } else {
                    k_final = pair_v * sin_v + k_local[j] * cos_v;
                }

                k_head[i] = k_final;
                k_cache_head[i] = __float2bfloat16(k_final);
                v_cache_head[i] = __float2bfloat16(v_head[i]);
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Phase 5: Causal Attention
// Each warp handles one (query_pos, q_head) pair
// =============================================================================

__device__ void prefill_mk_causal_attention(
    cg::grid_group& grid,
    const float* __restrict__ q,  // [seq_len, Q_SIZE]
    const float* __restrict__ k,  // [seq_len, KV_SIZE]
    const float* __restrict__ v,  // [seq_len, KV_SIZE]
    float* __restrict__ attn_out,  // [seq_len, Q_SIZE]
    int seq_len,
    float attn_scale
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Total work = seq_len * NUM_Q_HEADS
    int total_work = seq_len * NUM_Q_HEADS;
    int work_per_block = (total_work + num_blocks - 1) / num_blocks;
    int work_start = block_id * work_per_block;
    int work_end = min(work_start + work_per_block, total_work);

    constexpr int ELEMS_PER_LANE = HEAD_DIM / WARP_SIZE;  // 4

    for (int w_base = work_start; w_base < work_end; w_base += PREFILL_MK_NUM_WARPS) {
        int w_idx = w_base + warp_id;

        if (w_idx < work_end) {
            int q_pos = w_idx / NUM_Q_HEADS;
            int q_head = w_idx % NUM_Q_HEADS;
            int kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS);  // GQA mapping

            const float* q_vec = q + q_pos * Q_SIZE + q_head * HEAD_DIM;
            float* out_vec = attn_out + q_pos * Q_SIZE + q_head * HEAD_DIM;

            // Load Q values to registers
            float q_local[ELEMS_PER_LANE];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; e++) {
                q_local[e] = q_vec[lane_id + e * WARP_SIZE];
            }

            float out_acc[ELEMS_PER_LANE] = {0.0f};
            float max_score = -INFINITY;
            float sum_exp = 0.0f;

            // Process all KV positions up to and including q_pos (causal)
            for (int kv_pos = 0; kv_pos <= q_pos; kv_pos++) {
                const float* k_vec = k + kv_pos * KV_SIZE + kv_head * HEAD_DIM;
                const float* v_vec = v + kv_pos * KV_SIZE + kv_head * HEAD_DIM;

                // Compute dot product Q @ K
                float score = 0.0f;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; e++) {
                    score += q_local[e] * k_vec[lane_id + e * WARP_SIZE];
                }

                score = prefill_mk_warp_reduce_sum(score) * attn_scale;
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
                    out_acc[e] = out_acc[e] * exp_diff + weight * v_vec[lane_id + e * WARP_SIZE];
                }
            }

            // Write output
            float sum_inv = 1.0f / sum_exp;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; e++) {
                out_vec[lane_id + e * WARP_SIZE] = out_acc[e] * sum_inv;
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Phase 6: O Projection + Residual
// =============================================================================

__device__ void prefill_mk_o_proj_residual(
    cg::grid_group& grid,
    const float* __restrict__ attn_out,  // [seq_len, Q_SIZE]
    const __nv_bfloat16* __restrict__ o_weight,  // [HIDDEN_SIZE, Q_SIZE]
    const float* __restrict__ residual,  // [seq_len, HIDDEN_SIZE]
    float* __restrict__ hidden_out,  // [seq_len, HIDDEN_SIZE]
    int seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Total outputs = seq_len * HIDDEN_SIZE
    int total_outputs = seq_len * HIDDEN_SIZE;
    int outputs_per_block = (total_outputs + num_blocks - 1) / num_blocks;
    int output_start = block_id * outputs_per_block;
    int output_end = min(output_start + outputs_per_block, total_outputs);

    for (int out_base = output_start; out_base < output_end; out_base += PREFILL_MK_NUM_WARPS) {
        int out_idx = out_base + warp_id;

        if (out_idx < output_end) {
            int token_idx = out_idx / HIDDEN_SIZE;
            int dim_idx = out_idx % HIDDEN_SIZE;

            const float* attn_row = attn_out + token_idx * Q_SIZE;
            const __nv_bfloat16* o_row = o_weight + dim_idx * Q_SIZE;

            // Compute dot product with vec4 loads
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < Q_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * attn_row[k] +
                       __bfloat162float(w_ptr[1]) * attn_row[k+1] +
                       __bfloat162float(w_ptr[2]) * attn_row[k+2] +
                       __bfloat162float(w_ptr[3]) * attn_row[k+3];
            }

            sum = prefill_mk_warp_reduce_sum(sum);
            if (lane_id == 0) {
                hidden_out[out_idx] = sum + residual[out_idx];
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Phase 7: MLP Gate+Up (fused) + SiLU
// Single input load, two weight loads, two accumulations
// =============================================================================

__device__ void prefill_mk_mlp_gate_up(
    cg::grid_group& grid,
    const float* __restrict__ normalized,  // [seq_len, HIDDEN_SIZE]
    const __nv_bfloat16* __restrict__ gate_weight,  // [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    const __nv_bfloat16* __restrict__ up_weight,    // [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    float* __restrict__ mlp_out,  // [seq_len, INTERMEDIATE_SIZE]
    int seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Total outputs = seq_len * INTERMEDIATE_SIZE
    int total_outputs = seq_len * INTERMEDIATE_SIZE;
    int outputs_per_block = (total_outputs + num_blocks - 1) / num_blocks;
    int output_start = block_id * outputs_per_block;
    int output_end = min(output_start + outputs_per_block, total_outputs);

    for (int out_base = output_start; out_base < output_end; out_base += PREFILL_MK_NUM_WARPS) {
        int out_idx = out_base + warp_id;

        if (out_idx < output_end) {
            int token_idx = out_idx / INTERMEDIATE_SIZE;
            int dim_idx = out_idx % INTERMEDIATE_SIZE;

            const float* input_row = normalized + token_idx * HIDDEN_SIZE;
            const __nv_bfloat16* gate_row = gate_weight + dim_idx * HIDDEN_SIZE;
            const __nv_bfloat16* up_row = up_weight + dim_idx * HIDDEN_SIZE;

            float gate_sum = 0.0f;
            float up_sum = 0.0f;

            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 g_u2 = __ldg(reinterpret_cast<const uint2*>(gate_row + k));
                uint2 u_u2 = __ldg(reinterpret_cast<const uint2*>(up_row + k));
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u2);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u2);

                float in0 = input_row[k];
                float in1 = input_row[k+1];
                float in2 = input_row[k+2];
                float in3 = input_row[k+3];

                gate_sum += __bfloat162float(g_ptr[0]) * in0 +
                            __bfloat162float(g_ptr[1]) * in1 +
                            __bfloat162float(g_ptr[2]) * in2 +
                            __bfloat162float(g_ptr[3]) * in3;

                up_sum += __bfloat162float(u_ptr[0]) * in0 +
                          __bfloat162float(u_ptr[1]) * in1 +
                          __bfloat162float(u_ptr[2]) * in2 +
                          __bfloat162float(u_ptr[3]) * in3;
            }

            gate_sum = prefill_mk_warp_reduce_sum(gate_sum);
            up_sum = prefill_mk_warp_reduce_sum(up_sum);

            if (lane_id == 0) {
                mlp_out[out_idx] = prefill_mk_silu(gate_sum) * up_sum;
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Phase 8: MLP Down + Residual
// =============================================================================

__device__ void prefill_mk_mlp_down_residual(
    cg::grid_group& grid,
    const float* __restrict__ mlp_intermediate,  // [seq_len, INTERMEDIATE_SIZE]
    const __nv_bfloat16* __restrict__ down_weight,  // [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    const float* __restrict__ residual,  // [seq_len, HIDDEN_SIZE]
    float* __restrict__ hidden_out,  // [seq_len, HIDDEN_SIZE]
    int seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Total outputs = seq_len * HIDDEN_SIZE
    int total_outputs = seq_len * HIDDEN_SIZE;
    int outputs_per_block = (total_outputs + num_blocks - 1) / num_blocks;
    int output_start = block_id * outputs_per_block;
    int output_end = min(output_start + outputs_per_block, total_outputs);

    for (int out_base = output_start; out_base < output_end; out_base += PREFILL_MK_NUM_WARPS) {
        int out_idx = out_base + warp_id;

        if (out_idx < output_end) {
            int token_idx = out_idx / HIDDEN_SIZE;
            int dim_idx = out_idx % HIDDEN_SIZE;

            const float* mlp_row = mlp_intermediate + token_idx * INTERMEDIATE_SIZE;
            const __nv_bfloat16* down_row = down_weight + dim_idx * INTERMEDIATE_SIZE;

            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 4) {
                uint2 d_u2 = __ldg(reinterpret_cast<const uint2*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u2);

                sum += __bfloat162float(d_ptr[0]) * mlp_row[k] +
                       __bfloat162float(d_ptr[1]) * mlp_row[k+1] +
                       __bfloat162float(d_ptr[2]) * mlp_row[k+2] +
                       __bfloat162float(d_ptr[3]) * mlp_row[k+3];
            }

            sum = prefill_mk_warp_reduce_sum(sum);
            if (lane_id == 0) {
                hidden_out[out_idx] = sum + residual[out_idx];
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Final Norm (only last token)
// =============================================================================

__device__ void prefill_mk_final_norm(
    cg::grid_group& grid,
    const float* __restrict__ hidden,  // [seq_len, HIDDEN_SIZE]
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ output,  // [HIDDEN_SIZE]
    __nv_bfloat16* __restrict__ hidden_bf16_out,  // [HIDDEN_SIZE] for decode
    int seq_len
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Only block 0 processes final norm
    if (block_id == 0) {
        const float* in_row = hidden + (seq_len - 1) * HIDDEN_SIZE;

        __shared__ float smem_reduce[PREFILL_MK_NUM_WARPS];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += PREFILL_MK_BLOCK_SIZE) {
            float v = in_row[i];
            local_sum_sq += v * v;
        }

        local_sum_sq = prefill_mk_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < PREFILL_MK_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = prefill_mk_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + PREFILL_MK_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += PREFILL_MK_BLOCK_SIZE) {
            float v = in_row[i];
            float w = __bfloat162float(__ldg(weight + i));
            float result = v * rstd * w;
            output[i] = result;
            hidden_bf16_out[i] = __float2bfloat16(v);  // Pre-normalized for decode
        }
    }

    // No grid.sync needed - this is the last phase before LM head
}

// =============================================================================
// Main Prefill Megakernel
// =============================================================================

__global__ void __launch_bounds__(PREFILL_MK_BLOCK_SIZE, 1)
prefill_megakernel(
    const int* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_weight,
    const PrefillMKLayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    // Intermediate buffers (all float32)
    float* __restrict__ hidden,           // [seq_len, HIDDEN_SIZE]
    float* __restrict__ residual,         // [seq_len, HIDDEN_SIZE]
    float* __restrict__ normalized,       // [seq_len, HIDDEN_SIZE]
    float* __restrict__ q_proj,           // [seq_len, Q_SIZE]
    float* __restrict__ k_proj,           // [seq_len, KV_SIZE]
    float* __restrict__ v_proj,           // [seq_len, KV_SIZE]
    float* __restrict__ attn_out,         // [seq_len, Q_SIZE]
    float* __restrict__ mlp_intermediate, // [seq_len, INTERMEDIATE_SIZE]
    float* __restrict__ final_hidden,     // [HIDDEN_SIZE]
    __nv_bfloat16* __restrict__ hidden_bf16_out,  // [HIDDEN_SIZE] for decode
    int seq_len,
    int num_layers,
    int max_seq_len,
    float attn_scale
) {
    cg::grid_group grid = cg::this_grid();

    // Phase 1: Embedding lookup
    prefill_mk_embedding(grid, token_ids, embed_weight, hidden, seq_len);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const PrefillMKLayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_cache_layer_stride;

        // Phase 2: Input LayerNorm
        prefill_mk_rmsnorm(grid, hidden, w.input_layernorm_weight, normalized, residual, seq_len);

        // Phase 3: QKV Projection
        prefill_mk_qkv_projection(grid, normalized, w.q_proj_weight, w.k_proj_weight,
                                   w.v_proj_weight, q_proj, k_proj, v_proj, seq_len);

        // Phase 4: QK Norm + RoPE + KV Cache
        prefill_mk_qk_norm_rope_cache(grid, q_proj, k_proj, v_proj,
                                       w.q_norm_weight, w.k_norm_weight,
                                       cos_table, sin_table,
                                       layer_k_cache, layer_v_cache,
                                       seq_len, max_seq_len);

        // Phase 5: Causal Attention
        prefill_mk_causal_attention(grid, q_proj, k_proj, v_proj, attn_out, seq_len, attn_scale);

        // Phase 6: O Projection + Residual
        prefill_mk_o_proj_residual(grid, attn_out, w.o_proj_weight, residual, hidden, seq_len);

        // Phase 7: Post-attention LayerNorm (reuse normalized buffer)
        prefill_mk_rmsnorm(grid, hidden, w.post_attn_layernorm_weight, normalized, residual, seq_len);

        // Phase 8: MLP Gate+Up fused
        prefill_mk_mlp_gate_up(grid, normalized, w.gate_proj_weight, w.up_proj_weight,
                               mlp_intermediate, seq_len);

        // Phase 9: MLP Down + Residual
        prefill_mk_mlp_down_residual(grid, mlp_intermediate, w.down_proj_weight, residual, hidden, seq_len);
    }

    // Phase 10: Final Norm (only last token)
    prefill_mk_final_norm(grid, hidden, final_norm_weight, final_hidden, hidden_bf16_out, seq_len);
}

// =============================================================================
// LM Head Phase 1 (same as decode)
// =============================================================================

__global__ void prefill_mk_lm_head_phase1(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += PREFILL_MK_LM_BLOCK_SIZE) {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (PREFILL_MK_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, PREFILL_MK_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += PREFILL_MK_LM_BLOCK_SIZE / WARP_SIZE) {
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
        sum = prefill_mk_warp_reduce_sum(sum);

        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[PREFILL_MK_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[PREFILL_MK_LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < PREFILL_MK_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < PREFILL_MK_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

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

__global__ void prefill_mk_lm_head_phase2(
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
// Launch Function
// =============================================================================

extern "C" void launch_prefill_megakernel(
    const int* token_ids,
    int* output_token_id,
    int seq_len,
    const void* embed_weight,
    const PrefillMKLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    // Intermediate buffers (float32)
    void* hidden,
    void* residual,
    void* normalized,
    void* q_proj,
    void* k_proj,
    void* v_proj,
    void* attn_out,
    void* mlp_intermediate,
    void* final_hidden,
    void* hidden_bf16_out,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    void* kernel_args[] = {
        (void*)&token_ids,
        (void*)&embed_weight,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden,
        (void*)&residual,
        (void*)&normalized,
        (void*)&q_proj,
        (void*)&k_proj,
        (void*)&v_proj,
        (void*)&attn_out,
        (void*)&mlp_intermediate,
        (void*)&final_hidden,
        (void*)&hidden_bf16_out,
        (void*)&seq_len,
        (void*)&num_layers,
        (void*)&max_seq_len,
        (void*)&attn_scale
    };

    cudaLaunchCooperativeKernel(
        (void*)prefill_megakernel,
        dim3(PREFILL_MK_NUM_BLOCKS),
        dim3(PREFILL_MK_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    // LM Head (2-phase argmax)
    prefill_mk_lm_head_phase1<<<PREFILL_MK_LM_NUM_BLOCKS, PREFILL_MK_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)final_hidden,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    prefill_mk_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        PREFILL_MK_LM_NUM_BLOCKS
    );
}
