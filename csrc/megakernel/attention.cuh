#pragma once

#include "config.cuh"

// =============================================================================
// Attention Decode Implementation
// =============================================================================
//
// Computes attention for single query token against KV cache.
// out = softmax(Q @ K^T / sqrt(head_dim)) @ V
//
// Uses online softmax (flash-decoding style) to handle large cache_len
// without storing full attention matrix.
//
// GQA: num_q_heads = 16, num_kv_heads = 8, so 2 Q heads share each KV head
// =============================================================================

// Shared memory for attention decode
struct AttentionSmem {
    // Q vectors: [num_q_heads * head_dim]
    float q[Q_SIZE];

    // KV cache block: [KV_BLOCK_SIZE * head_dim] for K and V
    float k_block[KV_BLOCK_SIZE * HEAD_DIM];
    float v_block[KV_BLOCK_SIZE * HEAD_DIM];

    // Attention scores for current block: [KV_BLOCK_SIZE]
    float scores[KV_BLOCK_SIZE];

    // Per-head accumulators
    float out_accum[Q_SIZE];       // Running weighted sum of V
    float score_max[NUM_Q_HEADS];  // Running max score per head
    float score_sum[NUM_Q_HEADS];  // Running sum of exp(score - max) per head

    // Scratch for reductions
    float scratch[NUM_WARPS];
};

// -----------------------------------------------------------------------------
// Load a block of K or V cache into shared memory
// cache: [num_kv_heads, max_seq_len, head_dim]
// -----------------------------------------------------------------------------
__device__ __forceinline__ void load_kv_block(
    const __nv_bfloat16* __restrict__ cache,  // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ smem_block,            // [KV_BLOCK_SIZE * HEAD_DIM]
    int kv_head_idx,
    int block_start,
    int block_size,
    int max_seq_len,
    int cache_len
) {
    // cache layout: [num_kv_heads, max_seq_len, head_dim]
    // We load positions [block_start, block_start + block_size) for one KV head

    int head_offset = kv_head_idx * max_seq_len * HEAD_DIM;

    for (int i = threadIdx.x; i < block_size * HEAD_DIM; i += BLOCK_SIZE) {
        int pos_in_block = i / HEAD_DIM;
        int dim_idx = i % HEAD_DIM;
        int seq_pos = block_start + pos_in_block;

        if (seq_pos < cache_len) {
            int global_idx = head_offset + seq_pos * HEAD_DIM + dim_idx;
            smem_block[i] = __bfloat162float(cache[global_idx]);
        } else {
            smem_block[i] = 0.0f;
        }
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Compute dot product between Q head and K block
// q_head: [head_dim] one query head
// k_block: [block_size, head_dim] K vectors
// scores: [block_size] output scores
// -----------------------------------------------------------------------------
__device__ __forceinline__ void compute_qk_scores(
    const float* __restrict__ q_head,     // [head_dim]
    const float* __restrict__ k_block,    // [block_size * head_dim]
    float* __restrict__ scores,            // [block_size]
    float* __restrict__ scratch,
    int block_size,
    float scale
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Each thread computes partial dot products
    // Distribute K positions across threads

    for (int k_pos = 0; k_pos < block_size; k_pos++) {
        const float* k_vec = k_block + k_pos * HEAD_DIM;

        // Compute dot product Q @ K
        float dot = 0.0f;
        for (int d = threadIdx.x; d < HEAD_DIM; d += BLOCK_SIZE) {
            dot += q_head[d] * k_vec[d];
        }

        // Reduce across block
        dot = block_reduce_sum(dot, scratch);

        // Thread 0 writes the score
        if (threadIdx.x == 0) {
            scores[k_pos] = dot * scale;
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// Online softmax update
// Processes a block of scores and updates running max/sum/output
// -----------------------------------------------------------------------------
__device__ __forceinline__ void online_softmax_update(
    const float* __restrict__ new_scores,  // [block_size] new attention scores
    const float* __restrict__ v_block,     // [block_size * head_dim] V vectors
    float* __restrict__ out_accum,          // [head_dim] running output accumulator
    float& score_max,                       // running max score
    float& score_sum,                       // running sum of exp(score - max)
    int block_size
) {
    // Step 1: Find max of new scores
    float block_max = -INFINITY;
    for (int i = threadIdx.x; i < block_size; i += BLOCK_SIZE) {
        block_max = fmaxf(block_max, new_scores[i]);
    }

    // Reduce to find block max
    float* scratch = out_accum + HEAD_DIM;  // Reuse some scratch space
    block_max = block_reduce_max(block_max, scratch);

    // Step 2: Update running max and correction factor
    float new_max = fmaxf(score_max, block_max);
    float old_scale = expf(score_max - new_max);
    float new_scale_base = expf(block_max - new_max);

    // Step 3: Compute exp(score - new_max) for new scores and sum
    float block_sum = 0.0f;
    for (int i = threadIdx.x; i < block_size; i += BLOCK_SIZE) {
        float exp_score = expf(new_scores[i] - new_max);
        // Store normalized score temporarily
        const_cast<float*>(new_scores)[i] = exp_score;
        block_sum += exp_score;
    }
    block_sum = block_reduce_sum(block_sum, scratch);

    // Step 4: Update running sum
    float new_sum = score_sum * old_scale + block_sum;

    // Step 5: Rescale old accumulator and add new contribution
    // out_new = out_old * (old_sum * old_scale / new_sum) + sum(softmax_new * V) / new_sum

    float rescale = (score_sum * old_scale) / new_sum;
    float new_weight = 1.0f / new_sum;

    // Rescale existing accumulator
    for (int d = threadIdx.x; d < HEAD_DIM; d += BLOCK_SIZE) {
        out_accum[d] *= rescale;
    }
    __syncthreads();

    // Add new V contributions
    for (int k_pos = 0; k_pos < block_size; k_pos++) {
        float weight = new_scores[k_pos] * new_weight;
        const float* v_vec = v_block + k_pos * HEAD_DIM;

        for (int d = threadIdx.x; d < HEAD_DIM; d += BLOCK_SIZE) {
            out_accum[d] += weight * v_vec[d];
        }
    }
    __syncthreads();

    // Update running stats
    score_max = new_max;
    score_sum = new_sum;
}

// -----------------------------------------------------------------------------
// Full attention decode for all Q heads
// Handles GQA mapping (2 Q heads per KV head)
// -----------------------------------------------------------------------------
__device__ void attention_decode_full(
    const float* __restrict__ smem_q,           // [num_q_heads * head_dim] Q vectors
    const __nv_bfloat16* __restrict__ k_cache,  // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,  // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ smem_out,                // [num_q_heads * head_dim] output
    float* __restrict__ smem_k_block,            // [KV_BLOCK_SIZE * head_dim] temp buffer
    float* __restrict__ smem_v_block,            // [KV_BLOCK_SIZE * head_dim] temp buffer
    float* __restrict__ smem_scores,             // [KV_BLOCK_SIZE] temp buffer
    float* __restrict__ scratch,
    int cache_len,
    int max_seq_len,
    float scale
) {
    // Process each Q head
    // With GQA: Q heads 0,1 use KV head 0; Q heads 2,3 use KV head 1; etc.
    int gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;  // 2

    for (int q_head = 0; q_head < NUM_Q_HEADS; q_head++) {
        int kv_head = q_head / gqa_ratio;
        const float* q_vec = smem_q + q_head * HEAD_DIM;
        float* out_vec = smem_out + q_head * HEAD_DIM;

        // Initialize output accumulator to zero
        for (int d = threadIdx.x; d < HEAD_DIM; d += BLOCK_SIZE) {
            out_vec[d] = 0.0f;
        }
        __syncthreads();

        // Running softmax state
        float score_max = -INFINITY;
        float score_sum = 0.0f;

        // Process KV cache in blocks
        int num_blocks = (cache_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

        for (int blk = 0; blk < num_blocks; blk++) {
            int block_start = blk * KV_BLOCK_SIZE;
            int block_size = min(KV_BLOCK_SIZE, cache_len - block_start);

            // Load K block for this KV head
            load_kv_block(k_cache, smem_k_block, kv_head, block_start, block_size, max_seq_len, cache_len);

            // Load V block
            load_kv_block(v_cache, smem_v_block, kv_head, block_start, block_size, max_seq_len, cache_len);

            // Compute Q @ K^T scores
            compute_qk_scores(q_vec, smem_k_block, smem_scores, scratch, block_size, scale);

            // Online softmax update
            online_softmax_update(smem_scores, smem_v_block, out_vec, score_max, score_sum, block_size);
        }
    }
}

// -----------------------------------------------------------------------------
// Optimized attention decode - process multiple heads in parallel
// Each warp handles one Q head
// -----------------------------------------------------------------------------
__device__ void attention_decode_simple(
    const float* __restrict__ smem_q,           // [num_q_heads * head_dim]
    const __nv_bfloat16* __restrict__ k_cache,  // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,  // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ smem_out,                // [num_q_heads * head_dim]
    float* __restrict__ scratch,                 // [NUM_WARPS * 4] for per-warp state
    int cache_len,
    int max_seq_len,
    float scale
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;

    // Process Q heads in parallel - each warp handles one head at a time
    for (int q_base = 0; q_base < NUM_Q_HEADS; q_base += NUM_WARPS) {
        int q_head = q_base + warp_id;

        if (q_head < NUM_Q_HEADS) {
            int kv_head = q_head / gqa_ratio;
            const float* q_vec = smem_q + q_head * HEAD_DIM;
            float* out_vec = smem_out + q_head * HEAD_DIM;

            // Initialize output to zero
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                out_vec[d] = 0.0f;
            }

            // Online softmax state
            float running_max = -INFINITY;
            float running_sum = 0.0f;

            // Process cache positions
            for (int pos = 0; pos < cache_len; pos++) {
                const __nv_bfloat16* k_vec = k_cache + (kv_head * max_seq_len + pos) * HEAD_DIM;
                const __nv_bfloat16* v_vec = v_cache + (kv_head * max_seq_len + pos) * HEAD_DIM;

                // Compute Q @ K dot product (each lane handles HEAD_DIM/WARP_SIZE elements)
                float dot = 0.0f;
                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                    dot += q_vec[d] * __bfloat162float(k_vec[d]);
                }
                dot = warp_reduce_sum(dot);
                float score = dot * scale;

                // Online softmax update
                float new_max = fmaxf(running_max, score);
                float old_scale = expf(running_max - new_max);
                float new_weight = expf(score - new_max);
                float new_sum = running_sum * old_scale + new_weight;

                // Rescale existing output and add new V contribution
                float rescale = (running_sum > 0.0f) ? (running_sum * old_scale / new_sum) : 0.0f;
                float v_scale = new_weight / new_sum;

                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                    out_vec[d] = out_vec[d] * rescale + v_scale * __bfloat162float(v_vec[d]);
                }

                running_max = new_max;
                running_sum = new_sum;
            }
        }
    }
    __syncthreads();
}
