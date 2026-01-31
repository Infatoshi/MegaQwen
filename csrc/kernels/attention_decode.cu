/**
 * Attention decode kernel for single-token inference.
 *
 * Computes attention for a single query token against the full KV cache.
 * This is optimized for the decode phase where seq_len_q = 1.
 *
 * out = softmax(Q @ K^T / sqrt(head_dim)) @ V
 *
 * Supports GQA (grouped query attention) where n_q_heads > n_kv_heads.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

#define ATTN_WARP_SIZE 32

// Warp reduce max (for attention decode)
__device__ __forceinline__ float attn_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// Warp reduce sum (for attention decode)
__device__ __forceinline__ float attn_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduce max using shared memory - broadcasts result to all threads
__device__ __forceinline__ float attn_block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % ATTN_WARP_SIZE;
    int wid = threadIdx.x / ATTN_WARP_SIZE;
    int num_warps = (blockDim.x + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;

    // Reduce within warp
    val = attn_warp_reduce_max(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Warp 0 reduces across warps
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : -INFINITY;
        val = attn_warp_reduce_max(val);
        // Write final result to shared[0]
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    // All threads read the result
    return shared[0];
}

// Block reduce sum using shared memory - broadcasts result to all threads
__device__ __forceinline__ float attn_block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % ATTN_WARP_SIZE;
    int wid = threadIdx.x / ATTN_WARP_SIZE;
    int num_warps = (blockDim.x + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;

    // Reduce within warp
    val = attn_warp_reduce_sum(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Warp 0 reduces across warps
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = attn_warp_reduce_sum(val);
        // Write final result to shared[0]
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    // All threads read the result
    return shared[0];
}

/**
 * Attention decode kernel - one block per query head
 *
 * @param q Query tensor (batch, n_q_heads, 1, head_dim) in bf16
 * @param k_cache K cache (batch, n_kv_heads, max_seq_len, head_dim) in bf16
 * @param v_cache V cache (batch, n_kv_heads, max_seq_len, head_dim) in bf16
 * @param out Output tensor (batch, n_q_heads, 1, head_dim) in bf16
 * @param cache_len Number of valid tokens in cache (including current)
 * @param n_q_heads Number of query heads
 * @param n_kv_heads Number of KV heads
 * @param head_dim Head dimension
 * @param max_seq_len Maximum sequence length in cache
 * @param scale Attention scale (1 / sqrt(head_dim))
 */
__global__ void attention_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int cache_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale
) {
    extern __shared__ float shared[];

    int batch_idx = blockIdx.y;
    int q_head_idx = blockIdx.x;

    // GQA: map query head to kv head
    int n_groups = n_q_heads / n_kv_heads;
    int kv_head_idx = q_head_idx / n_groups;

    // Pointers for this head
    const __nv_bfloat16* q_head = q + (batch_idx * n_q_heads + q_head_idx) * head_dim;
    const __nv_bfloat16* k_head = k_cache + (batch_idx * n_kv_heads + kv_head_idx) * max_seq_len * head_dim;
    const __nv_bfloat16* v_head = v_cache + (batch_idx * n_kv_heads + kv_head_idx) * max_seq_len * head_dim;
    __nv_bfloat16* out_head = out + (batch_idx * n_q_heads + q_head_idx) * head_dim;

    // Load Q into registers (each thread loads part of the vector)
    float q_reg[8];  // Assume max 8 elements per thread
    int elems_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    elems_per_thread = min(elems_per_thread, 8);

    for (int i = 0; i < elems_per_thread; i++) {
        int idx = threadIdx.x * elems_per_thread + i;
        if (idx < head_dim) {
            q_reg[i] = __bfloat162float(q_head[idx]);
        } else {
            q_reg[i] = 0.0f;
        }
    }

    // Shared memory layout:
    // [0..num_warps): for reductions
    // [num_warps..num_warps + BLOCK_K): for attention weights
    int num_warps = (blockDim.x + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;
    float* reduce_shared = shared;
    float* attn_weights = shared + num_warps;

    // Phase 1: Compute all attention scores and find max
    float max_score = -INFINITY;

    // Each thread processes multiple KV positions
    for (int kv_pos = threadIdx.x; kv_pos < cache_len; kv_pos += blockDim.x) {
        const __nv_bfloat16* k_pos = k_head + kv_pos * head_dim;

        // Compute dot product Q @ K^T for this position
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            float q_val = __bfloat162float(q_head[i]);
            float k_val = __bfloat162float(k_pos[i]);
            score += q_val * k_val;
        }
        score *= scale;

        attn_weights[kv_pos] = score;
        max_score = fmaxf(max_score, score);
    }

    __syncthreads();

    // Reduce max across block
    max_score = attn_block_reduce_max(max_score, reduce_shared);
    __syncthreads();

    // Phase 2: Compute exp(score - max) and sum
    float sum_exp = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < cache_len; kv_pos += blockDim.x) {
        float score = attn_weights[kv_pos];
        float exp_score = expf(score - max_score);
        attn_weights[kv_pos] = exp_score;
        sum_exp += exp_score;
    }

    __syncthreads();

    // Reduce sum across block
    sum_exp = attn_block_reduce_sum(sum_exp, reduce_shared);
    __syncthreads();

    // Phase 3: Normalize and compute weighted sum of V
    float inv_sum = 1.0f / sum_exp;

    // Each thread accumulates its part of the output
    float out_accum[8] = {0.0f};

    for (int kv_pos = 0; kv_pos < cache_len; kv_pos++) {
        float weight = attn_weights[kv_pos] * inv_sum;
        const __nv_bfloat16* v_pos = v_head + kv_pos * head_dim;

        // Accumulate weighted V
        for (int i = 0; i < elems_per_thread; i++) {
            int idx = threadIdx.x * elems_per_thread + i;
            if (idx < head_dim) {
                float v_val = __bfloat162float(v_pos[idx]);
                out_accum[i] += weight * v_val;
            }
        }
    }

    // Store output
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = threadIdx.x * elems_per_thread + i;
        if (idx < head_dim) {
            out_head[idx] = __float2bfloat16(out_accum[i]);
        }
    }
}

/**
 * Attention decode kernel v2 - simpler single-pass approach
 * Each block handles one query head, threads cooperate on dot products
 */
__global__ void attention_decode_kernel_v2(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int cache_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale
) {
    // Shared memory layout:
    // - q_shared: head_dim floats for Q vector
    // - scores: cache_len floats for attention scores
    // - reduce_shared: num_warps floats for reductions
    extern __shared__ float shared[];

    int batch_idx = blockIdx.y;
    int q_head_idx = blockIdx.x;

    // GQA: map query head to kv head
    int n_groups = n_q_heads / n_kv_heads;
    int kv_head_idx = q_head_idx / n_groups;

    // Pointers
    const __nv_bfloat16* q_head = q + (batch_idx * n_q_heads + q_head_idx) * head_dim;
    const __nv_bfloat16* k_head = k_cache + (batch_idx * n_kv_heads + kv_head_idx) * max_seq_len * head_dim;
    const __nv_bfloat16* v_head = v_cache + (batch_idx * n_kv_heads + kv_head_idx) * max_seq_len * head_dim;
    __nv_bfloat16* out_head = out + (batch_idx * n_q_heads + q_head_idx) * head_dim;

    int num_warps = (blockDim.x + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;

    float* q_shared = shared;
    float* scores = shared + head_dim;
    float* reduce_shared = shared + head_dim + cache_len;

    // Load Q into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = __bfloat162float(q_head[i]);
    }
    __syncthreads();

    // Phase 1: Compute all attention scores Q @ K^T
    float local_max = -INFINITY;
    for (int kv_pos = threadIdx.x; kv_pos < cache_len; kv_pos += blockDim.x) {
        const __nv_bfloat16* k_pos = k_head + kv_pos * head_dim;

        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += q_shared[i] * __bfloat162float(k_pos[i]);
        }
        score *= scale;
        scores[kv_pos] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();

    // Reduce to find global max
    float max_score = attn_block_reduce_max(local_max, reduce_shared);
    __syncthreads();

    // Phase 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < cache_len; kv_pos += blockDim.x) {
        float exp_score = expf(scores[kv_pos] - max_score);
        scores[kv_pos] = exp_score;
        local_sum += exp_score;
    }
    __syncthreads();

    float sum_exp = attn_block_reduce_sum(local_sum, reduce_shared);
    __syncthreads();

    float inv_sum = 1.0f / sum_exp;

    // Phase 3: Compute weighted sum of V values
    // Each thread is responsible for a subset of output dimensions
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < cache_len; kv_pos++) {
            float weight = scores[kv_pos] * inv_sum;
            float v_val = __bfloat162float(v_head[kv_pos * head_dim + d]);
            acc += weight * v_val;
        }
        out_head[d] = __float2bfloat16(acc);
    }
}

// Wrapper function callable from PyTorch
extern "C" void launch_attention_decode(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    int batch_size,
    int cache_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale,
    cudaStream_t stream
) {
    // Use 128 threads per block (4 warps)
    int threads = 128;

    // Shared memory: Q (head_dim) + scores (cache_len) + reduce buffer (num_warps)
    int num_warps = (threads + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;
    int shared_mem = (head_dim + cache_len + num_warps) * sizeof(float);

    dim3 grid(n_q_heads, batch_size);

    attention_decode_kernel_v2<<<grid, threads, shared_mem, stream>>>(
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k_cache,
        (const __nv_bfloat16*)v_cache,
        (__nv_bfloat16*)out,
        cache_len,
        n_q_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        scale
    );
}
