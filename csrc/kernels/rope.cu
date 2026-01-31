/**
 * RoPE (Rotary Position Embedding) CUDA kernel.
 *
 * Applies rotary position embeddings to Q and K tensors.
 * Uses the "split-half" format: rotate first half with second half.
 *
 * out[..., :half] = x[..., :half] * cos - x[..., half:] * sin
 * out[..., half:] = x[..., half:] * cos + x[..., :half] * sin
 *
 * Computation in float32, output in bf16.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

/**
 * RoPE kernel - processes Q and K in one launch
 *
 * @param q Input Q tensor (batch * n_q_heads * seq_len, head_dim) in bf16
 * @param k Input K tensor (batch * n_kv_heads * seq_len, head_dim) in bf16
 * @param cos Cosine values (seq_len, head_dim) in bf16
 * @param sin Sine values (seq_len, head_dim) in bf16
 * @param q_out Output Q tensor (batch * n_q_heads * seq_len, head_dim) in bf16
 * @param k_out Output K tensor (batch * n_kv_heads * seq_len, head_dim) in bf16
 * @param seq_len Sequence length
 * @param n_q_heads Number of Q heads
 * @param n_kv_heads Number of KV heads
 * @param head_dim Head dimension
 */
__global__ void rope_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim
) {
    // Grid: (seq_len, max(n_q_heads, n_kv_heads), 2)
    // dim 2: 0 = Q, 1 = K
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int is_k = blockIdx.z;  // 0 for Q, 1 for K

    int n_heads = is_k ? n_kv_heads : n_q_heads;
    if (head_idx >= n_heads) return;

    // Compute pointers
    const __nv_bfloat16* input;
    __nv_bfloat16* output;

    if (is_k) {
        int offset = (seq_idx * n_kv_heads + head_idx) * head_dim;
        input = k + offset;
        output = k_out + offset;
    } else {
        int offset = (seq_idx * n_q_heads + head_idx) * head_dim;
        input = q + offset;
        output = q_out + offset;
    }

    // cos/sin for this position
    const __nv_bfloat16* cos_pos = cos + seq_idx * head_dim;
    const __nv_bfloat16* sin_pos = sin + seq_idx * head_dim;

    int half_dim = head_dim / 2;

    // Each thread processes one pair
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        // Load x[..., :half] and x[..., half:]
        float x0 = __bfloat162float(input[i]);
        float x1 = __bfloat162float(input[i + half_dim]);

        // Load cos and sin (already at full head_dim from precompute)
        float c = __bfloat162float(cos_pos[i]);
        float s = __bfloat162float(sin_pos[i]);

        // Apply rotation
        float out0 = x0 * c - x1 * s;
        float out1 = x1 * c + x0 * s;

        // Store as bf16
        output[i] = __float2bfloat16(out0);
        output[i + half_dim] = __float2bfloat16(out1);
    }
}

/**
 * Single-head RoPE kernel for decode (single token)
 * More efficient when processing one token at a time
 */
__global__ void rope_single_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ out,
    int n_heads,
    int head_dim,
    int position  // Position in the sequence (for cos/sin lookup)
) {
    int head_idx = blockIdx.x;
    if (head_idx >= n_heads) return;

    const __nv_bfloat16* x_head = x + head_idx * head_dim;
    __nv_bfloat16* out_head = out + head_idx * head_dim;

    // cos/sin for this position
    const __nv_bfloat16* cos_pos = cos + position * head_dim;
    const __nv_bfloat16* sin_pos = sin + position * head_dim;

    int half_dim = head_dim / 2;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        float x0 = __bfloat162float(x_head[i]);
        float x1 = __bfloat162float(x_head[i + half_dim]);

        float c = __bfloat162float(cos_pos[i]);
        float s = __bfloat162float(sin_pos[i]);

        float out0 = x0 * c - x1 * s;
        float out1 = x1 * c + x0 * s;

        out_head[i] = __float2bfloat16(out0);
        out_head[i + half_dim] = __float2bfloat16(out1);
    }
}

// Wrapper for prefill RoPE (Q and K together)
extern "C" void launch_rope(
    const void* q,
    const void* k,
    const void* cos,
    const void* sin,
    void* q_out,
    void* k_out,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    cudaStream_t stream
) {
    int threads = min(128, head_dim / 2);
    threads = ((threads + 31) / 32) * 32;

    dim3 grid(seq_len, max(n_q_heads, n_kv_heads), 2);

    rope_kernel<<<grid, threads, 0, stream>>>(
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)cos,
        (const __nv_bfloat16*)sin,
        (__nv_bfloat16*)q_out,
        (__nv_bfloat16*)k_out,
        seq_len,
        n_q_heads,
        n_kv_heads,
        head_dim
    );
}

// Wrapper for single-token RoPE (decode phase)
extern "C" void launch_rope_single(
    const void* x,
    const void* cos,
    const void* sin,
    void* out,
    int n_heads,
    int head_dim,
    int position,
    cudaStream_t stream
) {
    int threads = min(128, head_dim / 2);
    threads = ((threads + 31) / 32) * 32;

    rope_single_kernel<<<n_heads, threads, 0, stream>>>(
        (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)cos,
        (const __nv_bfloat16*)sin,
        (__nv_bfloat16*)out,
        n_heads,
        head_dim,
        position
    );
}
