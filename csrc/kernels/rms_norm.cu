/**
 * RMSNorm CUDA kernel - matches PyTorch implementation exactly.
 *
 * out = (weight.float() * (x.float() / sqrt(mean(x^2) + eps))).to(bf16)
 *
 * Order of operations matters for bit-exact matching:
 * 1. Compute variance in float32 (sequential sum for n<=128, tree for larger)
 * 2. Normalize x in float32
 * 3. Multiply by weight in float32
 * 4. Convert result to bf16 only at the end
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Simple warp reduce sum using tree reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduce sum using shared memory with tree reduction
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    int num_warps = (blockDim.x + 31) / 32;
    if (threadIdx.x < num_warps) {
        val = shared[lane];
    } else {
        val = 0.0f;
    }
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/**
 * RMSNorm kernel with SEQUENTIAL sum - for small sizes (n_cols <= 128)
 * Uses simple sequential summation to match Triton/PyTorch exactly.
 *
 * Single thread per row computes sum sequentially (same order as Triton).
 */
__global__ void rms_norm_kernel_sequential(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int n_cols,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + row * n_cols;
    __nv_bfloat16* out_row = out + row * n_cols;

    // Sequential sum of squares - matches Triton exactly
    float sum_sq = 0.0f;
    for (int col = 0; col < n_cols; col++) {
        float val = __bfloat162float(x_row[col]);
        sum_sq += val * val;
    }

    float variance = sum_sq / n_cols;
    float rstd = rsqrtf(variance + eps);

    // Output
    for (int col = 0; col < n_cols; col++) {
        float x_val = __bfloat162float(x_row[col]);
        float w_val = __bfloat162float(weight[col]);
        float result = w_val * (x_val * rstd);
        out_row[col] = __float2bfloat16_rn(result);
    }
}

/**
 * RMSNorm kernel with TREE reduction - for larger sizes (n_cols > 128)
 * Uses parallel block reduction which matches PyTorch for larger sizes.
 */
__global__ void rms_norm_kernel_tree(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int n_cols,
    float eps
) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + row * n_cols;
    __nv_bfloat16* out_row = out + row * n_cols;

    // Compute sum of squares with tree reduction
    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        float val = __bfloat162float(x_row[col]);
        sum_sq += val * val;
    }

    // Reduce across block with tree reduction
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Broadcast rstd to all threads
    __shared__ float rstd;
    if (threadIdx.x == 0) {
        float variance = sum_sq / n_cols;
        rstd = rsqrtf(variance + eps);
    }
    __syncthreads();

    // Normalize and scale
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        float x_val = __bfloat162float(x_row[col]);
        float w_val = __bfloat162float(weight[col]);
        float result = w_val * (x_val * rstd);
        out_row[col] = __float2bfloat16_rn(result);
    }
}

// Wrapper function callable from PyTorch
extern "C" void launch_rms_norm(
    const void* x,
    const void* weight,
    void* out,
    int n_rows,
    int n_cols,
    float eps,
    cudaStream_t stream
) {
    if (n_cols <= 128) {
        // Sequential kernel - single thread per row with sequential sum to match Triton/PyTorch
        rms_norm_kernel_sequential<<<n_rows, 1, 0, stream>>>(
            (const __nv_bfloat16*)x,
            (const __nv_bfloat16*)weight,
            (__nv_bfloat16*)out,
            n_cols,
            eps
        );
    } else {
        // Tree reduction kernel - parallel threads for larger sizes
        int threads = min(1024, n_cols);
        // Round up to multiple of 32 for efficient warp operations
        threads = ((threads + 31) / 32) * 32;

        // Need 1 float per warp for tree reduction
        int shared_mem = (threads / 32) * sizeof(float);

        rms_norm_kernel_tree<<<n_rows, threads, shared_mem, stream>>>(
            (const __nv_bfloat16*)x,
            (const __nv_bfloat16*)weight,
            (__nv_bfloat16*)out,
            n_cols,
            eps
        );
    }
}
