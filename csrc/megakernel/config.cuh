#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// =============================================================================
// Tunable Parameters
// =============================================================================

// Weight tiling for matvec
constexpr int TILE_ROWS = 16;           // Output elements per weight tile
constexpr int TILE_COLS = 1024;         // Input dimension (full hidden_size)

// Pipelining
constexpr int NUM_PIPELINE_STAGES = 3;  // Weight buffer stages

// Thread organization
constexpr int BLOCK_SIZE = 1024;        // 32 warps
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// Attention
constexpr int KV_BLOCK_SIZE = 64;       // KV cache positions per iteration

// =============================================================================
// Model Dimensions (Qwen3-0.6B)
// =============================================================================

constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;      // 2048
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM;    // 1024
constexpr float RMS_NORM_EPS = 1e-6f;

// =============================================================================
// Derived Constants
// =============================================================================

constexpr int ELEMENTS_PER_THREAD = TILE_COLS / BLOCK_SIZE;  // 4 for 1024/256
constexpr int TILE_SIZE_BYTES = TILE_ROWS * TILE_COLS * sizeof(__nv_bfloat16);

// Shared memory layout sizes
constexpr int WEIGHT_BUFFER_SIZE = NUM_PIPELINE_STAGES * TILE_SIZE_BYTES;
constexpr int ACTIVATION_BUFFER_SIZE = HIDDEN_SIZE * sizeof(float);
constexpr int REDUCTION_BUFFER_SIZE = NUM_WARPS * TILE_ROWS * sizeof(float);

// =============================================================================
// Utility Functions
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (wid == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();

    // Broadcast result
    if (wid == 0 && lane == 0) {
        shared[0] = val;
    }
    __syncthreads();

    return shared[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
    }
    __syncthreads();

    if (wid == 0 && lane == 0) {
        shared[0] = val;
    }
    __syncthreads();

    return shared[0];
}

// =============================================================================
// Async Copy Helpers (cp.async)
// =============================================================================

__device__ __forceinline__ void cp_async_cg(void* dst, const void* src, int bytes) {
    // Copy 16 bytes (128 bits) at a time using cp.async.cg
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(dst)),
          "l"(src),
          "n"(16)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
    if (n == 0) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else if (n == 1) {
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else if (n == 2) {
        asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    }
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}
