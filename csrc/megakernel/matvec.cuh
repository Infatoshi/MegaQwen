#pragma once

#include "config.cuh"

// =============================================================================
// Pipelined Matrix-Vector Multiplication
// =============================================================================
//
// Computes: out[M] = weights[M, K] @ activations[K]
//
// Strategy:
// - Stream weight tiles (TILE_ROWS x K) through shared memory
// - Keep activations in shared memory (loaded once)
// - Each tile produces TILE_ROWS output elements
// - Triple-buffer weight tiles for async load/compute overlap
//
// Thread assignment for matvec:
// - Each thread handles ELEMENTS_PER_THREAD elements of the dot product
// - Warp reduces partial sums
// - Block reduces across warps for final output
// =============================================================================

// Shared memory structure for pipelined matvec
struct MatvecSmem {
    // Weight tiles: NUM_PIPELINE_STAGES buffers of TILE_ROWS x TILE_COLS
    __nv_bfloat16 weight_tiles[NUM_PIPELINE_STAGES][TILE_ROWS][TILE_COLS];

    // Activations: single buffer, loaded once
    float activations[HIDDEN_SIZE];

    // Reduction scratch: per-warp partial sums for each output row
    float reduction[NUM_WARPS][TILE_ROWS];

    // Block-level scratch
    float scratch[NUM_WARPS];
};

// -----------------------------------------------------------------------------
// Load activation vector into shared memory
// -----------------------------------------------------------------------------
__device__ __forceinline__ void load_activations_to_smem(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ smem_activations,
    int size
) {
    // Each thread loads multiple elements
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE) {
        smem_activations[i] = __bfloat162float(input[i]);
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Async load one weight tile into shared memory
// -----------------------------------------------------------------------------
__device__ __forceinline__ void async_load_weight_tile(
    __nv_bfloat16* __restrict__ smem_tile,  // [TILE_ROWS][TILE_COLS]
    const __nv_bfloat16* __restrict__ weights,
    int tile_idx,
    int total_rows,
    int K
) {
    // weights layout: [M, K] row-major
    // tile covers rows [tile_idx * TILE_ROWS, (tile_idx + 1) * TILE_ROWS)

    int row_start = tile_idx * TILE_ROWS;
    int elements_per_tile = TILE_ROWS * K;

    // Each thread loads multiple 16-byte chunks
    // 16 bytes = 8 bf16 elements
    constexpr int CHUNK_SIZE = 8;
    int chunks_per_tile = elements_per_tile / CHUNK_SIZE;

    for (int chunk = threadIdx.x; chunk < chunks_per_tile; chunk += BLOCK_SIZE) {
        int elem_idx = chunk * CHUNK_SIZE;
        int row = elem_idx / K;
        int col = elem_idx % K;

        if (row_start + row < total_rows) {
            const void* src = &weights[(row_start + row) * K + col];
            void* dst = &smem_tile[row * K + col];

            // Use cp.async for 16-byte copy
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(dst)),
                  "l"(src)
                : "memory"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Compute matvec for one tile: TILE_ROWS outputs from TILE_ROWS x K weights
// -----------------------------------------------------------------------------
__device__ __forceinline__ void compute_tile_matvec(
    const __nv_bfloat16* __restrict__ smem_weights,  // [TILE_ROWS][K]
    const float* __restrict__ smem_activations,       // [K]
    float* __restrict__ smem_reduction,               // [NUM_WARPS][TILE_ROWS]
    float* __restrict__ output,                       // [TILE_ROWS] partial outputs
    int K
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Each warp processes all TILE_ROWS, with threads splitting the K dimension
    // Elements per thread for the dot product
    int elems_per_thread = K / BLOCK_SIZE;
    int start_k = threadIdx.x * elems_per_thread;

    // Process each output row
    #pragma unroll
    for (int row = 0; row < TILE_ROWS; row++) {
        float sum = 0.0f;

        // Each thread accumulates its portion of the dot product
        #pragma unroll 4
        for (int k = 0; k < elems_per_thread; k++) {
            int k_idx = start_k + k;
            float w = __bfloat162float(smem_weights[row * K + k_idx]);
            float a = smem_activations[k_idx];
            sum += w * a;
        }

        // Warp-level reduction
        sum = warp_reduce_sum(sum);

        // Lane 0 of each warp writes partial sum
        if (lane_id == 0) {
            smem_reduction[warp_id * TILE_ROWS + row] = sum;
        }
    }

    __syncthreads();

    // Final reduction across warps - first warp handles this
    if (warp_id == 0 && lane_id < TILE_ROWS) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            sum += smem_reduction[w * TILE_ROWS + lane_id];
        }
        output[lane_id] = sum;
    }

    __syncthreads();
}

// -----------------------------------------------------------------------------
// Full pipelined matvec: out[M] = weights[M, K] @ activations[K]
// -----------------------------------------------------------------------------
//
// Assumes activations are already in smem_activations
// Output is written to smem_output (or can be kept for further processing)
//
__device__ void pipelined_matvec(
    const __nv_bfloat16* __restrict__ weights,  // [M, K] in global memory
    float* __restrict__ smem_activations,        // [K] in shared memory
    __nv_bfloat16* __restrict__ smem_weight_tiles,  // [NUM_PIPELINE_STAGES][TILE_ROWS][K]
    float* __restrict__ smem_reduction,          // [NUM_WARPS * TILE_ROWS]
    float* __restrict__ output,                  // [M] output buffer
    int M,
    int K
) {
    int num_tiles = (M + TILE_ROWS - 1) / TILE_ROWS;
    int tile_stride = TILE_ROWS * K;

    // Pipeline prologue: load first tiles
    int tiles_to_prefetch = min(NUM_PIPELINE_STAGES, num_tiles);

    for (int t = 0; t < tiles_to_prefetch; t++) {
        __nv_bfloat16* tile_ptr = smem_weight_tiles + t * tile_stride;
        async_load_weight_tile(tile_ptr, weights, t, M, K);
        cp_async_commit();
    }

    // Main loop: compute tile t while loading tile t + NUM_PIPELINE_STAGES
    for (int t = 0; t < num_tiles; t++) {
        int stage = t % NUM_PIPELINE_STAGES;

        // Wait for tile t to be ready
        if (t < NUM_PIPELINE_STAGES) {
            cp_async_wait_group(NUM_PIPELINE_STAGES - 1 - t);
        } else {
            cp_async_wait_group(NUM_PIPELINE_STAGES - 1);
        }
        __syncthreads();

        // Start loading next tile if there are more
        int next_tile = t + NUM_PIPELINE_STAGES;
        if (next_tile < num_tiles) {
            int next_stage = next_tile % NUM_PIPELINE_STAGES;
            __nv_bfloat16* next_tile_ptr = smem_weight_tiles + next_stage * tile_stride;
            async_load_weight_tile(next_tile_ptr, weights, next_tile, M, K);
            cp_async_commit();
        }

        // Compute current tile
        __nv_bfloat16* current_tile = smem_weight_tiles + stage * tile_stride;
        float tile_output[TILE_ROWS];

        compute_tile_matvec(
            current_tile,
            smem_activations,
            smem_reduction,
            tile_output,
            K
        );

        // Write tile outputs to output buffer
        int out_start = t * TILE_ROWS;
        if (threadIdx.x < TILE_ROWS && out_start + threadIdx.x < M) {
            output[out_start + threadIdx.x] = tile_output[threadIdx.x];
        }

        __syncthreads();
    }

    // Ensure all async copies complete
    cp_async_wait_all();
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Variant that keeps output in shared memory for fusion
// -----------------------------------------------------------------------------
__device__ void pipelined_matvec_to_smem(
    const __nv_bfloat16* __restrict__ weights,
    float* __restrict__ smem_activations,
    __nv_bfloat16* __restrict__ smem_weight_tiles,
    float* __restrict__ smem_reduction,
    float* __restrict__ smem_output,  // Output stays in shared memory
    int M,
    int K
) {
    int num_tiles = (M + TILE_ROWS - 1) / TILE_ROWS;
    int tile_stride = TILE_ROWS * K;

    // Pipeline prologue
    int tiles_to_prefetch = min(NUM_PIPELINE_STAGES, num_tiles);

    for (int t = 0; t < tiles_to_prefetch; t++) {
        __nv_bfloat16* tile_ptr = smem_weight_tiles + t * tile_stride;
        async_load_weight_tile(tile_ptr, weights, t, M, K);
        cp_async_commit();
    }

    // Main loop
    for (int t = 0; t < num_tiles; t++) {
        int stage = t % NUM_PIPELINE_STAGES;

        if (t < NUM_PIPELINE_STAGES) {
            cp_async_wait_group(NUM_PIPELINE_STAGES - 1 - t);
        } else {
            cp_async_wait_group(NUM_PIPELINE_STAGES - 1);
        }
        __syncthreads();

        int next_tile = t + NUM_PIPELINE_STAGES;
        if (next_tile < num_tiles) {
            int next_stage = next_tile % NUM_PIPELINE_STAGES;
            __nv_bfloat16* next_tile_ptr = smem_weight_tiles + next_stage * tile_stride;
            async_load_weight_tile(next_tile_ptr, weights, next_tile, M, K);
            cp_async_commit();
        }

        __nv_bfloat16* current_tile = smem_weight_tiles + stage * tile_stride;

        // Inline tile computation to avoid extra buffer
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        int elems_per_thread = K / BLOCK_SIZE;
        int start_k = threadIdx.x * elems_per_thread;

        #pragma unroll
        for (int row = 0; row < TILE_ROWS; row++) {
            float sum = 0.0f;

            #pragma unroll 4
            for (int k = 0; k < elems_per_thread; k++) {
                int k_idx = start_k + k;
                float w = __bfloat162float(current_tile[row * K + k_idx]);
                float a = smem_activations[k_idx];
                sum += w * a;
            }

            sum = warp_reduce_sum(sum);

            if (lane_id == 0) {
                smem_reduction[warp_id * TILE_ROWS + row] = sum;
            }
        }

        __syncthreads();

        // Final reduction and write to smem_output
        int out_start = t * TILE_ROWS;
        if (warp_id == 0 && lane_id < TILE_ROWS && out_start + lane_id < M) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                sum += smem_reduction[w * TILE_ROWS + lane_id];
            }
            smem_output[out_start + lane_id] = sum;
        }

        __syncthreads();
    }

    cp_async_wait_all();
    __syncthreads();
}
