/**
 * Fused SiLU * mul CUDA kernel.
 *
 * out = SiLU(gate) * up = (gate * sigmoid(gate)) * up
 *
 * All computation done in float32 for maximum precision, converts to bf16 only at the end.
 * This matches Triton's ordering: (silu(gate.float()) * up.float()).to(bf16)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

/**
 * Fused SiLU(gate) * up kernel
 *
 * @param gate Input gate tensor in bf16
 * @param up Input up tensor in bf16
 * @param out Output tensor in bf16
 * @param n_elements Total number of elements
 */
__global__ void silu_mul_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        // Load gate and up, convert to float32
        float gate_val = __bfloat162float(gate[idx]);
        float up_val = __bfloat162float(up[idx]);

        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) in float32
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
        float silu_gate = gate_val * sigmoid_gate;

        // Keep multiply in float32, then convert to bf16 (matches Triton)
        float result = silu_gate * up_val;
        out[idx] = __float2bfloat16(result);
    }
}

// Vectorized version using bf16x2 for better memory throughput
__global__ void silu_mul_kernel_vec2(
    const __nv_bfloat162* __restrict__ gate,
    const __nv_bfloat162* __restrict__ up,
    __nv_bfloat162* __restrict__ out,
    int n_elements_vec2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements_vec2) {
        // Load bf16x2
        __nv_bfloat162 gate_vec = gate[idx];
        __nv_bfloat162 up_vec = up[idx];

        // Convert gate and up to float
        float gate_lo = __bfloat162float(__low2bfloat16(gate_vec));
        float gate_hi = __bfloat162float(__high2bfloat16(gate_vec));
        float up_lo = __bfloat162float(__low2bfloat16(up_vec));
        float up_hi = __bfloat162float(__high2bfloat16(up_vec));

        // SiLU in float32
        float sigmoid_lo = 1.0f / (1.0f + expf(-gate_lo));
        float sigmoid_hi = 1.0f / (1.0f + expf(-gate_hi));
        float silu_lo = gate_lo * sigmoid_lo;
        float silu_hi = gate_hi * sigmoid_hi;

        // Keep multiply in float32, then convert to bf16 (matches Triton)
        float result_lo = silu_lo * up_lo;
        float result_hi = silu_hi * up_hi;

        // Pack and store
        out[idx] = __halves2bfloat162(__float2bfloat16(result_lo), __float2bfloat16(result_hi));
    }
}

// Wrapper function callable from PyTorch
extern "C" void launch_silu_mul(
    const void* gate,
    const void* up,
    void* out,
    int n_elements,
    cudaStream_t stream
) {
    // Use vectorized version if elements are even
    if (n_elements % 2 == 0) {
        int n_vec2 = n_elements / 2;
        int threads = 256;
        int blocks = (n_vec2 + threads - 1) / threads;

        silu_mul_kernel_vec2<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat162*)gate,
            (const __nv_bfloat162*)up,
            (__nv_bfloat162*)out,
            n_vec2
        );
    } else {
        int threads = 256;
        int blocks = (n_elements + threads - 1) / threads;

        silu_mul_kernel<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)gate,
            (const __nv_bfloat16*)up,
            (__nv_bfloat16*)out,
            n_elements
        );
    }
}
