"""CUDA kernels for Qwen3 inference."""

import os

import torch
from torch.utils.cpp_extension import load_inline

_cuda_kernels = None


def _get_cuda_source(filename: str) -> str:
    """Read CUDA source file."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(kernel_dir, filename)) as f:
        return f.read()


def _compile_kernels():
    """Compile CUDA kernels using torch's JIT compilation."""
    global _cuda_kernels

    if _cuda_kernels is not None:
        return _cuda_kernels

    # Read CUDA sources
    rms_norm_src = _get_cuda_source("rms_norm.cu")
    silu_mul_src = _get_cuda_source("silu_mul.cu")
    rope_src = _get_cuda_source("rope.cu")
    attention_decode_src = _get_cuda_source("attention_decode.cu")

    # Combined source with Python bindings
    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Forward declarations
extern "C" void launch_rms_norm(
    const void* x,
    const void* weight,
    void* out,
    int n_rows,
    int n_cols,
    float eps,
    cudaStream_t stream
);

extern "C" void launch_silu_mul(
    const void* gate,
    const void* up,
    void* out,
    int n_elements,
    cudaStream_t stream
);

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
);

extern "C" void launch_rope_single(
    const void* x,
    const void* cos,
    const void* sin,
    void* out,
    int n_heads,
    int head_dim,
    int position,
    cudaStream_t stream
);

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
);

torch::Tensor cuda_rms_norm(torch::Tensor x, torch::Tensor weight, float eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto shape = x.sizes();
    int n_cols = shape[shape.size() - 1];
    int n_rows = x.numel() / n_cols;

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_rms_norm(
        x.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        n_rows,
        n_cols,
        eps,
        stream
    );

    return out;
}

torch::Tensor cuda_silu_mul(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(up.is_cuda(), "up must be a CUDA tensor");
    TORCH_CHECK(gate.dtype() == torch::kBFloat16, "gate must be bfloat16");
    TORCH_CHECK(up.dtype() == torch::kBFloat16, "up must be bfloat16");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have same shape");

    auto out = torch::empty_like(gate);
    int n_elements = gate.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_silu_mul(
        gate.data_ptr(),
        up.data_ptr(),
        out.data_ptr(),
        n_elements,
        stream
    );

    return out;
}

std::tuple<torch::Tensor, torch::Tensor> cuda_rope(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin
) {
    // q: (batch, n_q_heads, seq_len, head_dim)
    // k: (batch, n_kv_heads, seq_len, head_dim)
    // cos, sin: (seq_len, head_dim)
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "q and k must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q and k must be contiguous");

    int batch = q.size(0);
    int n_q_heads = q.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    int n_kv_heads = k.size(1);

    // Reshape for kernel: (batch * n_heads * seq_len, head_dim)
    auto q_flat = q.view({-1, head_dim});
    auto k_flat = k.view({-1, head_dim});

    auto q_out = torch::empty_like(q_flat);
    auto k_out = torch::empty_like(k_flat);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Reshape for kernel input format
    auto q_reshaped = q.permute({0, 2, 1, 3}).contiguous().view({seq_len, batch * n_q_heads, head_dim});
    auto k_reshaped = k.permute({0, 2, 1, 3}).contiguous().view({seq_len, batch * n_kv_heads, head_dim});

    auto q_out_reshaped = torch::empty_like(q_reshaped);
    auto k_out_reshaped = torch::empty_like(k_reshaped);

    launch_rope(
        q_reshaped.data_ptr(),
        k_reshaped.data_ptr(),
        cos.data_ptr(),
        sin.data_ptr(),
        q_out_reshaped.data_ptr(),
        k_out_reshaped.data_ptr(),
        seq_len,
        batch * n_q_heads,
        batch * n_kv_heads,
        head_dim,
        stream
    );

    // Reshape back to original format
    q_out = q_out_reshaped.view({seq_len, batch, n_q_heads, head_dim}).permute({1, 2, 0, 3}).contiguous();
    k_out = k_out_reshaped.view({seq_len, batch, n_kv_heads, head_dim}).permute({1, 2, 0, 3}).contiguous();

    return std::make_tuple(q_out, k_out);
}

torch::Tensor cuda_rope_single(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin,
    int position
) {
    // x: (n_heads, head_dim)
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int n_heads = x.size(0);
    int head_dim = x.size(1);

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_rope_single(
        x.data_ptr(),
        cos.data_ptr(),
        sin.data_ptr(),
        out.data_ptr(),
        n_heads,
        head_dim,
        position,
        stream
    );

    return out;
}

torch::Tensor cuda_attention_decode(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int cache_len
) {
    // q: (batch, n_q_heads, 1, head_dim)
    // k_cache, v_cache: (batch, n_kv_heads, max_seq_len, head_dim)
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");

    int batch = q.size(0);
    int n_q_heads = q.size(1);
    int head_dim = q.size(3);
    int n_kv_heads = k_cache.size(1);
    int max_seq_len = k_cache.size(2);

    float scale = 1.0f / sqrtf((float)head_dim);

    // Flatten q for kernel: (batch, n_q_heads, head_dim)
    auto q_flat = q.squeeze(2).contiguous();
    auto out = torch::empty_like(q_flat);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_attention_decode(
        q_flat.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        out.data_ptr(),
        batch,
        cache_len,
        n_q_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        scale,
        stream
    );

    // Reshape back to (batch, n_q_heads, 1, head_dim)
    return out.unsqueeze(2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &cuda_rms_norm, "CUDA RMSNorm");
    m.def("silu_mul", &cuda_silu_mul, "CUDA fused SiLU * mul");
    m.def("rope", &cuda_rope, "CUDA RoPE (prefill)");
    m.def("rope_single", &cuda_rope_single, "CUDA RoPE (decode single token)");
    m.def("attention_decode", &cuda_attention_decode, "CUDA attention decode");
}
"""

    combined_cuda_src = rms_norm_src + "\n\n" + silu_mul_src + "\n\n" + rope_src + "\n\n" + attention_decode_src

    _cuda_kernels = load_inline(
        name="qwen3_cuda_kernels",
        cpp_sources=[cpp_src],
        cuda_sources=[combined_cuda_src],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

    return _cuda_kernels


def get_kernels():
    """Get compiled CUDA kernels."""
    return _compile_kernels()
