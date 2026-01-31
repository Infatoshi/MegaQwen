"""Megakernel for Qwen3-0.6B transformer block."""

import os

import torch
from torch.utils.cpp_extension import load_inline

_megakernel = None


def _get_cuda_source(filename: str) -> str:
    """Read CUDA source file."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(kernel_dir, filename)) as f:
        return f.read()


def _compile_megakernel():
    """Compile the megakernel using torch's JIT compilation."""
    global _megakernel

    if _megakernel is not None:
        return _megakernel

    # Read all CUDA sources
    config_src = _get_cuda_source("config.cuh")
    matvec_src = _get_cuda_source("matvec.cuh")
    rmsnorm_src = _get_cuda_source("rmsnorm.cuh")
    rope_src = _get_cuda_source("rope.cuh")
    attention_src = _get_cuda_source("attention.cuh")
    transformer_src = _get_cuda_source("transformer_block.cu")

    # Python bindings
    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void launch_transformer_block(
    const void* hidden_states,
    void* output_states,
    const void* input_layernorm_weight,
    const void* q_proj_weight,
    const void* k_proj_weight,
    const void* v_proj_weight,
    const void* q_norm_weight,
    const void* k_norm_weight,
    const void* o_proj_weight,
    const void* post_attn_layernorm_weight,
    const void* gate_proj_weight,
    const void* up_proj_weight,
    const void* down_proj_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
);

torch::Tensor transformer_block(
    torch::Tensor hidden_states,
    torch::Tensor input_layernorm_weight,
    torch::Tensor q_proj_weight,
    torch::Tensor k_proj_weight,
    torch::Tensor v_proj_weight,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    torch::Tensor o_proj_weight,
    torch::Tensor post_attn_layernorm_weight,
    torch::Tensor gate_proj_weight,
    torch::Tensor up_proj_weight,
    torch::Tensor down_proj_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int position,
    int cache_len,
    int max_seq_len
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA tensor");
    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states must be bfloat16");

    int hidden_size = hidden_states.size(-1);
    int head_dim = 128;
    float attn_scale = 1.0f / sqrtf((float)head_dim);

    auto output = torch::empty_like(hidden_states);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_transformer_block(
        hidden_states.data_ptr(),
        output.data_ptr(),
        input_layernorm_weight.data_ptr(),
        q_proj_weight.data_ptr(),
        k_proj_weight.data_ptr(),
        v_proj_weight.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        o_proj_weight.data_ptr(),
        post_attn_layernorm_weight.data_ptr(),
        gate_proj_weight.data_ptr(),
        up_proj_weight.data_ptr(),
        down_proj_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        stream
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transformer_block", &transformer_block, "Megakernel transformer block");
}
"""

    # Combine CUDA sources (headers first, then implementation)
    # Note: We inline the headers into the main .cu file
    cuda_src = transformer_src

    _megakernel = load_inline(
        name="qwen3_megakernel",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-I" + os.path.dirname(os.path.abspath(__file__)),
        ],
        verbose=True,
    )

    return _megakernel


def get_megakernel():
    """Get compiled megakernel."""
    return _compile_megakernel()
