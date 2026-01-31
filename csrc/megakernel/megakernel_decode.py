"""
Megakernel decode module for Qwen3-0.6B.
Wraps the fused_decode_ldg.cu kernel with Python bindings.
Includes prefill support using fused_prefill.cu with cuBLAS.
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

_decode_kernel = None
_prefill_kernel = None
_fused_prefill_kernel = None

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
NUM_LAYERS = 28
VOCAB_SIZE = 151936


def _get_cuda_source(filename: str) -> str:
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(kernel_dir, filename)) as f:
        return f.read()


def _compile_decode_kernel():
    global _decode_kernel
    if _decode_kernel is not None:
        return _decode_kernel

    cuda_src = _get_cuda_source("fused_decode_ldg.cu")

    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Must match the struct in fused_decode_ldg.cu
struct LDGLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_ldg_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
);

class MegakernelDecoder {
public:
    MegakernelDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len) {

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;

        // Store layer weights
        layer_weights_tensors_ = layer_weights_flat;

        // Build layer weights structs
        layer_weights_.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();
        }

        // Copy layer weights to device
        d_layer_weights_ = torch::empty({num_layers * (int)sizeof(LDGLayerWeights)},
                                         torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_weights_.data_ptr(), layer_weights_.data(),
                   num_layers * sizeof(LDGLayerWeights), cudaMemcpyHostToDevice);

        // Allocate KV cache
        int kv_heads = 8;
        int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // Allocate intermediate buffers
        int hidden_size = 1024;
        int q_size = 16 * 128;
        int kv_size = 8 * 128;
        int intermediate_size = 3072;

        hidden_buffer_ = torch::empty({hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        g_activations_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_residual_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_q_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_k_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_v_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_attn_out_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_mlp_intermediate_ = torch::empty({intermediate_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_normalized_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        output_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

        position_ = 0;
        attn_scale_ = 1.0f / sqrtf(128.0f);
    }

    int decode_step(int input_token_id) {
        int cache_len = position_ + 1;

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_ldg_decode(
            input_token_id,
            (int*)output_token_.data_ptr(),
            embed_weight_.data_ptr(),
            (const LDGLayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            position_,
            cache_len,
            max_seq_len_,
            attn_scale_,
            stream
        );

        position_++;
        return output_token_.item<int>();
    }

    void reset() {
        position_ = 0;
        k_cache_.zero_();
        v_cache_.zero_();
    }

    int position() const { return position_; }

private:
    int num_layers_;
    int max_seq_len_;
    int position_;
    float attn_scale_;

    torch::Tensor embed_weight_;
    torch::Tensor final_norm_weight_;
    torch::Tensor lm_head_weight_;
    torch::Tensor cos_table_;
    torch::Tensor sin_table_;
    torch::Tensor d_layer_weights_;

    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<LDGLayerWeights> layer_weights_;

    torch::Tensor k_cache_, v_cache_;
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;
    torch::Tensor block_max_vals_, block_max_idxs_, output_token_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelDecoder>(m, "MegakernelDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step", &MegakernelDecoder::decode_step)
        .def("reset", &MegakernelDecoder::reset)
        .def("position", &MegakernelDecoder::position);
}
"""

    kernel_dir = os.path.dirname(os.path.abspath(__file__))

    _decode_kernel = load_inline(
        name="megakernel_decode",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-arch=sm_86",
            "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
        ],
        verbose=False,
    )

    return _decode_kernel


def _compile_prefill_kernel():
    """Compile the prefill kernel with cuBLAS support."""
    global _prefill_kernel
    if _prefill_kernel is not None:
        return _prefill_kernel

    cuda_src = _get_cuda_source("fused_prefill.cu")

    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <c10/cuda/CUDAStream.h>

// Must match the struct in fused_prefill.cu
struct PrefillLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_prefill_float(
    const int* input_token_ids,
    int* output_token_id,
    int seq_len,
    const void* embed_weight,
    const PrefillLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_float,
    void* residual,
    void* normalized,
    void* q_proj,
    void* k_proj,
    void* v_proj,
    void* attn_out,
    void* o_proj_out,
    void* mlp_norm,
    void* gate_out,
    void* up_out,
    void* mlp_intermediate,
    void* down_out,
    void* final_hidden,
    void* block_max_vals,
    void* block_max_idxs,
    void* hidden_bf16_out,
    int num_layers,
    int max_seq_len,
    float attn_scale,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
);

// For decode continuation after prefill
struct LDGLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_ldg_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
);

class MegakernelPrefillDecoder {
public:
    MegakernelPrefillDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len,
        int max_prefill_len = 512
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len), max_prefill_len_(max_prefill_len) {

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;

        layer_weights_tensors_ = layer_weights_flat;

        // Build layer weights structs (same layout for both prefill and decode)
        prefill_layer_weights_.resize(num_layers);
        decode_layer_weights_.resize(num_layers);

        for (int i = 0; i < num_layers; i++) {
            prefill_layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            prefill_layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            prefill_layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            prefill_layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            prefill_layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            prefill_layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            prefill_layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            prefill_layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            prefill_layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            prefill_layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            prefill_layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();

            // Decode uses same layout
            decode_layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            decode_layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            decode_layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            decode_layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            decode_layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            decode_layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            decode_layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            decode_layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            decode_layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            decode_layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            decode_layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();
        }

        // Copy prefill layer weights to device
        d_prefill_layer_weights_ = torch::empty({num_layers * (int)sizeof(PrefillLayerWeights)},
                                                 torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_prefill_layer_weights_.data_ptr(), prefill_layer_weights_.data(),
                   num_layers * sizeof(PrefillLayerWeights), cudaMemcpyHostToDevice);

        // Copy decode layer weights to device
        d_decode_layer_weights_ = torch::empty({num_layers * (int)sizeof(LDGLayerWeights)},
                                                torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_decode_layer_weights_.data_ptr(), decode_layer_weights_.data(),
                   num_layers * sizeof(LDGLayerWeights), cudaMemcpyHostToDevice);

        // Create cuBLAS handle
        cublasCreate(&cublas_handle_);

        // Allocate KV cache
        int kv_heads = 8;
        int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // Allocate prefill buffers (sized for max_prefill_len) - all BF16 for cuBLAS
        int hidden_size = 1024;
        int q_size = 16 * 128;
        int kv_size = 8 * 128;
        int intermediate_size = 3072;

        hidden_float_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        residual_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        normalized_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        q_proj_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        k_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        attn_out_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        o_proj_out_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        mlp_norm_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        gate_out_ = torch::empty({max_prefill_len, intermediate_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        up_out_ = torch::empty({max_prefill_len, intermediate_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        mlp_intermediate_ = torch::empty({max_prefill_len, intermediate_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        down_out_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        final_hidden_ = torch::empty({hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // Decode buffers (single token)
        hidden_buffer_ = torch::empty({hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        g_activations_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_residual_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_q_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_k_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_v_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_attn_out_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_mlp_intermediate_ = torch::empty({intermediate_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_normalized_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        output_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        input_tokens_ = torch::empty({max_prefill_len}, torch::dtype(torch::kInt32).device(torch::kCUDA));

        position_ = 0;
        attn_scale_ = 1.0f / sqrtf(128.0f);
    }

    ~MegakernelPrefillDecoder() {
        cublasDestroy(cublas_handle_);
    }

    int prefill_step(torch::Tensor input_token_ids) {
        // input_token_ids: 1D tensor of token IDs [seq_len]
        int seq_len = input_token_ids.size(0);
        if (seq_len > max_prefill_len_) {
            throw std::runtime_error("Prefill sequence length exceeds maximum");
        }
        if (seq_len == 0) {
            throw std::runtime_error("Empty input sequence");
        }

        // Copy input tokens to device
        input_tokens_.narrow(0, 0, seq_len).copy_(input_token_ids.to(torch::kCUDA));

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_prefill_float(
            (const int*)input_tokens_.data_ptr(),
            (int*)output_token_.data_ptr(),
            seq_len,
            embed_weight_.data_ptr(),
            prefill_layer_weights_.data(),  // Use HOST pointer, not device
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_float_.data_ptr(),
            residual_.data_ptr(),
            normalized_.data_ptr(),
            q_proj_.data_ptr(),
            k_proj_.data_ptr(),
            v_proj_.data_ptr(),
            attn_out_.data_ptr(),
            o_proj_out_.data_ptr(),
            mlp_norm_.data_ptr(),
            gate_out_.data_ptr(),
            up_out_.data_ptr(),
            mlp_intermediate_.data_ptr(),
            down_out_.data_ptr(),
            final_hidden_.data_ptr(),
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            hidden_buffer_.data_ptr(),  // bf16 output for decode continuation
            num_layers_,
            max_seq_len_,
            attn_scale_,
            cublas_handle_,
            stream
        );

        position_ = seq_len;
        return output_token_.item<int>();
    }

    int decode_step(int input_token_id) {
        int cache_len = position_ + 1;

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_ldg_decode(
            input_token_id,
            (int*)output_token_.data_ptr(),
            embed_weight_.data_ptr(),
            (const LDGLayerWeights*)d_decode_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            position_,
            cache_len,
            max_seq_len_,
            attn_scale_,
            stream
        );

        position_++;
        return output_token_.item<int>();
    }

    void reset() {
        position_ = 0;
        k_cache_.zero_();
        v_cache_.zero_();
    }

    int position() const { return position_; }

    torch::Tensor get_k_cache() const { return k_cache_; }
    torch::Tensor get_v_cache() const { return v_cache_; }

private:
    int num_layers_;
    int max_seq_len_;
    int max_prefill_len_;
    int position_;
    float attn_scale_;

    cublasHandle_t cublas_handle_;

    torch::Tensor embed_weight_;
    torch::Tensor final_norm_weight_;
    torch::Tensor lm_head_weight_;
    torch::Tensor cos_table_;
    torch::Tensor sin_table_;
    torch::Tensor d_prefill_layer_weights_;
    torch::Tensor d_decode_layer_weights_;

    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<PrefillLayerWeights> prefill_layer_weights_;
    std::vector<LDGLayerWeights> decode_layer_weights_;

    torch::Tensor k_cache_, v_cache_;

    // Prefill buffers
    torch::Tensor hidden_float_, residual_, normalized_;
    torch::Tensor q_proj_, k_proj_, v_proj_;
    torch::Tensor attn_out_, o_proj_out_, mlp_norm_;
    torch::Tensor gate_out_, up_out_, mlp_intermediate_, down_out_;
    torch::Tensor final_hidden_;
    torch::Tensor input_tokens_;

    // Decode buffers
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;

    // Shared
    torch::Tensor block_max_vals_, block_max_idxs_, output_token_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelPrefillDecoder>(m, "MegakernelPrefillDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>(),
             py::arg("embed_weight"),
             py::arg("layer_weights_flat"),
             py::arg("final_norm_weight"),
             py::arg("lm_head_weight"),
             py::arg("cos_table"),
             py::arg("sin_table"),
             py::arg("num_layers"),
             py::arg("max_seq_len"),
             py::arg("max_prefill_len") = 512)
        .def("prefill_step", &MegakernelPrefillDecoder::prefill_step)
        .def("decode_step", &MegakernelPrefillDecoder::decode_step)
        .def("reset", &MegakernelPrefillDecoder::reset)
        .def("position", &MegakernelPrefillDecoder::position)
        .def("get_k_cache", &MegakernelPrefillDecoder::get_k_cache)
        .def("get_v_cache", &MegakernelPrefillDecoder::get_v_cache);
}
"""

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    decode_cuda_src = _get_cuda_source("fused_decode_ldg.cu")

    _prefill_kernel = load_inline(
        name="megakernel_prefill",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src, decode_cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-arch=sm_86",
            "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
        ],
        extra_ldflags=["-lcublas"],
        verbose=False,
    )

    return _prefill_kernel


def _compile_fused_prefill_kernel():
    """Compile the fused prefill megakernel (no cuBLAS)."""
    global _fused_prefill_kernel
    if _fused_prefill_kernel is not None:
        return _fused_prefill_kernel

    cuda_src = _get_cuda_source("fused_prefill_megakernel.cu")

    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Must match the struct in fused_prefill_megakernel.cu
struct PrefillMKLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

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
);

// For decode continuation after prefill
struct LDGLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_ldg_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
);

class MegakernelFusedPrefillDecoder {
public:
    MegakernelFusedPrefillDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len,
        int max_prefill_len = 64
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len), max_prefill_len_(max_prefill_len) {

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;

        layer_weights_tensors_ = layer_weights_flat;

        // Build layer weights structs
        prefill_layer_weights_.resize(num_layers);
        decode_layer_weights_.resize(num_layers);

        for (int i = 0; i < num_layers; i++) {
            prefill_layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            prefill_layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            prefill_layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            prefill_layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            prefill_layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            prefill_layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            prefill_layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            prefill_layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            prefill_layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            prefill_layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            prefill_layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();

            // Decode uses same layout
            decode_layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            decode_layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            decode_layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            decode_layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            decode_layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            decode_layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            decode_layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            decode_layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            decode_layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            decode_layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            decode_layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();
        }

        // Copy prefill layer weights to device
        d_prefill_layer_weights_ = torch::empty({num_layers * (int)sizeof(PrefillMKLayerWeights)},
                                                 torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_prefill_layer_weights_.data_ptr(), prefill_layer_weights_.data(),
                   num_layers * sizeof(PrefillMKLayerWeights), cudaMemcpyHostToDevice);

        // Copy decode layer weights to device
        d_decode_layer_weights_ = torch::empty({num_layers * (int)sizeof(LDGLayerWeights)},
                                                torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_decode_layer_weights_.data_ptr(), decode_layer_weights_.data(),
                   num_layers * sizeof(LDGLayerWeights), cudaMemcpyHostToDevice);

        // Allocate KV cache
        int kv_heads = 8;
        int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // Allocate prefill buffers (all float32 for megakernel)
        int hidden_size = 1024;
        int q_size = 16 * 128;
        int kv_size = 8 * 128;
        int intermediate_size = 3072;

        hidden_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        residual_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        normalized_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        q_proj_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        k_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        v_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_out_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        mlp_intermediate_ = torch::empty({max_prefill_len, intermediate_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        final_hidden_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        // Decode buffers (single token)
        hidden_buffer_ = torch::empty({hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        g_activations_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_residual_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_q_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_k_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_v_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_attn_out_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_mlp_intermediate_ = torch::empty({intermediate_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_normalized_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        output_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        input_tokens_ = torch::empty({max_prefill_len}, torch::dtype(torch::kInt32).device(torch::kCUDA));

        position_ = 0;
        attn_scale_ = 1.0f / sqrtf(128.0f);
    }

    int prefill_step(torch::Tensor input_token_ids) {
        int seq_len = input_token_ids.size(0);
        if (seq_len > max_prefill_len_) {
            throw std::runtime_error("Prefill sequence length exceeds maximum");
        }
        if (seq_len == 0) {
            throw std::runtime_error("Empty input sequence");
        }

        // Copy input tokens to device
        input_tokens_.narrow(0, 0, seq_len).copy_(input_token_ids.to(torch::kCUDA));

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_prefill_megakernel(
            (const int*)input_tokens_.data_ptr(),
            (int*)output_token_.data_ptr(),
            seq_len,
            embed_weight_.data_ptr(),
            (const PrefillMKLayerWeights*)d_prefill_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_.data_ptr(),
            residual_.data_ptr(),
            normalized_.data_ptr(),
            q_proj_.data_ptr(),
            k_proj_.data_ptr(),
            v_proj_.data_ptr(),
            attn_out_.data_ptr(),
            mlp_intermediate_.data_ptr(),
            final_hidden_.data_ptr(),
            hidden_buffer_.data_ptr(),  // bf16 output for decode continuation
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            max_seq_len_,
            attn_scale_,
            stream
        );

        position_ = seq_len;
        return output_token_.item<int>();
    }

    int decode_step(int input_token_id) {
        int cache_len = position_ + 1;

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_ldg_decode(
            input_token_id,
            (int*)output_token_.data_ptr(),
            embed_weight_.data_ptr(),
            (const LDGLayerWeights*)d_decode_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            position_,
            cache_len,
            max_seq_len_,
            attn_scale_,
            stream
        );

        position_++;
        return output_token_.item<int>();
    }

    void reset() {
        position_ = 0;
        k_cache_.zero_();
        v_cache_.zero_();
    }

    int position() const { return position_; }
    int max_prefill_len() const { return max_prefill_len_; }

    torch::Tensor get_k_cache() const { return k_cache_; }
    torch::Tensor get_v_cache() const { return v_cache_; }

private:
    int num_layers_;
    int max_seq_len_;
    int max_prefill_len_;
    int position_;
    float attn_scale_;

    torch::Tensor embed_weight_;
    torch::Tensor final_norm_weight_;
    torch::Tensor lm_head_weight_;
    torch::Tensor cos_table_;
    torch::Tensor sin_table_;
    torch::Tensor d_prefill_layer_weights_;
    torch::Tensor d_decode_layer_weights_;

    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<PrefillMKLayerWeights> prefill_layer_weights_;
    std::vector<LDGLayerWeights> decode_layer_weights_;

    torch::Tensor k_cache_, v_cache_;

    // Prefill buffers (float32)
    torch::Tensor hidden_, residual_, normalized_;
    torch::Tensor q_proj_, k_proj_, v_proj_;
    torch::Tensor attn_out_, mlp_intermediate_;
    torch::Tensor final_hidden_;
    torch::Tensor input_tokens_;

    // Decode buffers
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;

    // Shared
    torch::Tensor block_max_vals_, block_max_idxs_, output_token_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelFusedPrefillDecoder>(m, "MegakernelFusedPrefillDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>(),
             py::arg("embed_weight"),
             py::arg("layer_weights_flat"),
             py::arg("final_norm_weight"),
             py::arg("lm_head_weight"),
             py::arg("cos_table"),
             py::arg("sin_table"),
             py::arg("num_layers"),
             py::arg("max_seq_len"),
             py::arg("max_prefill_len") = 64)
        .def("prefill_step", &MegakernelFusedPrefillDecoder::prefill_step)
        .def("decode_step", &MegakernelFusedPrefillDecoder::decode_step)
        .def("reset", &MegakernelFusedPrefillDecoder::reset)
        .def("position", &MegakernelFusedPrefillDecoder::position)
        .def("max_prefill_len", &MegakernelFusedPrefillDecoder::max_prefill_len)
        .def("get_k_cache", &MegakernelFusedPrefillDecoder::get_k_cache)
        .def("get_v_cache", &MegakernelFusedPrefillDecoder::get_v_cache);
}
"""

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    decode_cuda_src = _get_cuda_source("fused_decode_ldg.cu")

    _fused_prefill_kernel = load_inline(
        name="megakernel_fused_prefill",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src, decode_cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-arch=sm_86",
            "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
        ],
        verbose=False,
    )

    return _fused_prefill_kernel


def load_qwen3_weights(model_name="Qwen/Qwen3-0.6B"):
    """Load Qwen3-0.6B weights from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Extract weights
    state = model.state_dict()

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    final_norm_weight = state["model.norm.weight"].contiguous()
    lm_head_weight = state["lm_head.weight"].contiguous()

    # Build RoPE tables
    max_seq_len = 2048
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    # Extract layer weights
    layer_weights = []
    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}."
        layer_weights.extend([
            state[prefix + "input_layernorm.weight"].contiguous(),
            state[prefix + "self_attn.q_proj.weight"].contiguous(),
            state[prefix + "self_attn.k_proj.weight"].contiguous(),
            state[prefix + "self_attn.v_proj.weight"].contiguous(),
            state[prefix + "self_attn.q_norm.weight"].contiguous(),
            state[prefix + "self_attn.k_norm.weight"].contiguous(),
            state[prefix + "self_attn.o_proj.weight"].contiguous(),
            state[prefix + "post_attention_layernorm.weight"].contiguous(),
            state[prefix + "mlp.gate_proj.weight"].contiguous(),
            state[prefix + "mlp.up_proj.weight"].contiguous(),
            state[prefix + "mlp.down_proj.weight"].contiguous(),
        ])

    del model
    torch.cuda.empty_cache()

    return {
        "embed_weight": embed_weight,
        "layer_weights": layer_weights,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head_weight,
        "cos_table": cos_table,
        "sin_table": sin_table,
        "tokenizer": tokenizer,
    }


class MegakernelGenerator:
    """High-level generator using the megakernel."""

    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_seq_len=2048):
        weights = load_qwen3_weights(model_name)

        kernel = _compile_decode_kernel()

        self.decoder = kernel.MegakernelDecoder(
            weights["embed_weight"],
            weights["layer_weights"],
            weights["final_norm_weight"],
            weights["lm_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            NUM_LAYERS,
            max_seq_len,
        )

        self.tokenizer = weights["tokenizer"]
        self.max_seq_len = max_seq_len

    def generate(self, prompt, max_new_tokens=100, temperature=1.0, stop_tokens=None):
        """Generate text from a prompt."""
        self.decoder.reset()

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        # Process prompt tokens (prefill simulation - run each through decode)
        for token_id in input_ids[:-1]:
            self.decoder.decode_step(token_id)

        # Generate new tokens
        generated = []
        current_token = input_ids[-1]

        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            next_token = self.decoder.decode_step(current_token)

            if next_token in stop_tokens:
                break

            generated.append(next_token)
            current_token = next_token

            if self.decoder.position() >= self.max_seq_len - 1:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(self, prompt, max_new_tokens=100, stop_tokens=None):
        """Generate text with streaming output."""
        self.decoder.reset()

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        for token_id in input_ids[:-1]:
            self.decoder.decode_step(token_id)

        current_token = input_ids[-1]
        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            next_token = self.decoder.decode_step(current_token)

            if next_token in stop_tokens:
                break

            yield self.tokenizer.decode([next_token])
            current_token = next_token

            if self.decoder.position() >= self.max_seq_len - 1:
                break


class MegakernelPrefillGenerator:
    """High-level generator using prefill + decode kernels."""

    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_seq_len=2048, max_prefill_len=512):
        weights = load_qwen3_weights(model_name)

        kernel = _compile_prefill_kernel()

        self.decoder = kernel.MegakernelPrefillDecoder(
            weights["embed_weight"],
            weights["layer_weights"],
            weights["final_norm_weight"],
            weights["lm_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            NUM_LAYERS,
            max_seq_len,
            max_prefill_len,
        )

        self.tokenizer = weights["tokenizer"]
        self.max_seq_len = max_seq_len
        self.max_prefill_len = max_prefill_len

    def generate(self, prompt, max_new_tokens=100, stop_tokens=None):
        """Generate text from a prompt using prefill + decode."""
        self.decoder.reset()

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if len(input_ids) > self.max_prefill_len:
            raise ValueError(f"Prompt length {len(input_ids)} exceeds max prefill length {self.max_prefill_len}")

        # Prefill: process all prompt tokens at once
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)

        # Generate new tokens using decode
        generated = []
        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break

            generated.append(next_token)

            if self.decoder.position() >= self.max_seq_len - 1:
                break

            next_token = self.decoder.decode_step(next_token)

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(self, prompt, max_new_tokens=100, stop_tokens=None):
        """Generate text with streaming output using prefill + decode."""
        self.decoder.reset()

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if len(input_ids) > self.max_prefill_len:
            raise ValueError(f"Prompt length {len(input_ids)} exceeds max prefill length {self.max_prefill_len}")

        # Prefill
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)

        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break

            yield self.tokenizer.decode([next_token])

            if self.decoder.position() >= self.max_seq_len - 1:
                break

            next_token = self.decoder.decode_step(next_token)

    def prefill_only(self, prompt):
        """Run only prefill and return the first generated token."""
        self.decoder.reset()

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if len(input_ids) > self.max_prefill_len:
            raise ValueError(f"Prompt length {len(input_ids)} exceeds max prefill length {self.max_prefill_len}")

        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)

        return next_token, self.decoder.position()


class MegakernelFusedPrefillGenerator:
    """High-level generator using fused prefill megakernel (no cuBLAS).

    Optimized for short sequences (4-64 tokens) where the batched GEMV
    approach can outperform cuBLAS GEMM operations.
    """

    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_seq_len=2048, max_prefill_len=64):
        weights = load_qwen3_weights(model_name)

        kernel = _compile_fused_prefill_kernel()

        self.decoder = kernel.MegakernelFusedPrefillDecoder(
            weights["embed_weight"],
            weights["layer_weights"],
            weights["final_norm_weight"],
            weights["lm_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            NUM_LAYERS,
            max_seq_len,
            max_prefill_len,
        )

        self.tokenizer = weights["tokenizer"]
        self.max_seq_len = max_seq_len
        self.max_prefill_len = max_prefill_len

    def generate(self, prompt, max_new_tokens=100, stop_tokens=None):
        """Generate text from a prompt using fused prefill + decode."""
        self.decoder.reset()

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if len(input_ids) > self.max_prefill_len:
            raise ValueError(f"Prompt length {len(input_ids)} exceeds max prefill length {self.max_prefill_len}")

        # Prefill: process all prompt tokens at once
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)

        # Generate new tokens using decode
        generated = []
        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break

            generated.append(next_token)

            if self.decoder.position() >= self.max_seq_len - 1:
                break

            next_token = self.decoder.decode_step(next_token)

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(self, prompt, max_new_tokens=100, stop_tokens=None):
        """Generate text with streaming output using fused prefill + decode."""
        self.decoder.reset()

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if len(input_ids) > self.max_prefill_len:
            raise ValueError(f"Prompt length {len(input_ids)} exceeds max prefill length {self.max_prefill_len}")

        # Prefill
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)

        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break

            yield self.tokenizer.decode([next_token])

            if self.decoder.position() >= self.max_seq_len - 1:
                break

            next_token = self.decoder.decode_step(next_token)

    def prefill_only(self, prompt):
        """Run only prefill and return the first generated token."""
        self.decoder.reset()

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if len(input_ids) > self.max_prefill_len:
            raise ValueError(f"Prompt length {len(input_ids)} exceeds max prefill length {self.max_prefill_len}")

        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)

        return next_token, self.decoder.position()


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", action="store_true", help="Use cuBLAS prefill kernel")
    parser.add_argument("--fused-prefill", action="store_true", help="Use fused prefill megakernel (no cuBLAS)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--seq-len", type=int, default=None, help="Test with specific sequence length")
    args = parser.parse_args()

    if args.fused_prefill:
        print("Using fused prefill megakernel + decode kernels...")
        gen = MegakernelFusedPrefillGenerator(max_prefill_len=64)
    elif args.prefill:
        print("Using cuBLAS prefill + decode kernels...")
        gen = MegakernelPrefillGenerator()
    else:
        print("Using decode-only kernel...")
        gen = MegakernelGenerator()

    prompt = "Hello, my name is"

    if args.benchmark:
        use_prefill = args.prefill or args.fused_prefill

        # Create test sequence of specified length if provided
        if args.seq_len is not None:
            # Create a sequence of the specified length
            base_ids = gen.tokenizer.encode(prompt, add_special_tokens=True)
            if args.seq_len > len(base_ids):
                # Pad with a repeated token to reach desired length
                pad_token = base_ids[-1]
                input_ids = base_ids + [pad_token] * (args.seq_len - len(base_ids))
            else:
                input_ids = base_ids[:args.seq_len]
        else:
            input_ids = gen.tokenizer.encode(prompt, add_special_tokens=True)

        input_tensor = torch.tensor(input_ids, dtype=torch.int32)

        # Warmup
        for _ in range(3):
            gen.decoder.reset()
            if use_prefill:
                gen.decoder.prefill_step(input_tensor)
            else:
                for tid in input_ids[:-1]:
                    gen.decoder.decode_step(tid)
                gen.decoder.decode_step(input_ids[-1])

        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(20):
            gen.decoder.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            if use_prefill:
                gen.decoder.prefill_step(input_tensor)
            else:
                for tid in input_ids[:-1]:
                    gen.decoder.decode_step(tid)
                gen.decoder.decode_step(input_ids[-1])
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mode = "Fused megakernel" if args.fused_prefill else ("cuBLAS prefill" if args.prefill else "Decode-only")
        print(f"{mode} ({len(input_ids)} tokens):")
        print(f"  Mean: {sum(times)/len(times):.2f} ms")
        print(f"  Min: {min(times):.2f} ms")
        print(f"  Max: {max(times):.2f} ms")
        print(f"  Throughput: {len(input_ids) * 1000 / (sum(times)/len(times)):.1f} tokens/s")

    else:
        print("Testing generation...")
        output = gen.generate(prompt, max_new_tokens=20)
        print(f"Output: {output}")
