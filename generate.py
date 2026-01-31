"""End-to-end token generation using megakernels."""
import os
import time

import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model config for Qwen3-0.6B
NUM_LAYERS = 28
HIDDEN_SIZE = 1024
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_SEQ_LEN = 512


def compile_kernels():
    """Compile all megakernels."""
    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "megakernel")

    # Read kernel sources
    with open(os.path.join(kernel_dir, "transformer_block_v2.cu")) as f:
        decode_src = f.read()
    with open(os.path.join(kernel_dir, "lm_head.cu")) as f:
        lm_head_src = f.read()

    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Decode kernel
extern "C" void launch_transformer_block_v2(
    const void* hidden_states, void* output_states,
    const void* input_layernorm_weight,
    const void* q_proj_weight, const void* k_proj_weight, const void* v_proj_weight,
    const void* q_norm_weight, const void* k_norm_weight,
    const void* o_proj_weight,
    const void* post_attn_layernorm_weight,
    const void* gate_proj_weight, const void* up_proj_weight, const void* down_proj_weight,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* g_activations, void* g_residual, void* g_q, void* g_k, void* g_v,
    void* g_attn_out, void* g_mlp_intermediate, void* g_scratch,
    int position, int cache_len, int max_seq_len, float attn_scale, cudaStream_t stream
);

// LM head kernels
extern "C" void launch_rmsnorm_final(
    const void* input, const void* weight, void* output,
    int seq_len, cudaStream_t stream
);

extern "C" void launch_lm_head_last_token(
    const void* normalized, const void* weight, void* logits,
    int seq_len, cudaStream_t stream
);

// Global buffers for decode
struct DecodeBuffers {
    torch::Tensor activations, residual, q, k, v, attn_out, mlp_intermediate, scratch;
};
static DecodeBuffers* g_decode_buffers = nullptr;

void init_decode_buffers(torch::Device device) {
    if (g_decode_buffers != nullptr) return;
    g_decode_buffers = new DecodeBuffers();
    g_decode_buffers->activations = torch::zeros({1024}, torch::dtype(torch::kFloat32).device(device));
    g_decode_buffers->residual = torch::zeros({1024}, torch::dtype(torch::kFloat32).device(device));
    g_decode_buffers->q = torch::zeros({2048}, torch::dtype(torch::kFloat32).device(device));
    g_decode_buffers->k = torch::zeros({1024}, torch::dtype(torch::kFloat32).device(device));
    g_decode_buffers->v = torch::zeros({1024}, torch::dtype(torch::kFloat32).device(device));
    g_decode_buffers->attn_out = torch::zeros({2048}, torch::dtype(torch::kFloat32).device(device));
    g_decode_buffers->mlp_intermediate = torch::zeros({3072}, torch::dtype(torch::kFloat32).device(device));
    g_decode_buffers->scratch = torch::zeros({8192}, torch::dtype(torch::kFloat32).device(device));
}

torch::Tensor decode_layer(
    torch::Tensor hidden_states,
    torch::Tensor input_layernorm_weight,
    torch::Tensor q_proj_weight, torch::Tensor k_proj_weight, torch::Tensor v_proj_weight,
    torch::Tensor q_norm_weight, torch::Tensor k_norm_weight,
    torch::Tensor o_proj_weight, torch::Tensor post_attn_layernorm_weight,
    torch::Tensor gate_proj_weight, torch::Tensor up_proj_weight, torch::Tensor down_proj_weight,
    torch::Tensor cos_table, torch::Tensor sin_table,
    torch::Tensor k_cache, torch::Tensor v_cache,
    int position, int cache_len, int max_seq_len
) {
    init_decode_buffers(hidden_states.device());
    float attn_scale = 1.0f / sqrtf(128.0f);
    auto output = torch::empty_like(hidden_states);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_transformer_block_v2(
        hidden_states.data_ptr(), output.data_ptr(),
        input_layernorm_weight.data_ptr(),
        q_proj_weight.data_ptr(), k_proj_weight.data_ptr(), v_proj_weight.data_ptr(),
        q_norm_weight.data_ptr(), k_norm_weight.data_ptr(),
        o_proj_weight.data_ptr(), post_attn_layernorm_weight.data_ptr(),
        gate_proj_weight.data_ptr(), up_proj_weight.data_ptr(), down_proj_weight.data_ptr(),
        cos_table.data_ptr(), sin_table.data_ptr(),
        k_cache.data_ptr(), v_cache.data_ptr(),
        g_decode_buffers->activations.data_ptr(), g_decode_buffers->residual.data_ptr(),
        g_decode_buffers->q.data_ptr(), g_decode_buffers->k.data_ptr(), g_decode_buffers->v.data_ptr(),
        g_decode_buffers->attn_out.data_ptr(), g_decode_buffers->mlp_intermediate.data_ptr(),
        g_decode_buffers->scratch.data_ptr(),
        position, cache_len, max_seq_len, attn_scale, stream
    );
    return output;
}

torch::Tensor lm_head_last(
    torch::Tensor hidden_states,
    torch::Tensor norm_weight,
    torch::Tensor lm_weight
) {
    int seq_len = (hidden_states.dim() == 1) ? 1 : hidden_states.size(0);
    int vocab_size = lm_weight.size(0);

    auto normalized = torch::empty({seq_len, 1024}, torch::dtype(torch::kFloat32).device(hidden_states.device()));
    auto logits = torch::empty({vocab_size}, torch::dtype(torch::kBFloat16).device(hidden_states.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Handle 1D input (single token)
    auto input_2d = (hidden_states.dim() == 1) ? hidden_states.unsqueeze(0) : hidden_states;

    launch_rmsnorm_final(
        input_2d.data_ptr(), norm_weight.data_ptr(), normalized.data_ptr(),
        seq_len, stream
    );
    launch_lm_head_last_token(
        normalized.data_ptr(), lm_weight.data_ptr(), logits.data_ptr(),
        seq_len, stream
    );

    return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_layer", &decode_layer, "Decode one transformer layer");
    m.def("lm_head_last", &lm_head_last, "Final norm + LM head (last token)");
}
"""

    module = load_inline(
        name="megakernel_generate",
        cpp_sources=[cpp_src],
        cuda_sources=[decode_src, lm_head_src],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
        ],
        verbose=False,
    )
    return module


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 1000000.0, device="cuda"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


class MegakernelModel:
    """Wrapper for running Qwen3-0.6B with megakernels."""

    def __init__(self, model_name="Qwen/Qwen3-0.6B", device="cuda"):
        print("Loading model weights...")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.hf_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

        print("Compiling megakernels...")
        self.kernel = compile_kernels()

        print("Extracting weights...")
        self._extract_weights()

        print("Initializing KV caches...")
        self._init_kv_cache()

        # RoPE tables
        self.cos, self.sin = precompute_rope_freqs(HEAD_DIM, MAX_SEQ_LEN, 1000000.0, device)

    def _extract_weights(self):
        """Extract all layer weights from HuggingFace model."""
        sd = self.hf_model.state_dict()
        self.layer_weights = []

        for i in range(NUM_LAYERS):
            prefix = f"model.layers.{i}"
            weights = {
                'input_layernorm': sd[f"{prefix}.input_layernorm.weight"].contiguous(),
                'q_proj': sd[f"{prefix}.self_attn.q_proj.weight"].contiguous(),
                'k_proj': sd[f"{prefix}.self_attn.k_proj.weight"].contiguous(),
                'v_proj': sd[f"{prefix}.self_attn.v_proj.weight"].contiguous(),
                'q_norm': sd[f"{prefix}.self_attn.q_norm.weight"].contiguous(),
                'k_norm': sd[f"{prefix}.self_attn.k_norm.weight"].contiguous(),
                'o_proj': sd[f"{prefix}.self_attn.o_proj.weight"].contiguous(),
                'post_attn_layernorm': sd[f"{prefix}.post_attention_layernorm.weight"].contiguous(),
                'gate_proj': sd[f"{prefix}.mlp.gate_proj.weight"].contiguous(),
                'up_proj': sd[f"{prefix}.mlp.up_proj.weight"].contiguous(),
                'down_proj': sd[f"{prefix}.mlp.down_proj.weight"].contiguous(),
            }
            self.layer_weights.append(weights)

        # Final norm and LM head
        self.final_norm_weight = sd["model.norm.weight"].contiguous()
        self.lm_head_weight = sd["lm_head.weight"].contiguous()

        # Embedding
        self.embed_tokens = self.hf_model.model.embed_tokens

    def _init_kv_cache(self):
        """Initialize KV cache for all layers."""
        self.k_caches = []
        self.v_caches = []
        for _ in range(NUM_LAYERS):
            k = torch.zeros(NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
                           device=self.device, dtype=torch.bfloat16).contiguous()
            v = torch.zeros(NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
                           device=self.device, dtype=torch.bfloat16).contiguous()
            self.k_caches.append(k)
            self.v_caches.append(v)

    def reset_kv_cache(self):
        """Clear KV cache."""
        for k, v in zip(self.k_caches, self.v_caches):
            k.zero_()
            v.zero_()

    def decode_one_token(self, hidden_states, position, cache_len):
        """Run one token through all 28 layers using megakernels."""
        h = hidden_states

        for i in range(NUM_LAYERS):
            w = self.layer_weights[i]
            h = self.kernel.decode_layer(
                h.contiguous(),
                w['input_layernorm'], w['q_proj'], w['k_proj'], w['v_proj'],
                w['q_norm'], w['k_norm'], w['o_proj'], w['post_attn_layernorm'],
                w['gate_proj'], w['up_proj'], w['down_proj'],
                self.cos, self.sin,
                self.k_caches[i], self.v_caches[i],
                position, cache_len, MAX_SEQ_LEN
            )

        return h

    def get_next_token_logits(self, hidden_states):
        """Get logits for the last token."""
        return self.kernel.lm_head_last(
            hidden_states.contiguous(),
            self.final_norm_weight,
            self.lm_head_weight
        )

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7, do_sample: bool = True):
        """Generate text from prompt."""
        self.reset_kv_cache()

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        if prompt_len > MAX_SEQ_LEN - max_new_tokens:
            raise ValueError(f"Prompt too long: {prompt_len} tokens")

        # Prefill using HuggingFace (our prefill kernel is for small seqs only)
        # For a proper implementation, we'd use our prefill kernel for short prompts
        print(f"Prefill: {prompt_len} tokens...")

        # Get embeddings for prompt
        embeds = self.embed_tokens(input_ids)  # [1, seq_len, hidden_size]

        # Run HF prefill to populate KV cache
        with torch.no_grad():
            hf_out = self.hf_model(input_ids, use_cache=True)
            past_kv = hf_out.past_key_values

        # Copy HF's KV cache to our format
        for i in range(NUM_LAYERS):
            # HF cache: [batch, num_heads, seq_len, head_dim]
            # Our cache: [num_heads, max_seq_len, head_dim]
            k_hf = past_kv[i][0][0]  # [num_kv_heads, seq_len, head_dim]
            v_hf = past_kv[i][1][0]
            self.k_caches[i][:, :prompt_len, :] = k_hf
            self.v_caches[i][:, :prompt_len, :] = v_hf

        # Get last hidden state from HF output for starting decode
        # Actually we need to get the hidden states before LM head
        # Re-run to get hidden states
        hf_out = self.hf_model.model(input_ids, use_cache=False)
        last_hidden = hf_out.last_hidden_state[0, -1, :]  # [hidden_size]

        # Decode loop
        generated_ids = []
        current_pos = prompt_len - 1
        hidden = last_hidden

        print(f"Decoding {max_new_tokens} tokens...")
        decode_start = time.perf_counter()

        for i in range(max_new_tokens):
            current_pos += 1
            cache_len = current_pos + 1

            # Run through all layers
            hidden = self.decode_one_token(hidden, current_pos, cache_len)

            # Get logits
            logits = self.get_next_token_logits(hidden)

            # Sample
            if do_sample and temperature > 0:
                probs = torch.softmax(logits.float() / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = logits.argmax().item()

            generated_ids.append(next_token)

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break

            # Get embedding for next token
            next_embed = self.embed_tokens(torch.tensor([[next_token]], device=self.device))
            hidden = next_embed[0, 0, :]  # [hidden_size]

        decode_time = time.perf_counter() - decode_start
        tokens_generated = len(generated_ids)
        tokens_per_sec = tokens_generated / decode_time

        print(f"\nGenerated {tokens_generated} tokens in {decode_time*1000:.1f}ms")
        print(f"Throughput: {tokens_per_sec:.1f} tokens/sec")

        # Decode output
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return prompt + output_text, tokens_per_sec


def benchmark_decode_only(model, num_tokens=100):
    """Benchmark pure decode throughput (no prefill overhead)."""
    print(f"\n{'='*60}")
    print(f"Benchmarking pure decode: {num_tokens} tokens")
    print("="*60)

    model.reset_kv_cache()

    # Start with a dummy hidden state
    hidden = torch.randn(HIDDEN_SIZE, device=model.device, dtype=torch.bfloat16)

    # Warmup
    for pos in range(10):
        hidden = model.decode_one_token(hidden, pos, pos + 1)
        logits = model.get_next_token_logits(hidden)
        hidden = torch.randn(HIDDEN_SIZE, device=model.device, dtype=torch.bfloat16)

    model.reset_kv_cache()
    hidden = torch.randn(HIDDEN_SIZE, device=model.device, dtype=torch.bfloat16)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for pos in range(num_tokens):
        hidden = model.decode_one_token(hidden, pos, pos + 1)
        logits = model.get_next_token_logits(hidden)
        # Simulate getting next embedding (negligible time)
        hidden = torch.randn(HIDDEN_SIZE, device=model.device, dtype=torch.bfloat16)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = num_tokens / elapsed
    us_per_token = elapsed / num_tokens * 1e6

    print(f"Total time: {elapsed*1000:.1f}ms")
    print(f"Per token: {us_per_token:.1f}us")
    print(f"Throughput: {tokens_per_sec:.1f} tokens/sec")

    return tokens_per_sec


def main():
    print("="*60)
    print("Megakernel End-to-End Generation")
    print("="*60)

    model = MegakernelModel()

    # Benchmark pure decode
    tps = benchmark_decode_only(model, num_tokens=100)

    # Test generation
    print(f"\n{'='*60}")
    print("Test Generation")
    print("="*60)

    prompt = "The meaning of life is"
    print(f"\nPrompt: {prompt}")

    output, tps = model.generate(prompt, max_new_tokens=30, temperature=0.7)
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
