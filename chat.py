"""Interactive chat with Qwen3-0.6B using custom CUDA kernels."""
import os
import time

import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoTokenizer

# Model config
NUM_LAYERS = 28
HIDDEN_SIZE = 1024
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 3072
MAX_SEQ_LEN = 512


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 1000000.0, device="cuda"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def compile_kernel():
    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "megakernel")

    with open(os.path.join(kernel_dir, "fused_decode_ldg.cu")) as f:
        cuda_src = f.read()

    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

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

static std::vector<LDGLayerWeights> g_layer_weights;
static LDGLayerWeights* d_layer_weights = nullptr;
static torch::Tensor d_embed_weight;

void init_ldg_layer_weights(
    std::vector<torch::Tensor> input_layernorm_weights,
    std::vector<torch::Tensor> q_proj_weights,
    std::vector<torch::Tensor> k_proj_weights,
    std::vector<torch::Tensor> v_proj_weights,
    std::vector<torch::Tensor> q_norm_weights,
    std::vector<torch::Tensor> k_norm_weights,
    std::vector<torch::Tensor> o_proj_weights,
    std::vector<torch::Tensor> post_attn_layernorm_weights,
    std::vector<torch::Tensor> gate_proj_weights,
    std::vector<torch::Tensor> up_proj_weights,
    std::vector<torch::Tensor> down_proj_weights
) {
    int num_layers = input_layernorm_weights.size();
    g_layer_weights.resize(num_layers);

    for (int i = 0; i < num_layers; i++) {
        g_layer_weights[i].input_layernorm_weight = input_layernorm_weights[i].data_ptr();
        g_layer_weights[i].q_proj_weight = q_proj_weights[i].data_ptr();
        g_layer_weights[i].k_proj_weight = k_proj_weights[i].data_ptr();
        g_layer_weights[i].v_proj_weight = v_proj_weights[i].data_ptr();
        g_layer_weights[i].q_norm_weight = q_norm_weights[i].data_ptr();
        g_layer_weights[i].k_norm_weight = k_norm_weights[i].data_ptr();
        g_layer_weights[i].o_proj_weight = o_proj_weights[i].data_ptr();
        g_layer_weights[i].post_attn_layernorm_weight = post_attn_layernorm_weights[i].data_ptr();
        g_layer_weights[i].gate_proj_weight = gate_proj_weights[i].data_ptr();
        g_layer_weights[i].up_proj_weight = up_proj_weights[i].data_ptr();
        g_layer_weights[i].down_proj_weight = down_proj_weights[i].data_ptr();
    }

    if (d_layer_weights != nullptr) {
        cudaFree(d_layer_weights);
    }
    cudaMalloc(&d_layer_weights, num_layers * sizeof(LDGLayerWeights));
    cudaMemcpy(d_layer_weights, g_layer_weights.data(), num_layers * sizeof(LDGLayerWeights), cudaMemcpyHostToDevice);
}

void init_ldg_embed_weight(torch::Tensor embed_weight) {
    d_embed_weight = embed_weight;
}

int decode_ldg(
    int input_token_id,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor hidden_buffer,
    torch::Tensor g_activations,
    torch::Tensor g_residual,
    torch::Tensor g_q,
    torch::Tensor g_k,
    torch::Tensor g_v,
    torch::Tensor g_attn_out,
    torch::Tensor g_mlp_intermediate,
    torch::Tensor g_normalized,
    torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len
) {
    float attn_scale = 1.0f / sqrtf(128.0f);
    auto output_token = torch::empty({1}, torch::dtype(torch::kInt32).device(k_cache.device()));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_ldg_decode(
        input_token_id,
        output_token.data_ptr<int>(),
        d_embed_weight.data_ptr(),
        d_layer_weights,
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden_buffer.data_ptr(),
        g_activations.data_ptr(),
        g_residual.data_ptr(),
        g_q.data_ptr(),
        g_k.data_ptr(),
        g_v.data_ptr(),
        g_attn_out.data_ptr(),
        g_mlp_intermediate.data_ptr(),
        g_normalized.data_ptr(),
        block_max_vals.data_ptr(),
        block_max_idxs.data_ptr(),
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        stream
    );

    cudaStreamSynchronize(stream);
    return output_token.item<int>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_ldg_layer_weights", &init_ldg_layer_weights);
    m.def("init_ldg_embed_weight", &init_ldg_embed_weight);
    m.def("decode_ldg", &decode_ldg);
}
"""

    module = load_inline(
        name="ldg_kernel_chat",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
            "-lineinfo",
            "-maxrregcount=64",
        ],
        verbose=False,
    )

    return module


def load_weights_from_hf():
    """Load weights from HuggingFace without keeping the model."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    return state_dict


class MegakernelChat:
    def __init__(self):
        self.device = "cuda"

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

        print("Compiling custom CUDA kernels...")
        self.kernel = compile_kernel()

        print("Loading model weights...")
        state_dict = load_weights_from_hf()

        input_layernorm_weights = []
        q_proj_weights = []
        k_proj_weights = []
        v_proj_weights = []
        q_norm_weights = []
        k_norm_weights = []
        o_proj_weights = []
        post_attn_layernorm_weights = []
        gate_proj_weights = []
        up_proj_weights = []
        down_proj_weights = []

        for i in range(NUM_LAYERS):
            input_layernorm_weights.append(state_dict[f"model.layers.{i}.input_layernorm.weight"].contiguous())
            q_proj_weights.append(state_dict[f"model.layers.{i}.self_attn.q_proj.weight"].contiguous())
            k_proj_weights.append(state_dict[f"model.layers.{i}.self_attn.k_proj.weight"].contiguous())
            v_proj_weights.append(state_dict[f"model.layers.{i}.self_attn.v_proj.weight"].contiguous())
            q_norm_weights.append(state_dict[f"model.layers.{i}.self_attn.q_norm.weight"].contiguous())
            k_norm_weights.append(state_dict[f"model.layers.{i}.self_attn.k_norm.weight"].contiguous())
            o_proj_weights.append(state_dict[f"model.layers.{i}.self_attn.o_proj.weight"].contiguous())
            post_attn_layernorm_weights.append(state_dict[f"model.layers.{i}.post_attention_layernorm.weight"].contiguous())
            gate_proj_weights.append(state_dict[f"model.layers.{i}.mlp.gate_proj.weight"].contiguous())
            up_proj_weights.append(state_dict[f"model.layers.{i}.mlp.up_proj.weight"].contiguous())
            down_proj_weights.append(state_dict[f"model.layers.{i}.mlp.down_proj.weight"].contiguous())

        self.final_norm_weight = state_dict["model.norm.weight"].contiguous()
        self.lm_head_weight = state_dict["lm_head.weight"].contiguous()
        embed_weight = state_dict["model.embed_tokens.weight"].contiguous()

        self.cos_table, self.sin_table = precompute_rope_freqs(HEAD_DIM, MAX_SEQ_LEN, 1000000.0, self.device)

        # Initialize kernel weights
        self.kernel.init_ldg_layer_weights(
            input_layernorm_weights, q_proj_weights, k_proj_weights, v_proj_weights,
            q_norm_weights, k_norm_weights, o_proj_weights, post_attn_layernorm_weights,
            gate_proj_weights, up_proj_weights, down_proj_weights,
        )
        self.kernel.init_ldg_embed_weight(embed_weight)

        # Allocate buffers
        HIGHPAR_NUM_BLOCKS = 1184
        self.k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        self.v_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        self.hidden_buffer = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.bfloat16)
        self.g_activations = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.float32)
        self.g_residual = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.float32)
        self.g_q = torch.zeros(2048, device=self.device, dtype=torch.float32)
        self.g_k = torch.zeros(1024, device=self.device, dtype=torch.float32)
        self.g_v = torch.zeros(1024, device=self.device, dtype=torch.float32)
        self.g_attn_out = torch.zeros(2048, device=self.device, dtype=torch.float32)
        self.g_mlp_intermediate = torch.zeros(INTERMEDIATE_SIZE, device=self.device, dtype=torch.float32)
        self.g_normalized = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.float32)
        self.block_max_vals = torch.zeros(HIGHPAR_NUM_BLOCKS, device=self.device, dtype=torch.float32)
        self.block_max_idxs = torch.zeros(HIGHPAR_NUM_BLOCKS, device=self.device, dtype=torch.int32)

        print("Ready!")

    def generate(self, prompt: str, max_new_tokens: int = 100, show_speed: bool = True) -> str:
        """Generate response using 100% custom CUDA kernels."""

        # Reset KV cache
        self.k_cache.zero_()
        self.v_cache.zero_()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0].tolist()  # List of token IDs
        prompt_len = len(input_ids)

        if prompt_len >= MAX_SEQ_LEN - max_new_tokens:
            print(f"Warning: prompt too long ({prompt_len} tokens), truncating")
            input_ids = input_ids[-(MAX_SEQ_LEN - max_new_tokens - 1):]
            prompt_len = len(input_ids)

        total_start = time.perf_counter()

        # Sequential prefill using custom kernel (process each prompt token)
        prefill_start = time.perf_counter()
        first_generated_token = None
        for position, token_id in enumerate(input_ids):
            cache_len = position + 1
            output_token = self.kernel.decode_ldg(
                token_id,
                self.final_norm_weight,
                self.lm_head_weight,
                self.cos_table.contiguous(),
                self.sin_table.contiguous(),
                self.k_cache.contiguous(),
                self.v_cache.contiguous(),
                self.hidden_buffer,
                self.g_activations,
                self.g_residual,
                self.g_q,
                self.g_k,
                self.g_v,
                self.g_attn_out,
                self.g_mlp_intermediate,
                self.g_normalized,
                self.block_max_vals,
                self.block_max_idxs,
                NUM_LAYERS,
                position,
                cache_len,
                MAX_SEQ_LEN,
            )
            # Keep the last output - this is the first generated token
            first_generated_token = output_token
        prefill_time = time.perf_counter() - prefill_start

        # Start with the predicted next token from prefill
        current_token = first_generated_token
        generated_tokens = [current_token]  # Include the first predicted token

        # Check if first token is EOS
        if current_token == self.tokenizer.eos_token_id:
            decode_time = 0
            total_time = time.perf_counter() - total_start
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if show_speed:
                print(f"\n[prefill: {prompt_len} tok @ {prompt_len/prefill_time:.0f} tok/s | decode: 1 tok | total: {total_time*1000:.0f}ms]")
            return response

        # Decode using custom kernel (continue from first generated token)
        decode_start = time.perf_counter()

        for i in range(max_new_tokens - 1):  # -1 because we already have first token from prefill
            position = prompt_len + i  # first_generated_token goes at position prompt_len
            cache_len = position + 1

            next_token = self.kernel.decode_ldg(
                current_token,
                self.final_norm_weight,
                self.lm_head_weight,
                self.cos_table.contiguous(),
                self.sin_table.contiguous(),
                self.k_cache.contiguous(),
                self.v_cache.contiguous(),
                self.hidden_buffer,
                self.g_activations,
                self.g_residual,
                self.g_q,
                self.g_k,
                self.g_v,
                self.g_attn_out,
                self.g_mlp_intermediate,
                self.g_normalized,
                self.block_max_vals,
                self.block_max_idxs,
                NUM_LAYERS,
                position,
                cache_len,
                MAX_SEQ_LEN,
            )

            generated_tokens.append(next_token)
            current_token = next_token

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        decode_time = time.perf_counter() - decode_start
        total_time = time.perf_counter() - total_start

        # Decode tokens
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if show_speed:
            num_generated = len(generated_tokens)
            prefill_tps = prompt_len / prefill_time if prefill_time > 0 else 0
            decode_tps = num_generated / decode_time if decode_time > 0 else 0
            total_tokens = prompt_len + num_generated
            overall_tps = total_tokens / total_time if total_time > 0 else 0
            print(f"\n[prefill: {prompt_len} tok @ {prefill_tps:.0f} tok/s | "
                  f"decode: {num_generated} tok @ {decode_tps:.0f} tok/s | "
                  f"total: {total_time*1000:.0f}ms]")

        return response

    def chat(self):
        """Interactive chat loop."""
        print("\n" + "="*60)
        print("Qwen3-0.6B Chat (Custom CUDA Kernels)")
        print("="*60)
        print("Type 'quit' or 'exit' to end the conversation.")
        print("Type 'clear' to start a new conversation.")
        print("="*60 + "\n")

        conversation = []

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if user_input.lower() == 'clear':
                conversation = []
                print("\n[Conversation cleared]\n")
                continue

            # Build prompt (simple format for Qwen3)
            conversation.append({"role": "user", "content": user_input})

            # Format as chat with system prompt to suppress verbose thinking
            # /nothink disables Qwen3's chain-of-thought reasoning
            prompt = "<|im_start|>system\nYou are a helpful assistant. Be concise and direct. /nothink<|im_end|>\n"
            for msg in conversation:
                if msg["role"] == "user":
                    prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                else:
                    prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            # Debug: show prompt (uncomment to debug)
            # print(f"\n[DEBUG] Prompt ({len(self.tokenizer.encode(prompt))} tokens):\n{prompt}\n")

            # Generate
            print("Assistant: ", end="", flush=True)
            response = self.generate(prompt, max_new_tokens=200, show_speed=True)

            # Clean up response
            response = response.split("<|im_end|>")[0].strip()

            # Strip thinking tags if present (Qwen3 outputs <think>...</think> for reasoning)
            import re
            # Remove complete think blocks
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            # Remove unclosed think tags (if generation was cut off)
            if '<think>' in response:
                response = response.split('<think>')[0].strip()
            if not response:
                response = "[Model is thinking... try asking more directly]"

            print(response)

            conversation.append({"role": "assistant", "content": response})
            print()


def main():
    chat = MegakernelChat()
    chat.chat()


if __name__ == "__main__":
    main()
