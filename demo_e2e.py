"""
End-to-end demo comparing Torch vs Triton vs CUDA backends for Qwen3-0.6B.
Streams tokens to screen as they are generated.
"""

import time
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from kernels import get_kernels
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960


# ============================================================================
# Torch RMS Norm (sequential sum to match Triton/CUDA exactly)
# ============================================================================

def torch_rms_norm_sequential(x, weight, eps):
    """RMS norm using sequential sum - matches Triton/CUDA exactly for n_cols <= 128."""
    original_shape = x.shape
    n_cols = original_shape[-1]
    x_flat = x.view(-1, n_cols)
    x_f32 = x_flat.float()

    # Sequential sum of squares - matches Triton/CUDA accumulation order
    sum_sq = torch.zeros(x_flat.shape[0], 1, device=x.device, dtype=torch.float32)
    for i in range(n_cols):
        sum_sq += x_f32[:, i:i+1] ** 2

    variance = sum_sq / n_cols
    rstd = torch.rsqrt(variance + eps)
    x_normed = x_f32 * rstd
    result = (weight.float() * x_normed).to(x.dtype)
    return result.view(original_shape)


# ============================================================================
# Triton Kernels (copied from qwen3-0.6b.py)
# ============================================================================

@triton.jit
def rms_norm_sequential(
    x_ptr, weight_ptr, out_ptr, stride_x_row, n_cols, eps, BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm with sequential sum for small sizes (n_cols <= 128) - matches PyTorch."""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x_row
    # Sequential sum of squares - matches PyTorch for small sizes
    sum_sq = 0.0
    for i in range(n_cols):
        x_i = tl.load(x_ptr + row_start + i).to(tl.float32)
        sum_sq += x_i * x_i
    variance = sum_sq / n_cols
    rstd = tl.math.rsqrt(variance + eps)
    # Vectorized output
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = w * (x * rstd)
    tl.store(out_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def rms_norm_tree(
    x_ptr, weight_ptr, out_ptr, stride_x_row, n_cols, eps, BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm with tree reduction for medium sizes (128 < n_cols <= 4096) - matches PyTorch."""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x_row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    variance = sum_sq / n_cols
    rstd = tl.math.rsqrt(variance + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = w * (x * rstd)
    tl.store(out_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def rms_norm_multi_pass(
    x_ptr, weight_ptr, out_ptr, stride_x_row, n_cols, eps, BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm with multi-pass for very large sizes (n_cols > 4096)."""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x_row
    # Accumulate sum of squares across blocks
    sum_sq = 0.0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    variance = sum_sq / n_cols
    rstd = tl.math.rsqrt(variance + eps)
    # Output pass
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = w * (x * rstd)
        tl.store(out_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm wrapper with hybrid sum method to match PyTorch exactly."""
    assert x.is_contiguous()
    shape = x.shape
    x = x.view(-1, shape[-1])
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if n_cols <= 128:
        # Sequential sum for head_dim=128 (QK norm)
        rms_norm_sequential[(n_rows,)](x, weight, out, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    elif BLOCK_SIZE <= 4096:
        # Tree reduction for hidden_size=1024 (layer norm)
        rms_norm_tree[(n_rows,)](x, weight, out, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # Multi-pass for very large hidden sizes
        rms_norm_multi_pass[(n_rows,)](x, weight, out, x.stride(0), n_cols, eps, BLOCK_SIZE=4096)
    return out.view(shape)


@triton.jit
def silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    gate = tl.load(gate_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up
    tl.store(out_ptr + offset, out.to(tl.bfloat16), mask=mask)


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    BLOCK_SIZE = 1024
    silu_mul_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        gate.view(-1), up.view(-1), out.view(-1), n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 1000000.0, device="cuda"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def apply_rope_torch(q, k, cos, sin, position_ids):
    """PyTorch RoPE implementation."""
    pos = position_ids[0]
    cos_pos = cos[pos].unsqueeze(0).unsqueeze(0)
    sin_pos = sin[pos].unsqueeze(0).unsqueeze(0)
    q_fp32 = q.float()
    k_fp32 = k.float()
    cos_fp32 = cos_pos.float()
    sin_fp32 = sin_pos.float()
    half = q.shape[-1] // 2
    q1, q2 = q_fp32[..., :half], q_fp32[..., half:]
    k1, k2 = k_fp32[..., :half], k_fp32[..., half:]
    cos1, sin1 = cos_fp32[..., :half], sin_fp32[..., :half]
    q_rot = torch.cat([q1 * cos1 - q2 * sin1, q2 * cos1 + q1 * sin1], dim=-1)
    k_rot = torch.cat([k1 * cos1 - k2 * sin1, k2 * cos1 + k1 * sin1], dim=-1)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def attention_decode_torch(q, k_cache, v_cache, cache_len):
    """PyTorch attention decode."""
    batch, n_heads_q, _, head_dim = q.shape
    n_heads_kv = k_cache.shape[1]
    n_groups = n_heads_q // n_heads_kv
    k = k_cache[:, :, :cache_len, :]
    v = v_cache[:, :, :cache_len, :]
    k = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(batch, n_heads_q, cache_len, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(batch, n_heads_q, cache_len, head_dim)
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.to(q.dtype)


# ============================================================================
# Backend enum
# ============================================================================

class Backend:
    TORCH = "torch"
    TRITON = "triton"
    CUDA = "cuda"


# ============================================================================
# Model Components (Backend-switchable)
# ============================================================================

class Qwen3Attention:
    def __init__(self, layer_weights, config: Qwen3Config, layer_idx: int, backend: str, cuda_kernels=None):
        self.config = config
        self.layer_idx = layer_idx
        self.backend = backend
        self.cuda_kernels = cuda_kernels
        self.q_proj_weight = layer_weights["q_proj.weight"]
        self.k_proj_weight = layer_weights["k_proj.weight"]
        self.v_proj_weight = layer_weights["v_proj.weight"]
        self.o_proj_weight = layer_weights["o_proj.weight"]
        self.q_norm_weight = layer_weights["q_norm.weight"]
        self.k_norm_weight = layer_weights["k_norm.weight"]

    def rms_norm(self, x, weight):
        if self.backend == Backend.TORCH:
            return torch_rms_norm_sequential(x, weight, self.config.rms_norm_eps)
        elif self.backend == Backend.TRITON:
            return triton_rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)
        else:  # CUDA
            return self.cuda_kernels.rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)

    def apply_rope(self, q, k, cos, sin, position_ids):
        if self.backend == Backend.TORCH:
            return apply_rope_torch(q, k, cos, sin, position_ids)
        elif self.backend == Backend.TRITON:
            return apply_rope_torch(q, k, cos, sin, position_ids)  # Use torch for triton too
        else:  # CUDA
            pos = position_ids[0]
            cos_pos = cos[pos].contiguous()
            sin_pos = sin[pos].contiguous()
            return self.cuda_kernels.rope(q.contiguous(), k.contiguous(), cos_pos, sin_pos)

    def attention_decode(self, q, k_cache, v_cache, cache_len):
        if self.backend == Backend.CUDA:
            return self.cuda_kernels.attention_decode(q.contiguous(), k_cache.contiguous(), v_cache.contiguous(), cache_len)
        else:
            return attention_decode_torch(q, k_cache, v_cache, cache_len)

    def forward(self, hidden_states, cos, sin, position_ids, k_cache=None, v_cache=None, cache_position=0, is_prefill=True):
        batch, seq_len, _ = hidden_states.shape
        q = torch.nn.functional.linear(hidden_states, self.q_proj_weight)
        k = torch.nn.functional.linear(hidden_states, self.k_proj_weight)
        v = torch.nn.functional.linear(hidden_states, self.v_proj_weight)
        q = q.view(batch, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2).contiguous()

        # QK norm
        q = self.rms_norm(q.contiguous(), self.q_norm_weight)
        k = self.rms_norm(k.contiguous(), self.k_norm_weight)

        # RoPE
        q, k = self.apply_rope(q, k, cos, sin, position_ids)

        if is_prefill:
            n_groups = self.config.num_attention_heads // self.config.num_key_value_heads
            k_expanded = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
            k_expanded = k_expanded.reshape(batch, self.config.num_attention_heads, seq_len, self.config.head_dim)
            v_expanded = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
            v_expanded = v_expanded.reshape(batch, self.config.num_attention_heads, seq_len, self.config.head_dim)
            scale = 1.0 / (self.config.head_dim ** 0.5)
            scores = torch.matmul(q.float(), k_expanded.float().transpose(-2, -1)) * scale
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v_expanded.float()).to(q.dtype)
            if k_cache is not None:
                k_cache[:, :, :seq_len, :] = k
                v_cache[:, :, :seq_len, :] = v
        else:
            k_cache[:, :, cache_position:cache_position+1, :] = k
            v_cache[:, :, cache_position:cache_position+1, :] = v
            attn_out = self.attention_decode(q, k_cache, v_cache, cache_position + 1)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = torch.nn.functional.linear(attn_out, self.o_proj_weight)
        return output, k_cache, v_cache


class Qwen3MLP:
    def __init__(self, layer_weights, config: Qwen3Config, backend: str, cuda_kernels=None):
        self.config = config
        self.backend = backend
        self.cuda_kernels = cuda_kernels
        self.gate_proj_weight = layer_weights["gate_proj.weight"]
        self.up_proj_weight = layer_weights["up_proj.weight"]
        self.down_proj_weight = layer_weights["down_proj.weight"]

    def silu_mul(self, gate, up):
        if self.backend == Backend.TORCH:
            return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)
        elif self.backend == Backend.TRITON:
            return triton_silu_mul(gate, up)
        else:  # CUDA
            return self.cuda_kernels.silu_mul(gate, up)

    def forward(self, hidden_states):
        gate = torch.nn.functional.linear(hidden_states, self.gate_proj_weight)
        up = torch.nn.functional.linear(hidden_states, self.up_proj_weight)
        hidden = self.silu_mul(gate, up)
        return torch.nn.functional.linear(hidden, self.down_proj_weight)


class Qwen3Layer:
    def __init__(self, layer_weights, config: Qwen3Config, layer_idx: int, backend: str, cuda_kernels=None):
        self.config = config
        self.backend = backend
        self.cuda_kernels = cuda_kernels
        self.input_layernorm_weight = layer_weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = layer_weights["post_attention_layernorm.weight"]
        attn_weights = {k.replace("self_attn.", ""): v for k, v in layer_weights.items() if "self_attn" in k}
        mlp_weights = {k.replace("mlp.", ""): v for k, v in layer_weights.items() if "mlp" in k}
        self.self_attn = Qwen3Attention(attn_weights, config, layer_idx, backend, cuda_kernels)
        self.mlp = Qwen3MLP(mlp_weights, config, backend, cuda_kernels)

    def rms_norm(self, x, weight):
        if self.backend == Backend.TORCH:
            return torch_rms_norm_sequential(x, weight, self.config.rms_norm_eps)
        elif self.backend == Backend.TRITON:
            return triton_rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)
        else:  # CUDA
            return self.cuda_kernels.rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, position_ids, k_cache=None, v_cache=None, cache_position=0, is_prefill=True):
        residual = hidden_states
        hidden_states = self.rms_norm(hidden_states, self.input_layernorm_weight)
        hidden_states, k_cache, v_cache = self.self_attn.forward(
            hidden_states, cos, sin, position_ids, k_cache, v_cache, cache_position, is_prefill
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.rms_norm(hidden_states, self.post_attention_layernorm_weight)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, k_cache, v_cache


class Qwen3Model:
    def __init__(self, hf_model, config: Qwen3Config, backend: str, cuda_kernels=None):
        self.config = config
        self.backend = backend
        self.cuda_kernels = cuda_kernels
        self.device = next(hf_model.parameters()).device
        self.dtype = torch.bfloat16
        state_dict = hf_model.state_dict()
        self.embed_tokens = state_dict["model.embed_tokens.weight"]
        self.final_norm_weight = state_dict["model.norm.weight"]
        self.lm_head_weight = self.embed_tokens
        self.layers = []
        for i in range(config.num_hidden_layers):
            layer_weights = {k.replace(f"model.layers.{i}.", ""): v for k, v in state_dict.items() if f"model.layers.{i}." in k}
            self.layers.append(Qwen3Layer(layer_weights, config, i, backend, cuda_kernels))
        self.cos, self.sin = precompute_rope_freqs(config.head_dim, config.max_position_embeddings, config.rope_theta, self.device)

    def rms_norm(self, x, weight):
        if self.backend == Backend.TORCH:
            return torch_rms_norm_sequential(x, weight, self.config.rms_norm_eps)
        elif self.backend == Backend.TRITON:
            return triton_rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)
        else:  # CUDA
            return self.cuda_kernels.rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)

    def prefill(self, input_ids):
        batch, seq_len = input_ids.shape
        hidden_states = torch.nn.functional.embedding(input_ids, self.embed_tokens)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        kv_caches = []
        max_cache_len = seq_len + 512
        for layer in self.layers:
            k_cache = torch.zeros(batch, self.config.num_key_value_heads, max_cache_len, self.config.head_dim, device=self.device, dtype=self.dtype)
            v_cache = torch.zeros(batch, self.config.num_key_value_heads, max_cache_len, self.config.head_dim, device=self.device, dtype=self.dtype)
            hidden_states, k_cache, v_cache = layer.forward(hidden_states, self.cos, self.sin, position_ids, k_cache, v_cache, 0, is_prefill=True)
            kv_caches.append((k_cache, v_cache))
        hidden_states = self.rms_norm(hidden_states, self.final_norm_weight)
        logits = torch.nn.functional.linear(hidden_states, self.lm_head_weight)
        return logits, kv_caches, seq_len

    def decode_step(self, input_id, kv_caches, cache_position):
        hidden_states = torch.nn.functional.embedding(input_id, self.embed_tokens)
        position_ids = torch.tensor([[cache_position]], device=self.device)
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = kv_caches[i]
            hidden_states, k_cache, v_cache = layer.forward(hidden_states, self.cos, self.sin, position_ids, k_cache, v_cache, cache_position, is_prefill=False)
            new_kv_caches.append((k_cache, v_cache))
        hidden_states = self.rms_norm(hidden_states, self.final_norm_weight)
        logits = torch.nn.functional.linear(hidden_states, self.lm_head_weight)
        return logits, new_kv_caches

    @torch.no_grad()
    def generate_streaming(self, input_ids, tokenizer, max_new_tokens=100):
        """Generate tokens with streaming output."""
        # Prefill
        logits, kv_caches, cache_len = self.prefill(input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            # Decode token and print
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_str, end="", flush=True)

            # Check for EOS
            if next_token.item() == 151645:
                break

            # Generate next token
            logits, kv_caches = self.decode_step(next_token, kv_caches, cache_len + tokens_generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens_generated += 1

        print()  # Newline after generation
        return tokens_generated


# ============================================================================
# Main Demo
# ============================================================================

def run_demo():
    print("=" * 70)
    print(" End-to-End Demo: Torch vs Triton vs CUDA")
    print("=" * 70)

    # Load model
    print("\nLoading Qwen3-0.6B model...")
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    hf_model.eval()

    # Compile CUDA kernels
    print("Compiling CUDA kernels...")
    cuda_kernels = get_kernels()
    print("Done!\n")

    # Create config
    config = Qwen3Config()

    # Create models for each backend
    print("Creating backend models...")
    torch_model = Qwen3Model(hf_model, config, Backend.TORCH)
    triton_model = Qwen3Model(hf_model, config, Backend.TRITON)
    cuda_model = Qwen3Model(hf_model, config, Backend.CUDA, cuda_kernels)
    print("Done!\n")

    while True:
        print("-" * 70)
        prompt = input("Enter your question (or 'quit' to exit): ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        if not prompt:
            continue

        # Format with chat template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]

        print(f"\nInput tokens: {input_ids.shape[1]}")

        # Generate with each backend
        max_new_tokens = 100

        # Torch
        print("\n" + "=" * 70)
        print(" TORCH (Reference)")
        print("=" * 70)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch_tokens = torch_model.generate_streaming(input_ids.clone(), tokenizer, max_new_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        torch_time = t1 - t0
        print(f"[Generated {torch_tokens} tokens in {torch_time:.3f}s ({torch_tokens/torch_time:.1f} tok/s)]")

        # Triton
        print("\n" + "=" * 70)
        print(" TRITON")
        print("=" * 70)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        triton_tokens = triton_model.generate_streaming(input_ids.clone(), tokenizer, max_new_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        triton_time = t1 - t0
        print(f"[Generated {triton_tokens} tokens in {triton_time:.3f}s ({triton_tokens/triton_time:.1f} tok/s)]")

        # CUDA
        print("\n" + "=" * 70)
        print(" CUDA")
        print("=" * 70)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cuda_tokens = cuda_model.generate_streaming(input_ids.clone(), tokenizer, max_new_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cuda_time = t1 - t0
        print(f"[Generated {cuda_tokens} tokens in {cuda_time:.3f}s ({cuda_tokens/cuda_time:.1f} tok/s)]")

        # Summary
        print("\n" + "-" * 70)
        print("Performance Summary:")
        print(f"  Torch:  {torch_time:.3f}s ({torch_tokens/torch_time:.1f} tok/s)")
        print(f"  Triton: {triton_time:.3f}s ({triton_tokens/triton_time:.1f} tok/s)")
        print(f"  CUDA:   {cuda_time:.3f}s ({cuda_tokens/cuda_time:.1f} tok/s)")
        print()


if __name__ == "__main__":
    run_demo()
