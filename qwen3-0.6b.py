"""
Qwen3-0.6B inference with custom Triton kernels.
Separate prefill and decode phases with exact HuggingFace match verification.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
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
# Triton Kernels
# ============================================================================

@triton.jit
def rms_norm_sequential(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_x_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm with sequential sum for small sizes (n_cols <= 128).

    PyTorch uses sequential accumulation for small reductions, so we must match
    that to get identical results.
    """
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
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # Keep as bf16
    normalized = (x * rstd).to(tl.bfloat16)
    out = w * normalized
    tl.store(out_ptr + row_start + cols, out, mask=mask)


@triton.jit
def rms_norm_tree(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_x_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm with tree reduction for medium sizes (128 < n_cols <= 4096).

    PyTorch uses tree reduction for larger sizes, and tl.sum() matches this.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)

    # Tree reduction via tl.sum - matches PyTorch for n_cols >= 256
    sum_sq = tl.sum(x * x, axis=0)
    variance = sum_sq / n_cols
    rstd = tl.math.rsqrt(variance + eps)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # Keep as bf16
    normalized = (x * rstd).to(tl.bfloat16)
    out = w * normalized
    tl.store(out_ptr + row_start + cols, out, mask=mask)


@triton.jit
def rms_norm_multi_pass(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_x_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
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
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # Keep as bf16
        normalized = (x * rstd).to(tl.bfloat16)
        out = w * normalized
        tl.store(out_ptr + row_start + cols, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm wrapper with hybrid sum method to match PyTorch exactly.

    - Sequential sum for n_cols <= 128 (matches PyTorch's accumulation order)
    - Tree reduction for 128 < n_cols <= 4096 (tl.sum matches PyTorch)
    - Multi-pass for n_cols > 4096
    """
    assert x.is_contiguous()
    shape = x.shape
    x = x.view(-1, shape[-1])
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if n_cols <= 128:
        # Sequential sum for small sizes - matches PyTorch exactly
        rms_norm_sequential[(n_rows,)](
            x, weight, out,
            x.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif BLOCK_SIZE <= 4096:
        # Tree reduction for medium sizes - tl.sum matches PyTorch
        rms_norm_tree[(n_rows,)](
            x, weight, out,
            x.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Multi-pass for very large hidden sizes
        rms_norm_multi_pass[(n_rows,)](
            x, weight, out,
            x.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=4096,
        )
    return out.view(shape)


@triton.jit
def rope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    seq_len,
    n_q_heads,
    n_kv_heads,
    head_dim,
    stride_q_seq,
    stride_q_head,
    stride_k_seq,
    stride_k_head,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply rotary position embeddings to Q and K."""
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    is_q = tl.program_id(2) == 0

    # Determine if this is a Q or K head
    if is_q:
        if head_idx >= n_q_heads:
            return
        ptr = q_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head
        out_ptr = q_out_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head
    else:
        if head_idx >= n_kv_heads:
            return
        ptr = k_ptr + seq_idx * stride_k_seq + head_idx * stride_k_head
        out_ptr = k_out_ptr + seq_idx * stride_k_seq + head_idx * stride_k_head

    # Process in pairs (real, imag)
    half_dim = head_dim // 2
    for i in range(0, half_dim, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < half_dim

        # Load x[..., :half] and x[..., half:]
        x0 = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(ptr + cols + half_dim, mask=mask, other=0.0).to(tl.float32)

        # Load cos and sin for this position
        cos = tl.load(cos_ptr + seq_idx * head_dim + cols, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + seq_idx * head_dim + cols, mask=mask, other=0.0).to(tl.float32)

        # Apply rotation: [x0, x1] * [[cos, -sin], [sin, cos]]
        out0 = x0 * cos - x1 * sin
        out1 = x1 * cos + x0 * sin

        tl.store(out_ptr + cols, out0.to(tl.bfloat16), mask=mask)
        tl.store(out_ptr + cols + half_dim, out1.to(tl.bfloat16), mask=mask)


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 1000000.0, device="cuda"):
    """Precompute RoPE frequencies."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    # Expand to full head_dim by repeating
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def triton_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K tensors.

    q: (batch, n_q_heads, seq_len, head_dim)
    k: (batch, n_kv_heads, seq_len, head_dim)
    cos, sin: (max_seq_len, head_dim)
    position_ids: (batch, seq_len)
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]

    # Gather cos/sin for the specific positions
    # position_ids is (batch, seq_len), we need (seq_len, head_dim) for single batch
    pos = position_ids[0]  # Assume batch=1 for now
    cos_pos = cos[pos]  # (seq_len, head_dim)
    sin_pos = sin[pos]  # (seq_len, head_dim)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    BLOCK_SIZE = min(64, head_dim // 2)

    grid = (seq_len, max(n_q_heads, n_kv_heads), 2)

    rope_kernel[grid](
        q.view(batch * n_q_heads * seq_len, head_dim),
        k.view(batch * n_kv_heads * seq_len, head_dim),
        cos_pos,
        sin_pos,
        q_out.view(batch * n_q_heads * seq_len, head_dim),
        k_out.view(batch * n_kv_heads * seq_len, head_dim),
        seq_len,
        n_q_heads,
        n_kv_heads,
        head_dim,
        head_dim,
        head_dim,
        head_dim,
        head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q_out, k_out


def apply_rope_simple(q, k, cos, sin, position_ids):
    """Simple PyTorch RoPE implementation for correctness."""
    # q: (batch, n_heads, seq_len, head_dim)
    # cos, sin: (max_seq_len, head_dim)
    pos = position_ids[0]  # (seq_len,)
    cos_pos = cos[pos].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin_pos = sin[pos].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

    # Split into first and second half
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


@triton.jit
def silu_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU(gate) * up - fused activation."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    gate = tl.load(gate_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    # SiLU = x * sigmoid(x)
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up

    tl.store(out_ptr + offset, out.to(tl.bfloat16), mask=mask)


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up."""
    assert gate.shape == up.shape
    out = torch.empty_like(gate)
    n_elements = gate.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    silu_mul_kernel[grid](
        gate.view(-1),
        up.view(-1),
        out.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def attention_prefill_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    n_heads_q, n_heads_kv,
    seq_len,
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Prefill attention with causal mask and GQA support."""
    batch_idx = tl.program_id(2)
    head_idx = tl.program_id(1)
    m_block_idx = tl.program_id(0)

    # GQA: map query head to kv head
    kv_head_idx = head_idx // (n_heads_q // n_heads_kv)

    # Compute block start positions
    m_start = m_block_idx * BLOCK_M

    # Initialize pointers
    q_block_ptr = q_ptr + batch_idx * stride_qz + head_idx * stride_qh + m_start * stride_qm
    k_block_ptr = k_ptr + batch_idx * stride_kz + kv_head_idx * stride_kh
    v_block_ptr = v_ptr + batch_idx * stride_vz + kv_head_idx * stride_vh

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Load Q block
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    q_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    q = tl.load(q_block_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                mask=q_mask, other=0.0)

    # Iterate over K/V blocks
    for n_start in range(0, seq_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        # Load K block
        k_mask = (offs_n[:, None] < seq_len) & (offs_k[None, :] < head_dim)
        k = tl.load(k_block_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=k_mask, other=0.0)

        # Compute attention scores: Q @ K^T
        qk = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

        # Apply causal mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float('-inf'))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        # Load V block
        v_mask = (offs_n[:, None] < seq_len) & (offs_k[None, :] < head_dim)
        v = tl.load(v_block_ptr + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                    mask=v_mask, other=0.0)

        # Update accumulator
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32))

        m_i = m_new
        l_i = l_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    out_block_ptr = out_ptr + batch_idx * stride_oz + head_idx * stride_oh + m_start * stride_om
    out_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    tl.store(out_block_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
             acc.to(tl.bfloat16), mask=out_mask)


def triton_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Flash attention for prefill phase.

    q: (batch, n_heads_q, seq_len, head_dim)
    k: (batch, n_heads_kv, seq_len, head_dim)
    v: (batch, n_heads_kv, seq_len, head_dim)
    """
    batch, n_heads_q, seq_len, head_dim = q.shape
    n_heads_kv = k.shape[1]

    out = torch.empty_like(q)
    scale = 1.0 / (head_dim ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = head_dim

    grid = (triton.cdiv(seq_len, BLOCK_M), n_heads_q, batch)

    attention_prefill_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        n_heads_q, n_heads_kv,
        seq_len,
        head_dim,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def attention_decode_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: int,
) -> torch.Tensor:
    """Decode attention using PyTorch (single token attends to all cached tokens).

    q: (batch, n_heads_q, 1, head_dim)
    k_cache: (batch, n_heads_kv, max_seq_len, head_dim)
    v_cache: (batch, n_heads_kv, max_seq_len, head_dim)
    """
    batch, n_heads_q, _, head_dim = q.shape
    n_heads_kv = k_cache.shape[1]
    n_groups = n_heads_q // n_heads_kv

    # Slice cache to actual length
    k = k_cache[:, :, :cache_len, :]  # (batch, n_heads_kv, cache_len, head_dim)
    v = v_cache[:, :, :cache_len, :]

    # Expand KV for GQA
    k = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)  # (batch, n_heads_kv, n_groups, cache_len, head_dim)
    k = k.reshape(batch, n_heads_q, cache_len, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
    v = v.reshape(batch, n_heads_q, cache_len, head_dim)

    # Standard attention
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # (batch, n_heads_q, 1, cache_len)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())  # (batch, n_heads_q, 1, head_dim)

    return out.to(q.dtype)


# ============================================================================
# Model Components
# ============================================================================

class TritonQwen3Attention:
    """Attention layer using Triton kernels."""

    def __init__(self, layer_weights, config: Qwen3Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx

        # Load weights
        self.q_proj_weight = layer_weights["q_proj.weight"]
        self.k_proj_weight = layer_weights["k_proj.weight"]
        self.v_proj_weight = layer_weights["v_proj.weight"]
        self.o_proj_weight = layer_weights["o_proj.weight"]
        self.q_norm_weight = layer_weights["q_norm.weight"]
        self.k_norm_weight = layer_weights["k_norm.weight"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        cache_position: int = 0,
        is_prefill: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape

        # QKV projections
        q = torch.nn.functional.linear(hidden_states, self.q_proj_weight)
        k = torch.nn.functional.linear(hidden_states, self.k_proj_weight)
        v = torch.nn.functional.linear(hidden_states, self.v_proj_weight)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2).contiguous()

        # Apply QK norm (Qwen3 specific)
        q = triton_rms_norm(q.contiguous(), self.q_norm_weight, self.config.rms_norm_eps)
        k = triton_rms_norm(k.contiguous(), self.k_norm_weight, self.config.rms_norm_eps)

        # Apply RoPE
        q, k = apply_rope_simple(q, k, cos, sin, position_ids)

        if is_prefill:
            # Prefill: use flash attention
            # Expand KV for GQA during prefill
            n_groups = self.config.num_attention_heads // self.config.num_key_value_heads
            k_expanded = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
            k_expanded = k_expanded.reshape(batch, self.config.num_attention_heads, seq_len, self.config.head_dim)
            v_expanded = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
            v_expanded = v_expanded.reshape(batch, self.config.num_attention_heads, seq_len, self.config.head_dim)

            # Use PyTorch attention for correctness (Triton flash attention can be added later)
            scale = 1.0 / (self.config.head_dim ** 0.5)
            scores = torch.matmul(q.float(), k_expanded.float().transpose(-2, -1)) * scale
            # Causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v_expanded.float()).to(q.dtype)

            # Update cache
            if k_cache is not None:
                k_cache[:, :, :seq_len, :] = k
                v_cache[:, :, :seq_len, :] = v
        else:
            # Decode: update cache and attend to all cached tokens
            k_cache[:, :, cache_position:cache_position+1, :] = k
            v_cache[:, :, cache_position:cache_position+1, :] = v
            attn_out = attention_decode_torch(q, k_cache, v_cache, cache_position + 1)

        # Reshape and output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = torch.nn.functional.linear(attn_out, self.o_proj_weight)

        return output, k_cache, v_cache


class TritonQwen3MLP:
    """MLP layer using Triton kernels."""

    def __init__(self, layer_weights, config: Qwen3Config):
        self.config = config
        self.gate_proj_weight = layer_weights["gate_proj.weight"]
        self.up_proj_weight = layer_weights["up_proj.weight"]
        self.down_proj_weight = layer_weights["down_proj.weight"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.linear(hidden_states, self.gate_proj_weight)
        up = torch.nn.functional.linear(hidden_states, self.up_proj_weight)

        # Fused SiLU * up
        hidden = triton_silu_mul(gate, up)

        output = torch.nn.functional.linear(hidden, self.down_proj_weight)
        return output


class TritonQwen3Layer:
    """Single transformer layer."""

    def __init__(self, layer_weights, config: Qwen3Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx

        # RMSNorm weights
        self.input_layernorm_weight = layer_weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = layer_weights["post_attention_layernorm.weight"]

        # Sublayers
        attn_weights = {k.replace("self_attn.", ""): v for k, v in layer_weights.items() if "self_attn" in k}
        mlp_weights = {k.replace("mlp.", ""): v for k, v in layer_weights.items() if "mlp" in k}

        self.self_attn = TritonQwen3Attention(attn_weights, config, layer_idx)
        self.mlp = TritonQwen3MLP(mlp_weights, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        cache_position: int = 0,
        is_prefill: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = triton_rms_norm(hidden_states, self.input_layernorm_weight, self.config.rms_norm_eps)
        hidden_states, k_cache, v_cache = self.self_attn.forward(
            hidden_states, cos, sin, position_ids, k_cache, v_cache, cache_position, is_prefill
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = triton_rms_norm(hidden_states, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, k_cache, v_cache


class TritonQwen3Model:
    """Full Qwen3 model with Triton kernels."""

    def __init__(self, hf_model, config: Qwen3Config):
        self.config = config
        self.device = next(hf_model.parameters()).device
        self.dtype = torch.bfloat16

        # Extract weights from HF model
        state_dict = hf_model.state_dict()

        # Embedding
        self.embed_tokens = state_dict["model.embed_tokens.weight"]

        # Final norm
        self.final_norm_weight = state_dict["model.norm.weight"]

        # LM head (tied with embedding in Qwen3)
        self.lm_head_weight = self.embed_tokens  # Tied weights

        # Layers
        self.layers = []
        for i in range(config.num_hidden_layers):
            layer_weights = {
                k.replace(f"model.layers.{i}.", ""): v
                for k, v in state_dict.items()
                if f"model.layers.{i}." in k
            }
            self.layers.append(TritonQwen3Layer(layer_weights, config, i))

        # Precompute RoPE frequencies
        self.cos, self.sin = precompute_rope_freqs(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            self.device,
        )

    def prefill(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Prefill phase: process the entire prompt.

        Returns logits and KV cache for all layers.
        """
        batch, seq_len = input_ids.shape

        # Embedding lookup
        hidden_states = torch.nn.functional.embedding(input_ids, self.embed_tokens)

        # Position IDs
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        # Initialize KV caches
        kv_caches = []
        max_cache_len = seq_len + 256  # Allow for some decode tokens

        for layer in self.layers:
            k_cache = torch.zeros(
                batch, self.config.num_key_value_heads, max_cache_len, self.config.head_dim,
                device=self.device, dtype=self.dtype
            )
            v_cache = torch.zeros(
                batch, self.config.num_key_value_heads, max_cache_len, self.config.head_dim,
                device=self.device, dtype=self.dtype
            )

            hidden_states, k_cache, v_cache = layer.forward(
                hidden_states, self.cos, self.sin, position_ids,
                k_cache, v_cache, 0, is_prefill=True
            )
            kv_caches.append((k_cache, v_cache))

        # Final norm
        hidden_states = triton_rms_norm(hidden_states, self.final_norm_weight, self.config.rms_norm_eps)

        # LM head
        logits = torch.nn.functional.linear(hidden_states, self.lm_head_weight)

        return logits, kv_caches, seq_len

    def decode_step(
        self,
        input_id: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        cache_position: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Decode phase: process a single token.

        input_id: (batch, 1)
        """
        # Embedding lookup
        hidden_states = torch.nn.functional.embedding(input_id, self.embed_tokens)

        # Position IDs
        position_ids = torch.tensor([[cache_position]], device=self.device)

        # Process through layers
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = kv_caches[i]
            hidden_states, k_cache, v_cache = layer.forward(
                hidden_states, self.cos, self.sin, position_ids,
                k_cache, v_cache, cache_position, is_prefill=False
            )
            new_kv_caches.append((k_cache, v_cache))

        # Final norm
        hidden_states = triton_rms_norm(hidden_states, self.final_norm_weight, self.config.rms_norm_eps)

        # LM head
        logits = torch.nn.functional.linear(hidden_states, self.lm_head_weight)

        return logits, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        """Generate tokens using greedy decoding."""
        # Prefill
        logits, kv_caches, cache_len = self.prefill(input_ids)

        # Get first generated token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens = [next_token]

        # Decode loop
        for i in range(max_new_tokens - 1):
            logits, kv_caches = self.decode_step(next_token, kv_caches, cache_len + i)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

            # Stop on EOS
            if next_token.item() == 151645:  # eos_token_id
                break

        return torch.cat([input_ids] + generated_tokens, dim=1)


# ============================================================================
# Verification
# ============================================================================

def verify_single_prompt(tokenizer, hf_model, triton_model, prompt: str, max_new_tokens: int = 30, debug: bool = False) -> bool:
    """Verify Triton matches HuggingFace for a single prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    if debug:
        print(f"  Input length: {input_ids.shape[1]}")

    # HuggingFace generation (greedy)
    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Triton generation (greedy)
    triton_output = triton_model.generate(input_ids, max_new_tokens=max_new_tokens)

    # Compare
    min_len = min(len(hf_output[0]), len(triton_output[0]))
    match = torch.all(hf_output[0][:min_len] == triton_output[0][:min_len])

    return match.item(), hf_output, triton_output


def verify_outputs():
    """Verify Triton implementation matches HuggingFace on multiple prompts."""
    print("Loading HuggingFace model...")
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    hf_model.eval()

    print("Building Triton model...")
    config = Qwen3Config()
    triton_model = TritonQwen3Model(hf_model, config)

    # Test prompts - varying lengths and content
    test_prompts = [
        ("The capital of France is", 30),
        ("Write a Python function to compute fibonacci numbers", 50),
        ("Explain quantum computing in simple terms", 40),
        ("1 + 1 =", 10),
        ("The quick brown fox", 20),
        ("In machine learning, gradient descent is used to", 35),
    ]

    all_match = True
    results = []

    for i, (prompt, max_tokens) in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        print(f"{'='*60}")

        match, hf_output, triton_output = verify_single_prompt(
            tokenizer, hf_model, triton_model, prompt, max_tokens
        )

        hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
        triton_text = tokenizer.decode(triton_output[0], skip_special_tokens=True)

        print(f"HF output: {hf_text[-100:]}...")
        print(f"Triton output: {triton_text[-100:]}...")
        print(f"Match: {match}")

        results.append((prompt[:40], match))
        if not match:
            all_match = False
            min_len = min(len(hf_output[0]), len(triton_output[0]))
            for j in range(min_len):
                if hf_output[0][j] != triton_output[0][j]:
                    print(f"  First mismatch at position {j}: HF={hf_output[0][j].item()}, Triton={triton_output[0][j].item()}")
                    break

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for prompt, match in results:
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {prompt}...")

    passed = sum(1 for _, m in results if m)
    print(f"\nPassed: {passed}/{len(results)}")

    return all_match


def main():
    all_match = verify_outputs()
    if all_match:
        print("\n[SUCCESS] All tests passed - Triton implementation matches HuggingFace exactly!")
    else:
        print("\n[MISMATCH] Some outputs differ - debugging needed")


if __name__ == "__main__":
    main()
