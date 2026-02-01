"""
Comprehensive benchmark for Qwen3-0.6B across all frameworks.

Measures throughput, power consumption, and quality metrics sequentially.
"""

import os
import sys
import time
import gc
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F

MODEL_NAME = "Qwen/Qwen3-0.6B"
GGUF_PATH = "/tmp/qwen3_gguf/qwen3-0.6b-f16.gguf"
TEST_PROMPT = "Write a detailed essay about lobsters, covering their biology, habitat, and importance to marine ecosystems."
MAX_TOKENS = 200
VOCAB_SIZE = 151936


@dataclass
class BenchmarkResult:
    framework: str
    supported: bool
    tok_per_s: Optional[float] = None
    avg_power_w: Optional[float] = None
    peak_power_w: Optional[float] = None
    tok_per_joule: Optional[float] = None
    argmax_match: Optional[float] = None
    kl_divergence: Optional[float] = None
    error: Optional[str] = None


class PowerMonitor:
    def __init__(self, gpu_id: int = 0, sample_interval_ms: int = 50):
        self.gpu_id = gpu_id
        self.sample_interval_ms = sample_interval_ms
        self.readings = []
        self._stop = False
        self._thread = None

    def _read_power(self) -> float:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", f"--id={self.gpu_id}"],
                capture_output=True, text=True, timeout=1
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _monitor_loop(self):
        while not self._stop:
            self.readings.append(self._read_power())
            time.sleep(self.sample_interval_ms / 1000.0)

    def start(self):
        self.readings = []
        self._stop = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[float, float]:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)
        if not self.readings:
            return 0.0, 0.0
        return max(self.readings), sum(self.readings) / len(self.readings)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    time.sleep(2)


def kl_divergence(ref_logits: torch.Tensor, tgt_logits: torch.Tensor) -> float:
    ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
    tgt_log_probs = F.log_softmax(tgt_logits.float(), dim=-1)
    ref_probs = ref_log_probs.exp()
    kl = (ref_probs * (ref_log_probs - tgt_log_probs)).sum(dim=-1).mean()
    return float(kl.item())


# =============================================================================
# HuggingFace
# =============================================================================

def benchmark_huggingface(monitor: PowerMonitor, ref_logits=None) -> Tuple[BenchmarkResult, Optional[torch.Tensor]]:
    print("\n" + "="*60)
    print("HUGGINGFACE")
    print("="*60)
    clear_gpu()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True
        )
        model.eval()

        # Warmup
        print("Warming up...")
        input_ids = tokenizer(TEST_PROMPT, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()

        # Benchmark with power
        print(f"Generating {MAX_TOKENS} tokens...")
        monitor.start()
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=MAX_TOKENS, do_sample=False)

        torch.cuda.synchronize()
        end = time.perf_counter()
        peak_power, avg_power = monitor.stop()

        gen_time = end - start
        tokens = output.shape[1] - input_ids.shape[1]
        tok_per_s = tokens / gen_time
        energy_joules = avg_power * gen_time
        tok_per_joule = tokens / energy_joules if energy_joules > 0 else 0

        # Get logits for quality comparison
        with torch.no_grad():
            logits_out = model(input_ids, use_cache=False)
        last_logits = logits_out.logits[:, -1, :].clone()

        del model, tokenizer
        clear_gpu()

        print(f"  Throughput: {tok_per_s:.1f} tok/s")
        print(f"  Power: {avg_power:.1f}W avg, {peak_power:.1f}W peak")
        print(f"  Energy: {tok_per_joule:.2f} tok/J")

        return BenchmarkResult(
            "HuggingFace", True, tok_per_s, avg_power, peak_power, tok_per_joule, 1.0, 0.0
        ), last_logits

    except Exception as e:
        import traceback
        traceback.print_exc()
        clear_gpu()
        return BenchmarkResult("HuggingFace", False, error=str(e)), None


# =============================================================================
# Megakernel
# =============================================================================

def benchmark_megakernel(monitor: PowerMonitor, ref_logits: torch.Tensor) -> BenchmarkResult:
    print("\n" + "="*60)
    print("MEGAKERNEL")
    print("="*60)
    clear_gpu()

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        from chat import MegakernelChat, NUM_LAYERS, MAX_SEQ_LEN
        from transformers import AutoTokenizer

        print("Loading megakernel...")
        chat = MegakernelChat()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

        # Warmup
        print("Warming up...")
        _ = chat.generate(TEST_PROMPT, max_new_tokens=10, show_speed=False)
        torch.cuda.synchronize()

        # Benchmark with power
        print(f"Generating {MAX_TOKENS} tokens...")
        chat.k_cache.zero_()
        chat.v_cache.zero_()

        monitor.start()
        torch.cuda.synchronize()
        start = time.perf_counter()

        output = chat.generate(TEST_PROMPT, max_new_tokens=MAX_TOKENS, show_speed=False)

        torch.cuda.synchronize()
        end = time.perf_counter()
        peak_power, avg_power = monitor.stop()

        gen_time = end - start
        tokens = len(tokenizer.encode(output)) - len(tokenizer.encode(TEST_PROMPT))
        tok_per_s = tokens / gen_time
        energy_joules = avg_power * gen_time
        tok_per_joule = tokens / energy_joules if energy_joules > 0 else 0

        # Quality check - get logits for last position
        chat.k_cache.zero_()
        chat.v_cache.zero_()
        input_ids = tokenizer.encode(TEST_PROMPT)

        for pos, token_id in enumerate(input_ids[:-1]):
            chat.kernel.decode_ldg(
                token_id, chat.final_norm_weight, chat.lm_head_weight,
                chat.cos_table, chat.sin_table,
                chat.k_cache, chat.v_cache,
                chat.hidden_buffer, chat.g_activations, chat.g_residual,
                chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
                chat.g_mlp_intermediate, chat.g_normalized,
                chat.block_max_vals, chat.block_max_idxs,
                NUM_LAYERS, pos, pos + 1, MAX_SEQ_LEN,
            )

        token_id, logits = chat.kernel.decode_ldg_with_logits(
            input_ids[-1], chat.final_norm_weight, chat.lm_head_weight,
            chat.cos_table, chat.sin_table,
            chat.k_cache, chat.v_cache,
            chat.hidden_buffer, chat.g_activations, chat.g_residual,
            chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
            chat.g_mlp_intermediate, chat.g_normalized,
            chat.block_max_vals, chat.block_max_idxs,
            NUM_LAYERS, len(input_ids) - 1, len(input_ids), MAX_SEQ_LEN,
        )

        mega_logits = logits.unsqueeze(0)
        kl = kl_divergence(ref_logits, mega_logits)
        argmax_match = 1.0 if ref_logits.argmax().item() == mega_logits.argmax().item() else 0.0

        del chat
        clear_gpu()

        print(f"  Throughput: {tok_per_s:.1f} tok/s")
        print(f"  Power: {avg_power:.1f}W avg, {peak_power:.1f}W peak")
        print(f"  Energy: {tok_per_joule:.2f} tok/J")
        print(f"  KL Divergence: {kl:.6f}")
        print(f"  Argmax Match: {argmax_match:.0%}")

        return BenchmarkResult(
            "Megakernel", True, tok_per_s, avg_power, peak_power, tok_per_joule, argmax_match, kl
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        clear_gpu()
        return BenchmarkResult("Megakernel", False, error=str(e))


# =============================================================================
# vLLM
# =============================================================================

def benchmark_vllm(monitor: PowerMonitor, ref_logits: torch.Tensor) -> BenchmarkResult:
    print("\n" + "="*60)
    print("vLLM")
    print("="*60)
    clear_gpu()

    try:
        from vllm import LLM, SamplingParams

        print("Loading vLLM...")
        llm = LLM(model=MODEL_NAME, dtype="bfloat16", gpu_memory_utilization=0.8)

        # Warmup
        print("Warming up...")
        params = SamplingParams(temperature=0.0, max_tokens=10)
        _ = llm.generate([TEST_PROMPT], params, use_tqdm=False)

        # Benchmark with power
        print(f"Generating {MAX_TOKENS} tokens...")
        params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

        monitor.start()
        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = llm.generate([TEST_PROMPT], params, use_tqdm=False)

        torch.cuda.synchronize()
        end = time.perf_counter()
        peak_power, avg_power = monitor.stop()

        gen_time = end - start
        tokens = len(outputs[0].outputs[0].token_ids) if outputs else MAX_TOKENS
        tok_per_s = tokens / gen_time
        energy_joules = avg_power * gen_time
        tok_per_joule = tokens / energy_joules if energy_joules > 0 else 0

        # Argmax check
        params_check = SamplingParams(temperature=0.0, max_tokens=1)
        check_output = llm.generate([TEST_PROMPT], params_check, use_tqdm=False)
        vllm_pred = check_output[0].outputs[0].token_ids[0] if check_output[0].outputs[0].token_ids else -1
        hf_pred = ref_logits.argmax().item()
        argmax_match = 1.0 if vllm_pred == hf_pred else 0.0

        del llm
        clear_gpu()

        print(f"  Throughput: {tok_per_s:.1f} tok/s")
        print(f"  Power: {avg_power:.1f}W avg, {peak_power:.1f}W peak")
        print(f"  Energy: {tok_per_joule:.2f} tok/J")
        print(f"  Argmax Match: {argmax_match:.0%}")

        return BenchmarkResult(
            "vLLM", True, tok_per_s, avg_power, peak_power, tok_per_joule, argmax_match, None
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        clear_gpu()
        return BenchmarkResult("vLLM", False, error=str(e))


# =============================================================================
# llama.cpp
# =============================================================================

def benchmark_llamacpp(monitor: PowerMonitor, ref_logits: torch.Tensor) -> BenchmarkResult:
    print("\n" + "="*60)
    print("LLAMA.CPP")
    print("="*60)
    clear_gpu()

    if not os.path.exists(GGUF_PATH):
        return BenchmarkResult("llama.cpp", False, error="GGUF file not found")

    try:
        from llama_cpp import Llama

        print("Loading model...")
        llm = Llama(
            model_path=GGUF_PATH,
            n_gpu_layers=-1,
            n_ctx=512,
            verbose=False,
        )

        # Warmup
        print("Warming up...")
        _ = llm(TEST_PROMPT, max_tokens=10)

        # Benchmark with power
        print(f"Generating {MAX_TOKENS} tokens...")

        monitor.start()
        start = time.perf_counter()

        output = llm(TEST_PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)

        end = time.perf_counter()
        peak_power, avg_power = monitor.stop()

        gen_time = end - start
        tokens = output['usage']['completion_tokens']
        tok_per_s = tokens / gen_time
        energy_joules = avg_power * gen_time
        tok_per_joule = tokens / energy_joules if energy_joules > 0 else 0

        # Argmax check - get first predicted token
        check_output = llm(TEST_PROMPT, max_tokens=1, temperature=0.0)
        # llama.cpp doesn't easily expose token IDs, so we skip argmax match

        del llm
        clear_gpu()

        print(f"  Throughput: {tok_per_s:.1f} tok/s")
        print(f"  Power: {avg_power:.1f}W avg, {peak_power:.1f}W peak")
        print(f"  Energy: {tok_per_joule:.2f} tok/J")

        return BenchmarkResult(
            "llama.cpp", True, tok_per_s, avg_power, peak_power, tok_per_joule, None, None
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        clear_gpu()
        return BenchmarkResult("llama.cpp", False, error=str(e))


# =============================================================================
# Main
# =============================================================================

def format_results(results: List[BenchmarkResult]) -> str:
    lines = [
        "# Full Framework Benchmark - Qwen3-0.6B",
        "",
        f"**Prompt**: \"{TEST_PROMPT[:50]}...\"",
        f"**Max Tokens**: {MAX_TOKENS}",
        "",
        "## Results",
        "",
        "| Framework | tok/s | Avg Power (W) | Peak Power (W) | tok/J | Argmax Match | KL Div | Status |",
        "|-----------|-------|---------------|----------------|-------|--------------|--------|--------|",
    ]

    for r in results:
        if r.supported:
            tok_s = f"{r.tok_per_s:.1f}" if r.tok_per_s else "-"
            avg_p = f"{r.avg_power_w:.1f}" if r.avg_power_w else "-"
            peak_p = f"{r.peak_power_w:.1f}" if r.peak_power_w else "-"
            tok_j = f"{r.tok_per_joule:.2f}" if r.tok_per_joule else "-"
            argmax = f"{r.argmax_match:.0%}" if r.argmax_match is not None else "-"
            kl = f"{r.kl_divergence:.6f}" if r.kl_divergence is not None else "-"
            lines.append(f"| {r.framework} | {tok_s} | {avg_p} | {peak_p} | {tok_j} | {argmax} | {kl} | ok |")
        else:
            lines.append(f"| {r.framework} | - | - | - | - | - | - | {r.error or 'skipped'} |")

    return "\n".join(lines)


def main() -> int:
    print("="*60)
    print("FULL FRAMEWORK BENCHMARK - Qwen3-0.6B")
    print("="*60)

    monitor = PowerMonitor(gpu_id=0, sample_interval_ms=50)
    results = []

    # HuggingFace (reference)
    hf_result, ref_logits = benchmark_huggingface(monitor)
    results.append(hf_result)

    if ref_logits is None:
        print("ERROR: HuggingFace failed, cannot continue")
        return 1

    # Other frameworks
    results.append(benchmark_megakernel(monitor, ref_logits))
    results.append(benchmark_vllm(monitor, ref_logits))
    results.append(benchmark_llamacpp(monitor, ref_logits))

    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    output = format_results(results)
    print(output)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "full_results.md")
    with open(output_path, "w") as f:
        f.write(output)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
