"""Comprehensive benchmark suite: PyTorch vs Megakernel."""
import gc
import sys
import time
from dataclasses import dataclass

import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "csrc/megakernel")

@dataclass
class BenchmarkResult:
    name: str
    tokens: int
    time_s: float
    toks_per_s: float
    ms_per_tok: float

def run_benchmark(name: str, generate_fn, warmup: int = 2, runs: int = 3, tokens: int = 100) -> BenchmarkResult:
    """Run benchmark with warmup and multiple runs."""
    for _ in range(warmup):
        generate_fn()

    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate_fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    return BenchmarkResult(
        name=name,
        tokens=tokens,
        time_s=avg_time,
        toks_per_s=tokens / avg_time,
        ms_per_tok=avg_time * 1000 / tokens
    )

def benchmark_pytorch_hf(decode_tokens: int = 100) -> BenchmarkResult:
    """Benchmark HuggingFace PyTorch model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()

    input_ids = tokenizer("Hello", return_tensors="pt").input_ids.cuda()

    def generate():
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=decode_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

    result = run_benchmark("PyTorch (HF)", generate, tokens=decode_tokens)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result

def benchmark_megakernel(decode_tokens: int = 100) -> BenchmarkResult:
    """Benchmark megakernel."""
    from megakernel_decode import MegakernelGenerator

    print("  Loading megakernel...")
    gen = MegakernelGenerator()

    def generate():
        gen.decoder.reset()
        gen.generate("Hello", max_new_tokens=decode_tokens)

    result = run_benchmark("Megakernel", generate, tokens=decode_tokens)

    del gen
    gc.collect()
    torch.cuda.empty_cache()

    return result

def main():
    print("=" * 60)
    print("MegaQwen Benchmark Suite")
    print("=" * 60)
    print()

    results = []

    print("[1/2] PyTorch (HuggingFace)")
    results.append(benchmark_pytorch_hf())

    print("[2/2] Megakernel")
    results.append(benchmark_megakernel())

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"{'Backend':<25} {'tok/s':>10} {'ms/tok':>10} {'Speedup':>10}")
    print("-" * 60)

    baseline = results[0].toks_per_s
    for r in results:
        speedup = r.toks_per_s / baseline
        print(f"{r.name:<25} {r.toks_per_s:>10.1f} {r.ms_per_tok:>10.2f} {speedup:>10.2f}x")

if __name__ == "__main__":
    main()
