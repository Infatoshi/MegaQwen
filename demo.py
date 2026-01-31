"""Quick demo of the megakernel for Qwen3-0.6B."""
import sys
import time
sys.path.insert(0, "csrc/megakernel")

from megakernel_decode import MegakernelGenerator

print("=" * 60)
print("MegaQwen Demo - Qwen3-0.6B Megakernel Inference")
print("=" * 60)
print()

print("Loading model and compiling kernels...")
gen = MegakernelGenerator()

# Quick warmup
print("Warming up...")
gen.generate("Hello", max_new_tokens=5)
print("Ready!\n")

prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain quantum computing in one sentence:",
]

for prompt in prompts:
    print("-" * 60)
    print(f"Prompt: {prompt}")
    print()

    gen.decoder.reset()

    start = time.perf_counter()
    tokens_generated = 0

    print("Response: ", end="", flush=True)
    for token_str in gen.generate_stream(prompt, max_new_tokens=50):
        print(token_str, end="", flush=True)
        tokens_generated += 1

    elapsed = time.perf_counter() - start
    print()
    print(f"\n[{tokens_generated} tokens in {elapsed:.2f}s = {tokens_generated/elapsed:.1f} tok/s]")
    print()

print("=" * 60)
print("Demo complete!")
