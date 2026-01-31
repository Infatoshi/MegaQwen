"""Benchmark prefill vs decode-only for prompt processing."""
import time

import torch

print("Loading generators...")

from megakernel_decode import MegakernelGenerator, MegakernelPrefillGenerator

gen_decode = MegakernelGenerator()
gen_prefill = MegakernelPrefillGenerator()

# Test prompts of various lengths
test_prompts = [
    "Hello",
    "The quick brown fox",
    "The quick brown fox jumps over the lazy dog",
    "Once upon a time, in a land far far away, there lived a brave knight",
    "Artificial intelligence has transformed the way we interact with technology, enabling new possibilities that were once thought impossible",
]

print("\n" + "=" * 70)
print("Prefill vs Decode-only Benchmark")
print("=" * 70)

for prompt in test_prompts:
    tokens = gen_decode.tokenizer.encode(prompt, add_special_tokens=False)
    num_tokens = len(tokens)

    # Warmup
    for _ in range(3):
        gen_decode.decoder.reset()
        for tok in tokens:
            gen_decode.decoder.decode_step(tok)

        gen_prefill.decoder.reset()
        input_tensor = torch.tensor(tokens, dtype=torch.int32)
        gen_prefill.decoder.prefill_step(input_tensor)

    torch.cuda.synchronize()

    # Benchmark decode-only
    decode_times = []
    for _ in range(20):
        gen_decode.decoder.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for tok in tokens:
            gen_decode.decoder.decode_step(tok)
        torch.cuda.synchronize()
        end = time.perf_counter()
        decode_times.append((end - start) * 1000)

    # Benchmark prefill
    prefill_times = []
    input_tensor = torch.tensor(tokens, dtype=torch.int32)
    for _ in range(20):
        gen_prefill.decoder.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        gen_prefill.decoder.prefill_step(input_tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()
        prefill_times.append((end - start) * 1000)

    decode_mean = sum(decode_times) / len(decode_times)
    decode_min = min(decode_times)
    prefill_mean = sum(prefill_times) / len(prefill_times)
    prefill_min = min(prefill_times)

    speedup = decode_min / prefill_min

    print(f"\nPrompt: '{prompt[:40]}{'...' if len(prompt) > 40 else ''}'")
    print(f"  Tokens: {num_tokens}")
    print(f"  Decode-only: {decode_min:.2f} ms (mean: {decode_mean:.2f} ms)")
    print(f"  Prefill:     {prefill_min:.2f} ms (mean: {prefill_mean:.2f} ms)")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"  Decode throughput:  {num_tokens / (decode_min/1000):.0f} tok/s")
    print(f"  Prefill throughput: {num_tokens / (prefill_min/1000):.0f} tok/s")

print("\n" + "=" * 70)
print("Correctness verification:")
print("=" * 70)

# Verify outputs match
for prompt in test_prompts[:3]:
    tokens = gen_decode.tokenizer.encode(prompt, add_special_tokens=False)

    gen_decode.decoder.reset()
    for tok in tokens:
        decode_result = gen_decode.decoder.decode_step(tok)

    gen_prefill.decoder.reset()
    input_tensor = torch.tensor(tokens, dtype=torch.int32)
    prefill_result = gen_prefill.decoder.prefill_step(input_tensor)

    match = "MATCH" if decode_result == prefill_result else "MISMATCH"
    print(f"  '{prompt[:30]}...': {match}")
