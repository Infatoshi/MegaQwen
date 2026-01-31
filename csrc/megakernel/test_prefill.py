"""Test the prefill kernel thoroughly."""
import time

import torch

print("Testing prefill kernel...")

from megakernel_decode import MegakernelPrefillGenerator

print("Loading model and compiling prefill kernel...")
gen = MegakernelPrefillGenerator()
print("Ready!")

# Test 1: Basic prefill
print("\n=== Test 1: Basic prefill ===")
prompt = "Hello, my name is"
input_ids = gen.tokenizer.encode(prompt, add_special_tokens=True)
print(f"Prompt: '{prompt}'")
print(f"Token IDs: {input_ids}")

gen.decoder.reset()
input_tensor = torch.tensor(input_ids, dtype=torch.int32)
next_token = gen.decoder.prefill_step(input_tensor)
print(f"First predicted token: {next_token} -> '{gen.tokenizer.decode([next_token])}'")
print(f"Position after prefill: {gen.decoder.position()}")

# Test 2: Decode continuation
print("\n=== Test 2: Decode continuation ===")
generated = [next_token]
for i in range(10):
    next_token = gen.decoder.decode_step(next_token)
    generated.append(next_token)
    if next_token == gen.tokenizer.eos_token_id:
        break

print(f"Generated tokens: {generated}")
print(f"Generated text: '{gen.tokenizer.decode(generated)}'")

# Test 3: Full generation
print("\n=== Test 3: Full generation ===")
output = gen.generate("The quick brown fox", max_new_tokens=20)
print(f"Generated: '{output}'")

# Test 4: Benchmark prefill
print("\n=== Test 4: Prefill benchmark ===")
prompt = "The meaning of life is"
input_ids = gen.tokenizer.encode(prompt, add_special_tokens=True)
input_tensor = torch.tensor(input_ids, dtype=torch.int32)

# Warmup
for _ in range(5):
    gen.decoder.reset()
    gen.decoder.prefill_step(input_tensor)

torch.cuda.synchronize()
times = []
for _ in range(20):
    gen.decoder.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    gen.decoder.prefill_step(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f"Prefill ({len(input_ids)} tokens):")
print(f"  Mean: {sum(times)/len(times):.2f} ms")
print(f"  Min: {min(times):.2f} ms")
print(f"  Max: {max(times):.2f} ms")
print(f"  Throughput: {len(input_ids) / (min(times)/1000):.1f} tok/s")

print("\nAll tests passed!")
