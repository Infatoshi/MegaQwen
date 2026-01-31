"""Compare prefill vs decode-only outputs."""
import torch

print("Comparing prefill vs decode-only...")

from megakernel_decode import MegakernelGenerator, MegakernelPrefillGenerator

print("Loading decode-only generator...")
gen_decode = MegakernelGenerator()
print("Loading prefill generator...")
gen_prefill = MegakernelPrefillGenerator()

prompt = "The quick brown fox"
print(f"\nPrompt: '{prompt}'")

# Decode-only approach
print("\n=== Decode-only ===")
gen_decode.decoder.reset()
input_ids = gen_decode.tokenizer.encode(prompt, add_special_tokens=True)
print(f"Input tokens: {input_ids}")

# Run tokens through decode
for tok in input_ids[:-1]:
    gen_decode.decoder.decode_step(tok)
current = input_ids[-1]
decode_tokens = []
for _ in range(10):
    current = gen_decode.decoder.decode_step(current)
    decode_tokens.append(current)
print(f"Generated tokens: {decode_tokens}")
print(f"Generated text: '{gen_decode.tokenizer.decode(decode_tokens)}'")

# Prefill approach
print("\n=== Prefill + Decode ===")
gen_prefill.decoder.reset()
input_tensor = torch.tensor(input_ids, dtype=torch.int32)
current = gen_prefill.decoder.prefill_step(input_tensor)
prefill_tokens = []
for _ in range(10):
    if current == gen_prefill.tokenizer.eos_token_id:
        break
    prefill_tokens.append(current)
    current = gen_prefill.decoder.decode_step(current)
print(f"Generated tokens: {prefill_tokens}")
print(f"Generated text: '{gen_prefill.tokenizer.decode(prefill_tokens)}'")

# Compare
print("\n=== Comparison ===")
if decode_tokens[:5] == prefill_tokens[:5]:
    print("First 5 tokens MATCH!")
else:
    print("MISMATCH!")
    print(f"  Decode:  {decode_tokens[:5]}")
    print(f"  Prefill: {prefill_tokens[:5]}")
