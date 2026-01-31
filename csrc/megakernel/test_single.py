"""Test single-token prefill vs decode."""
import torch
from megakernel_decode import MegakernelGenerator, MegakernelPrefillGenerator

print("Loading generators...")
gen_decode = MegakernelGenerator()
gen_prefill = MegakernelPrefillGenerator()

print("\nSingle-token test:")
token_id = 785  # "The"

# Decode approach
gen_decode.decoder.reset()
result_decode = gen_decode.decoder.decode_step(token_id)
print(f"Decode result for token {token_id}: {result_decode}")

# Prefill approach
gen_prefill.decoder.reset()
input_tensor = torch.tensor([token_id], dtype=torch.int32)
result_prefill = gen_prefill.decoder.prefill_step(input_tensor)
print(f"Prefill result for token {token_id}: {result_prefill}")

if result_decode == result_prefill:
    print("MATCH!")
else:
    print(f"MISMATCH! decode={result_decode}, prefill={result_prefill}")

# Test with 2 tokens
print("\nTwo-token test:")
tokens = [785, 3974]  # "The quick"

gen_decode.decoder.reset()
gen_decode.decoder.decode_step(tokens[0])
result_decode = gen_decode.decoder.decode_step(tokens[1])
print(f"Decode result: {result_decode}")

gen_prefill.decoder.reset()
input_tensor = torch.tensor(tokens, dtype=torch.int32)
result_prefill = gen_prefill.decoder.prefill_step(input_tensor)
print(f"Prefill result: {result_prefill}")

if result_decode == result_prefill:
    print("MATCH!")
else:
    print(f"MISMATCH! decode={result_decode}, prefill={result_prefill}")
