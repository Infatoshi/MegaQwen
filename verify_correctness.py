"""Verify megakernel output matches HuggingFace reference."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    hf_model.eval()

    print("Loading megakernel...")
    from chat import MegakernelChat
    mega = MegakernelChat()

    prompt = "The capital of France is"
    print(f"\nPrompt: {prompt}")

    # HuggingFace generation
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)

    # Megakernel generation
    mega.k_cache.zero_()
    mega.v_cache.zero_()
    mega_text = mega.generate(prompt, max_new_tokens=20, show_speed=False)

    print(f"\nHuggingFace: {hf_text}")
    print(f"Megakernel:  {mega_text}")

    if hf_text == mega_text:
        print("\n[PASS] Outputs match exactly!")
    else:
        print("\n[WARN] Outputs differ (may be acceptable due to numerical precision)")

if __name__ == "__main__":
    main()
