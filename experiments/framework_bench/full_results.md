# Full Framework Benchmark - Qwen3-0.6B

**Prompt**: "Write a detailed essay about lobsters, covering th..."
**Max Tokens**: 200

## Results

| Framework | tok/s | Avg Power (W) | Peak Power (W) | tok/J | Argmax Match | KL Div | Status |
|-----------|-------|---------------|----------------|-------|--------------|--------|--------|
| HuggingFace | 59.0 | 185.8 | 191.7 | 0.32 | 100% | 0.000000 | ok |
| Megakernel | 157.8 | 204.9 | 232.5 | 0.77 | 0% | 0.000582 | ok |
| vLLM | 107.4 | 195.6 | 205.7 | 0.55 | 100% | - | ok |
| llama.cpp | 50.3 | 194.5 | 200.7 | 0.26 | - | - | ok |