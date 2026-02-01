# MegaQwen

Custom CUDA megakernel for Qwen3-0.6B inference achieving **240 tok/s decode** on RTX 3090 (4.1x faster than HuggingFace).

## Performance

| Backend | Decode (tok/s) | Speedup |
|---------|---------------|---------|
| TensorRT-LLM | 355 | 6.0x |
| **Megakernel** | **240** | **4.1x** |
| vLLM | 107 | 1.8x |
| SGLang | 107 | 1.8x |
| ExLlamaV2 | 98 | 1.7x |
| HuggingFace | 59 | 1.0x |

**Note**: Decode throughput depends on context length. At position 1: 242 tok/s, at position 200: 142 tok/s. See [experiments/RESULTS.md](experiments/RESULTS.md) for full benchmarks.

## What is a Megakernel?

A megakernel fuses an entire transformer block into a single CUDA kernel launch, eliminating kernel launch overhead and intermediate memory traffic. This implementation:

- Fuses RMSNorm, QKV projection, RoPE, attention, O projection, and MLP into one kernel
- Uses `__ldg()` for cached weight reads via texture cache
- Employs cooperative groups for grid-wide synchronization
- Implements online softmax for memory-efficient attention

## Requirements

- NVIDIA GPU with compute capability 8.6+ (RTX 3090, A100, etc.)
- CUDA 11.8+
- Python 3.10+

## Installation

```bash
git clone https://github.com/Infatoshi/MegaQwen.git
cd MegaQwen

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers triton
```

## Usage

### Interactive Chat
```bash
python chat.py
```

### Run Benchmarks
```bash
python benchmark_suite.py
```

### Verify Correctness
```bash
python verify_correctness.py
```

## Documentation

- [Architecture & Technical Details](docs/ARCHITECTURE.md)
- [Technical Specification](SPEC.md)

## License

MIT
