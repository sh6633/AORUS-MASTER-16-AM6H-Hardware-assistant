# GIGABYTE AORUS MASTER 16 AM6H RAG Assistant

This project is a lightweight RAG-based hardware assistant for answering product specification questions about the **GIGABYTE AORUS MASTER 16 AM6H** laptop family.

The system is designed for a resource-limited consumer laptop scenario and uses:

- Pure Python RAG logic, without LangChain or LlamaIndex
- `uv` for Python environment and dependency management
- `llama.cpp` through `llama-cpp-python` for local GGUF inference
- Natural-language specification chunks for Traditional Chinese and English questions

## Product Scope

The product family is:

```text
AORUS MASTER 16 AM6H
```

The variant models are:

```text
AORUS MASTER 16 BZH
AORUS MASTER 16 BYH
AORUS MASTER 16 BXH
```

The main known difference between the three variants is the GPU:

```text
BZH: NVIDIA GeForce RTX 5090 Laptop GPU, 24GB GDDR7, 175W
BYH: NVIDIA GeForce RTX 5080 Laptop GPU, 16GB GDDR7, 175W
BXH: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12GB GDDR7, 140W
```

## RAG Design

The retrieval data is stored in:

```text
data/processed/chunks_natural.json
```

Each chunk represents one product specification category, such as:

```text
GPU
Display
Memory
Storage
Ports
Battery
```

The retrieval pipeline in `src/rag_natural.py` works as follows:

```text
User question
-> tokenize question and chunk text
-> compute BM25 score
-> compute token-count cosine similarity
-> final score = BM25 + similarity * 10
-> remove chunks with BM25 = 0
-> take top 3 chunks
-> build prompt
-> stream answer with llama.cpp
```

This version does not use a neural embedding model. The similarity score is based on token frequency overlap, so the method is lightweight and CPU-friendly.

## Model Choice

Recommended model:

```text
Qwen2.5-3B-Instruct GGUF Q4_K_M
```

Expected path:

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
```

The model file is not included in this repository because GGUF files are large. Download the model separately and place it in the `models/` directory.

This model was selected because:

- It is small enough for consumer laptop use
- Q4 quantization keeps memory usage low
- Qwen models handle Chinese and English reasonably well
- It can run through `llama.cpp` without relying on a cloud API

## 4GB VRAM Constraint

The default configuration can run with:

```python
N_GPU_LAYERS = 0
```

This keeps inference in CPU mode and uses little to no VRAM, satisfying the 4GB VRAM constraint.

Optional GPU offload can be enabled by increasing:

```python
N_GPU_LAYERS
```

For example:

```python
N_GPU_LAYERS = 10
```

VRAM usage should be monitored with:

```powershell
nvidia-smi -l 1
```

In local testing on an RTX 4060 Laptop GPU, GPU offload used less than 4GB VRAM.

## Setup

Install dependencies with `uv`:

```powershell
uv sync
```

If `llama-cpp-python` needs to be installed manually, CPU mode can be installed with:

```powershell
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

CUDA acceleration is optional and requires a CUDA-enabled `llama-cpp-python` build that matches the local CUDA runtime.

## Run The Assistant

Start the interactive assistant:

```powershell
uv run python src\main.py
```

Example questions:

```text
BZH、BYH、BXH 主要差在哪？
這台電腦有低藍光認證嗎？
Can I charge it through Thunderbolt?
Which model has the highest VRAM?
```

## Run Benchmark

Benchmark questions are stored in:

```text
examples/AORUS_MASTER16_AM6H_100Q_Benchmark.txt
```

Run a small test:

```powershell
uv run python src\run_benchmark.py --limit 10
```

Run the full benchmark:

```powershell
uv run python src\run_benchmark.py
```

The output is saved to:

```text
data/processed/benchmark_results.json
```

Each result contains:

```json
{
  "question": "...",
  "answer": "...",
  "TTFT": 0.58,
  "TPS": 28.5,
  "token": 91,
  "retrieved_chunks": []
}
```

## Evaluation Notes

The benchmark records:

- **TTFT**: time to first token
- **TPS**: tokens per second
- Generated token count
- Retrieved chunks for qualitative inspection

In one local benchmark run with 200 questions:

```text
Average TTFT: about 0.58 seconds
Average TPS: about 28.5 tokens/second
Average output length: about 92 tokens
```

Product-specific questions generally perform well when the needed information exists in `chunks_natural.json`.

Known limitations:

- The current retriever is lexical, not embedding-based.
- Some out-of-scope questions may still retrieve weakly related chunks.
- If the official specification page does not contain a field, such as detailed cooling technology, the assistant should not invent it.
- The quality depends heavily on the completeness of `chunks_natural.json`.

## Repository Notes

Large files are intentionally excluded:

```text
models/*.gguf
data/processed/benchmark_results.json
```

The repository should include the source code, benchmark question file, and curated natural chunks.
