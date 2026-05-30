from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
NATURAL_CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks_natural.json"
MODEL_PATH = BASE_DIR / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

TOP_K = 3
N_CTX = 4096
N_GPU_LAYERS = 10
MAX_TOKENS = 512
TEMPERATURE = 0.2
