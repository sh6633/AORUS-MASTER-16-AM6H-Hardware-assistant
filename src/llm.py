import time
from collections.abc import Iterator

from config import MAX_TOKENS, MODEL_PATH, N_CTX, N_GPU_LAYERS, TEMPERATURE


class LocalLLM:
    def __init__(self) -> None:
        from llama_cpp import Llama

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )
        #print("使用GPU數量 = ", N_GPU_LAYERS)

    def stream(self, prompt: str) -> Iterator[tuple[str, dict | None]]:
        started_at = time.perf_counter()
        first_token_at = None
        token_count = 0

        stream = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise product specification assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens = MAX_TOKENS,
            temperature = TEMPERATURE,
            stream = True,
        )

        for event in stream:
            delta = event["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if not token:
                continue

            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            token_count += 1
            yield token, None

        ended_at = time.perf_counter()
        ttft = (first_token_at - started_at) if first_token_at else None
        decode_time = ended_at - (first_token_at or started_at)
        tps = token_count / decode_time if decode_time > 0 else 0.0
        yield "", {"ttft": ttft, "tps": tps, "tokens": token_count}
