from rag_natural import NaturalRAGPipeline
from llm import LocalLLM


FALLBACK_ANSWER = "不好意思喔，這個問題我沒辦法回答喔  (｀・ω・´)b"


def answer_stream(question: str, rag: NaturalRAGPipeline, llm: LocalLLM):
    # Natural chunk RAG flow changed here:
    # 1. Compare the question with every chunk["text"] using BM25 + similarity.
    # 2. Take the top 3 chunks.
    # 3. Put the selected chunk text into the prompt.
    # 4. Stream the answer through llama.cpp.
    chunks = rag.retrieve(question)
    prompt = rag.build_prompt(question, chunks)
    return chunks, llm.stream(prompt)


def main() -> None:
    rag = NaturalRAGPipeline()
    llm = LocalLLM()

    print("哈囉，我是 AORUS MASTER 16 AM6H 的硬體小助手，你可以問我任何有關該設備問題喔 (｀・ω・´)b")
    print("Using natural chunks: BM25 + similarity -> top 3 chunks -> llama.cpp streaming.")
    print("Type a question, or type exit to quit.\n")

    while True:
        question = input("Q: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        chunks, stream = answer_stream(question, rag, llm)
        print("\nRetrieved chunks:")
        if not chunks:
            print("- none")
            print(f"\nA: {FALLBACK_ANSWER}")
            print("\nMetrics: TTFT=n/a, TPS=0.00, tokens=0\n")
            continue

        for chunk in chunks:
            print(
                f"- {chunk['chunk_id']} "
                f"(score={chunk['_score']}, bm25={chunk['_bm25']}, similarity={chunk['_similarity']})"
            )
        print("\nA: ", end="", flush=True)

        metrics = None
        for token, maybe_metrics in stream:
            if token:
                print(token, end="", flush=True)
            if maybe_metrics:
                metrics = maybe_metrics

        print()
        if metrics:
            ttft = metrics["ttft"]
            ttft_text = f"{ttft:.3f}s" if ttft is not None else "n/a"
            print(f"\nMetrics: TTFT={ttft_text}, TPS={metrics['tps']:.2f}, tokens={metrics['tokens']}\n")


if __name__ == "__main__":
    main()
