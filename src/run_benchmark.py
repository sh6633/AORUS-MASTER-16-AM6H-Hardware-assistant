import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from llm import LocalLLM
from rag_natural import NaturalRAGPipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS_PATH = BASE_DIR / "examples" / "AORUS_MASTER16_AM6H_100Q_Benchmark.txt"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "benchmark_results.json"
FALLBACK_ANSWER = "不好意思喔，這個問題我沒辦法回答喔  (｀・ω・´)b"
EMOTICON = "(｀・ω・´)b"

QUESTION_RE = re.compile(r"^Q(?P<number>\d+)[\.:]\s*(?P<question>.+)$")


def load_questions(path: Path) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = QUESTION_RE.match(line.strip())
        if not match:
            continue
        questions.append(
            {
                "number": int(match.group("number")),
                "question": match.group("question").strip(),
            }
        )
    return questions


def answer_question(question: str, rag: NaturalRAGPipeline, llm: LocalLLM) -> dict:
    chunks = rag.retrieve(question)
    if not chunks:
        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "TTFT": None,
            "TPS": 0.0,
            "token": 0,
            "retrieved_chunks": [],
        }

    prompt = rag.build_prompt(question, chunks)

    answer_parts = []
    metrics = None
    for token, maybe_metrics in llm.stream(prompt):
        if token:
            answer_parts.append(token)
        if maybe_metrics:
            metrics = maybe_metrics

    metrics = metrics or {"ttft": None, "tps": 0.0, "tokens": 0}
    answer = "".join(answer_parts).strip()
    if EMOTICON not in answer:
        answer = f"{answer} {EMOTICON}".strip()
    return {
        "question": question,
        "answer": answer,
        "TTFT": metrics["ttft"],
        "TPS": metrics["tps"],
        "token": metrics["tokens"],
        "retrieved_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "score": chunk.get("_score"),
                "bm25": chunk.get("_bm25"),
                "similarity": chunk.get("_similarity"),
            }
            for chunk in chunks
        ],
    }


def run_benchmark(questions_path: Path, output_path: Path, limit: int | None) -> None:
    questions = load_questions(questions_path)
    if limit is not None:
        questions = questions[:limit]

    rag = NaturalRAGPipeline()
    llm = LocalLLM()

    results = []
    total = len(questions)
    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{total}] Q{item['number']}: {item['question']}")
        result = answer_question(item["question"], rag, llm)
        result["number"] = item["number"]
        results.append(result)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "questions_path": str(questions_path),
        "total_questions": len(results),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved benchmark results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark questions and save answers with metrics.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(args.questions, args.output, args.limit)


if __name__ == "__main__":
    main()
