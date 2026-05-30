import json
import math
import re
from collections import Counter
from pathlib import Path

from config import NATURAL_CHUNKS_PATH, TOP_K




#定義一個字中文字是token，一串英文/數字算一個token
TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
STOP_TOKENS = {
    "這",
    "台",
    "筆",
    "電",
    "的",
    "嗎",
    "是",
    "有",
    "哪",
    "個",
    "該",
    "今",
    "天",
    "幾",
    "號",
    "呢",
    "你",
    "我",
    "請",
    "問",
    "一",
    "下",
    "好",
    "阿",
    "了",
    "在",
    "為",
    "何",
}


#把一段文字切成token ==> ["gpu", "memory", "顯", "存"]
def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOP_TOKENS
    ]


def load_chunks(path: Path = NATURAL_CHUNKS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class NaturalRAGPipeline:
    def __init__(self, chunks_path: Path = NATURAL_CHUNKS_PATH, top_k: int = TOP_K) -> None:
        self.chunks = load_chunks(chunks_path)
        self.top_k = top_k
        #將chunk讀入，tokenize化之後(也是list)存入一個list
        self.chunk_tokens = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.doc_freq = self._build_doc_freq()
        self.avg_doc_len = self._average_doc_length()

    #統計這個詞在chunk出現幾次。
    def _build_doc_freq(self) -> Counter:
        doc_freq = Counter()
        for tokens in self.chunk_tokens:
            doc_freq.update(set(tokens))
        return doc_freq

    #平均 chunk 長度
    def _average_doc_length(self) -> float:
        if not self.chunk_tokens:
            return 0.0
        return sum(len(tokens) for tokens in self.chunk_tokens) / len(self.chunk_tokens)

    def _bm25_score(self, query_tokens: list[str], chunk_index: int) -> float:
        tokens = self.chunk_tokens[chunk_index]
        if not tokens:
            return 0.0

        #統計這個chunk的詞的次數
        term_counts = Counter(tokens)
        #總共有多少個chunk
        doc_count = len(self.chunk_tokens)
        #這個chunk的長度
        doc_len = len(tokens)
        k1 = 1.5
        b = 0.75
        score = 0.0

        for token in query_tokens:
            tf = term_counts[token]
            if tf == 0:
                continue
            df = self.doc_freq[token]
            #(跨)chunk越少出現的詞，idf越高，代表越有辨識度
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            #BM25公式，它同時考慮詞在 chunk 裡出現幾次。這個詞稀不稀有。chunk 是否太長。
            denominator = tf + k1 * (1 - b + b * doc_len / (self.avg_doc_len or 1))
            score += idf * (tf * (k1 + 1)) / denominator
        #想成這句話(tokenize)之後跟某個chunk有多相關。
        return score

    def _similarity_score(self, query_tokens: list[str], chunk_index: int) -> float:
        query_counts = Counter(query_tokens)
        chunk_counts = Counter(self.chunk_tokens[chunk_index])
        if not query_counts or not chunk_counts:
            return 0.0

        #取都有出現的KEY
        shared = set(query_counts) & set(chunk_counts)
        #對應的KEY相乘再相加，類似內積
        dot = sum(query_counts[token] * chunk_counts[token] for token in shared)
        #算向量長度
        query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
        chunk_norm = math.sqrt(sum(value * value for value in chunk_counts.values()))
        if query_norm == 0 or chunk_norm == 0:
            return 0.0
        return dot / (query_norm * chunk_norm)

    def retrieve(self, question: str) -> list[dict]:
        query_tokens = tokenize(question)
        scored = []

        for index, chunk in enumerate(self.chunks):
            bm25 = self._bm25_score(query_tokens, index)
            similarity = self._similarity_score(query_tokens, index)
            if bm25 == 0:
                continue
            weighted_similarity = similarity * 10
            score = bm25 + weighted_similarity
            scored.append(
                {
                    "chunk": chunk,
                    "score": score,
                    "bm25": bm25,
                    "similarity": similarity,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        if not scored:
            return []
        return [
            {
                **item["chunk"],
                "_score": round(item["score"], 4),
                "_bm25": round(item["bm25"], 4),
                "_similarity": round(item["similarity"], 4),
            }
            for item in scored[: self.top_k]
        ]

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[{index}] {chunk['chunk_id']}\n{chunk['text']}"
            for index, chunk in enumerate(chunks, start=1)
        )

        return f"""You are answering questions about GIGABYTE AORUS MASTER 16 AM6H product specifications.

Basic product facts:
- AORUS MASTER 16 AM6H is the product family.
- AORUS MASTER 16 BZH, AORUS MASTER 16 BYH, and AORUS MASTER 16 BXH are variant models under this product family.

Use only the context below. If the context does not contain the answer, say This is a very complicated question, I'm sorry, I cannot answer it.(窩不知道(｀・ω・´)b).
Use only the context below. If the context does not contain the answer, say This is a very complicated question, I'm sorry, I cannot answer it.(窩不知道(｀・ω・´)b).
Use only the context below. If the context does not contain the answer, say This is a very complicated question, I'm sorry, I cannot answer it.(窩不知道(｀・ω・´)b).

If numeric specifications are present in the context, quote them exactly.

End every answer with this exact emoticon: (｀・ω・´)b

Answer in the same language as the user's question. If the user asks in Traditional Chinese, answer in Traditional Chinese. If the user asks in English, answer in English. If the question mixes Chinese and English, follow the main language used by the user.
Question:
{question}

Context:
{context}

Answer:"""
