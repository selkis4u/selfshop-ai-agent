import os
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions


def simple_tokenize(text: str) -> List[str]:
    """한글 및 영문 단어 단위 간단 토크나이저"""
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())
    return tokens


class HybridPolicyRetriever:
    """
    BM25(키워드) + ChromaDB(시맨틱 임베딩) 기반 매장 규정 하이브리드 검색기
    - Reciprocal Rank Fusion(RRF)을 통해 두 검색 결과 순위를 결합합니다.
    """

    def __init__(
        self,
        docs_dir: str = "data/store_policies",
        collection_name: str = "store_policies",
        embedding_model_name: str = "jhgan/ko-sroberta-multitask"
    ):
        self.docs_dir = docs_dir
        self.collection_name = collection_name
        self.documents: List[str] = []
        self.bm25: BM25Okapi = None

        # 1. 문서 청크 분할 및 로드
        self._load_documents()

        # 2. BM25 인덱스 생성
        tokenized_corpus = [simple_tokenize(doc) for doc in self.documents]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. ChromaDB 임베딩 모델 및 벡터 스토어 초기화
        # 허깅페이스 한국어 지원 경량 임베딩 모델 사용
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn
        )

        # 4. 벡터 스토어 문서 색인
        self._index_chroma()

    def _load_documents(self):
        """텍스트 파일을 읽어 조항 단위(줄바꿈 2회 기준)로 청킹"""
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            return

        for fname in os.listdir(self.docs_dir):
            if fname.endswith(".txt") or fname.endswith(".md"):
                fpath = os.path.join(self.docs_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 조항별 빈 줄 분할
                    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                    self.documents.extend(chunks)

    def _index_chroma(self):
        """ChromaDB에 문서 청크 등록"""
        if not self.documents:
            return

        # 중복 인덱싱 방지를 위해 기존 컬렉션 비우기
        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        ids = [f"doc_{i}" for i in range(len(self.documents))]
        self.collection.add(
            ids=ids,
            documents=self.documents,
            metadatas=[{"source": "policy_manual", "chunk_id": i} for i in range(len(self.documents))]
        )

    def search_bm25(self, query: str, top_k: int = 3) -> List[int]:
        """BM25 키워드 검색 -> 상위 문서 인덱스 목록 반환"""
        if not self.bm25:
            return []
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return top_indices

    def search_vector(self, query: str, top_k: int = 3) -> List[int]:
        """ChromaDB 시맨틱 검색 -> 상위 문서 인덱스 목록 반환"""
        if not self.documents:
            return []
        results = self.collection.query(query_texts=[query], n_results=min(top_k, len(self.documents)))
        ids = results["ids"][0] if results["ids"] else []
        indices = [int(doc_id.replace("doc_", "")) for doc_id in ids]
        return indices

    def search_hybrid(self, query: str, top_k: int = 2, rrf_k: int = 60) -> str:
        """
        RRF (Reciprocal Rank Fusion)를 통한 하이브리드 검색
        Score = 1 / (rrf_k + rank_bm25) + 1 / (rrf_k + rank_vector)
        최종 1~2개 핵심 청크를 하나의 텍스트로 병합 반환합니다.
        """
        if not self.documents:
            return "참조할 매장 운영 규정이 인덱스에 존재하지 않습니다."

        bm25_indices = self.search_bm25(query, top_k=5)
        vector_indices = self.search_vector(query, top_k=5)

        rrf_scores: Dict[int, float] = {}

        # BM25 순위 점수 누적
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Vector 순위 점수 누적
        for rank, idx in enumerate(vector_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # 점수 기준 정렬
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        selected_indices = sorted_indices[:top_k]

        selected_texts = [self.documents[i] for i in selected_indices]
        return "\n\n".join(selected_texts)