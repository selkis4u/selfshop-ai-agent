import pytest
from src.rag.retriever import HybridPolicyRetriever


@pytest.fixture(scope="module")
def retriever():
    return HybridPolicyRetriever(
        docs_dir="data/store_policies",
        collection_name="test_store_policies"
    )


def test_retriever_initialization(retriever):
    """문서 로드 및 인덱싱 상태 검증"""
    assert len(retriever.documents) >= 4
    assert retriever.bm25 is not None


def test_hybrid_search_approval_policy(retriever):
    """점주 사전 승인 및 한도 금액 관련 질의 테스트"""
    query = "발주 금액 30,000원 초과 시 점주 승인 지침"
    context = retriever.search_hybrid(query, top_k=1)
    
    # 3조(승인 지침) 내용이 반환되어야 함
    assert "제3조" in context
    assert "점주 사전 승인" in context


def test_hybrid_search_specific_item_policy(retriever):
    """특정 품목(과일류) 발주 수량 규정 질의 테스트"""
    query = "신선 과일 apple 유통기한 소량 발주 기준"
    context = retriever.search_hybrid(query, top_k=1)

    # 2조(수량 산정) 내용이 반환되어야 함
    assert "제2조" in context
    assert "apple" in context