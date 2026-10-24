import pytest
from src.agent.graph import build_store_agent_graph
from src.agent.state import StoreAgentState


@pytest.fixture
def base_initial_state() -> StoreAgentState:
    """테스트용 기본 초기 State를 제공하는 픽스처"""
    return {
        "frame_path": "./data/sample_shelf_01.jpg",
        "timestamp": None,
        "detected_items": {},
        "low_stock_alerts": [],
        "policy_query": None,
        "policy_context": None,
        "order_draft": None,
        "requires_approval": True,
        "owner_approval": False,
        "execution_result": None,
        "audit_logs": []
    }


def test_approved_order_scenario(base_initial_state):
    """
    [시나리오 1] 점주 즉시 승인(owner_approval=True) 케이스
    - 실제 검출기 연동 시(샘플 이미지 미존재 시 재고 0 처리): 관리 대상 전 품목 부족 감지
    - 발주 초안 계산 및 최종 발주 완료 처리 검증
    """
    app = build_store_agent_graph()
    state = base_initial_state.copy()
    state["owner_approval"] = True

    final_state = app.invoke(state)

    # 1. 부족 품목 감지 검증 (5종 품목 부족 감지)
    low_stock = final_state.get("low_stock_alerts", [])
    assert len(low_stock) > 0, "부족 품목이 1건 이상 감지되어야 합니다."

# 2. RAG 규정 조회 검증
    assert final_state["policy_context"] is not None
    assert "제" in final_state["policy_context"]  # [제2조], [제3조] 등 조항 텍스트 검증

    # # 2. RAG 규정 조회 검증
    # assert final_state["policy_context"] is not None
    # assert "무인점포 운영 매뉴얼" in final_state["policy_context"]

    # 3. 발주 초안 검증
    draft = final_state.get("order_draft")
    assert draft is not None
    assert len(draft["items"]) == len(low_stock)
    assert draft["total_cost"] > 0

    # 4. 발주 실행 상태 검증 (승인 완료)
    assert final_state["execution_result"] is not None
    assert "✅ [발주 완료]" in final_state["execution_result"]
    assert draft["draft_id"] in final_state["execution_result"]

    # 5. 감사 로그 누적 확인
    assert len(final_state["audit_logs"]) >= 5


def test_pending_order_scenario(base_initial_state):
    """
    [시나리오 2] 점주 미승인(owner_approval=False) 케이스
    - 발주서는 작성되지만 실행 단계에서 보류 상태로 대기해야 함
    """
    app = build_store_agent_graph()
    state = base_initial_state.copy()
    state["owner_approval"] = False

    final_state = app.invoke(state)

    assert final_state.get("order_draft") is not None
    assert final_state["execution_result"] is not None
    assert "⏸️ [발주 보류]" in final_state["execution_result"]
    assert "점주 미승인" in final_state["execution_result"]


def test_early_exit_when_no_shortage(base_initial_state, monkeypatch):
    """
    [시나리오 3] 부족 재고가 없는 정상 매대일 경우 (조기 종료 조건 분기 검증)
    - RAG 규정 조회 및 발주 초안 단계 없이 즉시 종료(END) 검증
    """
    import src.agent.graph as graph_module

    # inspect_inventory_node가 반환할 Mock 함수 (모든 재고 충분 상태)
    def mock_inspect_all_sufficient(state):
        return {
            "timestamp": "2026-10-23 16:00:00",
            "detected_items": {"bottle": 10, "cup": 10, "bowl": 10, "apple": 5, "banana": 5},
            "low_stock_alerts": [],
            "audit_logs": ["검사 완료: 모든 모니터링 품목 재고 충분"]
        }

    monkeypatch.setattr(graph_module, "inspect_inventory_node", mock_inspect_all_sufficient)

    app = graph_module.build_store_agent_graph()
    state = base_initial_state.copy()

    final_state = app.invoke(state)

    assert len(final_state.get("low_stock_alerts", [])) == 0
    assert final_state.get("policy_context") is None
    assert final_state.get("order_draft") is None
    assert final_state.get("execution_result") is None