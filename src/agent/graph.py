from langgraph.graph import StateGraph, END
from src.agent.state import StoreAgentState
from src.agent.nodes import (
    inspect_inventory_node,
    retrieve_policy_node,
    create_order_draft_node,
    execute_order_node
)


def route_after_inspection(state: StoreAgentState) -> str:
    """
    [조건부 라우터 1]
    매대 분석 결과 부족 재고가 있으면 규정 조회(RAG)로 이동하고,
    부족 재고가 전혀 없으면 불필요한 연산 없이 즉시 파이프라인을 종료합니다.
    """
    low_stock = state.get("low_stock_alerts", [])
    if low_stock:
        return "retrieve_policy"
    return END


def route_after_draft(state: StoreAgentState) -> str:
    """
    [조건부 라우터 2]
    발주 초안 생성 후 승인 필요 여부(requires_approval) 및
    점주 승인 상태(owner_approval)를 확인하여 분기합니다.
    """
    requires_approval = state.get("requires_approval", True)
    owner_approval = state.get("owner_approval", False)

    # 승인이 필요하지 않거나 이미 승인된 경우 실행 노드로 이동
    if not requires_approval or owner_approval:
        return "execute_order"
    
    # 미승인 상태인 경우 실행 노드로 가되 보류 처리되도록 넘김
    return "execute_order"


def build_store_agent_graph():
    """
    무인점포 자율 운영 에이전트의 StateGraph를 조립하고 컴파일합니다.
    """
    workflow = StateGraph(StoreAgentState)

    # 1. 노드 등록
    workflow.add_node("inspect_inventory", inspect_inventory_node)
    workflow.add_node("retrieve_policy", retrieve_policy_node)
    workflow.add_node("create_order_draft", create_order_draft_node)
    workflow.add_node("execute_order", execute_order_node)

    # 2. 진입점(Entry Point) 설정
    workflow.set_entry_point("inspect_inventory")

    # 3. 엣지 및 조건부 분기 연결
    # 매대 검사 -> 재고 부족 시 RAG 조회 / 정상일 시 즉시 종료
    workflow.add_conditional_edges(
        "inspect_inventory",
        route_after_inspection,
        {
            "retrieve_policy": "retrieve_policy",
            END: END
        }
    )

    # 규정 조회 -> 발주서 초안 생성
    workflow.add_edge("retrieve_policy", "create_order_draft")

    # 발주 초안 생성 -> 발주 실행/보류 분기
    workflow.add_conditional_edges(
        "create_order_draft",
        route_after_draft,
        {
            "execute_order": "execute_order"
        }
    )

    # 발주 처리 완료 후 파이프라인 종료
    workflow.add_edge("execute_order", END)

    # 그래프 컴파일
    app = workflow.compile()
    return app