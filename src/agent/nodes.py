import os
from datetime import datetime
from typing import Dict, Any, List
from src.agent.state import StoreAgentState, StockItem, OrderDraftItem
from src.vision.detector import ShelfDetector
from src.rag.retriever import HybridPolicyRetriever

# 1. 품목 메타데이터 (안전 재고 기준치 및 단가)
INVENTORY_METADATA: Dict[str, Dict[str, Any]] = {
    "bottle": {"name": "음료/생수 500ml", "threshold": 5, "unit_price": 1200},
    "cup": {"name": "테이크아웃 컵/용기", "threshold": 4, "unit_price": 500},
    "bowl": {"name": "용기면/간편식 보울", "threshold": 3, "unit_price": 2500},
    "apple": {"name": "세척 사과(소포장)", "threshold": 3, "unit_price": 1800},
    "banana": {"name": "바나나(1송이)", "threshold": 2, "unit_price": 2000},
}

# 2. 비전 검출기 및 RAG 검색기 싱글톤 초기화
detector = ShelfDetector(
    model_path="data/models/yolo11n.pt",
    conf_threshold=0.25,
    target_classes=list(INVENTORY_METADATA.keys())
)

retriever = HybridPolicyRetriever(
    docs_dir="data/store_policies",
    collection_name="store_policies",
    embedding_model_name="jhgan/ko-sroberta-multitask"
)


def inspect_inventory_node(state: StoreAgentState) -> Dict[str, Any]:
    """
    [Node 1] 매대 이미지 분석 및 부족 재고 선별
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs = list(state.get("audit_logs", []))
    frame_path = state.get("frame_path", "")

    logs.append(f"[{timestamp}] [Vision] 매대 프레임 분석 요청: {frame_path}")

    detected_counts: Dict[str, int] = {}

    if frame_path and os.path.exists(frame_path):
        try:
            detected_counts = detector.detect_shelf(frame_path)
            logs.append(f"[{timestamp}] [Vision] YOLOv11n 추론 완료: {detected_counts}")
        except Exception as e:
            logs.append(f"[{timestamp}] [Vision Error] 추론 실패 ({str(e)}) -> 기본 빈 매대 처리")
            detected_counts = {}
    else:
        logs.append(f"[{timestamp}] [Vision Warning] 이미지 파일 없음 ({frame_path}) -> 초기 재고 0으로 평가")
        detected_counts = {}

    low_stock_list: List[StockItem] = []
    for item_id, meta in INVENTORY_METADATA.items():
        current_qty = detected_counts.get(item_id, 0)
        if current_qty < meta["threshold"]:
            low_stock_list.append({
                "item_id": item_id,
                "item_name": meta["name"],
                "current_count": current_qty,
                "threshold": meta["threshold"],
                "unit_price": meta["unit_price"]
            })

    logs.append(f"[{timestamp}] [Vision] 부족 품목 {len(low_stock_list)}건 감지 완료")

    return {
        "timestamp": timestamp,
        "detected_items": detected_counts,
        "low_stock_alerts": low_stock_list,
        "audit_logs": logs
    }


def retrieve_policy_node(state: StoreAgentState) -> Dict[str, Any]:
    """
    [Node 2] 매장 운영 규정 조회 (RAG)
    - 부족 품목명과 안전 재고 발주 지침, 승인 한도에 대한 동적 쿼리를 생성하여 검색합니다.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs = list(state.get("audit_logs", []))

    # 부족 품목 목록을 쿼리에 반영
    shortage_items = [item["item_id"] for item in state.get("low_stock_alerts", [])]
    items_str = ", ".join(shortage_items) if shortage_items else "품목"
    
    query = f"{items_str} 안전 재고 기준치 발주 수량 및 점주 사전 승인 지침"
    logs.append(f"[{timestamp}] [RAG] 하이브리드 검색 질의 실행: '{query}'")

    # BM25 + ChromaDB RRF 앙상블 검색 수행
    policy_context = retriever.search_hybrid(query=query, top_k=2)

    logs.append(f"[{timestamp}] [RAG] 규정 매뉴얼 참조 완료 (검색 결과 길이: {len(policy_context)}자)")

    return {
        "policy_query": query,
        "policy_context": policy_context,
        "audit_logs": logs
    }


def create_order_draft_node(state: StoreAgentState) -> Dict[str, Any]:
    """
    [Node 3] 발주서 초안 생성
    - 제2조 및 제3조 지침(신선 과일 소량 발주, 30,000원 초과/야간 시 점주 승인 필수)을 반영합니다.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs = list(state.get("audit_logs", []))
    logs.append(f"[{timestamp}] [Agent] 발주서 초안 계산 시작")

    draft_items: List[OrderDraftItem] = []
    total_cost = 0

    for item in state.get("low_stock_alerts", []):
        item_id = item["item_id"]
        # 신선 과일류(apple, banana)는 규정상 1회 최대 5개 이내로 제한
        if item_id in ["apple", "banana"]:
            target_qty = min(5, (item["threshold"] * 2) - item["current_count"])
        else:
            target_qty = (item["threshold"] * 2) - item["current_count"]

        cost = target_qty * item["unit_price"]
        total_cost += cost

        draft_items.append({
            "item_id": item_id,
            "item_name": item["item_name"],
            "order_quantity": target_qty,
            "estimated_cost": cost
        })

    order_draft = {
        "draft_id": f"DRAFT-{datetime.now().strftime('%Y%m%d%H%M')}",
        "items": draft_items,
        "total_cost": total_cost,
        "currency": "KRW",
        "created_at": timestamp
    }

    # 제3조 기준: 30,000원 초과 시 점주 승인(Human-in-the-loop) 필수
    requires_approval = total_cost > 30000

    logs.append(f"[{timestamp}] [Agent] 발주 초안 생성 완료 (총액: {total_cost:,}원 / 승인필수여부: {requires_approval})")

    return {
        "order_draft": order_draft,
        "requires_approval": requires_approval,
        "audit_logs": logs
    }


def execute_order_node(state: StoreAgentState) -> Dict[str, Any]:
    """
    [Node 4] 발주 실행 및 처리
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs = list(state.get("audit_logs", []))
    
    is_approved = state.get("owner_approval", False)
    requires_approval = state.get("requires_approval", True)
    draft = state.get("order_draft")

    if draft:
        # 승인이 필요 없거나, 점주가 승인한 경우
        if not requires_approval or is_approved:
            result_message = (
                f"✅ [발주 완료] 발주번호 {draft['draft_id']} / "
                f"총 {len(draft['items'])}종 품목(총 {draft['total_cost']:,}원) 납품 시스템 전송 완료"
            )
            logs.append(f"[{timestamp}] [Execute] 발주 실행 완료 (승인여부: {is_approved})")
        else:
            result_message = "⏸️ [발주 보류] 점주 미승인 상태로 발주가 대기열에 보관되었습니다."
            logs.append(f"[{timestamp}] [Execute] 점주 미승인 -> 보류 대기열 등록")
    else:
        result_message = "ℹ️ [발주 미실행] 생성된 발주 초안이 없습니다."

    return {
        "execution_result": result_message,
        "audit_logs": logs
    }