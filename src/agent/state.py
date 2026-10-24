from typing import TypedDict, List, Dict, Any, Optional

class StockItem(TypedDict):
    """개별 재고 품목의 상세 정보 스키마"""
    item_id: str
    item_name: str
    current_count: int
    threshold: int
    unit_price: int

class OrderDraftItem(TypedDict):
    """발주 품목 세부 항목"""
    item_id: str
    item_name: str
    order_quantity: int
    estimated_cost: int

class StoreAgentState(TypedDict):
    """
    무인점포 AI 에이전트 그래프 전체에서 공유되는 상태 규격 (State Schema)
    """
    # 1. 입력 입력 데이터
    frame_path: str                          # 분석할 매대 이미지/프레임 경로
    timestamp: Optional[str]                 # 감지 시각

    # 2. 비전 분석 결과
    detected_items: Dict[str, int]           # 객체 인식 결과 (클래스명: 검출 수량)
    low_stock_alerts: List[StockItem]        # 임계치 미달로 감지된 부족 품목 목록

    # 3. RAG 규정 검색 컨텍스트
    policy_query: Optional[str]              # 규정 검색에 사용된 쿼리
    policy_context: Optional[str]            # RAG로 검색/리랭킹된 매장 운영 규정 본문

    # 4. 의사결정 및 발주 생성
    order_draft: Optional[Dict[str, Any]]    # 생성된 발주서 초안 (품목, 총액, 공급업체 등)
    requires_approval: bool                  # 점주 승인 필요 여부 (기본 True)
    owner_approval: bool                     # 점주의 승인 여부 (True/False)

    # 5. 실행 결과 및 감사(Audit) 로그
    execution_result: Optional[str]          # 최종 API 전송 결과 또는 실패 메시지
    audit_logs: List[str]                    # 에이전트 워크플로우 진행 기록