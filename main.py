import os
import cv2
import numpy as np
from src.agent.graph import build_store_agent_graph


def create_dummy_shelf_image(image_path: str = "./data/sample_shelf_01.jpg"):
    """
    테스트용 매대 이미지가 없을 경우 기본 프레임을 자동 생성합니다.
    """
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if not os.path.exists(image_path):
        # 640x640 크기의 가상 프레임 생성
        img = np.full((640, 640, 3), 240, dtype=np.uint8)
        # 매대 선반 라인 시뮬레이션
        cv2.line(img, (0, 320), (640, 320), (100, 100, 100), 4)
        cv2.line(img, (0, 520), (640, 520), (100, 100, 100), 4)
        cv2.putText(img, "SHELF TEST FRAME", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        cv2.imwrite(image_path, img)
        print(f"[Setup] 테스트용 매대 이미지 자동 생성 완료: {image_path}")


def run_scenario(scenario_name: str, owner_approval: bool):
    print("\n" + "=" * 70)
    print(f"▶ [시나리오 실행] {scenario_name} (점주 승인 플래그: {owner_approval})")
    print("=" * 70)

    # 1. StateGraph 워크플로우 컴파일
    app = build_store_agent_graph()

    # 2. 초기 State 주입
    frame_path = "./data/sample_shelf_01.jpg"
    initial_state = {
        "frame_path": frame_path,
        "timestamp": None,
        "detected_items": {},
        "low_stock_alerts": [],
        "policy_query": None,
        "policy_context": None,
        "order_draft": None,
        "requires_approval": True,
        "owner_approval": owner_approval,
        "execution_result": None,
        "audit_logs": []
    }

    # 3. 워크플로우 실행
    final_state = app.invoke(initial_state)

    # 4. 결과 출력
    print("\n[1] 비전 검출(YOLOv11n) 및 재고 분석 요약")
    detected = final_state.get("detected_items", {})
    print(f"- 카메라 검출 객체: {detected if detected else '0건 (빈 매대 또는 미인식)'}")
    print(f"- 안전 재고 미달 품목:")
    for alert in final_state.get("low_stock_alerts", []):
        print(f"  * {alert['item_name']} ({alert['item_id']}): 현재 {alert['current_count']}개 / 기준치 {alert['threshold']}개")

    print("\n[2] 하이브리드 RAG (BM25 + ChromaDB) 참조 규정")
    print(f"- 실행된 검색 질의: '{final_state.get('policy_query')}'")
    print(f"- 참조된 운영 지침:")
    print("-" * 50)
    print(final_state.get("policy_context", "규정 없음"))
    print("-" * 50)

    print("\n[3] 발주서 초안 (Order Draft)")
    draft = final_state.get("order_draft")
    if draft:
        print(f"- 발주 ID: {draft['draft_id']}")
        print(f"- 총 예상 금액: {draft['total_cost']:,} {draft['currency']}")
        print(f"- 점주 승인 필수 여부: {final_state.get('requires_approval')}")
        print("- 세부 발주 품목:")
        for item in draft["items"]:
            print(f"  · {item['item_name']}: +{item['order_quantity']}개 (비용: {item['estimated_cost']:,}원)")

    print("\n[4] 최종 실행 결과")
    print(f"- {final_state.get('execution_result')}")

    print("\n[5] 에이전트 감사 로그 (Audit Logs)")
    for log in final_state.get("audit_logs", []):
        print(f"  {log}")


def main():
    # 0. 테스트 이미지 확보
    create_dummy_shelf_image()

    # 시나리오 1: 점주 승인이 완료된 발주 케이스
    run_scenario("시나리오 1: 점주 승인(owner_approval=True) 케이스", owner_approval=True)

    # 시나리오 2: 점주 미승인 보류 케이스
    run_scenario("시나리오 2: 점주 미승인(owner_approval=False) 케이스", owner_approval=False)


if __name__ == "__main__":
    main()