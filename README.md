# SelfShop AI Agent: 무인점포 자율 운영 AI 에이전트 시스템

`SelfShop AI Agent`는 컴퓨터 비전 기반 실시간 재고 파악, 운영 매뉴얼 준수를 위한 Hybrid RAG 규정 검색, 그리고 점주 사전 승인(Human-in-the-loop) 거버넌스를 결합한 **무인점포 자율 운영 및 자동 발주 에이전트 시스템**입니다.

---

## 1. 주요 기능 (Key Features)

- **실시간 매대 비전 검출 (Vision-based Stock Detection)**:
  - Ultralytics의 경량화된 최신 `YOLOv11 Nano(yolo11n)` 모델을 탑재하여 매대 영상/이미지 내 품목별 수량을 실시간 집계합니다.
- **하이브리드 RAG 규정 조회 (Hybrid Policy Retriever)**:
  - 키워드 매칭(BM25)과 시맨틱 문맥 임베딩(ChromaDB + `ko-sroberta-multitask`)을 **RRF(Reciprocal Rank Fusion, k=60)** 알고리즘으로 결합하여 매장 운영 매뉴얼에서 최적의 발주 및 승인 지침을 도출합니다.
- **에이전트 워크플로우 제어 (Stateful Agent Orchestration)**:
  - `LangGraph` 기반의 상태 그래프(StateGraph)로 전체 파이프라인의 상태 전이를 통제하며, 예산 한도(30,000원) 및 신선식품 소량 발주 규칙을 엄격히 적용합니다.
- **인간 개입형 승인 통제 (Human-in-the-loop Governance)**:
  - 예산 초과 또는 무인 운영 시간대 발생 건에 대해 점주 승인 플래그(`owner_approval`)를 검증하여 발주 실행 및 보류를 분기합니다.
- **감사 추적성 (Audit Trail)**:
  - 전 단계의 추론, RAG 쿼리, 견적 산출, 승인 내역을 타임스탬프와 함께 State 내 `audit_logs`로 영구 기록합니다.

---

## 2. 기술 스택 (Tech Stack)

| 구분 | 적용 기술 및 라이브러리 |
| :--- | :--- |
| **Language & Runtime** | Python 3.11+ |
| **Agent Orchestration** | LangGraph, LangChain Core |
| **Computer Vision** | YOLOv11 Nano (`ultralytics`), OpenCV (`opencv-python`) |
| **Hybrid RAG** | ChromaDB, Rank-BM25, Sentence-Transformers (`jhgan/ko-sroberta-multitask`) |
| **Testing Framework** | Pytest, Monkeypatch |

---

## 3. 프로젝트 구조 (Directory Structure)

```text
selfshop-ai-agent/
├── data/
│   ├── models/                  # YOLOv11n 모델 가중치 캐시 (yolo11n.pt)
│   ├── store_policies/          # 무인점포 운영 규정 원본 (policy_manual.txt)
│   └── sample_shelf_01.jpg      # 테스트용 매대 입력 이미지
├── src/
│   ├── agent/
│   │   ├── state.py             # TypedDict 기반 전역 StoreAgentState 정의
│   │   ├── nodes.py             # 비전/RAG/초안생성/발주실행 비즈니스 노드
│   │   └── graph.py             # LangGraph 상태 그래프 조립 및 조건부 라우터
│   ├── vision/
│   │   └── detector.py          # YOLOv11n 기반 매대 상품 검출기 (ShelfDetector)
│   └── rag/
│       └── retriever.py         # BM25 + ChromaDB 결합 RRF 하이브리드 검색기
├── tests/
│   ├── test_agent_workflow.py   # 점주 승인/보류/조기종료 시나리오 pytest
│   ├── test_vision.py           # YOLOv11n 추론 및 반환 인터페이스 단위 테스트
│   └── test_rag.py              # BM25+ChromaDB 규정 검색 단위 테스트
├── main.py                      # 전체 에이전트 구동 진입점 (시나리오 실행기)
├── pytest.ini                   # pytest 경로 및 모듈 인식 설정
└── requirements.txt             # 의존성 패키지 목록