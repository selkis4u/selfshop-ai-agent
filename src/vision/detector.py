import os
from typing import Dict, Any, List, Optional
from collections import Counter
from ultralytics import YOLO


class ShelfDetector:
    """
    YOLOv11n 기반 매대 상품 검출기
    - 입력 이미지 프레임을 분석하여 클래스별 검출 수량을 집계합니다.
    """

    def __init__(
        self,
        model_path: str = "data/models/yolo11n.pt",
        conf_threshold: float = 0.25,
        target_classes: Optional[List[str]] = None
    ):
        """
        Args:
            model_path: YOLOv11 모델 가중치 파일 경로 (.pt)
            conf_threshold: 검출 신뢰도 임계치 (기본값: 0.25)
            target_classes: 모니터링할 특정 클래스 목록 (None일 경우 전체 검출)
        """
        self.conf_threshold = conf_threshold
        self.target_classes = set(target_classes) if target_classes else None

        # 가중치 디렉터리 확인
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        print(f"[Vision] YOLOv11 모델 로드 중: {model_path}")
        # 모델 파일이 없으면 ultralytics가 최신 사전 학습 가중치를 자동 다운로드합니다.
        self.model = YOLO(model_path)
        print("[Vision] YOLOv11 모델 로드 완료.")

    def detect_shelf(self, image_path: str) -> Dict[str, int]:
        """
        이미지 파일 경로를 입력받아 클래스별 객체 개수를 카운트하여 반환합니다.

        Returns:
            Dict[str, int]: {"bottle": 4, "cup": 2, ...} 형태의 집계 딕셔너리
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[Vision Error] 이미지 파일을 찾을 수 없습니다: {image_path}")

        # 추론 실행 (verbose=False로 터미널 출력 최소화)
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False
        )

        detected_labels: List[str] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for cls_idx in boxes.cls:
                label_name = self.model.names[int(cls_idx)]
                
                # 타깃 클래스 필터링이 설정되어 있다면 필터 적용
                if self.target_classes and label_name not in self.target_classes:
                    continue
                    
                detected_labels.append(label_name)

        # 수량 집계
        counts = dict(Counter(detected_labels))
        return counts

    def detect_and_save_visual(self, image_path: str, output_path: str) -> Dict[str, int]:
        """
        바운딩 박스가 렌더링된 이미지를 저장하고 검출 결과를 반환합니다. (디버깅/모니터링용)
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False
        )
        
        counts = self.detect_shelf(image_path)
        
        # 첫 번째 결과 프레임 저장
        for result in results:
            result.save(filename=output_path)
            break

        return counts