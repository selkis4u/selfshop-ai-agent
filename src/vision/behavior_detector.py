import math
from typing import Dict, Any, List, Optional
from ultralytics import YOLO
import numpy as np


class BehaviorDetector:
    """
    YOLOv11-Pose 기반 점포 내 이상행동(실신/쓰러짐, 급격한 난동 등) 감지기
    """

    def __init__(self, model_path: str = "data/models/yolo11n-pose.pt"):
        print(f"[Vision] YOLOv11 Pose 모델 로드 중: {model_path}")
        self.model = YOLO(model_path)
        self.prev_keypoints: Optional[np.ndarray] = None

    def analyze_frame(self, frame) -> Dict[str, Any]:
        """
        단일 프레임에서 인물 골격을 추출하고 이상행동 여부를 판별합니다.
        """
        results = self.model.track(source=frame, persist=True, verbose=False)
        alerts = []

        for result in results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue

            # keypoints shape: (num_persons, 17, 3) -> [x, y, conf]
            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes

            for person_idx, kpts in enumerate(keypoints_data):
                # 1. 쓰러짐(Fall Down) 감지 로직
                # 머리/코(0), 어깨 중앙((5+6)/2), 골반 중앙((11+12)/2)의 각도 및 박스 종횡비
                if boxes is not None and len(boxes.xywh) > person_idx:
                    _, _, w, h = boxes.xywh[person_idx].cpu().numpy()
                    
                    # 서 있는 사람은 보통 h > w (종횡비 w/h < 1.0)
                    # 바닥에 쓰러지면 w > h 또는 w/h >= 1.2
                    aspect_ratio = w / max(h, 1e-5)
                    
                    nose_y = kpts[0][1]
                    hip_y = (kpts[11][1] + kpts[12][1]) / 2.0

                    # 머리가 골반보다 아래에 있거나 가로로 누워있는 상태
                    if aspect_ratio >= 1.2 or (nose_y > hip_y and kpts[0][2] > 0.5):
                        alerts.append({
                            "type": "FALL_DOWN",
                            "severity": "CRITICAL",
                            "message": f"인물 #{person_idx} 쓰러짐/실신 의심 감지"
                        })

                # 2. 난동/급격한 손짓(Vandalism/Rapid Motion) 감지
                # 손목(9:좌, 10:우) 좌표의 이전 프레임 대비 변위 측정
                # (프레임 간 연속 추적이 연결된 경우 동작)

        return {
            "has_anomaly": len(alerts) > 0,
            "alerts": alerts
        }