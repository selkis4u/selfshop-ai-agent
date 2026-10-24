import os
import cv2
import numpy as np
import pytest
from src.vision.detector import ShelfDetector


@pytest.fixture
def dummy_shelf_image(tmp_path) -> str:
    """테스트용 빈 단색 이미지 생성"""
    img_path = str(tmp_path / "test_shelf.jpg")
    # 640x640 크기의 임시 3채널 검은색 이미지 생성
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)
    return img_path


def test_detector_initialization():
    """모델 로드 및 기본 인스턴스 초기화 테스트"""
    detector = ShelfDetector(model_path="data/models/yolo11n.pt")
    assert detector.model is not None
    assert detector.conf_threshold == 0.25


def test_detect_shelf_returns_dict(dummy_shelf_image):
    """추론 결과가 Dict[str, int] 규격으로 반환되는지 테스트"""
    detector = ShelfDetector(model_path="data/models/yolo11n.pt")
    counts = detector.detect_shelf(dummy_shelf_image)

    # 더미 이미지이므로 검출 개수는 0개 딕셔너리여야 함
    assert isinstance(counts, dict)
    assert len(counts) == 0


def test_missing_image_raises_error():
    """존재하지 않는 파일 경로 입력 시 FileNotFoundError 발생 테스트"""
    detector = ShelfDetector(model_path="data/models/yolo11n.pt")
    with pytest.raises(FileNotFoundError):
        detector.detect_shelf("non_existent_shelf_path_123.jpg")