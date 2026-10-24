import os
import cv2
from typing import Union, Optional


class ShelfCamera:
    """
    매대 모니터링용 실시간 카메라 캡처 클래스
    - 로컬 USB/노트북 웹캠(디바이스 인덱스 정수: 0, 1 등) 지원
    - 스마트폰 IP Webcam 스트림(HTTP/RTSP URL 문자열) 지원
    """

    def __init__(self, source: Union[int, str] = 0):
        """
        Args:
            source: 0 (노트북 기본 웹캠) 또는 "http://192.168.0.x:8080/video" (스마트폰 앱 스트림 URL)
        """
        self.source = source

    def capture_frame(self, save_path: str = "data/live_shelf.jpg") -> str:
        """
        카메라에서 현재 단일 프레임을 캡처하여 디스크에 저장하고 경로를 반환합니다.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"[Camera] 카메라 소스 연결 시도: {self.source}")
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError(f"[Camera Error] 카메라에 연결할 수 없습니다. 소스: {self.source}")

        # 첫 프레임은 노출/화이트밸런스 조정 중일 수 있으므로 2~3프레임 버리고 최신 프레임 캡처
        for _ in range(5):
            ret, frame = cap.read()

        cap.release()

        if not ret or frame is None:
            raise RuntimeError("[Camera Error] 프레임을 정상적으로 읽어오지 못했습니다.")

        cv2.imwrite(save_path, frame)
        print(f"[Camera] 실시간 프레임 캡처 성공: {save_path} ({frame.shape[1]}x{frame.shape[0]})")
        return save_path

    def preview_stream(self):
        """
        카메라 앵글 및 매대 구도를 실시간으로 확인하는 미리보기 창을 띄웁니다.
        'q'를 누르면 창이 닫힙니다.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[Camera Error] 카메라 열기 실패: {self.source}")
            return

        print("[Camera] 실시간 프리뷰 시작 ('q' 키를 누르면 종료됩니다)")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(
                frame,
                "Press 'q' to exit preview",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.imshow("Shelf Live Preview", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()