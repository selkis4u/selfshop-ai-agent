import os
import time
import threading
import cv2
from ultralytics import YOLO

# CPU 스레드 최적화 (코어 과열 방지)
os.environ["OMP_NUM_THREADS"] = "4"

class ThreadedStreamCapture:
    """무선 스트림 버퍼 누적을 차단하고 항상 최신 1프레임만 유지하는 리더"""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.ret = ret
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()

# 1. ONNX 모델 로드 (없을 경우 .pt를 CPU 모드로 실행)
model_path = "yolo11n-pose.onnx" if os.path.exists("yolo11n-pose.onnx") else "yolo11n-pose.pt"
print(f"[Vision] CPU 추론 엔진 로드: {model_path}")
model = YOLO(model_path, task="pose")

# 2. 스마트폰 IP 주소 (본인 스마트폰 IP 입력)
stream_url = "http://192.168.219.128:4747/video"
stream = ThreadedStreamCapture(stream_url)

print("▶ CPU 전용 실시간 모니터링 시작 ('q' 누르면 종료)")

prev_time = time.time()
frame_interval = 2  # 2프레임마다 1회 추론 (CPU 점유율 50% 방어)
frame_count = 0

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    frame_count += 1
    display_frame = frame

    # N프레임마다 CPU 추론 수행
    if frame_count % frame_interval == 0:
        results = model.track(
            source=frame,
            device="cpu",
            persist=True,
            verbose=False,
            imgsz=480
        )
        result = results[0]
        display_frame = result.plot()

        # 쓰러짐 감지 (종횡비 + 머리 높이)
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is not None and keypoints is not None:
            for i in range(len(boxes)):
                xywh = boxes.xywh[i].cpu().numpy()
                w, h = xywh[2], xywh[3]
                aspect_ratio = w / max(h, 1e-5)

                kpts = keypoints.data[i].cpu().numpy()
                nose_y = kpts[0][1]
                hip_y = (kpts[11][1] + kpts[12][1]) / 2.0

                if aspect_ratio >= 1.2 or (kpts[0][2] > 0.4 and nose_y > hip_y):
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 70), (0, 0, 255), -1)
                    cv2.putText(
                        display_frame,
                        "🚨 FALL-DOWN DETECTED!",
                        (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2
                    )

    # FPS 계산 및 출력
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(display_frame, f"CPU FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Store Anomaly Monitor (CPU Mode)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()