import cv2
from ultralytics import YOLO

# 1. 초경량 포즈 추정 모델 로드 (자동 다운로드)
model = YOLO("yolo11n-pose.pt")

# 2. 웹캠 연결 (노트북 기본 웹캠은 0, 외장/스마트폰 캠은 해당 인덱스나 URL)
cap = cv2.VideoCapture("http://192.168.219.128:4747/video")
#cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다. 연결 상태를 확인해주세요.")
    exit()

print("▶ 실시간 이상행동 감지 시작 ('q'를 누르면 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추적 및 관절 추출 (추론 속도 최적화)
    results = model.track(source=frame, persist=True, verbose=False)

    for result in results:
        # 감지된 사람들의 바운딩 박스와 관절 좌표
        boxes = result.boxes
        keypoints = result.keypoints

        # 기본 시각화 (인물 골격 선 그리기)
        frame = result.plot()

        if boxes is not None and keypoints is not None:
            for i in range(len(boxes)):
                # 바운딩 박스 크기 (x_center, y_center, width, height)
                xywh = boxes.xywh[i].cpu().numpy()
                w, h = xywh[2], xywh[3]

                # 관절 좌표 (17개: 0번 코, 5~6번 어깨, 11~12번 골반)
                kpts = keypoints.data[i].cpu().numpy()
                nose_y = kpts[0][1]
                nose_conf = kpts[0][2]
                hip_y = (kpts[11][1] + kpts[12][1]) / 2.0

                # [판정 로직 1] 종횡비 판정: 가로 폭(w)이 세로 높이(h)보다 현저히 크면 누워있는 상태
                aspect_ratio = w / max(h, 1e-5)

                # [판정 로직 2] 관절 역전: 머리(코)가 골반보다 아래에 위치하는 경우
                is_head_below_hip = (nose_conf > 0.4 and nose_y > hip_y)

                if aspect_ratio >= 1.2 or is_head_below_hip:
                    # 화면 상단에 붉은색 경고 박스 표시
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "🚨 WARNING: FALL-DOWN / ABNORMAL POSTURE DETECTED!",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

    cv2.imshow("Real-Time Store Anomaly Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()