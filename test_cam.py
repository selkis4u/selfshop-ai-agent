from src.vision.camera import ShelfCamera

# 1. 노트북 내장/USB 웹캠을 사용할 경우: source=0
# 2. 스마트폰(DroidCam 또는 IP Webcam 앱)을 쓸 경우: source="http://192.168.0.x:4747/video"
camera = ShelfCamera(source="http://192.168.219.128:4747/video")

# 실시간 미리보기 창으로 화각 확인 (창 닫으려면 영문 'q' 입력)
camera.preview_stream()

# 단일 프레임 캡처 저장 테스트
captured_file = camera.capture_frame("data/live_shelf.jpg")
print(f"캡처 완료 파일: {captured_file}")