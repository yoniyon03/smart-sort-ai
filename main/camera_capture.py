# 아두이노 연결
import cv2
import os
import time
import serial

# --- 1. 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(PROJECT_ROOT, "input_images")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# --- 2. Arduino 포트 및 카메라 설정 ---
# ARDUINO_PORT 윈도우는 COM 포트로 수정
ARDUINO_PORT = "/dev/cu.usbmodem141011"
BAUD_RATE = 115200

# CAMERA_INDEX pc 마다 다를 수 있으니 0, 1 바꿔보기
CAMERA_INDEX = 0

DELAY_SECONDS = 0.01

def _sharpness_score(frame):
    """ 분산으로 선명도 측정 """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main_capture():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
        return

    # 1. 카메라 자동 설정 끄기 (가능한 경우)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 = OFF / 1 = ON
    cap.set(cv2.CAP_PROP_FOCUS, 0) # 포커스 수동 고정 (지원하는 카메라만), 숫자 크게 바꿔가면서 테스트

    # 2. 자동 노출 끄고 셔터 수동
    # 백엔드마다 조금씩 다른데, 0.25=수동, 0.75=자동 식으로 동작
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    # 3. 셔터(노출시간) 짧게 - 숫자 줄일수록 더 어둡지만 덜 흔들림
    SHUTTER_SPEED = -10  # -7. -6도 시험
    cap.set(cv2.CAP_PROP_EXPOSURE, SHUTTER_SPEED)

    # 4. 이득/ISO 비슷한 역할 (밝기 보정용)
    cap.set(cv2.CAP_PROP_GAIN, 4)       # 0~10 정도에서 테스트
    cap.set(cv2.CAP_PROP_ISO_SPEED, 1200)    # 지원되는 카메라이면 유지

    # 노트북/USB 카메라마다 지원되는 속성 다르므로 찍어보기
    print("[INFO] EXPOSURE: ", cap.get(cv2.CAP_PROP_EXPOSURE))
    print("[INFO] GAIN    : ", cap.get(cv2.CAP_PROP_GAIN))

    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Arduino 포트({ARDUINO_PORT} @ {BAUD_RATE}bps) 연결 성공.")
    except serial.SerialException:
        print(f"[ERROR] Arduino 포트({ARDUINO_PORT})를 열 수 없습니다.")
        print("[INFO] 1. Arduino가 연결됐는지 확인하세요.")
        print("[INFO] 2. ARDUINO_PORT 이름이 정확한지 확인하세요.")
        cap.release()
        return

    print(f"[INFO] Arduino 'READY' 신호를 기다리는 중...")

    last_sensor_state = "1"

    while True:
        line = ser.readline().decode('utf-8').strip()
        if line == "READY":
            print("[SUCCESS] Arduino 'READY' 신호 확인! 연결 성공.")
            ser.reset_input_buffer()
            break
        elif line:
            print(f"[SYNC] Arduino 부팅 신호 수신: {line}")

    print("==================================================")
    print(f"[INFO] 카메라가 연결되었습니다. (저장 폴더: {SAVE_FOLDER})")
    print(f"[INFO] Arduino에서 '0' (물체 감지) 신호를 대기합니다...")
    print(f"[INFO] (종료하려면 터미널에서 Ctrl + C 를 누르세요)")
    print("==================================================")

    # last_sensor_state = "Unknown"

    pending_captures = []  # "촬영 예약 목록" (예: [10:05, 10:07, 10:09])
    COOLDOWN = 0.5  # 같은 물체는 최소 0.5초 간격으로만 촬영
    last_trigger_time = 0.0

    try:
        while True:
            # 카메라 설정을 유지하고 버퍼를 비우기 위함
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다. (루프 1)")
                time.sleep(0.1)  # 잠시 대기 후 재시도
                continue

            line = ser.readline().decode('utf-8').strip()
            now = time.time()

            if line:
                # 수정: 0이 들어오면 → 일정 간격으로 단 1번만 예약
                if line == "0":
                    if now - last_trigger_time > COOLDOWN:
                        capture_time = now + DELAY_SECONDS
                        pending_captures.append(capture_time)
                        last_trigger_time = now
                        print(f"[SIGNAL] 0 감지 → 촬영 예약됨 (delay={DELAY_SECONDS}s)")

            triggered = []
            for ct in pending_captures:
                if now >= ct:
                    print("[CAPTURE] 예약된 촬영 실행!")

                    best_frame = None
                    best_score = -1

                    # 선명한 사진 선택
                    for _ in range(5):
                        r2, f2 = cap.read()
                        if not r2:
                            continue
                        score = _sharpness_score(f2)
                        if score > best_score:
                            best_score = score
                            best_frame = f2
                        time.sleep(0.005)

                    if best_frame is not None:
                        filename = f"capture_{int(time.time())}.jpg"
                        save_path = os.path.join(SAVE_FOLDER, filename)
                        cv2.imwrite(save_path, best_frame)
                        print(f"[SUCCESS] 저장 완료 → {save_path}")
                    else:
                        print("[ERROR] best_frame 생성 실패")

                    triggered.append(ct)

            # 예약 제거
            if triggered:
                pending_captures = [t for t in pending_captures if t not in triggered]

    except KeyboardInterrupt:
        print("\n[INFO] 강제 종료.")
    finally:
        print("[INFO] 카메라와 Arduino 포트를 종료합니다.")
        cap.release()
        ser.close()


if __name__ == "__main__":
    main_capture()


# # enter 키 누를 때마다 사진 1장씩 찍히는 코드 (USB 카메라 설정 포함)
# import cv2
# import os
# import time
#
# # --- 1. 경로 설정 ---
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# SAVE_FOLDER = os.path.join(PROJECT_ROOT, "input_images")
# os.makedirs(SAVE_FOLDER, exist_ok=True)
#
# CAMERA_INDEX = 0   # 필요하면 1, 2 로 바꿔가며 테스트
#
# def _sharpness_score(frame):
#     """라플라시안 분산으로 선명도 측정"""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray, cv2.CV_64F).var()
#
# def main_capture():
#     cap = cv2.VideoCapture(CAMERA_INDEX)
#
#     if not cap.isOpened():
#         print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
#         print("[INFO] USB 연결을 확인하거나, CAMERA_INDEX를 1, 2 등으로 바꿔보세요.")
#         return
#
#     # ── 여기부터 아까 자동 캡처 코드와 동일한 카메라 설정 ──
#     # 1) 자동 포커스 끄고 수동 포커스(지원되는 카메라만)
#     cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # 0 = OFF
#     cap.set(cv2.CAP_PROP_FOCUS, 0)       # 숫자 바꿔가며 테스트 (0~255 범위인 경우가 많음)
#
#     # 2) 자동 노출 끄고 수동으로
#     cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 백엔드마다 0.25가 수동, 0.75가 자동인 경우 많음
#
#     # 3) 셔터(노출시간) 짧게: 숫자 작을수록 더 어둡지만 덜 흔들림
#     SHUTTER_SPEED = -10  # -7, -8 등도 시험
#     cap.set(cv2.CAP_PROP_EXPOSURE, SHUTTER_SPEED)
#
#     # 4) Gain / ISO로 밝기 보정
#     cap.set(cv2.CAP_PROP_GAIN, 4)          # 0~10 정도에서 조정
#     cap.set(cv2.CAP_PROP_ISO_SPEED, 1200)  # 지원 안 하면 무시됨
#
#     print("==================================================")
#     print(f"[INFO] 카메라가 연결되었습니다. (저장 폴더: {SAVE_FOLDER})")
#     print(f"[INFO] 현재 EXPOSURE:", cap.get(cv2.CAP_PROP_EXPOSURE))
#     print(f"[INFO] 현재 GAIN    :", cap.get(cv2.CAP_PROP_GAIN))
#     print(f"[INFO] 'Enter' 키를 누르면 사진이 캡처됩니다.")
#     print(f"[INFO] 'q'를 입력하고 Enter를 누르면 종료됩니다.")
#     print("==================================================")
#
#     try:
#         while True:
#             user_input = input()  # 여기서 Enter 입력을 기다림
#
#             if user_input.lower() == 'q':
#                 print("[INFO] 'q' 입력. 프로그램을 종료합니다.")
#                 break
#
#             # Enter만 눌린 경우(user_input == "") 실제 캡처 수행
#             print("[CAPTURE] 촬영 시도... 가장 선명한 프레임을 선택합니다.")
#
#             best_frame = None
#             best_score = -1
#
#             # 연속으로 N장 읽어 가장 선명한 한 장 선택
#             N = 5
#             for _ in range(N):
#                 ret, frame = cap.read()
#                 if not ret:
#                     continue
#
#                 score = _sharpness_score(frame)
#                 if score > best_score:
#                     best_score = score
#                     best_frame = frame
#
#                 time.sleep(0.005)  # 버퍼 약간 이동용(너무 길게 두면 다시 흔들릴 수 있음)
#
#             if best_frame is None:
#                 print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다.")
#                 continue
#
#             filename = f"capture_{int(time.time())}.jpg"
#             save_path = os.path.join(SAVE_FOLDER, filename)
#
#             cv2.imwrite(save_path, best_frame)
#             print(f"\n[SUCCESS] 캡처 성공! 이미지가 '{save_path}'에 저장되었습니다.")
#             print(f"[INFO] (run_project.py가 1초 안에 이 파일을 처리합니다...)")
#
#     except KeyboardInterrupt:
#         print("\n[INFO] 강제 종료.")
#     finally:
#         print("[INFO] 카메라를 종료합니다.")
#         cap.release()
#
#
# if __name__ == "__main__":
#     main_capture()