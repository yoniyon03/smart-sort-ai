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

def main_capture():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
        return

    # 1. 카메라 자동 설정 끄기
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 자동 초점 끄기
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 셔터 속도/노출 수동 모드로 (0=수동, 1=자동모드끄기)

    # 2. 셔터 속도 빠르게 고정
    # (값이 작을수록 셔터가 빨라짐: -13 ~ 0)
    # (예: -6 = 1/60초, -7 = 1/125초, -8 = 1/250초 ...)
    # (카메라 기종마다 다름, -7 ~ -10 사이로 테스트 필요)
    SHUTTER_SPEED = -8  # 1/250초 예시
    cap.set(cv2.CAP_PROP_EXPOSURE, SHUTTER_SPEED)

    # 3. ISO(감도) 설정
    # 셔터가 빨라지면 사진이 "어두워지므로" ISO를 높여서 보정
    # (M2 맥북 내장 카메라는 이 설정이 안 먹힐 수 있음)
    cap.set(cv2.CAP_PROP_ISO_SPEED, 800)

    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"[INFO] 카메라 셔터 속도(Exposure) 설정 시도... 실제 적용 값: {exposure}")

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
            last_sensor_state = "1"
            break
        elif line:
            print(f"[SYNC] Arduino 부팅 신호 수신: {line}")
            last_sensor_state = line

    print("==================================================")
    print(f"[INFO] 카메라가 연결되었습니다. (저장 폴더: {SAVE_FOLDER})")
    print(f"[INFO] Arduino에서 '0' (물체 감지) 신호를 대기합니다...")
    print(f"[INFO] (종료하려면 터미널에서 Ctrl + C 를 누르세요)")
    print("==================================================")

    # last_sensor_state = "Unknown"

    pending_captures = []  # "촬영 예약 목록" (예: [10:05, 10:07, 10:09])

    try:
        while True:
            # 카메라 설정을 유지하고 버퍼를 비우기 위함
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다. (루프 1)")
                time.sleep(0.1)  # 잠시 대기 후 재시도
                continue

            line = ser.readline().decode('utf-8').strip()

            if line:
                if line == "0" and last_sensor_state == "1":
                    capture_time = time.time() + DELAY_SECONDS
                    pending_captures.append(capture_time)
                    print(f"\n[SIGNAL] '0' 신호 수신! {len(pending_captures)}번째 촬영 예약 (5초 후)")

                last_sensor_state = line

            now = time.time()
            triggered_captures = []

            # 예약 목록을 순회
            for capture_time in pending_captures:

                # 현재 시간이 예약된 시간을 지났다면
                if now >= capture_time:
                    print(f"[CAPTURE] 예약된 캡처 실행!")

                    ret_cap, frame_cap = cap.read()
                    if not ret_cap:
                        print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다. (루프 2)")
                    else:
                        filename = f"capture_{int(time.time())}.jpg"
                        save_path = os.path.join(SAVE_FOLDER, filename)
                        cv2.imwrite(save_path, frame_cap)
                        print(f"[SUCCESS] 캡처 성공! 이미지가 '{save_path}'에 저장되었습니다.")
                        print(f"[INFO] (run_project.py가 1초 안에 이 파일을 처리합니다...)")

                    triggered_captures.append(capture_time)

            # 촬영 완료된 항목들 "예약 목록"에서 제거
            if triggered_captures:
                pending_captures = [t for t in pending_captures if t not in triggered_captures]

    except KeyboardInterrupt:
        print("\n[INFO] 강제 종료.")
    finally:
        print("[INFO] 카메라와 Arduino 포트를 종료합니다.")
        cap.release()
        ser.close()


if __name__ == "__main__":
    main_capture()


# # enter 키 누를 때마다 사진 1장씩 찍히는 코드
# import cv2
# import os
# import time
#
# # --- 1. 경로 설정 ---
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# SAVE_FOLDER = os.path.join(PROJECT_ROOT, "input_images")
# os.makedirs(SAVE_FOLDER, exist_ok=True)
#
#
# def main_capture():
#     # --- 2. capture ---
#     CAMERA_INDEX = 0
#     cap = cv2.VideoCapture(CAMERA_INDEX)
#
#     if not cap.isOpened():
#         print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
#         print("[INFO] USB 연결을 확인하거나, CAMERA_INDEX를 2로 바꿔보세요.")
#         return
#
#     print("==================================================")
#     print(f"[INFO] 카메라가 연결되었습니다. (저장 폴더: {SAVE_FOLDER})")
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
#             # 4. Enter가 눌리면 카메라에서 프레임 읽기
#             ret, frame = cap.read()
#             if not ret:
#                 print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다.")
#                 continue
#
#             # 5. 파일 이름
#             filename = f"capture_{int(time.time())}.jpg"
#             save_path = os.path.join(SAVE_FOLDER, filename)
#
#             # 6. 파일로 저장
#             cv2.imwrite(save_path, frame)
#
#             print(f"\n[SUCCESS] 캡처 성공! 이미지가 '{save_path}'에 저장되었습니다.")
#             print(f"[INFO] (run_project.py가 1초 안에 이 파일을 처리합니다...)")
#
#     except KeyboardInterrupt:
#         print("\n[INFO] 강제 종료.")
#     finally:
#         # 7. 종료 시 카메라 닫기
#         print("[INFO] 카메라를 종료합니다.")
#         cap.release()
#
#
# if __name__ == "__main__":
#     main_capture()