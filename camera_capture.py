# 아두이노 연결
import cv2
import os
import time
import serial

# --- 1. 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(PROJECT_ROOT, "test_originals")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# --- 2. Arduino 포트 및 카메라 설정 ---
ARDUINO_PORT = "/dev/cu.usbmodem141011"
BAUD_RATE = 115200
CAMERA_INDEX = 0

DELAY_SECONDS = 5.0

def main_capture():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
        return

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

    while True:
        line = ser.readline().decode('utf-8').strip()
        if line == "READY":
            print("[SUCCESS] Arduino 'READY' 신호 확인! 연결 성공.")
            break
        elif line:
            print(f"[SYNC] Arduino 부팅 신호 수신: {line}")

    print("==================================================")
    print(f"[INFO] 카메라가 연결되었습니다. (저장 폴더: {SAVE_FOLDER})")
    print(f"[INFO] Arduino에서 '0' (물체 감지) 신호를 대기합니다...")
    print(f"[INFO] (종료하려면 터미널에서 Ctrl + C 를 누르세요)")
    print("==================================================")

    last_sensor_state = "Unknown"

    pending_captures = []  # "촬영 예약 목록" (예: [10:05, 10:07, 10:09])

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()

            if line:
                if last_sensor_state == "Unknown":
                    print(f"[INFO] 센서 초기 상태 수신: {line} ('1'=안 가려짐, '0'=가려짐)")
                    last_sensor_state = line
                    continue  # (사진 안 찍고 다음 루프로)

                if line == "0" and last_sensor_state == "1":
                    capture_time = time.time() + DELAY_SECONDS  # 5초 뒤 시간
                    pending_captures.append(capture_time)
                    print(f"\n[SIGNAL] '0' 신호 수신! {len(pending_captures)}번째 촬영 예약 (5초 후)")

                last_sensor_state = line

            now = time.time()
            triggered_captures = []

            # (예약 목록을 순회)
            for capture_time in pending_captures:

                # (현재 시간이 예약된 시간을 지났다면)
                if now >= capture_time:
                    print(f"[CAPTURE] 예약된 캡처 실행!")

                    ret, frame = cap.read()
                    if not ret:
                        print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다.")
                    else:
                        filename = f"capture_{int(time.time())}.jpg"
                        save_path = os.path.join(SAVE_FOLDER, filename)
                        cv2.imwrite(save_path, frame)
                        print(f"[SUCCESS] 캡처 성공! 이미지가 '{save_path}'에 저장되었습니다.")
                        print(f"[INFO] (run_project.py가 1초 안에 이 파일을 처리합니다...)")

                    triggered_captures.append(capture_time)

            # (촬영 완료된 항목들을 "예약 목록"에서 제거)
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


# enter 키 누를 때마다 사진 1장씩 찍히는 코드
# import cv2
# import os
# import time
#
# # --- 1. 경로 설정 ---
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# SAVE_FOLDER = os.path.join(PROJECT_ROOT, "test_originals")
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