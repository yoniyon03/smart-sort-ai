# # enter 키 누를 때마다 사진 1장씩 찍히는 코드
#
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
#     # 0번 - 나영 iPhone
#     # 1번 - 나영 맥북 내장 카메라
#     # 2번 - USB 카메라
#     CAMERA_INDEX = 1
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
#             # --- 3. 'c' 키 대신 Enter 키 입력 대기 ---
#             user_input = input()  # 여기서 Enter 입력을 기다림
#
#             if user_input.lower() == 'q':
#                 print("[INFO] 'q' 입력. 프로그램을 종료합니다.")
#                 break
#
#             # 4. Enter가 눌리면 그 즉시 카메라에서 프레임 읽기
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


# 아두이노 연결
import cv2
import os
import time
import serial

# --- 1. 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(PROJECT_ROOT, "test_originals")
os.makedirs(SAVE_FOLDER, exist_ok=True)
# -----------------------------------------------

# --- 2. Arduino 포트 및 카메라 설정 ---
# (Mac에서 'ls /dev/cu.*'로 찾은 Arduino 포트 이름)
ARDUINO_PORT = "/dev/cu.usbmodem1101"

# (아두이노 코드의 Serial.begin(115200)과 일치)
BAUD_RATE = 115200

# 0번 - 나영 iPhone
# 1번 - 나영 맥북 내장 카메라
# 2번 - USB 카메라
CAMERA_INDEX = 1


# -----------------------------------------------

def main_capture():
    # 3. 카메라 연결
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
        return

    # 4. Arduino 시리얼 포트 연결
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Arduino 포트({ARDUINO_PORT} @ {BAUD_RATE}bps) 연결 성공.")
    except serial.SerialException:
        print(f"[ERROR] Arduino 포트({ARDUINO_PORT})를 열 수 없습니다.")
        print("[INFO] 1. Arduino가 연결됐는지 확인하세요.")
        print("[INFO] 2. ARDUINO_PORT 이름이 정확한지 확인하세요.")
        cap.release()
        return

    print("==================================================")
    print(f"[INFO] 카메라가 연결되었습니다. (저장 폴더: {SAVE_FOLDER})")
    print(f"[INFO] Arduino에서 '0' (물체 감지) 신호를 대기합니다...")
    print(f"[INFO] (종료하려면 터미널에서 Ctrl + C 를 누르세요)")
    print("==================================================")

    # 5. 1.9초마다 100장 찍히는 걸 방지하기 위한 "상태" 변수
    # (HIGH = 1 = 안 가려짐, LOW = 0 = 가려짐)
    last_sensor_state = "1"

    try:
        while True:
            # 6. Arduino가 보낸 "0" 또는 "1" 신호를 한 줄씩 읽음
            line = ser.readline().decode('utf-8').strip()

            # (신호가 들어왔을 때만)
            if line:
                # 7. (핵심) 신호가 "0"이고, *이전* 상태가 "1"이었을 때
                if line == "0" and last_sensor_state == "1":
                    print(f"\n[SIGNAL] '0' 신호 수신! (물체 감지)")

                    # 8. 그 즉시 카메라에서 프레임 읽기
                    ret, frame = cap.read()
                    if not ret:
                        print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다.")
                        continue  # (다음 신호를 기다림)

                    # 9. 파일 이름 (타임스탬프)
                    filename = f"capture_{int(time.time())}.jpg"
                    save_path = os.path.join(SAVE_FOLDER, filename)

                    # 10. 파일로 저장
                    cv2.imwrite(save_path, frame)

                    print(f"[SUCCESS] 캡처 성공! 이미지가 '{save_path}'에 저장되었습니다.")
                    print(f"[INFO] (run_project.py가 1초 안에 이 파일을 처리합니다...)")

                # 11. 현재 상태를 "마지막 상태"로 저장
                last_sensor_state = line

    except KeyboardInterrupt:
        print("\n[INFO] 강제 종료.")
    finally:
        # 12. 종료 시 카메라와 시리얼 포트 닫기
        print("[INFO] 카메라와 Arduino 포트를 종료합니다.")
        cap.release()
        ser.close()


if __name__ == "__main__":
    main_capture()