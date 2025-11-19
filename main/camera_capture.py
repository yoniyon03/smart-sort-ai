# 아두이노 연결
import cv2
import os
import time
import serial

# --- 1. 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(PROJECT_ROOT, "input_images")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ★ 서보 명령 공유 파일
SERVO_CMD_PATH = os.path.join(PROJECT_ROOT, "servo_cmd.txt")

# --- 2. Arduino 포트 및 카메라 설정 ---
ARDUINO_PORT = "/dev/cu.usbmodem3"
BAUD_RATE = 115200

CAMERA_INDEX = 0
DELAY_SECONDS = 0.01


def _sharpness_score(frame):
    """ 분산으로 선명도 측정 """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _check_and_send_servo_cmd(ser, last_sent_angle):
    """
    servo_cmd.txt에 새 각도가 있으면
    아두이노로 'SERVO <angle>\\n' 전송하고, 파일을 비운다.
    같은 각도를 연속으로 보내는 건 막기 위해 last_sent_angle 사용.
    """
    try:
        if not os.path.exists(SERVO_CMD_PATH):
            return last_sent_angle

        with open(SERVO_CMD_PATH, "r") as f:
            content = f.read().strip()

        if not content:
            return last_sent_angle

        try:
            angle = int(content)
        except ValueError:
            print(f"[SERVO] servo_cmd.txt 내용이 숫자가 아님: '{content}'")
            # 내용이 이상하면 그냥 파일만 비우고 끝
            open(SERVO_CMD_PATH, "w").close()
            return last_sent_angle

        # 이미 같은 각도 보냈다면 다시 안 보냄 (원하면 이 조건 빼도 됨)
        if angle == last_sent_angle:
            return last_sent_angle

        cmd = f"SERVO {angle}\n".encode("utf-8")
        ser.write(cmd)
        print(f"[SERVO] 아두이노로 명령 전송: {cmd!r}")

        # 한 번 보낸 뒤 파일 비워서 중복 전송 방지
        open(SERVO_CMD_PATH, "w").close()

        return angle

    except Exception as e:
        print(f"[SERVO] 명령 처리 중 오류: {e}")
        return last_sent_angle


def main_capture():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
        return

    # 카메라 설정
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    SHUTTER_SPEED = -10
    cap.set(cv2.CAP_PROP_EXPOSURE, SHUTTER_SPEED)
    cap.set(cv2.CAP_PROP_GAIN, 4)
    cap.set(cv2.CAP_PROP_ISO_SPEED, 1200)

    print("[INFO] EXPOSURE: ", cap.get(cv2.CAP_PROP_EXPOSURE))
    print("[INFO] GAIN    : ", cap.get(cv2.CAP_PROP_GAIN))

    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.05)  # timeout 살짝 짧게
        print(f"[INFO] Arduino 포트({ARDUINO_PORT} @ {BAUD_RATE}bps) 연결 성공.")
    except serial.SerialException:
        print(f"[ERROR] Arduino 포트({ARDUINO_PORT})를 열 수 없습니다.")
        print("[INFO] 1. Arduino가 연결됐는지 확인하세요.")
        print("[INFO] 2. ARDUINO_PORT 이름이 정확한지 확인하세요.")
        cap.release()
        return

    print(f"[INFO] Arduino 'READY' 신호를 기다리는 중...")

    while True:
        line = ser.readline().decode("utf-8").strip()
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

    pending_captures = []
    COOLDOWN = 0.5
    last_trigger_time = 0.0

    # ★ 마지막으로 보낸 서보 각도
    last_servo_angle = None

    try:
        while True:
            # 1) 카메라 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다. (루프 1)")
                time.sleep(0.1)
                continue

            now = time.time()

            # 2) 아두이노에서 들어온 한 줄 읽기
            line = ser.readline().decode("utf-8").strip()
            if line:
                # 센서 이벤트
                if line == "0":
                    if now - last_trigger_time > COOLDOWN:
                        capture_time = now + DELAY_SECONDS
                        pending_captures.append(capture_time)
                        last_trigger_time = now
                        print(f"[SIGNAL] 0 감지 → 촬영 예약됨 (delay={DELAY_SECONDS}s)")
                elif line == "1":
                    # 필요하면 HIGH 로그
                    pass
                elif line.startswith("SERVO_OK"):
                    print(f"[SERIAL] 아두이노 응답: {line}")
                else:
                    # 기타 디버그용 출력
                    print(f"[SERIAL] 기타 수신: {line}")

            # 3) ★ 서보 명령 파일 확인 후 아두이노로 명령 전송
            last_servo_angle = _check_and_send_servo_cmd(ser, last_servo_angle)

            # 4) 예약된 캡처 실행
            triggered = []
            for ct in pending_captures:
                if now >= ct:
                    print("[CAPTURE] 예약된 촬영 실행!")

                    best_frame = None
                    best_score = -1
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
# # run_project.py 에서 WATCH_FOLDER = os.path.join(PROJECT_ROOT, "input_images") 로 되어 있으니까
# # 여기 폴더 이름도 반드시 동일해야 함
# SAVE_FOLDER = os.path.join(PROJECT_ROOT, "input_images")
# os.makedirs(SAVE_FOLDER, exist_ok=True)
#
# CAMERA_INDEX = 0   # 필요하면 1, 2 로 바꿔가며 테스트
#
#
# def _sharpness_score(frame):
#     """라플라시안 분산으로 선명도 측정"""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray, cv2.CV_64F).var()
#
#
# def main_capture():
#     cap = cv2.VideoCapture(CAMERA_INDEX)
#
#     if not cap.isOpened():
#         print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
#         print("[INFO] USB 연결을 확인하거나, CAMERA_INDEX를 1, 2 등으로 바꿔보세요.")
#         return
#
#     # ── 카메라 설정 (흔들림 줄이기용) ──
#     cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # 0 = OFF
#     cap.set(cv2.CAP_PROP_FOCUS, 0)       # 0~255 범위에서 바꿔가며 테스트
#
#     cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 수동 노출
#     SHUTTER_SPEED = -10                  # 더 짧게/길게 찍고 싶으면 숫자 조정
#     cap.set(cv2.CAP_PROP_EXPOSURE, SHUTTER_SPEED)
#
#     cap.set(cv2.CAP_PROP_GAIN, 4)
#     cap.set(cv2.CAP_PROP_ISO_SPEED, 1200)  # 지원 안 하는 카메라는 무시됨
#
#     print("==================================================")
#     print(f"[INFO] 카메라가 연결되었습니다. (저장 폴더: {SAVE_FOLDER})")
#     print(f"[INFO] 현재 EXPOSURE:", cap.get(cv2.CAP_PROP_EXPOSURE))
#     print(f"[INFO] 현재 GAIN    :", cap.get(cv2.CAP_PROP_GAIN))
#     print(f"[INFO] Enter → 사진 캡처, 'q' 입력 후 Enter → 종료")
#     print("==================================================")
#
#     try:
#         while True:
#             user_input = input()
#
#             if user_input.lower() == 'q':
#                 print("[INFO] 'q' 입력. 프로그램을 종료합니다.")
#                 break
#
#             # Enter만 눌렸을 때 캡처
#             print("[CAPTURE] 촬영 시도... 가장 선명한 프레임을 선택합니다.")
#
#             best_frame = None
#             best_score = -1
#
#             # 연속 N장 중 가장 선명한 한 장 선택
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
#                 time.sleep(0.005)
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
#             print(f"[INFO] run_project.py 가 이 파일을 곧 처리할 거야.")
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