# enter 키 누를 때마다 사진 1장씩 찍히는 코드

import cv2
import os
import time

# --- 1. 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(PROJECT_ROOT, "test_originals")
os.makedirs(SAVE_FOLDER, exist_ok=True)


def main_capture():
    # --- 2. capture ---
    # 0번 - 나영 iPhone
    # 1번 - 나영 맥북 내장 카메라
    CAMERA_INDEX = 0
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
        print("[INFO] USB 연결을 확인하거나, CAMERA_INDEX를 2로 바꿔보세요.")
        return

    print("==================================================")
    print(f"[INFO] 카메라가 연결되었습니다. (저장 폴더: {SAVE_FOLDER})")
    print(f"[INFO] 'Enter' 키를 누르면 사진이 캡처됩니다.")
    print(f"[INFO] 'q'를 입력하고 Enter를 누르면 종료됩니다.")
    print("==================================================")

    try:
        while True:
            # --- 3. 'c' 키 대신 Enter 키 입력 대기 ---
            user_input = input()  # 여기서 Enter 입력을 기다림

            if user_input.lower() == 'q':
                print("[INFO] 'q' 입력. 프로그램을 종료합니다.")
                break

            # 4. Enter가 눌리면 그 즉시 카메라에서 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다.")
                continue

            # 5. 파일 이름
            filename = f"capture_{int(time.time())}.jpg"
            save_path = os.path.join(SAVE_FOLDER, filename)

            # 6. 파일로 저장
            cv2.imwrite(save_path, frame)

            print(f"\n[SUCCESS] 캡처 성공! 이미지가 '{save_path}'에 저장되었습니다.")
            print(f"[INFO] (run_project.py가 1초 안에 이 파일을 처리합니다...)")

    except KeyboardInterrupt:
        print("\n[INFO] 강제 종료.")
    finally:
        # 7. 종료 시 카메라 닫기
        print("[INFO] 카메라를 종료합니다.")
        cap.release()


if __name__ == "__main__":
    main_capture()