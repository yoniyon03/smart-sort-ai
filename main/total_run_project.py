from ultralytics import YOLO
import cv2
import os
import glob
import time
import serial

from utils.ocr_module import run_ocr
from utils.hsv_module import get_color

from device_api import (
    MANAGER_ID,
    fetch_setup,
    build_rule_maps,
    send_sorting_result,
    send_error_event,
)

# True = 엔터 눌러 수동 촬영 모드
# 센서 없이 YOLO + 서버 + 서보 테스트 --> True
# 센서 포함 테스트 --> False
TEST_MODE = True

# --- 경로 / 환경 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_ROOT = os.path.join(PROJECT_ROOT, "..", "training")

WATCH_FOLDER = os.path.join(PROJECT_ROOT, "input_images")
ERROR_ROOT = os.path.join(PROJECT_ROOT, "results_error")
OCR_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "results_text")
COLOR_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "results_color")
ERROR_TEXT_OUTPUT_FOLDER = os.path.join(ERROR_ROOT, "text")
ERROR_COLOR_OUTPUT_FOLDER = os.path.join(ERROR_ROOT, "color")

os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(OCR_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(COLOR_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ERROR_ROOT, exist_ok=True)
os.makedirs(ERROR_TEXT_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ERROR_COLOR_OUTPUT_FOLDER, exist_ok=True)

# --- 아두이노 / 카메라 설정 ---
SERIAL_PORT = "/dev/cu.usbmodem141011"   # 네 환경에 맞게 수정 (윈도우면 "COM3" 이런 식)
BAUD_RATE = 115200
CAMERA_INDEX = 0

# --- YOLO 모델 로드 ---
TRAINED_MODEL_PATH = os.path.join(
    TRAINING_ROOT,
    "runs",
    "detect",
    "train12",
    "weights",
    "best.pt",
)

if not os.path.exists(TRAINED_MODEL_PATH):
    print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {TRAINED_MODEL_PATH}")
    print("[INFO] 'main.py'를 실행하여 모델을 먼저 학습시키세요.")
    exit()

print("[INFO] Loading trained YOLO model...")
model = YOLO(TRAINED_MODEL_PATH)
CLASS_NAMES = model.names
print(f"[INFO] Model loaded. Classes: {CLASS_NAMES}")

# --- 공통 파라미터 ---
KNOWN_TEXTS = []
KNOWN_COLORS = []

YOLO_CONF_THRESHOLD = 0.70
OCR_CONF_THRESHOLD = 0.70


def _sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# def capture_image_from_camera(cap, save_folder):
#     """센서가 0 들어왔을 때 카메라에서 여러 장 찍고 가장 선명한 한 장 저장"""
#     best_frame = None
#     best_score = -1
#
#     N = 5
#     for _ in range(N):
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         score = _sharpness_score(frame)
#         if score > best_score:
#             best_score = score
#             best_frame = frame
#         time.sleep(0.005)
#
#     if best_frame is None:
#         print("[ERROR] 카메라에서 프레임을 읽지 못했습니다.")
#         # 카메라 타임아웃 에러 전송
#         send_error_event(
#             manager_id=MANAGER_ID,
#             error_code="CAMERA_TIMEOUT",
#             image_path=None,
#         )
#         return None
#
#     filename = f"capture_{int(time.time())}.jpg"
#     save_path = os.path.join(save_folder, filename)
#     cv2.imwrite(save_path, best_frame)
#     print(f"[CAPTURE] 이미지 저장: {save_path}")
#     return save_path


def manual_capture_from_keyboard(cap, save_folder):
    """
    센서 없이 개발자가 직접 Enter 눌러서 테스트하는 모드.
    'q' 입력 시 종료.
    """
    print("==================================================")
    print("[TEST MODE] Enter → 사진 캡처")
    print("[TEST MODE] 'q' + Enter → 종료")
    print("==================================================")

    while True:
        user_input = input()

        if user_input.lower() == "q":
            print("[TEST MODE] 종료합니다.")
            return None  # 메인 루프로 나감

        print("[CAPTURE] 촬영 중... 가장 선명한 프레임을 찾는 중...")

        best_frame = None
        best_score = -1

        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                continue
            score = _sharpness_score(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
            time.sleep(0.005)

        if best_frame is None:
            print("[ERROR] 카메라에서 프레임을 읽지 못했습니다.")
            continue

        filename = f"capture_{int(time.time())}.jpg"
        save_path = os.path.join(save_folder, filename)

        cv2.imwrite(save_path, best_frame)
        print(f"[SUCCESS] 촬영 완료 → {save_path}")

        # run_project.py가 처리할 수 있게 바로 반환
        return save_path


def process_image(image_path, conf_threshold=0.7, ocr_good_threshold=0.7):
    """YOLO + OCR + HSV 실행해서 텍스트/색상 정보 추출"""
    if not os.path.exists(image_path):
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {image_path}")
        return None, None

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"[ERROR] cv2가 이미지를 읽지 못했습니다: {image_path}")
        return None, None

    orig_h, orig_w, _ = original_image.shape
    print(f"\n--- Processing Image: {image_path} (Size: {orig_w}x{orig_h}) ---")

    results = model(original_image)
    result = results[0]

    final_data = {
        "text": None,
        "text_conf": 0.0,
        "color": None,
        "object_type": None,
        "low_confidence_skips": [],
        "had_text": False,
        "had_color": False,
    }

    found_good_text = False

    for box in result.boxes:
        conf = box.conf[0].item()
        cls_index = int(box.cls[0].item())
        class_name = CLASS_NAMES[cls_index]

        nx1, ny1, nx2, ny2 = box.xyxyn[0].tolist()
        x1 = int(nx1 * orig_w)
        y1 = int(ny1 * orig_h)
        x2 = int(nx2 * orig_w)
        y2 = int(ny2 * orig_h)

        if class_name == "marker_text":
            bw = x2 - x1
            bh = y2 - y1
            pad = int(0.2 * max(bw, bh))
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(orig_w, x2 + pad)
            y2 = min(orig_h, y2 + pad)

        cropped_img = original_image[y1:y2, x1:x2]

        if class_name == "marker_text":
            final_data["had_text"] = True

            if found_good_text:
                print("[SKIP] 이미 신뢰도 높은 텍스트를 인식해서 나머지 marker_text는 OCR를 건너뜁니다.")
                continue

            if conf < conf_threshold:
                print(f"[WARN] 'marker_text' confidence is low ({conf * 100:.0f}%), 하지만 OCR를 시도합니다.")
                final_data["low_confidence_skips"].append(
                    f"{class_name}_lowconf ({conf * 100:.0f}%)"
                )
            else:
                print(f"[YOLO] Found 'marker_text' (Conf: {conf * 100:.0f}%). Sending to OCR...")

            temp_crop_path = os.path.join(PROJECT_ROOT, "temp_ocr_image.jpg")
            cv2.imwrite(temp_crop_path, cropped_img)
            print(f"[DEBUG] OCR용 *컬러 원본* 크롭 이미지 저장: {temp_crop_path}")

            extracted_text, ocr_confidence = run_ocr(temp_crop_path, multi_angle=True)

            if extracted_text:
                final_data["text"] = extracted_text
                final_data["text_conf"] = ocr_confidence
                print(f"[SUCCESS] OCR Result: {final_data['text']} (Conf: {ocr_confidence:.2f})")
                if ocr_confidence >= ocr_good_threshold:
                    found_good_text = True
            else:
                print("[INFO] OCR module ran, but found no text.")

            try:
                os.remove(temp_crop_path)
            except Exception as e:
                print(f"[WARN] Failed to remove temp crop file: {e}")

        elif class_name == "marker_color":
            final_data["had_color"] = True

            if conf < conf_threshold:
                print(
                    f"[WARN] 'marker_color' confidence is low ({conf * 100:.0f}%), "
                    f"하지만 HSV 분석을 시도합니다."
                )
                final_data["low_confidence_skips"].append(
                    f"{class_name}_lowconf ({conf * 100:.0f}%)"
                )
            else:
                print(f"[YOLO] Found 'marker_color' (Conf: {conf * 100:.0f}%). Sending to HSV...")

            detected_color = get_color(cropped_img)
            final_data["color"] = detected_color
            print(f"[SUCCESS] Color Result: {final_data['color']}")

        elif class_name == "objects":
            print(f"[YOLO] Found 'objects' (Conf: {conf * 100:.0f}%)")
            final_data["object_type"] = "object"

    return result.plot(), final_data


def classify_and_act(image_path, annotated_image, extracted_data, text_rules, color_rules, ser):
    """YOLO/OCR 결과 보고 폴더 분류 + 서버 전송 + 서보 제어까지 한 번에"""

    detected_text = extracted_data.get("text")
    detected_text_conf = extracted_data.get("text_conf", 0.0)
    detected_color = extracted_data.get("color")
    had_text = extracted_data.get("had_text", False)
    had_color = extracted_data.get("had_color", False)
    low_conf_list = extracted_data.get("low_confidence_skips", [])

    rule_id = None
    chute_id = None
    servo_deg = None
    save_path = None
    is_success = False
    error_code = None

    base_name = os.path.basename(image_path)
    save_name = f"{os.path.splitext(base_name)[0]}_result.jpg"

    # ─────────────────────────────────────
    # ① 텍스트 기준 정상/에러 분기
    # ─────────────────────────────────────
    if detected_text and detected_text_conf >= OCR_CONF_THRESHOLD:
        # 텍스트는 읽혔는데, 분류 규칙에 있는지 확인
        rule_info = text_rules.get(detected_text)

        if rule_info:
            # ✅ 정상 텍스트 인식 + 규칙 있음
            save_path = os.path.join(OCR_OUTPUT_FOLDER, save_name)
            is_success = True
            print("[ROUTE] 정상 텍스트 결과 폴더로 저장")

            rule_id = rule_info["ruleId"]
            chute_id = rule_info["chuteId"]
            servo_deg = rule_info.get("servoDeg")

        else:
            # 텍스트는 읽었는데, 규칙에 없는 텍스트 → OCR_FAIL
            save_path = os.path.join(ERROR_TEXT_OUTPUT_FOLDER, save_name)

            # 'UNKNOWN' 같은 라벨은 UNKNOWN_ERROR로 처리
            if detected_text.strip().upper() == "UNKNOWN":
                error_code = "UNKNOWN_ERROR"
                print("[ROUTE] UNKNOWN_ERROR: UNKNOWN 텍스트 라벨 감지")
            else:
                error_code = "OCR_FAIL"
                print("[ROUTE] OCR_FAIL: 분류 규칙에 없는 텍스트 라벨")

    # ─────────────────────────────────────
    # ② 색상 기준 정상/에러 분기
    #    (텍스트가 없고, 색상만 가지고 분류하는 경우)
    # ─────────────────────────────────────
    elif (not detected_text) and (detected_color in KNOWN_COLORS):
        rule_info = color_rules.get(detected_color)

        if rule_info:
            # ✅ 정상 색상 인식 + 규칙 있음
            save_path = os.path.join(COLOR_OUTPUT_FOLDER, save_name)
            is_success = True
            print("[ROUTE] 정상 색상 결과 폴더로 저장")

            rule_id = rule_info["ruleId"]
            chute_id = rule_info["chuteId"]
            servo_deg = rule_info.get("servoDeg")
        else:
            # 색상은 읽었는데, 규칙에 없음 → 일단 UNKNOWN_ERROR로 통일
            save_path = os.path.join(ERROR_COLOR_OUTPUT_FOLDER, save_name)
            error_code = "UNKNOWN_ERROR"
            print("[ROUTE] UNKNOWN_ERROR: 분류 규칙에 없는 색상 라벨")

    # ─────────────────────────────────────
    # ③ 텍스트는 읽혔지만, 신뢰도 낮거나 marker_text low conf
    #    → OCR_FAIL
    # ─────────────────────────────────────
    elif (
        (detected_text and detected_text_conf < OCR_CONF_THRESHOLD)
        or any("marker_text" in s for s in low_conf_list)
    ):
        save_path = os.path.join(ERROR_TEXT_OUTPUT_FOLDER, save_name)
        error_code = "OCR_FAIL"
        print("[ROUTE] OCR_FAIL: OCR 신뢰도 낮음 / marker_text low confidence")

    # ─────────────────────────────────────
    # ④ 색상 쪽 이상 (예상 밖 색상/low conf)
    #    → UNKNOWN_ERROR
    # ─────────────────────────────────────
    elif (
        (detected_color and detected_color not in KNOWN_COLORS)
        or any("marker_color" in s for s in low_conf_list)
    ):
        save_path = os.path.join(ERROR_COLOR_OUTPUT_FOLDER, save_name)
        error_code = "UNKNOWN_ERROR"
        print("[ROUTE] UNKNOWN_ERROR: 색상 정보 이상(예상 밖 색상 또는 low confidence)")

    # ─────────────────────────────────────
    # ⑤ 그 외 (마지막 fallback)
    #    - had_text=True & text 없음 → 빈 라벨/읽기 불가 → UNKNOWN_ERROR
    #    - had_color=True & color 없음 → UNKNOWN_ERROR
    #    - 둘 다 false → NO_MARKER
    # ─────────────────────────────────────
    else:
        if had_text and not detected_text:
            save_path = os.path.join(ERROR_TEXT_OUTPUT_FOLDER, save_name)
            error_code = "UNKNOWN_ERROR"
            print("[ROUTE] UNKNOWN_ERROR: marker_text는 있었으나 OCR 결과 없음(빈 라벨 등)")

        elif had_color and not detected_color:
            save_path = os.path.join(ERROR_COLOR_OUTPUT_FOLDER, save_name)
            error_code = "UNKNOWN_ERROR"
            print("[ROUTE] UNKNOWN_ERROR: marker_color는 있었으나 HSV 결과 없음")

        else:
            # YOLO가 marker_text / marker_color 둘 다 못 본 경우 → NO_MARKER
            save_path = os.path.join(ERROR_ROOT, save_name)
            error_code = "NO_MARKER"
            print("[ROUTE] NO_MARKER: YOLO가 표식을 탐지하지 못함 (너무 빠르게 지나간 경우 등)")

    # ─────────────────────────────────────
    # ⑥ 파일 저장 + 서버 통신 + 서보 제어
    # ─────────────────────────────────────
    try:
        cv2.imwrite(save_path, annotated_image)

        if is_success:
            print(f"[SUCCESS] 결과 이미지를 '{save_path}'에 저장했습니다.")

            # --- 서보 제어 ---
            if servo_deg is not None:
                if ser is None:
                    # 서보 연결이 아예 안 된 상태 → SERVO_ERROR
                    print("[SERVO] 포트 미연결: SERVO_ERROR")
                    send_error_event(
                        manager_id=MANAGER_ID,
                        error_code="SERVO_ERROR",
                        rule_id=rule_id,
                        chute_id=chute_id,
                        image_path=save_path,
                    )
                else:
                    try:
                        print(f"[SERVO] move to {servo_deg} deg")
                        ser.write(f"SERVO {servo_deg}\n".encode("utf-8"))

                        # 물건 떨어질 시간
                        time.sleep(0.5)

                        print("[SERVO] go HOME")
                        ser.write(b"HOME\n")

                        # 응답 읽기 (SERVO_OK ...)
                        reply = ser.readline().decode("utf-8", errors="ignore").strip()
                        if reply:
                            print(f"[SERVO] from Arduino: {reply}")
                    except Exception as e:
                        print(f"[SERVO] 명령 전송 중 오류: {e}")
                        # 서보 관련 예외는 SERVO_ERROR로 보고
                        send_error_event(
                            manager_id=MANAGER_ID,
                            error_code="SERVO_ERROR",
                            rule_id=rule_id,
                            chute_id=chute_id,
                            image_path=save_path,
                        )

            # --- 서버로 sorting-result 전송 ---
            if rule_id is not None and chute_id is not None:
                send_sorting_result(
                    manager_id=MANAGER_ID,
                    rule_id=rule_id,
                    chute_id=chute_id,
                    image_path=save_path,
                )
            else:
                # (이 상황은 거의 없겠지만 방어적으로 UNKNOWN_ERROR)
                if not error_code:
                    error_code = "UNKNOWN_ERROR"
                send_error_event(
                    manager_id=MANAGER_ID,
                    error_code=error_code,
                    image_path=save_path,
                )

        else:
            # 분류 실패 케이스 → 에러 코드 필수로 하나 있어야 함
            print(f"[INFO] 예외 항목 이미지를 '{save_path}'에 저장했습니다.")
            if not error_code:
                error_code = "UNKNOWN_ERROR"

            send_error_event(
                manager_id=MANAGER_ID,
                error_code=error_code,
                rule_id=rule_id,
                chute_id=chute_id,
                image_path=save_path,
            )

    except Exception as e:
        print(f"[ERROR] 결과 이미지 저장 실패: {e}")
        # 저장 자체가 안 되면, 이미지 없이 UNKNOWN_ERROR 날려줌
        send_error_event(
            manager_id=MANAGER_ID,
            error_code="UNKNOWN_ERROR",
            rule_id=rule_id,
            chute_id=chute_id,
            image_path=None,
        )


if __name__ == "__main__":
    # 1) 서버에서 분류 규칙 받아오기
    setup_json = fetch_setup(MANAGER_ID)
    TEXT_RULES, COLOR_RULES = build_rule_maps(setup_json)

    # 서버에서 받은 규칙을 기준으로 "정상 텍스트/색상 목록" 자동 세팅
    global KNOWN_TEXTS, KNOWN_COLORS
    KNOWN_TEXTS = list(TEXT_RULES.keys())
    KNOWN_COLORS = list(COLOR_RULES.keys())

    print("[INFO] KNOWN_TEXTS (from server):", KNOWN_TEXTS)
    print("[INFO] KNOWN_COLORS (from server):", KNOWN_COLORS)

    # 2) 아두이노 시리얼, 카메라 열기
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"[SERIAL] Connected to {SERIAL_PORT} @ {BAUD_RATE}")
    except serial.SerialException:
        print(f"[ERROR] Arduino 포트를 열 수 없습니다: {SERIAL_PORT}")
        ser = None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")

        # CAMERA_TIMEOUT 에러 전송
        send_error_event(
            manager_id=MANAGER_ID,
            error_code="CAMERA_TIMEOUT",
            image_path=None,
        )
        if ser is not None:
            ser.close()
        exit()

    # 카메라 설정 (필요하면 조정)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -10)
    cap.set(cv2.CAP_PROP_GAIN, 4)

    print("[INFO] EXPOSURE:", cap.get(cv2.CAP_PROP_EXPOSURE))
    print("[INFO] GAIN    :", cap.get(cv2.CAP_PROP_GAIN))

    # 3) 아두이노 READY 기다리기
    if ser is not None:
        print("[INFO] Arduino 'READY' 신호를 기다리는 중...")
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line in ("READY", "SENSOR_READY"):
                print(f"[SUCCESS] Arduino 신호 수신: {line}")
                ser.reset_input_buffer()
                break
            elif line:
                print(f"[SYNC] Arduino 부팅 신호: {line}")

    print("==================================================")
    print(f"[INFO] 센서 신호를 대기합니다. (Arduino에서 '0' 들어오면 촬영 + 분류 실행)")
    print("==================================================")

    COOLDOWN = 0.5
    last_trigger_time = 0.0

    try:
        while True:
            # ─────────────────────────────
            # ① TEST_MODE (수동 촬영 모드)
            # ─────────────────────────────
            if TEST_MODE:
                # 센서 안 쓰고, 엔터 칠 때마다 한 장씩 촬영 → 바로 분류
                image_path = manual_capture_from_keyboard(cap, WATCH_FOLDER)
                if not image_path:
                    # None이면 q 눌렀거나 오류 → 살짝 쉬고 계속 / 원하면 break로 바꿔도 됨
                    time.sleep(0.1)
                    continue

                # YOLO/OCR 처리
                annotated_image, extracted_data = process_image(
                    image_path,
                    conf_threshold=YOLO_CONF_THRESHOLD,
                )
                if annotated_image is None:
                    print("[WARN] 이미지 처리 실패")
                    continue

                print(f"\n--- Final Data for {os.path.basename(image_path)} ---")
                print(extracted_data)

                # 분류 + 서버 전송 + (서보 제어는 ser 연결돼 있으면 실행)
                classify_and_act(
                    image_path,
                    annotated_image,
                    extracted_data,
                    TEXT_RULES,
                    COLOR_RULES,
                    ser,
                )

                # 수동 모드는 한 바퀴 돌고 다시 while 처음으로
                continue

            # ─────────────────────────────
            # ② 자동 센서 모드
            # ─────────────────────────────
            # 센서 신호 읽기
            line = ser.readline().decode("utf-8", errors="ignore").strip() if ser else ""
            now = time.time()

            if line:
                if line == "0":
                    if now - last_trigger_time > COOLDOWN:
                        last_trigger_time = now
                        print("[SIGNAL] 센서 0 감지 → 촬영 + 분류 시작")

                        # ① 사진 캡처
                        image_path = capture_image_from_camera(cap, WATCH_FOLDER)
                        if not image_path:
                            continue

                        # ② YOLO/OCR 처리
                        annotated_image, extracted_data = process_image(
                            image_path,
                            conf_threshold=YOLO_CONF_THRESHOLD,
                        )
                        if annotated_image is None:
                            print("[WARN] 이미지 처리 실패")
                            continue

                        print(f"\n--- Final Data for {os.path.basename(image_path)} ---")
                        print(extracted_data)

                        # ③ 분류 + 서버 전송 + 서보 제어
                        classify_and_act(
                            image_path,
                            annotated_image,
                            extracted_data,
                            TEXT_RULES,
                            COLOR_RULES,
                            ser,
                        )

                elif line.startswith("SERVO_OK"):
                    print(f"[SERIAL] from Arduino: {line}")
                # 필요하면 '1' 신호 등도 처리

            # 너무 바쁘지 않게 살짝 딜레이
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] 사용자가 종료했습니다.")
    finally:
        print("[INFO] 자원 정리 중...")
        if ser is not None:
            try:
                ser.write(b"HOME\n")
                reply = ser.readline().decode("utf-8", errors="ignore").strip()
                if reply:
                    print(f"[SERVO] from Arduino: {reply}")
            except Exception as e:
                print(f"[SERVO] HOME 명령 실패: {e}")
            ser.close()
        cap.release()
        print("[INFO] 프로그램 종료 완료.")