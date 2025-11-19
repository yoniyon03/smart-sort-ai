from ultralytics import YOLO
import cv2
import os
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
TEST_MODE = False

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
SERIAL_PORT = "/dev/cu.usbmodem141011"   # 환경에 맞게 수정 (윈도우면 "COM3" 이런 식)
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

# --- 센서로 사진 찍기 ---
def capture_image_from_camera(cap, save_folder):
    """센서가 0 들어왔을 때 카메라에서 여러 장 찍고 가장 선명한 한 장 저장"""
    best_frame = None
    best_score = -1

    N = 5
    for _ in range(N):
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
        # 카메라 타임아웃 에러 전송
        send_error_event(
            manager_id=MANAGER_ID,
            error_code="CAMERA_TIMEOUT",
            image_path=None,
        )
        return None

    filename = f"capture_{int(time.time())}.jpg"
    save_path = os.path.join(save_folder, filename)
    cv2.imwrite(save_path, best_frame)
    print(f"[CAPTURE] 이미지 저장: {save_path}")
    return save_path

# --- enter 키로 사진 찍기 ---
def manual_capture_from_keyboard(cap, save_folder):
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


# YOLO + OCR + HSV 실행해서 텍스트/색상 정보 추출
def process_image(image_path, conf_threshold=0.7, ocr_good_threshold=0.7):
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


# YOLO/OCR 결과 보고 폴더 분류 + 서버 전송 + 서보 제어까지 한 번에
def classify_and_act(image_path, annotated_image, extracted_data, text_rules, color_rules, ser):
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
    servo_error = False
    servo_moved = False  # ← 항상 초기화

    base_name = os.path.basename(image_path)
    save_name = f"{os.path.splitext(base_name)[0]}_result.jpg"

    # ① 텍스트 기준 정상/에러 분기
    if detected_text and detected_text_conf >= OCR_CONF_THRESHOLD:
        rule_info = text_rules.get(detected_text)

        if rule_info:
            save_path = os.path.join(OCR_OUTPUT_FOLDER, save_name)
            is_success = True
            print("[ROUTE] 정상 텍스트 결과 폴더로 저장")

            rule_id = rule_info["ruleId"]
            chute_id = rule_info["chuteId"]
            servo_deg = rule_info.get("servoDeg")

        else:
            save_path = os.path.join(ERROR_TEXT_OUTPUT_FOLDER, save_name)

            if detected_text.strip().upper() == "UNKNOWN":
                error_code = "UNKNOWN_ERROR"
                print("[ROUTE] UNKNOWN_ERROR: UNKNOWN 텍스트 라벨 감지")
            else:
                error_code = "OCR_FAIL"
                print("[ROUTE] OCR_FAIL: 분류 규칙에 없는 텍스트 라벨")

    # ② 색상 기준 정상/에러 분기
    elif (not detected_text) and (detected_color in KNOWN_COLORS):
        rule_info = color_rules.get(detected_color)

        if rule_info:
            save_path = os.path.join(COLOR_OUTPUT_FOLDER, save_name)
            is_success = True
            print("[ROUTE] 정상 색상 결과 폴더로 저장")

            rule_id = rule_info["ruleId"]
            chute_id = rule_info["chuteId"]
            servo_deg = rule_info.get("servoDeg")
        else:
            save_path = os.path.join(ERROR_COLOR_OUTPUT_FOLDER, save_name)
            error_code = "UNKNOWN_ERROR"
            print("[ROUTE] UNKNOWN_ERROR: 분류 규칙에 없는 색상 라벨")

    # ③ 텍스트는 읽혔지만 신뢰도 낮음
    elif (
        (detected_text and detected_text_conf < OCR_CONF_THRESHOLD)
        or any("marker_text" in s for s in low_conf_list)
    ):
        save_path = os.path.join(ERROR_TEXT_OUTPUT_FOLDER, save_name)
        error_code = "OCR_FAIL"
        print("[ROUTE] OCR_FAIL: OCR 신뢰도 낮음 / marker_text low confidence")

    # ④ 색상 쪽 이상
    elif (
        (detected_color and detected_color not in KNOWN_COLORS)
        or any("marker_color" in s for s in low_conf_list)
    ):
        save_path = os.path.join(ERROR_COLOR_OUTPUT_FOLDER, save_name)
        error_code = "UNKNOWN_ERROR"
        print("[ROUTE] UNKNOWN_ERROR: 색상 정보 이상(예상 밖 색상 또는 low confidence)")

    # ⑤ 기타 fallback
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
            save_path = os.path.join(ERROR_ROOT, save_name)
            error_code = "NO_MARKER"
            print("[ROUTE] NO_MARKER: YOLO가 표식을 탐지하지 못함 (너무 빠르게 지나간 경우 등)")

    # ⑥ 파일 저장 + 서버 통신 + 서보 제어
    try:
        cv2.imwrite(save_path, annotated_image)

        if is_success:
            print(f"[SUCCESS] 결과 이미지를 '{save_path}'에 저장했습니다.")

            # --- 서보 제어 ---
            if servo_deg is not None:
                if ser is None:
                    print("[SERVO] 포트 미연결: SERVO_ERROR")
                    servo_error = True
                    send_error_event(
                        manager_id=MANAGER_ID,
                        error_code="SERVO_ERROR",
                        rule_id=None,
                        chute_id=None,
                        image_path=save_path,
                    )
                else:
                    try:
                        main_deg = servo_deg

                        # ==========================
                        # ① 90도 이하: 5초 후 0도로 복귀
                        # ==========================
                        if servo_deg <= 90:
                            PRE_KICK_WAIT = 5

                            print(f"[SERVO] SIMPLE MOVE → main={main_deg}°, then HOME after {PRE_KICK_WAIT}s")

                            # 분류 각도로 이동
                            print(f"[SERVO] → {main_deg}° (분류 위치 이동)")
                            ser.write(f"SERVO {main_deg}\n".encode("utf-8"))
                            time.sleep(PRE_KICK_WAIT)

                            # 0도로 복귀
                            print("[SERVO] → 0° (back to HOME)")
                            ser.write(b"SERVO 0\n")
                            time.sleep(0.3)

                        # ==========================
                        # ② 90도 초과: 킥 모션 유지
                        # ==========================
                        else:
                            PRE_KICK_WAIT = 10  # 항상 8초
                            KICK_DELTA = 30  # 항상 30도
                            KICK_WAIT = 0.25

                            back_deg = max(0, min(180, main_deg - KICK_DELTA))

                            print(
                                f"[SERVO] KICK START → main={main_deg}°, "
                                f"back={back_deg}°, PRE_WAIT={PRE_KICK_WAIT}s"
                            )

                            # 1) 분류 위치 이동
                            print(f"[SERVO] → {main_deg}° (분류 위치 이동)")
                            ser.write(f"SERVO {main_deg}\n".encode("utf-8"))
                            time.sleep(PRE_KICK_WAIT)

                            # 2) 살짝 뒤로 당기기 (킥 준비)
                            print(f"[SERVO] → {back_deg}° (kick back)")
                            ser.write(f"SERVO {back_deg}\n".encode("utf-8"))
                            time.sleep(KICK_WAIT)

                            # 3) 다시 앞으로 밀기 (실제 킥)
                            print(f"[SERVO] → {main_deg}° (kick forward)")
                            ser.write(f"SERVO {main_deg}\n".encode("utf-8"))
                            time.sleep(KICK_WAIT)

                        # 공통: 아두이노 응답 한 번 읽기
                        reply = ser.readline().decode("utf-8", errors="ignore").strip()
                        if reply:
                            print(f"[SERVO] from Arduino: {reply}")

                        servo_moved = True

                    except Exception as e:
                        print(f"[SERVO] 명령 전송 중 오류: {e}")
                        servo_error = True
                        send_error_event(
                            manager_id=MANAGER_ID,
                            error_code="SERVO_ERROR",
                            rule_id=rule_id,
                            chute_id=chute_id,
                            image_path=save_path,
                        )

            # --- 정상 분류 이력 전송 (서보 에러 없을 때만) ---
            if rule_id is not None and chute_id is not None and not servo_error:
                send_sorting_result(
                    manager_id=MANAGER_ID,
                    rule_id=rule_id,
                    chute_id=chute_id,
                    image_path=save_path,
                )
            else:
                if servo_error and not error_code:
                    error_code = "SERVO_ERROR"
                if error_code:
                    send_error_event(
                        manager_id=MANAGER_ID,
                        error_code=error_code,
                        rule_id=None,
                        chute_id=None,
                        image_path=save_path,
                    )

        else:
            print(f"[INFO] 예외 항목 이미지를 '{save_path}'에 저장했습니다.")
            if not error_code:
                error_code = "UNKNOWN_ERROR"

            send_error_event(
                manager_id=MANAGER_ID,
                error_code=error_code,
                rule_id=None,
                chute_id=None,
                image_path=save_path,
            )

    except Exception as e:
        print(f"[ERROR] 결과 이미지 저장 실패: {e}")
        send_error_event(
            manager_id=MANAGER_ID,
            error_code="UNKNOWN_ERROR",
            rule_id=None,
            chute_id=None,
            image_path=None,
        )

    # 🔚 항상 3개 값 리턴
    return is_success, servo_moved, save_path


if __name__ == "__main__":
    # 1) 서버에서 분류 규칙 받아오기
    setup_json = fetch_setup(MANAGER_ID)
    TEXT_RULES, COLOR_RULES = build_rule_maps(setup_json)

    # 서버에서 받은 규칙을 기준으로 "정상 텍스트/색상 목록" 자동 세팅
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

    REAR_TIMEOUT_SEC = 1.0  # 뒤 센서 기다릴 최대 시간 (1초 예시)

    COOLDOWN = 0.5
    last_trigger_time = 0.0

    # ★ 현재 물건 상태 (앞 센서가 본 직후부터 뒤 센서 검수까지)
    current_item = None  # {"sorted_ok": bool, "image_path": str}
    waiting_for_rear = False
    rear_deadline = 0.0

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

    try:
        while True:
            # ① TEST_MODE (수동 촬영)
            if TEST_MODE:
                image_path = manual_capture_from_keyboard(cap, WATCH_FOLDER)
                if not image_path:
                    time.sleep(0.1)
                    continue

                annotated_image, extracted_data = process_image(
                    image_path,
                    conf_threshold=YOLO_CONF_THRESHOLD,
                )
                if annotated_image is None:
                    print("[WARN] 이미지 처리 실패")
                    continue

                print(f"\n--- Final Data for {os.path.basename(image_path)} ---")
                print(extracted_data)

                classify_and_act(
                    image_path,
                    annotated_image,
                    extracted_data,
                    TEXT_RULES,
                    COLOR_RULES,
                    ser,
                )
                continue

            # ② 자동 센서 모드
            line = ser.readline().decode("utf-8", errors="ignore").strip() if ser else ""
            now = time.time()

            # # ★ 뒤 센서 타임아웃 처리 (원하면 사용)
            # if waiting_for_rear and now > rear_deadline:
            #     print("[CHECK] 뒤 센서 타임아웃 → 검수 종료")
            #     waiting_for_rear = False
            #     current_item = None

            # 디버그용: 들어오는 모든 문자열 찍어보기
            if line:
                print(f"[SERIAL-DEBUG] got line: {repr(line)}")

            # ───────────── 앞 포토센서 (감지용) ─────────────
            if line == "0":  # 앞 센서에서 오는 신호
                if now - last_trigger_time > COOLDOWN:
                    last_trigger_time = now
                    print("[SIGNAL] 앞 센서 감지 → 촬영 + 분류 시작")

                    # 센서 들어올 때마다 HOME으로 초기화
                    if ser is not None:
                        print("[SERVO] sensor triggered → HOME (초기화)")
                        ser.write(b"HOME\n")
                        time.sleep(0.05)

                    # ① 사진 캡처
                    image_path = capture_image_from_camera(cap, WATCH_FOLDER)
                    if not image_path:
                        # 여기서 카메라 타임아웃 에러는 이미 전송됨
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
                    is_success, servo_moved, saved_path = classify_and_act(
                        image_path,
                        annotated_image,
                        extracted_data,
                        TEXT_RULES,
                        COLOR_RULES,
                        ser,
                    )

                    # ★ 이 물건에 대한 상태를 저장 → 뒤 센서가 나중에 검사
                    current_item = {
                        "sorted_ok": bool(is_success and servo_moved),  # 분류 성공 + 서보까지 보냄
                        "image_path": saved_path,
                    }
                    waiting_for_rear = True
                    # rear_deadline = time.time() + REAR_TIMEOUT_SEC

            # ───────────── 뒤 포토센서 (검수용) ─────────────
            # elif line == "CHECK_0":  # 아두이노에서 검수 센서용으로 보내는 문자열
            #     print("[SENSOR] 뒤 검수 센서 감지")
            #
            #     if waiting_for_rear and current_item is not None:
            #         # 1) 앞 센서 + 분류 + 서보까지 정상 동작했는데도
            #         #    뒤 검사 센서에 걸렸다 → 서보가 물건을 못 날린 것
            #         if current_item["sorted_ok"]:
            #             print("[ERROR] SERVO_ERROR: 분류/서보는 정상인데 뒤 센서에서 물체 감지")
            #             send_error_event(
            #                 manager_id=MANAGER_ID,
            #                 error_code="SERVO_ERROR",
            #                 rule_id=None,
            #                 chute_id=None,
            #                 image_path=current_item["image_path"],
            #             )
            #         # 2) 앞 센서는 감지했지만 분류/서보가 정상적으로 안 됐는데
            #         #    뒤 센서에서 물체가 나왔다 → 카메라/인식 계열 문제로 간주
            #         else:
            #             print("[ERROR] CAMERA_ERROR: 앞 센서 감지 + 사진/인식 쪽 문제 추정")
            #             send_error_event(
            #                 manager_id=MANAGER_ID,
            #                 error_code="CAMERA_ERROR",
            #                 rule_id=None,
            #                 chute_id=None,
            #                 image_path=current_item["image_path"],
            #             )
            #
            #         # 이 물건에 대한 검수 끝
            #         waiting_for_rear = False
            #         current_item = None
            #
            #     else:
            #         # 3) 앞 센서 기록 없이 뒤 센서만 감지됨
            #         #    → 앞 포토센서가 물건을 못 본 상황 (포토센서 에러)
            #         print("[ERROR] PHOTO_SENSOR_ERROR: 앞 센서 미감지 + 뒤 센서 감지")
            #         send_error_event(
            #             manager_id=MANAGER_ID,
            #             error_code="PHOTO_SENSOR_ERROR",
            #             rule_id=None,
            #             chute_id=None,
            #             image_path=None,
            #         )
            elif line == "CHECK_0":
                print("[SENSOR] 뒤 검수 센서 감지")

                if waiting_for_rear and current_item is not None:
                    # 앞 센서도 봤고, 뒤 센서도 물체를 본 상황 → SERVO_ERROR
                    print("[ERROR] SERVO_ERROR: 앞/뒤 센서 모두 감지됨 → 물건이 끝까지 갔음")
                    send_error_event(
                        manager_id=MANAGER_ID,
                        error_code="SERVO_ERROR",
                        rule_id=None,
                        chute_id=None,
                        image_path=current_item["image_path"],
                    )

                    waiting_for_rear = False
                    current_item = None

                else:
                    # 앞 센서 기록 없이 뒤 센서만 감지됨 → 포토센서 에러
                    print("[ERROR] PHOTO_SENSOR_ERROR: 앞 센서 미감지 + 뒤 센서 감지")
                    send_error_event(
                        manager_id=MANAGER_ID,
                        error_code="PHOTO_SENSOR_ERROR",
                        rule_id=None,
                        chute_id=None,
                        image_path=None,
                    )

            elif line.startswith("SERVO_OK"):
                print(f"[SERIAL] from Arduino: {line}")

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