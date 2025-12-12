from ultralytics import YOLO
import cv2
import os
import time
import serial
import torch

from utils.ocr_module import run_ocr
from utils.hsv_module import get_color

from device_api import (
    MANAGER_ID,
    fetch_setup,
    build_rule_maps,
    send_sorting_result,
    send_error_event,
)

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# True = 엔터 눌러 수동 촬영 모드 (manual_capture_from_keyboard 코드 실행 시 True)
# 센서 없이 YOLO + 서버 + 서보 테스트 --> True
# 센서 포함 테스트 --> False
TEST_MODE = False

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

SERIAL_PORT = "/dev/cu.usbmodem141011"   # 환경에 맞게 수정 (윈도우면 "COM3" 이런 식)
BAUD_RATE = 115200
CAMERA_INDEX = 0

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
DEVICE = pick_device()
print(f"[INFO] Using device: {DEVICE}")

model = YOLO(TRAINED_MODEL_PATH)
model.to(DEVICE)
CLASS_NAMES = model.names
print(f"[INFO] Model loaded. Classes: {CLASS_NAMES}")

KNOWN_TEXTS = []
KNOWN_COLORS = []

YOLO_CONF_THRESHOLD = 0.70 # YOLO가 물체를 찾았다고 확신하는 최소 확률 (70%)
OCR_CONF_THRESHOLD = 0.70 # OCR이 글자를 읽었다고 확신하는 최소 확률 (70%)

def _sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def capture_image_from_camera(cap, save_folder):
    # 카메라 버퍼 지우기
    for _ in range(3):
        cap.grab()

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

# enter 키로 사진 찍기 (디버깅용)
def manual_capture_from_keyboard(cap, save_folder):
    print("==================================================")
    print("[TEST MODE] Enter → 사진 캡처")
    print("[TEST MODE] 'q' + Enter → 종료")
    print("==================================================")

    while True:
        user_input = input()

        if user_input.lower() == "q":
            print("[TEST MODE] 종료합니다.")
            return None

        print("[CAPTURE] 촬영 중, 가장 선명한 프레임을 찾는 중...")

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
        print(f"[SUCCESS] 촬영 완료 --> {save_path}")

        return save_path


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
        "low_confidence_skips": [], # 신뢰도 낮아서 스킵된 항목들
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

        # 텍스트 라벨
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

        # 색상 라벨
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
                print(f"[YOLO] Found 'marker_color' (Conf: {conf * 100:.0f}%). Sending to HSV")

            detected_color = get_color(cropped_img)
            final_data["color"] = detected_color
            print(f"[SUCCESS] Color Result: {final_data['color']}")

        # 일반 물체
        elif class_name == "objects":
            print(f"[YOLO] Found 'objects' (Conf: {conf * 100:.0f}%)")
            final_data["object_type"] = "object"

    return result.plot(), final_data

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
    servo_moved = False

    base_name = os.path.basename(image_path)
    save_name = f"{os.path.splitext(base_name)[0]}_result.jpg"

    # 텍스트 인식됨 + 신뢰도가 기준치 이상
    if detected_text and detected_text_conf >= OCR_CONF_THRESHOLD:
        rule_info = text_rules.get(detected_text)

        # 등록된 규칙 있으면 정상 분류
        if rule_info:
            save_path = os.path.join(OCR_OUTPUT_FOLDER, save_name)
            is_success = True
            print("[ROUTE] 정상 텍스트 결과 폴더로 저장")

            rule_id = rule_info["ruleId"]
            chute_id = rule_info["chuteId"]
            servo_deg = rule_info.get("servoDeg")

        # 븐류 규칙에 없으면 에러
        else:
            save_path = os.path.join(ERROR_TEXT_OUTPUT_FOLDER, save_name)

            if detected_text.strip().upper() == "UNKNOWN":
                error_code = "UNKNOWN_ERROR"
                print("[ROUTE] UNKNOWN_ERROR: UNKNOWN 텍스트 라벨 감지")

            else:
                error_code = "OCR_FAIL"
                print("[ROUTE] OCR_FAIL: 분류 규칙에 없는 텍스트 라벨")

    # 텍스트 없음 + 규칙에 존재하는 색상 인식
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

    # 텍스트는 인식됨 but 텍스트/YOLO 신뢰도 낮음
    elif (
        (detected_text and detected_text_conf < OCR_CONF_THRESHOLD)
        or any("marker_text" in s for s in low_conf_list)
    ):
        save_path = os.path.join(ERROR_TEXT_OUTPUT_FOLDER, save_name)
        error_code = "OCR_FAIL"
        print("[ROUTE] OCR_FAIL: OCR 신뢰도 낮음 / marker_text low confidence")

    # 색상 정보에 없는 색상
    elif (
        (detected_color and detected_color not in KNOWN_COLORS)
        or any("marker_color" in s for s in low_conf_list)
    ):
        save_path = os.path.join(ERROR_COLOR_OUTPUT_FOLDER, save_name)
        error_code = "UNKNOWN_ERROR"
        print("[ROUTE] UNKNOWN_ERROR: 색상 정보 이상(예상 밖 색상 또는 low confidence)")

    # 그 외
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

    try:
        cv2.imwrite(save_path, annotated_image)

        if is_success:
            print(f"[SUCCESS] 결과 이미지를 '{save_path}'에 저장했습니다.")

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

                        # 90도 이하: 5초 후 0도로 복귀
                        if servo_deg <= 90:
                            PRE_KICK_WAIT = 2

                            print(f"[SERVO] SIMPLE MOVE --> main={main_deg}°, then HOME after {PRE_KICK_WAIT}s")

                            print(f"[SERVO] --> {main_deg}° (분류 위치 이동)")
                            ser.write(f"SERVO {main_deg}\n".encode("utf-8"))
                            time.sleep(PRE_KICK_WAIT)

                            print("[SERVO] --> 0° (back to HOME)")
                            ser.write(b"SERVO 0\n")
                            time.sleep(0.3)

                        # 90도 초과: 킥 모션 유지
                        else:
                            PRE_KICK_WAIT = 6 # 분류 위치까지 가는 시간
                            KICK_DELTA = 40 # 킥 반동 각도
                            KICK_WAIT = 0.25 # 킥 속도

                            back_deg = max(0, min(180, main_deg - KICK_DELTA))

                            print(
                                f"[SERVO] KICK START --> main={main_deg}°, "
                                f"back={back_deg}°, PRE_WAIT={PRE_KICK_WAIT}s"
                            )

                            print(f"[SERVO] --> {main_deg}° (분류 위치 이동)")
                            ser.write(f"SERVO {main_deg}\n".encode("utf-8"))
                            time.sleep(PRE_KICK_WAIT)

                            print(f"[SERVO] --> {back_deg}° (kick back)")
                            ser.write(f"SERVO {back_deg}\n".encode("utf-8"))
                            time.sleep(KICK_WAIT)

                            print(f"[SERVO] --> {main_deg}° (kick forward)")
                            ser.write(f"SERVO {main_deg}\n".encode("utf-8"))
                            time.sleep(KICK_WAIT)

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

    return is_success, servo_moved, save_path


if __name__ == "__main__":
    setup_json = fetch_setup(MANAGER_ID)
    TEXT_RULES, COLOR_RULES = build_rule_maps(setup_json)

    KNOWN_TEXTS = list(TEXT_RULES.keys())
    KNOWN_COLORS = list(COLOR_RULES.keys())

    print("[INFO] KNOWN_TEXTS (from server):", KNOWN_TEXTS)
    print("[INFO] KNOWN_COLORS (from server):", KNOWN_COLORS)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"[SERIAL] Connected to {SERIAL_PORT} @ {BAUD_RATE}")
    except serial.SerialException:
        print(f"[ERROR] Arduino 포트를 열 수 없습니다: {SERIAL_PORT}")
        ser = None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    print("[INFO] BUFFERSIZE  :", cap.get(cv2.CAP_PROP_BUFFERSIZE))
    print("[INFO] FRAMEWIDTH  :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("[INFO] FRAMEHEIGHT :", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] FPS         :", cap.get(cv2.CAP_PROP_FPS))


    if not cap.isOpened():
        print(f"[ERROR] {CAMERA_INDEX}번 카메라를 열 수 없습니다.")

        send_error_event(
            manager_id=MANAGER_ID,
            error_code="CAMERA_TIMEOUT",
            image_path=None,
        )
        if ser is not None:
            ser.close()
        exit()

    REAR_TIMEOUT_SEC = 1.0
    COOLDOWN = 0.5
    last_trigger_time = 0.0

    # 검수용
    current_item = None
    waiting_for_rear = False
    rear_deadline = 0.0

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
            # 키보드 엔터로 수동 촬영 (디버깅용)
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

            # 자동 센서 촬영
            line = ser.readline().decode("utf-8", errors="ignore").strip() if ser else ""
            now = time.time()

            if line:
                print(f"[SERIAL-DEBUG] got line: {repr(line)}")

            # 앞 포토센서 감지 ('0' 수신) -> 촬영, 분류 시작
            if line == "0":
                start_time = time.time()

                if now - last_trigger_time > COOLDOWN:
                    last_trigger_time = now
                    print("[SIGNAL] 앞 센서 감지 --> 촬영 + 분류 시작")

                    if ser is not None:
                        print("[SERVO] sensor triggered → HOME (초기화)")
                        ser.write(b"HOME\n")
                        time.sleep(0.05)

                    # 이미지 캡쳐
                    image_path = capture_image_from_camera(cap, WATCH_FOLDER)
                    if not image_path:
                        continue

                    # YOLO/OCR/HSV
                    annotated_image, extracted_data = process_image(
                        image_path,
                        conf_threshold=YOLO_CONF_THRESHOLD,
                    )
                    if annotated_image is None:
                        print("[WARN] 이미지 처리 실패")
                        continue

                    print(f"\n--- Final Data for {os.path.basename(image_path)} ---")
                    print(extracted_data)

                    # 분류 + 서버에 전송
                    is_success, servo_moved, saved_path = classify_and_act(
                        image_path,
                        annotated_image,
                        extracted_data,
                        TEXT_RULES,
                        COLOR_RULES,
                        ser,
                    )

                    # 성능평가
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    print(f"[METRIC] Latency: {latency_ms:.1f} ms")

                    current_item = {
                        "sorted_ok": bool(is_success and servo_moved),
                        "image_path": saved_path,
                    }
                    waiting_for_rear = True

            # 검수용 센서
            elif line == "CHECK_0":
                print("[SENSOR] 뒤 검수 센서 감지")

                # 앞 센서 보지 않고 뒤만 본 경우 -> PHOTO_SENSOR_ERROR
                if not waiting_for_rear:
                    print("[ERROR] PHOTO_SENSOR_ERROR: 앞 센서 미감지 + 뒤 센서 감지")
                    send_error_event(
                        manager_id=MANAGER_ID,
                        error_code="PHOTO_SENSOR_ERROR",
                        rule_id=None,
                        chute_id=None,
                        image_path=None,
                    )
                    continue

                # 앞 센서는 봤지만 current_item 데이터가 없는 경우
                if current_item is None:
                    print("[ERROR] PHOTO_SENSOR_ERROR: 상태 없음 + 뒤 센서 감지")
                    send_error_event(
                        manager_id=MANAGER_ID,
                        error_code="PHOTO_SENSOR_ERROR",
                        rule_id=None,
                        chute_id=None,
                        image_path=None,
                    )
                    waiting_for_rear = False
                    continue

                # 정상 분류
                if current_item["sorted_ok"]:
                    print("[INFO] 정상 분류된 물건이 뒤 센서를 통과")
                    send_error_event(
                        manager_id=MANAGER_ID,
                        error_code="SERVO_ERROR",
                        rule_id=None,
                        chute_id=None,
                        image_path=current_item["image_path"],
                    )
                    waiting_for_rear = False
                    current_item = None
                    continue

                # 분류 실패 + 뒤 센서 감지
                print("[INFO] 앞에서 에러 처리된 물건이 뒤 센서를 통과")

                waiting_for_rear = False
                current_item = None
                continue

            elif line.startswith("SERVO_OK"):
                print(f"[SERIAL] from Arduino: {line}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] 사용자가 종료했습니다.")
    finally:
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