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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))        # main 폴더
TRAINING_ROOT = os.path.join(PROJECT_ROOT, "..", "training")     # training 폴더
SERIAL_PORT = "/dev/cu.usbmodem141011"  # 네 환경에 맞게
BAUD_RATE = 115200

# camera_capture.py 가 읽는 서보 명령 파일
SERVO_CMD_FILE = os.path.join(PROJECT_ROOT, "servo_cmd.txt")

# 1. main.py가 만든 학습된 모델 경로 로드
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


if __name__ == "__main__":
    setup_json = fetch_setup(MANAGER_ID)
    TEXT_RULES, COLOR_RULES = build_rule_maps(setup_json)

    WATCH_FOLDER = os.path.join(PROJECT_ROOT, "input_images")
    ERROR_ROOT = os.path.join(PROJECT_ROOT, "results_error")

    OCR_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "results_text")
    COLOR_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "results_color")

    ERROR_TEXT_OUTPUT_FOLDER = os.path.join(ERROR_ROOT, "text")
    ERROR_COLOR_OUTPUT_FOLDER = os.path.join(ERROR_ROOT, "color")

    os.makedirs(OCR_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(COLOR_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ERROR_ROOT, exist_ok=True)
    os.makedirs(ERROR_TEXT_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ERROR_COLOR_OUTPUT_FOLDER, exist_ok=True)

    KNOWN_TEXTS = ["대형", "중형", "소형"]
    KNOWN_COLORS = ["RED", "GREEN", "BLUE", "YELLOW"]

    YOLO_CONF_THRESHOLD = 0.70
    OCR_CONF_THRESHOLD = 0.70

    processed_files = set()

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[SERVO] Connected to {SERIAL_PORT} @ {BAUD_RATE}")
    except serial.SerialException:
        print(f"[ERROR] 서보용 Arduino 포트를 열 수 없습니다: {SERIAL_PORT}")
        ser = None

    print("==================================================")
    print(f"[INFO] '{WATCH_FOLDER}' 폴더를 감시합니다...")
    print(f"[INFO] 텍스트({KNOWN_TEXTS}) 결과는 '.../results_text' 폴더에 저장됩니다.")
    print(f"[INFO] 색상({KNOWN_COLORS}) 결과는 '.../results_color' 폴더에 저장됩니다.")
    print(f"[INFO] 그 외 모든 예외 항목은 '.../results_error' 폴더에 저장됩니다.")
    print(f"[INFO] (종료하려면 터미널에서 Ctrl + C 를 누르세요)")
    print("==================================================")

    try:
        while True:
            current_files = set()
            image_types = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
            for img_type in image_types:
                current_files.update(glob.glob(os.path.join(WATCH_FOLDER, img_type)))

            new_files = current_files - processed_files

            if new_files:
                print(f"\n[INFO] {len(new_files)}개의 새 이미지를 감지했습니다. 처리를 시작합니다...")
                for image_path in new_files:
                    annotated_image, extracted_data = process_image(
                        image_path,
                        conf_threshold=YOLO_CONF_THRESHOLD,
                    )

                    if annotated_image is not None:
                        print(f"\n--- Final Data for {os.path.basename(image_path)} ---")
                        print(extracted_data)

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

                        # ① 정상 텍스트 인식
                        if (detected_text in KNOWN_TEXTS) and (
                            detected_text_conf >= OCR_CONF_THRESHOLD
                        ):
                            save_path = os.path.join(OCR_OUTPUT_FOLDER, save_name)
                            is_success = True
                            print("[ROUTE] 정상 텍스트 결과 폴더로 저장")

                            rule_info = TEXT_RULES.get(detected_text)
                            if rule_info:
                                rule_id = rule_info["ruleId"]
                                chute_id = rule_info["chuteId"]
                                servo_deg = rule_info.get("servoDeg")
                            else:
                                is_success = False
                                error_code = "RULE_NOT_FOUND"
                                print("[WARN] TEXT_RULES에 해당 텍스트 규칙이 없습니다. RULE_NOT_FOUND")

                        # ② 정상 색상 인식
                        elif (not detected_text) and (detected_color in KNOWN_COLORS):
                            save_path = os.path.join(COLOR_OUTPUT_FOLDER, save_name)
                            is_success = True
                            print("[ROUTE] 정상 색상 결과 폴더로 저장")

                            rule_info = COLOR_RULES.get(detected_color)
                            if rule_info:
                                rule_id = rule_info["ruleId"]
                                chute_id = rule_info["chuteId"]
                                servo_deg = rule_info.get("servoDeg")
                            else:
                                is_success = False
                                error_code = "RULE_NOT_FOUND"
                                print("[WARN] COLOR_RULES에 해당 색상 규칙이 없습니다. RULE_NOT_FOUND")

                        # ③ 텍스트 오류
                        elif (
                            (detected_text and detected_text_conf < OCR_CONF_THRESHOLD)
                            or any("marker_text" in s for s in low_conf_list)
                        ):
                            save_path = os.path.join(ERROR_TEXT_OUTPUT_FOLDER, save_name)
                            error_code = "OCR_LOW_CONFIDENCE"
                            print("[ROUTE] 텍스트 오류 폴더(results_error/text)로 저장")

                        # ④ 색상 오류
                        elif (
                            (detected_color and detected_color not in KNOWN_COLORS)
                            or any("marker_color" in s for s in low_conf_list)
                        ):
                            save_path = os.path.join(ERROR_COLOR_OUTPUT_FOLDER, save_name)
                            error_code = "COLOR_UNEXPECTED"
                            print("[ROUTE] 색상 오류 폴더(results_error/color)로 저장")

                        # ⑤ 그 외
                        else:
                            save_path = os.path.join(ERROR_ROOT, save_name)
                            if had_text and not detected_text:
                                error_code = "OCR_NO_TEXT"
                            elif had_color and not detected_color:
                                error_code = "HSV_NO_COLOR"
                            else:
                                error_code = "YOLO_NO_DETECTION"
                            print("[ROUTE] 기타 에러 폴더(results_error)로 저장")

                        # ── ⑥ 파일 저장 & 서버 통신 ──
                        try:
                            cv2.imwrite(save_path, annotated_image)

                            if is_success:
                                print(f"[SUCCESS] 결과 이미지를 '{save_path}'에 저장했습니다.")

                                if rule_info and "servoDeg" in rule_info and ser is not None:
                                    angle = rule_info["servoDeg"]
                                    print(f"[SERVO] Move to angle: {angle}")
                                    ser.write(f"SERVO {angle}\n".encode("utf-8"))

                                    time.sleep(0.5)  # 물체가 떨어질 시간 (필요에 따라 조절)

                                    print("[SERVO] Return HOME")
                                    ser.write(b"HOME\n")

                                if ser is not None:
                                    # 어떤 rule_info 를 썼는지 기억하려면 위에서 따로 저장해둬도 되고,
                                    # 가장 마지막에 썼던 rule_info 를 하나 변수에 저장해둬도 돼.
                                    # 여기서는 간단히 servo_deg 를 위에서 같이 빼왔다고 가정할게.
                                    if "servoDeg" in rule_info:
                                        angle = rule_info["servoDeg"]
                                        print(f"[SERVO] move to {angle} deg")
                                        ser.write(f"SERVO {angle}\n".encode("utf-8"))
                                        # 응답 한번 읽어보기 (선택)
                                        try:
                                            reply = ser.readline().decode("utf-8", errors="ignore").strip()
                                            if reply:
                                                print(f"[SERVO] from Arduino: {reply}")
                                        except Exception:
                                            pass

                                # 분류 성공 → sorting-result 이벤트 전송
                                if rule_id is not None and chute_id is not None:
                                    send_sorting_result(
                                        manager_id=MANAGER_ID,
                                        rule_id=rule_id,
                                        chute_id=chute_id,
                                        image_path=save_path,
                                    )
                                else:
                                    # ruleId / chuteId가 없다면 설정 오류로 에러 보고
                                    send_error_event(
                                        manager_id=MANAGER_ID,
                                        error_code=error_code or "RULE_NOT_FOUND",
                                        image_path=save_path,
                                    )

                        except Exception as e:
                            print(f"[ERROR] 결과 이미지 저장 실패: {e}")

                    else:
                        print(f"[WARN] '{image_path}' 처리에 실패했습니다.")

                    processed_files.add(image_path)

            time.sleep(1)


    except KeyboardInterrupt:
        print("\n[INFO] 감시를 종료합니다.")

    finally:
        print("[INFO] 프로그램이 종료되었습니다.")
        if 'ser' in locals() and ser is not None:
            try:
                print("[SERVO] go HOME")
                ser.write(b"HOME\n")
                reply = ser.readline().decode("utf-8", errors="ignore").strip()

                if reply:
                    print(f"[SERVO] from Arduino: {reply}")

            except Exception as e:

                print(f"[SERVO] HOME 명령 실패: {e}")

            ser.close()