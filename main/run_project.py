from ultralytics import YOLO
import cv2
import os
import glob
import time

from utils.ocr_module import run_ocr
from utils.hsv_module import get_color

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))        # main 폴더
TRAINING_ROOT = os.path.join(PROJECT_ROOT, "..", "training")     # training 폴더

# 1. main.py가 만든 학습된 모델 경로 로드
TRAINED_MODEL_PATH = os.path.join(
    TRAINING_ROOT,
    "runs",
    "detect",
    "train14",
    "weights",
    "best.pt"
)

# 2. 모델 로드
if not os.path.exists(TRAINED_MODEL_PATH):
    print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {TRAINED_MODEL_PATH}")
    print("[INFO] 'main.py'를 실행하여 모델을 먼저 학습시키세요.")
    exit()

print("[INFO] Loading trained YOLO model...")
model = YOLO(TRAINED_MODEL_PATH)
CLASS_NAMES = model.names
print(f"[INFO] Model loaded. Classes: {CLASS_NAMES}")

# 사진 1장 들어왔을 때 처리하는 과정
def process_image(image_path, conf_threshold=0.7):

    if not os.path.exists(image_path):
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {image_path}")
        return None, None

    # 고해상도 원본 이미지를 로드
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"[ERROR] cv2가 이미지를 읽지 못했습니다: {image_path}")
        return None, None

    orig_h, orig_w, _ = original_image.shape
    print(f"\n--- Processing Image: {image_path} (Size: {orig_w}x{orig_h}) ---")

    # YOLO 모델 실행
    results = model(original_image)
    result = results[0]

    final_data = {
        "text": None,
        "text_conf": 0.0, # OCR 신뢰도
        "color": None,
        "object_type": None,
        "low_confidence_skips": [],
        "had_text": False, # results_text를 본 적 있는지
        "had_color": False, # results_color를 본 적 있는지
    }

    for box in result.boxes:
        conf = box.conf[0].item()
        cls_index = int(box.cls[0].item())
        class_name = CLASS_NAMES[cls_index]

        nx1, ny1, nx2, ny2 = box.xyxyn[0].tolist()
        x1 = int(nx1 * orig_w)
        y1 = int(ny1 * orig_h)
        x2 = int(nx2 * orig_w)
        y2 = int(ny2 * orig_h)

        cropped_img = original_image[y1:y2, x1:x2]

        # text 처리 로직
        if class_name == 'marker_text':
            final_data["had_text"] = True

            if conf < conf_threshold:
                skip_info = f"{class_name} ({conf * 100:.0f}%)"
                print(f"[SKIP] Found 'marker_text' but confidence is too low ({conf * 100:.0f}%)")
                final_data["low_confidence_skips"].append(skip_info)
                continue


            print(f"[YOLO] Found 'marker_text' (Conf: {conf * 100:.0f}%). Sending to OCR...")

            temp_crop_path = os.path.join(PROJECT_ROOT, "temp_ocr_image.jpg")
            cv2.imwrite(temp_crop_path, cropped_img)
            print(f"[DEBUG] OCR용 *컬러 원본* 크롭 이미지 저장: {temp_crop_path}")

            # 파일 경로 전달
            extracted_text, ocr_confidence = run_ocr(temp_crop_path, multi_angle=False)

            if extracted_text:
                final_data["text"] = extracted_text
                final_data["text_conf"] = ocr_confidence
                print(f"[SUCCESS] OCR Result: {final_data['text']} (Conf: {ocr_confidence:.2f})")
            else:
                print("[INFO] OCR module ran, but found no text.")

            try:
                os.remove(temp_crop_path)
            except Exception as e:
                print(f"[WARN] Failed to remove temp crop file: {e}")

        # color 처리 로직
        elif class_name == 'marker_color':
            final_data["had_color"] = True

            if conf < conf_threshold:
                skip_info = f"{class_name} ({conf * 100:.0f}%)"
                print(f"[SKIP] Found 'marker_color' but confidence is too low ({conf * 100:.0f}%)")
                final_data["low_confidence_skips"].append(skip_info)
                continue

            print(f"[YOLO] Found 'marker_color' (Conf: {conf * 100:.0f}%). Sending to HSV...")
            detected_color = get_color(cropped_img)
            final_data["color"] = detected_color
            print(f"[SUCCESS] Color Result: {final_data['color']}")

        elif class_name == 'objects':
            print(f"[YOLO] Found 'objects' (Conf: {conf * 100:.0f}%)")
            final_data["object_type"] = "object"

    # 결과 이미지 반환
    return result.plot(), final_data

# 메인 실행
if __name__ == "__main__":

    WATCH_FOLDER = os.path.join(PROJECT_ROOT, "input_images")
    ERROR_ROOT = os.path.join(PROJECT_ROOT, "results_error")

    # 정상 폴더
    OCR_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "results_text")
    COLOR_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "results_color")

    # 에러 상세 폴더
    ERROR_TEXT_OUTPUT_FOLDER = os.path.join(ERROR_ROOT, "text")
    ERROR_COLOR_OUTPUT_FOLDER = os.path.join(ERROR_ROOT, "color")

    # 폴더 생성
    os.makedirs(OCR_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(COLOR_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ERROR_ROOT, exist_ok=True)
    os.makedirs(ERROR_TEXT_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ERROR_COLOR_OUTPUT_FOLDER, exist_ok=True)

    KNOWN_TEXTS = ['대형', '중형', '소형']
    KNOWN_COLORS = ['RED', 'GREEN', 'BLUE', 'YELLOW']

    YOLO_CONF_THRESHOLD = 0.70  # "탐지" 컷오프 (YOLO)
    OCR_CONF_THRESHOLD = 0.70  # "인식" 컷오프 (PaddleOCR)

    processed_files = set()

    print(f"==================================================")
    print(f"[INFO] '{WATCH_FOLDER}' 폴더를 감시합니다...")
    print(f"[INFO] 텍스트({KNOWN_TEXTS}) 결과는 '..._ocr' 폴더에 저장됩니다.")
    print(f"[INFO] 색상({KNOWN_COLORS}) 결과는 '..._color' 폴더에 저장됩니다.")
    print(f"[INFO] 그 외 모든 예외 항목은 '..._error' 폴더에 저장됩니다.")
    print(f"[INFO] (종료하려면 터미널에서 Ctrl + C 를 누르세요)")
    print(f"==================================================")

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

                    # 컷오프 70% (0.7)를 process_image 함수로 전달
                    annotated_image, extracted_data = process_image(image_path, conf_threshold=YOLO_CONF_THRESHOLD)

                    if annotated_image is not None:
                        print(f"\n--- Final Data for {os.path.basename(image_path)} ---")
                        print(extracted_data)

                        # 예외 분류 저장 로직
                        base_name = os.path.basename(image_path)
                        save_name = f"{os.path.splitext(base_name)[0]}_result.jpg"

                        save_path = None
                        is_success = False

                        detected_text = extracted_data.get("text")
                        detected_text_conf = extracted_data.get("text_conf", 0.0)
                        detected_color = extracted_data.get("color")
                        low_conf_list = extracted_data.get("low_confidence_skips", [])

                        # 1. 정상 텍스트 인식
                        if (detected_text in KNOWN_TEXTS) and (detected_text_conf >= OCR_CONF_THRESHOLD):
                            save_path = os.path.join(OCR_OUTPUT_FOLDER, save_name)
                            is_success = True
                            print("[ROUTE] 정상 텍스트 결과 폴더로 저장")

                        # 2. 정상 색상 인식
                        elif (not detected_text) and (detected_color in KNOWN_COLORS):
                            save_path = os.path.join(COLOR_OUTPUT_FOLDER, save_name)
                            is_success = True
                            print("[ROUTE] 정상 색상 결과 폴더로 저장")

                        # 3. 텍스트 오류 케이스 (텍스트는 인식했으나 신뢰도가 낮음, 혹은 low_confidence_skips 안에 marker_text 관련 내용이 있음)
                        elif (
                                (detected_text and detected_text_conf < OCR_CONF_THRESHOLD) or
                                any("marker_text" in s for s in low_conf_list)
                        ):
                            save_path = os.path.join(ERROR_TEXT_OUTPUT_FOLDER, save_name)
                            print("[ROUTE] 텍스트 오류 폴더(results_error/text)로 저장")

                        # 4. 색상 오류 케이스 (색상은 나왔으나 우리가 아는 색상이 아님, 혹은 low_confidence_skips 안에 color 관련 내용이 있음)
                        elif (
                                (detected_color and detected_color not in KNOWN_COLORS) or
                                any("marker_color" in s for s in low_conf_list)
                        ):
                            save_path = os.path.join(ERROR_COLOR_OUTPUT_FOLDER, save_name)
                            print("[ROUTE] 색상 오류 폴더(results_error/color)로 저장")

                        # 5. 그 외 애매한 모든 케이스 --> 최상위 에러 폴더
                        else:
                            save_path = os.path.join(ERROR_ROOT, save_name)
                            print("[ROUTE] 기타 에러 폴더(results_error)로 저장")


                        # 파일 저장 및 로그
                        try:
                            cv2.imwrite(save_path, annotated_image)

                            if is_success:
                                print(f"[SUCCESS] 결과 이미지를 '{save_path}'에 저장했습니다.")
                            else:
                                print(f"[INFO] 예외 항목(알 수 없는 텍스트/색상 또는 70% 미만) 이미지를 '{save_path}'에 저장했습니다.")

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