from ultralytics import YOLO
import cv2
import os
import numpy as np
import glob
import time  # time 모듈

# utils 폴더에서 우리가 만든 모듈을 가져옵니다.
from utils.ocr_module import run_ocr
from utils.hsv_module import get_color

# --- ⚠️ 중요: PROJECT_ROOT 정의 (파일 상단) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. 학습된 모델 경로
TRAINED_MODEL_PATH = "/Users/skdod/runs/detect/train10/weights/best.pt"

# 2. 모델 로드 (한 번만)
if not os.path.exists(TRAINED_MODEL_PATH):
    print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {TRAINED_MODEL_PATH}")
    print("[INFO] 'main.py'를 실행하여 모델을 먼저 학습시키세요.")
    exit()

print("[INFO] Loading trained YOLO model...")
model = YOLO(TRAINED_MODEL_PATH)
CLASS_NAMES = model.names
print(f"[INFO] Model loaded. Classes: {CLASS_NAMES}")


# --- 경로 설정 끝 ---


def process_image(image_path, conf_threshold=0.7):
    """
    (이 함수는 '슬리퍼'를 인식하게 된 최종 수정본입니다 - 수정 X)
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {image_path}")
        return None, None

    # 1. 고해상도 원본 이미지를 로드
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"[ERROR] cv2가 이미지를 읽지 못했습니다: {image_path}")
        return None, None

    orig_h, orig_w, _ = original_image.shape
    print(f"\n--- Processing Image: {image_path} (Size: {orig_w}x{orig_h}) ---")

    # 2. YOLO 모델 실행
    results = model(original_image)
    result = results[0]

    final_data = {
        "text": None,
        "color": None,
        "object_type": None
    }

    for box in result.boxes:
        conf = box.conf[0].item()
        cls_index = int(box.cls[0].item())
        class_name = CLASS_NAMES[cls_index]

        if conf < conf_threshold:
            print(f"[SKIP] Found '{class_name}' but confidence is too low ({conf * 100:.0f}%)")
            continue

        nx1, ny1, nx2, ny2 = box.xyxyn[0].tolist()
        x1 = int(nx1 * orig_w)
        y1 = int(ny1 * orig_h)
        x2 = int(nx2 * orig_w)
        y2 = int(ny2 * orig_h)

        cropped_img = original_image[y1:y2, x1:x2]

        if class_name == 'marker_text':
            print(f"[YOLO] Found 'marker_text' (Conf: {conf * 100:.0f}%). Sending to OCR...")

            # --- 1. 그레이스케일 + 이진화 ---
            try:
                gray_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                _, binary_img = cv2.threshold(gray_cropped_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            except Exception as e:
                print(f"[WARN] Failed to preprocess image for OCR: {e}")
                continue

                # --- 2. 임시 파일로 저장 (이진화된 이미지) ---
            temp_crop_path = os.path.join(PROJECT_ROOT, "temp_ocr_image.jpg")
            cv2.imwrite(temp_crop_path, binary_img)

            # --- 3. 파일 경로를 전달 ---
            extracted_texts = run_ocr(temp_crop_path)

            if extracted_texts:
                final_data["text"] = extracted_texts[0]
                print(f"[SUCCESS] OCR Result: {final_data['text']}")
            else:
                print("[INFO] OCR module ran, but found no text.")

            # --- 4. 임시 파일 삭제 ---
            try:
                os.remove(temp_crop_path)
            except Exception as e:
                print(f"[WARN] Failed to remove temp crop file: {e}")

        elif class_name == 'marker_color':
            print(f"[YOLO] Found 'marker_color' (Conf: {conf * 100:.0f}%). Sending to HSV...")
            detected_color = get_color(cropped_img)
            final_data["color"] = detected_color
            print(f"[SUCCESS] Color Result: {final_data['color']}")

        elif class_name == 'objects':
            print(f"[YOLO] Found 'objects' (Conf: {conf * 100:.0f}%)")
            final_data["object_type"] = "Box"

    # --- (수정) 렉 안 걸리게 True 대신, '결과 이미지'를 반환 ---
    return result.plot(), final_data


# --- ⚠️ 메인 실행 (결과 이미지를 폴더에 저장하는 로직) ---
if __name__ == "__main__":

    WATCH_FOLDER = os.path.join(PROJECT_ROOT, "test_originals")

    # --- 1. (추가) 결과 저장용 폴더 지정 ---
    OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "test_originals_ocr")
    # 폴더가 없으면 자동 생성
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # ------------------------------------

    processed_files = set()

    print(f"==================================================")
    print(f"[INFO] '{WATCH_FOLDER}' 폴더를 감시합니다...")
    print(f"[INFO] 결과는 '{OUTPUT_FOLDER}' 폴더에 저장됩니다.")
    print(f"[INFO] (종료하려면 터미널에서 Ctrl + C 를 누르세요)")
    print(f"==================================================")

    try:
        while True:
            current_files = set()
            image_types = ("*.jpg", "*.jpeg", "*.png")
            for img_type in image_types:
                current_files.update(glob.glob(os.path.join(WATCH_FOLDER, img_type)))

            new_files = current_files - processed_files

            if new_files:
                print(f"\n[INFO] {len(new_files)}개의 새 이미지를 감지했습니다. 처리를 시작합니다...")
                for image_path in new_files:

                    # --- 2. (수정) annotated_image 변수 다시 받기 ---
                    annotated_image, extracted_data = process_image(image_path, conf_threshold=0.45)  # (45% 유지)

                    if annotated_image is not None:
                        print(f"\n--- Final Data for {os.path.basename(image_path)} ---")
                        print(extracted_data)

                        # --- 3. (수정) 렉 걸리는 창 띄우기 대신 파일로 저장 ---
                        try:
                            # 원본 파일명 + "_result.jpg"로 저장
                            base_name = os.path.basename(image_path)
                            save_name = f"{os.path.splitext(base_name)[0]}_result.jpg"
                            save_path = os.path.join(OUTPUT_FOLDER, save_name)

                            cv2.imwrite(save_path, annotated_image)
                            print(f"[SUCCESS] 결과 이미지를 '{save_path}'에 저장했습니다.")
                        except Exception as e:
                            print(f"[ERROR] 결과 이미지 저장 실패: {e}")
                        # -----------------------------------------------

                    else:
                        print(f"[WARN] '{image_path}' 처리에 실패했습니다.")

                    processed_files.add(image_path)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] 감시를 종료합니다.")
    finally:
        # (창을 안 띄웠으니 닫을 필요 없음)
        print("[INFO] 프로그램이 종료되었습니다.")