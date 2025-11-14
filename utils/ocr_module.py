from paddleocr import PaddleOCR
import cv2
import os

try:
    print("[INFO] Loading PaddleOCR 'korean' model...")
    ocr = PaddleOCR(
        lang='korean',
        use_angle_cls=True
    )
    print("[INFO] PaddleOCR model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load PaddleOCR model: {e}")
    ocr = None


def _run_ocr_on_image(image_array):
    if ocr is None:
        return []

    result = ocr.predict(input=image_array)

    extracted_results = []

    try:
        if result and isinstance(result, list) and len(result) > 0:
            result_data = result[0]

            # 딕셔너리 안에 'rec_texts'와 'rec_scores'가 있는지 확인
            if isinstance(result_data, dict) and 'rec_texts' in result_data and 'rec_scores' in result_data:

                # 'rec_texts'와 'rec_scores' 리스트를 함께 순회
                for text, confidence in zip(result_data['rec_texts'], result_data['rec_scores']):
                    extracted_results.append((text, confidence))

            # 혹시라도 예전 포맷으로 나올 경우
            elif isinstance(result_data, list):
                for line_data in result_data:
                    text = line_data[1][0]
                    confidence = line_data[1][1]
                    extracted_results.append((text, confidence))

    except Exception as e:
        print(f"[ERROR] Failed to parse OCR result: {e}")
        print(f"[ERROR] Crashing line_data was: {result}")  # 오류 내용 출력

    return extracted_results


def run_ocr(image_path, multi_angle=False):
    """
    (메인 호출용)
    이미지 경로를 받아 OCR을 수행하고 (최고 텍스트, 최고 신뢰도)를 반환합니다.

    multi_angle=True: -20도 ~ +20도까지 5번 돌려서 제일 높은 점수를 찾음 (느리지만 정확)
    multi_angle=False: 0도(원본)만 테스트 (빠름)
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] OCR image file not found: {image_path}")
        return None, 0.0

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] cv2가 이미지를 읽지 못했습니다: {image_path}")
        return None, 0.0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    best_text = None
    best_conf = 0.0  # (최저 0.0으로 시작)

    # (테스트할 각도 리스트)
    if multi_angle:
        angles = range(-20, 21, 10)  # -20, -10, 0, 10, 20 (5번)
    else:
        angles = [0]  # 0도 (1번)

    for angle in angles:
        rotated_image = image
        if angle != 0:
            # 1. 회전 행렬 계산 및 이미지 회전
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, M, (w, h),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # 2. (내부) OCR 실행
        results = _run_ocr_on_image(rotated_image)

        if results:
            # (여러 텍스트가 찾아질 수 있으니, 그중 최고 신뢰도를 찾음)
            current_best_conf = 0.0
            current_best_text = ""
            for text, confidence in results:
                if confidence > current_best_conf:
                    current_best_conf = confidence
                    current_best_text = text

            print(f"[DEBUG-OCR-Angle] {angle}도: '{current_best_text}' (Conf: {current_best_conf:.2f})")

            # 3. "전체 최고" 신뢰도 갱신
            if current_best_conf > best_conf:
                best_conf = current_best_conf
                best_text = current_best_text

    return best_text, best_conf  # (텍스트, 신뢰도) 2개 반환


# --- 테스트용 코드 ---
# if __name__ == "__main__":
#     test_image_path = os.path.join(os.path.dirname(__file__), "../test_originals/테스트할이미지파일명")
#
#     if os.path.exists(test_image_path):
#         print(f"--- Running OCR test on {test_image_path} ---")
#         texts = run_ocr(test_image_path)
#         print(f"--- OCR Test Done ---")
#         print(f"All Extracted Text: {texts}")
#     else:
#         print(f"[ERROR] Test image not found: {test_image_path}")