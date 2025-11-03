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


# PaddleOCR한테 임시 파일(temp_ocr_image.jpg) 분석하라고 시킴
def run_ocr(image_path):
    # 이미지 경로를 입력받아 OCR을 수행, 추출된 텍스트들을 리스트로 반환
    if ocr is None:
        print("[ERROR] OCR model is not loaded.")
        return []

    if not os.path.exists(image_path):
        print(f"[ERROR] OCR image file not found: {image_path}")
        return []

    print(f"[DEBUG-OCR] Predicting text from file: {image_path}")
    result = ocr.predict(input=image_path)

    extracted_texts = []

    # PaddleOCR이 ['객체 이름']와 ['정확도']처럼 복잡한 딕셔너리(dict) 형식으로 결과를 주는데,
    # 이 부분을 파싱(parsing)해서 extracted_texts.append(text): ['객체 이름'] 이라는
    # 깔끔한 파이썬 리스트로 만들어 run_project.py에게 둘려줌
    try:
        if result and isinstance(result, list) and len(result) > 0:
            result_data = result[0]

            # 딕셔너리 안에 'rec_texts'와 'rec_scores'가 있는지 확인
            if isinstance(result_data, dict) and 'rec_texts' in result_data and 'rec_scores' in result_data:

                # 'rec_texts'와 'rec_scores' 리스트를 함께 순회
                for text, confidence in zip(result_data['rec_texts'], result_data['rec_scores']):
                    print(f"[DEBUG-OCR] Found text: '{text}' with confidence: {confidence}")
                    extracted_texts.append(text)

            # 혹시라도 예전 포맷으로 나올 경우
            elif isinstance(result_data, list):
                for line_data in result_data:
                    text = line_data[1][0]
                    confidence = line_data[1][1]
                    print(f"[DEBUG-OCR] Found text (old format): '{text}' with confidence: {confidence}")
                    extracted_texts.append(text)

    except Exception as e:
        print(f"[ERROR] Failed to parse OCR result: {e}")
        print(f"[ERROR] Crashing line_data was: {result}")  # 오류 내용 출력

    return extracted_texts


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