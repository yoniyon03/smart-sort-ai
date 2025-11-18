from paddleocr import PaddleOCR
import cv2
import os
import numpy as np

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
    """ PaddleOCR에 이미지를 한 번 넣고 (text, score) 리스트를 뽑아오는 가장 낮은 레벨 함수
    image_array: BGR 또는 RGB ndarray
    """
    if ocr is None:
        return []

    result = ocr.predict(input=image_array)
    extracted_results = []

    try:
        if result and isinstance(result, list) and len(result) > 0:
            result_data = result[0]

            # 새 포맷: dict 안에 rec_texts / rex_scores
            if isinstance(result_data, dict) and 'rec_texts' in result_data and 'rec_scores' in result_data:
                for text, confidence in zip(result_data['rec_texts'], result_data['rec_scores']):
                    extracted_results.append((text, confidence))

            # 옛날 포맷 대비
            elif isinstance(result_data, list):
                for line_data in result_data:
                    text = line_data[1][0]
                    confidence = line_data[1][1]
                    extracted_results.append((text, confidence))

    except Exception as e:
        print(f"[ERROR] Failed to parse OCR result: {e}")
        print(f"[ERROR] Crashing line_data was: {result}")  # 오류 내용 출력

    return extracted_results


# 전처리 함수 (약하게/강하게)
def _preprocess_for_cor(image_bgr, strong=False):
    """
    OCR 전에 글자 가독성 높여주는 전처리
    strong=True이면 이진화 + 조금 더 센 보정
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # 1. 그레이 + 크기 통일 (작은 크롭이면 키워주기)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 🔽 target_h를 320 -> 260 정도로만 (조금 덜 키워서 연산량 살짝 줄이기)
    target_h = 260
    if h < target_h:
        scale = target_h / float(h)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3. 강한 모드: adaptive threshold로 첫 글자만 뽑기
    if strong:
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25, 15
        )
        # 살짝 팽창시켜 글자 두껍게
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)

    # 3 채널로 다시 변환해 PaddleOCR로 넣기
    proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return proc


# 각도 리스트 + 전처리까지 포함한 한 번의 시도
def _try_ocr_with_angles(image_bgr, angles, strong=False):
    """
    여러 각도에서 OCR을 돌려보고, 그 중 최고 신뢰도 결과 하나만 반환
    strong=True이면 강한 전처리 모드
    """
    (h, w) = image_bgr.shape[:2]
    center = [w // 2, h // 2]

    best_text = None
    best_conf = 0.0

    # 🔽 “이 이상이면 더 돌려도 의미 없다” 기준
    EARLY_BREAK_CONF = 0.98

    for angle in angles:
        rotated = image_bgr
        if angle != 0:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image_bgr, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

        # 전처리 적용
        processed = _preprocess_for_cor(rotated, strong=strong)
        results = _run_ocr_on_image(processed)

        if not results:
            continue

        current_best_conf = 0.0
        current_best_text = ""
        for text, confidence in results:
            if confidence > current_best_conf:
                current_best_conf = confidence
                current_best_text = text

        print(f"[DEBUG-OCR-Angle][strong={strong}] {angle}도: "
              f"'{current_best_text}' (Conf: {current_best_conf:.2f})")

        if current_best_conf > best_conf:
            best_conf = current_best_conf
            best_text = current_best_text

        # 이미 충분히 높은 신뢰도면 남은 각도는 안 돌리고 탈출
        if best_conf >= EARLY_BREAK_CONF:
            print(f"[DEBUG-OCR] 신뢰도 {best_conf:.2f} ≥ {EARLY_BREAK_CONF:.2f}, "
                  f"나머지 각도는 스킵합니다.")
            break

    return best_text, best_conf

def run_ocr(image_path, multi_angle=False):
    """
    메인 호출용
    이미지 경로를 받아 OCR을 수행하고 (최고 텍스트, 최고 신뢰도)를 반환
    multi_angle=False : 빠른 시도 1번 (0도 / 약한 전처리)
    multi_angle=True  : 1단계(빠른 시도) 후, 신뢰도 낮으면
                        2단계(강한 전처리 + 다각도 TTA)까지 돌림
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] OCR image file not found: {image_path}")
        return None, 0.0

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] cv2가 이미지를 읽지 못했습니다: {image_path}")
        return None, 0.0

    # 1단계: 빠른 약한 전처리 + 0도만
    print("[DEBUG-OCR] 1단계: fast pass (angle=0, weak preprocess)")
    best_text, best_conf = _try_ocr_with_angles(
        image_bgr=image,
        angles=[0],
        strong=False
    )

    # multi_angle=False이면 여기서 바로 종료
    if not multi_angle:
        return best_text, best_conf

    # 1단계에서 이미 충분히 잘 나온 경우 2단계 스킵
    FAST_GOOD_THRESHOLD = 0.70  # 0.8 이상이면 그냥 이 결과 믿고 끝내기
    if best_conf >= FAST_GOOD_THRESHOLD:
        print(f"[DEBUG-OCR] 1단계 신뢰도 {best_conf:.2f} ≥ {FAST_GOOD_THRESHOLD:.2f}, "
              f"2단계 strong multi-angle 생략.")
        return best_text, best_conf

    # 2단계: 신뢰도가 낮을 때만 강한 모드 + 다각도
    print("[DEBUG-OCR] 신뢰도 낮음 -> 2단계 strong multi-angle 시도")

    # 각도 범위 줄이기: [-20, -10, 0, 10, 20] → [-15, 0, 15]
    angles_second_stage = [-15, 0, 15]

    text2, conf2 = _try_ocr_with_angles(
        image_bgr=image,
        angles=angles_second_stage,
        strong=True
    )

    if conf2 > best_conf:
        best_text, best_conf = text2, conf2

    return best_text, best_conf