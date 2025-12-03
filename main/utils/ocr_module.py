from paddleocr import PaddleOCR
import cv2
import os
import numpy as np

# [초기화] PaddleOCR 모델 로드
# OCR 모델은 로딩하는 데 시간이 오래 걸리므로
# 프로그램 시작 시 딱 한 번만 전역 변수로 로드
try:
    print("[INFO] Loading PaddleOCR 'korean' model...")
    ocr = PaddleOCR(
        lang='en',
        use_angle_cls=True
    )
    print("[INFO] PaddleOCR model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load PaddleOCR model: {e}")
    ocr = None


# PaddleOCR 라이브러리에 이미지를 실제로 집어넣고 결과를 받아오는 함수
# 입력: 이미지 배열 (BGR)
# 출력: [(텍스트, 신뢰도), (텍스트, 신뢰도), ...] 형태의 리스트
def _run_ocr_on_image(image_array):
    if ocr is None:
        return []

    # 실제 예측 수행
    result = ocr.predict(input=image_array)
    extracted_results = []

    try:
        # 결과 유효한 지 확인
        if result and isinstance(result, list) and len(result) > 0:
            result_data = result[0]

            # case 1: PaddleOCR 포맷 (딕셔너리 형태)
            if isinstance(result_data, dict) and 'rec_texts' in result_data and 'rec_scores' in result_data:
                for text, confidence in zip(result_data['rec_texts'], result_data['rec_scores']):
                    extracted_results.append((text, confidence))

            # case 2: PaddleOCR 포맷 (리스트 형태)
            elif isinstance(result_data, list):
                for line_data in result_data:
                    # line_data: [ [좌표], ('텍스트', 신뢰도) ]
                    text = line_data[1][0]
                    confidence = line_data[1][1]
                    extracted_results.append((text, confidence))

    except Exception as e:
        print(f"[ERROR] Failed to parse OCR result: {e}")
        print(f"[ERROR] Crashing line_data was: {result}")

    return extracted_results


# OCR 인식률을 높이기 위해 이미지 전처리
def _preprocess_for_cor(image_bgr, strong=False):
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # 1. 그레이 + 크기 통일 (작은 크롭이면 키워주기)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 2. 해상도 조정 (작은 스티커 크기는 뭉개지기 쉬움 --> 강제로 키우기)
    target_h = 260
    if h < target_h:
        scale = target_h / float(h)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 3. 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 이진화 및 두께 보정 (배경 노이즈가 심하거나 글자가 너무 얇을 때)
    if strong:
        # 그림자만 있어도 글자만 검은색/흰색으로 분리
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25, 15
        )

        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)

    # PaddleOCR은 3채널 이미지 --> BGR로 다시 변환
    proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return proc


# 이미지를 여러 각도로 회전시켜가며 OCR을 시도 (가장 높은 신뢰도가 나온 결과를 선택)
def _try_ocr_with_angles(image_bgr, angles, strong=False):
    (h, w) = image_bgr.shape[:2]
    center = [w // 2, h // 2]

    best_text = None
    best_conf = 0.0

    # 최적화: 신뢰도 이 값 이상 나오면 다른 각도는 시도하지 않고 즉시 종료
    EARLY_BREAK_CONF = 0.98

    for angle in angles:
        rotated = image_bgr

        # 이미지를 지정한 각도만큼 회전
        if angle != 0:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image_bgr, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

        # 전처리 적용
        processed = _preprocess_for_cor(rotated, strong=strong)

        # OCR 실행
        results = _run_ocr_on_image(processed)

        if not results:
            continue

        # 이번 각도에서 나온 결과 중 가장 신뢰도 높은 결과 찾기
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

        # 이미 충분히 높은 신뢰도라면 남은 각도는 돌리지 않고 종료
        if best_conf >= EARLY_BREAK_CONF:
            print(f"[DEBUG-OCR] 신뢰도 {best_conf:.2f} ≥ {EARLY_BREAK_CONF:.2f}, "
                  f"나머지 각도는 스킵합니다.")
            break

    return best_text, best_conf

# 메인 함수
def run_ocr(image_path, multi_angle=False):
    if not os.path.exists(image_path):
        print(f"[ERROR] OCR image file not found: {image_path}")
        return None, 0.0

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] cv2가 이미지를 읽지 못했습니다: {image_path}")
        return None, 0.0

    # 1단계: 빠른 약한 전처리 + 원본(0도) 검사
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
    FAST_GOOD_THRESHOLD = 0.70
    if best_conf >= FAST_GOOD_THRESHOLD:
        print(f"[DEBUG-OCR] 1단계 신뢰도 {best_conf:.2f} ≥ {FAST_GOOD_THRESHOLD:.2f}, "
              f"2단계 strong multi-angle 생략.")
        return best_text, best_conf

    # 2단계: 신뢰도가 낮을 때만 강한 모드 + 다각도
    print("[DEBUG-OCR] 신뢰도 낮음 -> 2단계 strong multi-angle 시도")

    # 세 방향으로 돌려보면서 적용
    angles_second_stage = [-15, 0, 15]

    text2, conf2 = _try_ocr_with_angles(
        image_bgr=image,
        angles=angles_second_stage,
        strong=True
    )

    # 2단계 결과가 더 좋으면 갱신
    if conf2 > best_conf:
        best_text, best_conf = text2, conf2

    return best_text, best_conf