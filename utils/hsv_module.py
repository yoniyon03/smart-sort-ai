# utils/hsv_module.py

import cv2
import numpy as np


def get_color(image_array):
    """
    잘라낸 이미지를 받아 HSV로 변환하고,
    평균 색상값을 기반으로 색상 이름을 반환합니다.
    """
    if image_array is None or image_array.size == 0:
        return "Unknown"

    # 1. BGR 이미지를 HSV로 변환
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

    # 2. 이미지의 평균 H, S, V 값 계산
    # H(Hue): 색상, S(Saturation): 채도, V(Value): 명도
    # h_val = hsv_image[:, :, 0].mean() # 평균 색상
    s_val = hsv_image[:, :, 1].mean()  # 평균 채도
    v_val = hsv_image[:, :, 2].mean()  # 평균 명도

    # 3. 채도(S)나 명도(V)가 너무 낮으면 '무채색'으로 판단
    if s_val < 50 or v_val < 50:
        return "GRAY/BLACK/WHITE"  # (필요시 세분화)

    # 4. H(색상) 값으로 색상 판단
    # OpenCV H 범위: 0 ~ 179
    # H 값은 중앙값이나 최빈값을 쓰는 것이 더 정확하지만,
    # 스티커는 단색이므로 평균값으로도 충분할 수 있습니다.
    h_val = hsv_image[:, :, 0].mean()

    color_name = "Unknown"

    # H 값 범위로 색상 정의 (이 범위는 조명/카메라에 따라 튜닝 필요)
    if (0 <= h_val <= 10) or (170 <= h_val <= 179):
        color_name = "RED"
    elif 11 <= h_val <= 25:
        color_name = "ORANGE"  # (필요시)
    elif 26 <= h_val <= 34:
        color_name = "YELLOW"
    elif 35 <= h_val <= 75:
        color_name = "GREEN"
    elif 76 <= h_val <= 130:
        color_name = "BLUE"
    elif 131 <= h_val <= 169:
        color_name = "PURPLE/PINK"  # (필요시)

    return color_name


# --- 이 파일 자체를 테스트할 때만 실행 ---
if __name__ == "__main__":
    # 테스트용 단색 이미지 생성 (파란색)
    test_blue_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_blue_img[:] = (255, 0, 0)  # BGR로 파란색

    color = get_color(test_blue_img)
    print(f"Test Image 1 (Blue) -> Detected: {color}")

    # 테스트용 단색 이미지 생성 (빨간색)
    test_red_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_red_img[:] = (0, 0, 255)  # BGR로 빨간색

    color = get_color(test_red_img)
    print(f"Test Image 2 (Red) -> Detected: {color}")