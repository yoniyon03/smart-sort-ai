import cv2


def get_color(image_array):
    """
    잘라낸 이미지를 받아 HSV로 변환,
    평균 색상값을 기반으로 색상 이름을 반환
    """
    if image_array is None or image_array.size == 0:
        return "Unknown"

    # 1. BGR 이미지를 HSV로 변환
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

    # 2. 이미지의 평균 H, S, V 값 계산
    h_val = hsv_image[:, :, 0].mean()  # 평균 H (색상)
    s_val = hsv_image[:, :, 1].mean()  # 평균 S (채도)
    v_val = hsv_image[:, :, 2].mean()  # 평균 V (명도)

    # --- 1. 디버그 로그 ---
    print(f"\n[DEBUG-HSV] ---------------------------------")
    print(f"[DEBUG-HSV] 크롭된 이미지의 평균 H={h_val:.2f}, S={s_val:.2f}, V={v_val:.2f}")
    # ---------------------------------------------

    # 3. 채도(S)나 명도(V)가 너무 낮으면 '무채색'으로 판단
    if s_val < 50 or v_val < 50:
        print("[DEBUG-HSV] 판정: 채도/명도 낮음 -> GRAY/BLACK/WHITE")
        return "GRAY/BLACK/WHITE"

    # 4. H(색상) 값으로 색상 판단
    color_name = "Unknown"

    # --- 4가지 색상 전용 범위 ---
    if (0 <= h_val <= 10) or (170 <= h_val <= 179):
        color_name = "RED"

    elif 25 <= h_val <= 40:
        color_name = "YELLOW"

    elif 60 <= h_val <= 90:
        color_name = "GREEN"

    elif 100 <= h_val <= 130:
        color_name = "BLUE"
    # -----------------------------------------------

    # --- 5. 최종 판정 결과도 출력 ---
    print(f"[DEBUG-HSV] H 값({h_val:.2f}) -> 최종 판정: {color_name}")
    print(f"[DEBUG-HSV] ---------------------------------\n")

    return color_name