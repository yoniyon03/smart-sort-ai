import cv2
import numpy as np

# run_project.py로부터 marker_color 스티커의 컬러 원본(BGR) 이미지를 받아옴
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

    # --- 1. 여기서 H,S,V 값을 직접 출력! ---
    print(f"\n[DEBUG-HSV] ---------------------------------")
    print(f"[DEBUG-HSV] 크롭된 이미지의 평균 H={h_val:.2f}, S={s_val:.2f}, V={v_val:.2f}")
    # ---------------------------------------------

    # 3. 채도(S)나 명도(V)가 너무 낮으면 '무채색'으로 판단
    if s_val < 50 or v_val < 50:
        print("[DEBUG-HSV] 판정: 채도/명도 낮음 -> GRAY/BLACK/WHITE")
        return "GRAY/BLACK/WHITE"

        # 4. H(색상) 값으로 색상 판단
    h_val = hsv_image[:, :, 0].mean()
    color_name = "Unknown"

    # # 조명 있다고 가정? (보정 X)
    # if (0 <= h_val <= 10) or (170 <= h_val <= 179):
    #     color_name = "RED"
    # elif 11 <= h_val <= 25:
    #     color_name = "ORANGE"
    # elif 26 <= h_val <= 34:
    #     color_name = "YELLOW"
    # elif 35 <= h_val <= 75:
    #     color_name = "GREEN"
    # elif 76 <= h_val <= 130:
    #     color_name = "BLUE"
    # elif 131 <= h_val <= 169:
    #     color_name = "PURPLE/PINK"

    # 보정
    if (0 <= h_val <= 10) or (170 <= h_val <= 179):
        color_name = "RED"
    elif 11 <= h_val <= 25:
        color_name = "ORANGE"
    elif 26 <= h_val <= 34:
        color_name = "YELLOW"

    # --- "GREEN" 범위를 110까지 늘림 ---
    elif 35 <= h_val <= 110:
        color_name = "GREEN"

    # --- "BLUE" 시작을 111부터로 미룸 ---
    elif 111 <= h_val <= 130:
        color_name = "BLUE"

    elif 131 <= h_val <= 169:
        color_name = "PURPLE/PINK"

    # --- 최종 판정 결과도 출력 ---
    print(f"[DEBUG-HSV] H 값({h_val:.2f}) -> 최종 판정: {color_name}")
    print(f"[DEBUG-HSV] ---------------------------------\n")
    # ----------------------------------------

    return color_name

# # --- 테스트할 때만 실행 ---
# if __name__ == "__main__":
#     test_blue_img = np.zeros((100, 100, 3), dtype=np.uint8)
#     test_blue_img[:] = (255, 0, 0)
#     color = get_color(test_blue_img)
#     print(f"Test Image 1 (Blue) -> Detected: {color}")