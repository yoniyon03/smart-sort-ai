# 특정 이미지의 HSV 가 어떻게 추출 되는지 체크하는 코드
import cv2
import numpy as np

# 테스트용 이미지 경로 (본인 이미지로 바꿔도 됨)
def show_hsv(img_path="../test_originals"):
    # 이미지 불러오기 (BGR)
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

    # BGR → HSV 변환
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # ---- 1. 픽셀 하나 찍어서 값 확인 ----
    # 이미지 중앙 픽셀 값
    h, w, _ = hsv.shape
    center_pixel = hsv[h // 2, w // 2]
    print(f"[중앙 HSV] H={center_pixel[0]}, S={center_pixel[1]}, V={center_pixel[2]}")

    # ---- 2. HSV 분리 시각화 ----
    hue, sat, val = cv2.split(hsv)

    # Hue를 컬러맵으로 보기 좋게 표시
    hue_color = cv2.applyColorMap(hue, cv2.COLORMAP_HSV)

    cv2.imshow("Original (BGR)", bgr)
    cv2.imshow("Hue channel", hue)
    cv2.imshow("Hue colorized", hue_color)
    cv2.imshow("Saturation", sat)
    cv2.imshow("Value", val)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_hsv()