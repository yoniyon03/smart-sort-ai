import cv2
import numpy as np

# 테스트용 이미지 경로 (본인 이미지로 바꿔도 됨)
def show_hsv(img_path="../input_imagesinput_images"):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    h, w, _ = hsv.shape
    center_pixel = hsv[h // 2, w // 2]
    print(f"[중앙 HSV] H={center_pixel[0]}, S={center_pixel[1]}, V={center_pixel[2]}")

    hue, sat, val = cv2.split(hsv)

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