# HSV 변환 및 저장용
import cv2

def convert_to_hsv(image_path="images/stickers.png", save_path="output_hsv.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("이미지를 불러올 수 없습니다.")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w, _ = hsv_img.shape
    center_pixel = hsv_img[h // 2, w // 2]
    print("중앙 픽셀 HSV =", center_pixel)

    cv2.imshow("Original (BGR)", img)
    cv2.imshow("Converted (HSV)", hsv_img)

    cv2.imwrite(save_path, hsv_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    convert_to_hsv()