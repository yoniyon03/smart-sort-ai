# ** HSV 변환 및 저장용 **
import cv2

def convert_to_hsv(image_path="images/stickers.png", save_path="output_hsv.jpg"):
    # 테스트용 이미지 불러오기
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("이미지를 불러올 수 없습니다.")

    # BGR → HSV 변환
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 특정 좌표 픽셀 값 확인 (예: 이미지 중앙)
    h, w, _ = hsv_img.shape
    center_pixel = hsv_img[h // 2, w // 2]
    print("중앙 픽셀 HSV =", center_pixel)

    # 결과 출력
    cv2.imshow("Original (BGR)", img)
    cv2.imshow("Converted (HSV)", hsv_img)

    # 저장도 가능
    cv2.imwrite(save_path, hsv_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    convert_to_hsv()