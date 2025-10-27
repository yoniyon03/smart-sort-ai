# ** YOLO 코드 **
from ultralytics import YOLO
import cv2, os

def run_infer(image_path="images/필기체4.jpeg"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    # YOLOv8n 모델 로드
    models = YOLO("yolov8n.pt")

    # bus.jpg 이미지에서 객체 탐지
    results = models("images/필기체4.jpeg", imgsz=640)

    # 결과 시각화
    img = results[0].plot()
    cv2.imshow("YOLOv8n Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_infer()