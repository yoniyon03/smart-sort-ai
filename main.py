# 학습용 - 전이학습 코드
from ultralytics import YOLO
import torch, os

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon(M1/M2/M3)
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    data_yaml = "./data.yaml"  # <- 필요하면 절대경로로 수정
    device = pick_device()
    print(f"[INFO] Using device: {device}")

    # 1) 사전학습 가중치 불러오기 (COCO로 학습된 초경량 모델)
    model = YOLO("yolov8n.pt")

    # 2) 전이학습(Transfer Learning)
    results = model.train(
        data=data_yaml,
        imgsz=640,
        epochs=50,         # 데이터 적으면 50~80, 많으면 100+
        batch=16,          # GPU 메모리 부족하면 8로
        device=device,
        patience=0,       # early stopping 여유 (0으로 하면 early stopping X)
        freeze=10          # 데이터 적을 때 백본 일부 고정; 많으면 0
    )
    print("[INFO] Train done. Best metrics:", results.results_dict)

    # 3) 검증 (val 세트로 mAP/PR/Recall 등 확인)
    val_metrics = model.val(data=data_yaml, device=device)
    print("[INFO] Val mAP50-95:", val_metrics.box.map)

    # 4) 예측 테스트 (아래 경로에 있는 이미지/폴더로 바꾸기)
    test_source = "./dataset/images/val"   # 샘플로 val 폴더에 대해 예측
    preds = model.predict(
        source=test_source,
        conf=0.25,
        iou=0.5,
        device=device,
        save=True,         # runs/detect/predict 폴더에 결과 이미지 저장
        project="runs",
        name="quick_check",
        exist_ok=True
    )
    print(f"[INFO] Prediction saved to: {preds[0].save_dir}")

if __name__ == "__main__":
    main()