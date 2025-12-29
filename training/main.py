from ultralytics import YOLO
import torch, os

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    data_yaml = "./data.yaml"
    device = pick_device()
    print(f"[INFO] Using device: {device}")

    model = YOLO("yolov8n.pt")

    # 전이학습
    results = model.train(
        data=data_yaml,
        imgsz=960,          # 640 --> 960 수정 (이미지 크기 너무 작으면 인식 잘 안 됨)
        epochs=50,
        batch=8,            # imgsz 960으로 늘리면서 batch도 16 --> 8로 수정 (메모리 부족 방지)
        device=device,
        patience=0,
        freeze=10
    )
    print("[INFO] Train done. Best metrics:", results.results_dict)

    val_metrics = model.val(data=data_yaml, device=device)
    print("[INFO] Val mAP50-95:", val_metrics.box.map)

    test_source = "./dataset/images/val"
    preds = model.predict(
        source=test_source,
        conf=0.25,
        iou=0.5,
        device=device,
        save=True,
        project="runs",
        name="quick_check",
        exist_ok=True
    )
    print(f"[INFO] Prediction saved to: {preds[0].save_dir}")

if __name__ == "__main__":
    main()