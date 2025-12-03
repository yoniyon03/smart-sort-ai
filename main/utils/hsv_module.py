import cv2, numpy as np

# 각도(deg, 0~360) 기준 밴드
BANDS = {
    "RED":    [(0, 20), (340, 360)],
    "YELLOW": [(50, 90)],
    "GREEN":  [(90, 160)],
    "BLUE":   [(190, 260)],
}

def _in_band(d, ranges):
    return any(lo <= d <= hi for lo, hi in ranges)

def _band_name(deg):
    for name, rng in BANDS.items():
        if _in_band(deg, rng):
            return name
    return "Unknown"

def get_color(bgr_crop):
    if bgr_crop is None or bgr_crop.size == 0:
        return "Unknown"

    # 1. 테두리 축소
    h, w = bgr_crop.shape[:2]
    border = max(6, int(min(h, w) * 0.06))
    x1, y1 = border, border
    x2, y2 = max(x1+1, w-border), max(y1+1, h-border)
    bgr = bgr_crop[y1:y2, x1:x2]
    bgr = cv2.GaussianBlur(bgr, (3, 3), 0)

    # 2. HSV + S/V 마스크
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mask = (S >= 70) & (V >= 50) & (V <= 235)
    if not np.any(mask):
        return "Unknown"

    # 3. 경계/반사 제거
    k = max(1, int(min(hsv.shape[0], hsv.shape[1]) * 0.015))
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1).astype(bool)
    if not np.any(mask):
        return "Unknown"

    # 4. Hue(0~179) --> degree(0~360), 원형 K-Means (K=2)
    h_deg = (H[mask].astype(np.float32) * 2.0)
    if h_deg.size < 30:
        return _band_name(np.median(h_deg)) if h_deg.size else "Unknown"

    # (cos, sin) 공간으로 투영 (원형 거리)
    theta = np.deg2rad(h_deg)
    feats = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    # 간단 K-Means (K=2), 반복 10
    # 초기 중심은 두 개의 랜덤 샘플
    rng = np.random.default_rng(0)
    centers = feats[rng.choice(len(feats), size=2, replace=False)]
    for _ in range(10):
        d0 = np.sum((feats - centers[0])**2, axis=1)
        d1 = np.sum((feats - centers[1])**2, axis=1)
        labels = (d1 < d0).astype(np.int32)

        # 업데이트
        c0 = feats[labels == 0].mean(axis=0) if np.any(labels == 0) else centers[0]
        c1 = feats[labels == 1].mean(axis=0) if np.any(labels == 1) else centers[1]

        # 정규화(단위벡터)
        def _nz(v):
            n = np.linalg.norm(v) + 1e-9
            return v / n
        new_centers = np.stack([_nz(c0), _nz(c1)], axis=0)
        if np.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers

    # 5. 가장 큰 클러스터 선택 --> 각도
    n0, n1 = np.sum(labels == 0), np.sum(labels == 1)
    main_c = centers[0] if n0 >= n1 else centers[1]
    main_deg = (np.rad2deg(np.arctan2(main_c[1], main_c[0])) + 360.0) % 360.0

    # 6) 노랑-빨강 경계 안정화하는 규칙
    # 50~90도이면 기본 노란색 우선
    # 완전 빨강이면 main_deg가 0~20 또는 340~360 근처로 분명하게 위치해야 함
    if 50 <= main_deg <= 90:
        return "YELLOW"

    if _in_band(main_deg, BANDS["RED"]):
        # 붉은 표 비율이 확실히 높을 때만 RED
        red_ratio = np.mean((h_deg <= 20) | (h_deg >= 340))
        if red_ratio >= 0.55:
            return "RED"
        # 그렇지 않다면 주변 밴드로 재판정
        return "YELLOW" if 45 <= main_deg <= 60 else "Unknown"

    return _band_name(main_deg)