# utils/hsv_module.py
import cv2
import numpy as np

# 색상 밴드(각도, degree). OpenCV Hue=0~179이므로 *2 해서 0~360 기준으로 생각.
# 필요하면 +-5~10도씩 조금씩 조정하면서 현장에 맞추세요.
RANGES_DEG = {
    "RED":    [(0, 20), (340, 360)],   # 빨강은 랩어라운드(두 구간)
    "YELLOW": [(20, 45)],
    "GREEN":  [(45, 90)],
    "BLUE":   [(95, 140)],
}

def _shrink_border(img, border=6):
    h, w = img.shape[:2]
    x1 = border
    y1 = border
    x2 = w - border
    y2 = h - border
    if x2 <= x1 or y2 <= y1:  # 너무 작아지는 경우 방지
        return img
    return img[y1:y2, x1:x2]

def _mask_s_v(hsv, s_min=60, v_min=40, v_max=240):
    H, S, V = cv2.split(hsv)
    mask = (S >= s_min) & (V >= v_min) & (V <= v_max)
    return mask

def _circular_mean_deg(h_arr_deg):
    """Hue 원형 평균 (degree)."""
    if len(h_arr_deg) == 0:
        return None
    # degree -> rad
    rad = np.deg2rad(h_arr_deg)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    ang = np.arctan2(mean_sin, mean_cos)  # -pi..pi
    deg = np.rad2deg(ang)
    if deg < 0:
        deg += 360.0
    return deg

def _in_ranges(deg, ranges):
    for lo, hi in ranges:
        if lo <= deg <= hi:
            return True
    return False

def _vote_color_by_ranges(h_deg):
    """밴드에 가장 많이 속한 색으로 투표."""
    votes = {k: 0 for k in RANGES_DEG.keys()}
    for d in h_deg:
        for cname, bands in RANGES_DEG.items():
            if _in_ranges(d, bands):
                votes[cname] += 1
                break
    # 최다 득표
    best = max(votes.items(), key=lambda x: x[1])
    return best[0], votes

def _rgb_dominance(bgr_roi):
    # 중앙 50% ROI의 평균 B,G,R
    h, w = bgr_roi.shape[:2]
    y1, y2 = int(h*0.25), int(h*0.75)
    x1, x2 = int(w*0.25), int(w*0.75)
    roi = bgr_roi[y1:y2, x1:x2]
    b, g, r = cv2.mean(roi)[:3]
    if r > g and r > b: dom = "R"
    elif g > r and g > b: dom = "G"
    elif b > r and b > g: dom = "B"
    else: dom = "N"
    return (b, g, r), dom

def get_color(image_array):
    """
    YOLO 크롭 이미지를 받아 Red/Green/Yellow/Blue 중 하나를 반환.
    - 테두리 축소 → HSV 변환 → 채도/명도 마스크 → Hue 원형 평균/투표
    - 경계(빨강↔노랑 등)에서는 RGB 우세채널로 보정
    """
    if image_array is None or image_array.size == 0:
        return "Unknown"

    # 1) 테두리 오염 제거 + 약한 블러로 노이즈 감소
    bgr = _shrink_border(image_array, border=6)
    bgr = cv2.GaussianBlur(bgr, (3,3), 0)

    # 2) HSV 변환 (OpenCV: H=0..179, S=0..255, V=0..255)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 3) 색이 뚜렷한 픽셀만(채도/명도 마스크)
    mask = _mask_s_v(hsv, s_min=60, v_min=40, v_max=245)

    if not np.any(mask):
        return "Unknown"

    H, S, V = cv2.split(hsv)
    # 0..179 -> 0..360(deg)
    h_deg = (H[mask].astype(np.float32) * 2.0)

    # 4) 빨강 랩어라운드 보정: 빨강 성분이 많은지 먼저 확인
    #    (0~20) + (340~360)에 해당하는 픽셀 비율
    red_ratio = (np.sum((h_deg <= 20) | (h_deg >= 340)) / len(h_deg))

    # 5) 투표/원형 평균으로 1차 분류
    voted_color, votes = _vote_color_by_ranges(h_deg)
    mean_deg = _circular_mean_deg(h_deg)

    # 6) 경계 보정: 빨강 vs 노랑이 비슷하게 나오면 RGB 우세 채널로 결정
    (b_val, g_val, r_val), dom = _rgb_dominance(bgr)

    # 빨강/노랑 경계(예: 18~30도 부근)에서 보정
    if mean_deg is not None and 18 <= mean_deg <= 30:
        if dom == "R":
            voted_color = "RED"
        elif dom == "G":
            voted_color = "YELLOW"

    # 빨강 랩어라운드에서 빨강 표가 많으면 RED 우선
    if red_ratio >= 0.25:
        voted_color = "RED"

    # 7) 품질 체크: 유효 픽셀 수가 너무 적으면 Unknown
    if len(h_deg) < 50:
        return "Unknown"

    # 디버그: 필요시 프린트
    # print(f"[HSV] mean_deg={mean_deg:.1f}, red_ratio={red_ratio:.2f}, votes={votes}, RGB=({r_val:.0f},{g_val:.0f},{b_val:.0f}), dom={dom}")

    return voted_color