import json
import mimetypes
from datetime import datetime

import requests  # pip install requests

BASE_URL = "https://smartparcel-api.azurewebsites.net"
# 계정에 맞게 id 번호 수정
MANAGER_ID = 2


def now_iso_with_offset():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pretty(resp, title=""):
    print(f"\n=== {title} ===")
    print("Status:", resp.status_code)
    ctype = resp.headers.get("Content-Type", "")
    try:
        if "application/json" in ctype:
            print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
        else:
            print(resp.text)
    except Exception:
        print(resp.text)


# 1. 디바이스 초기 설정 받아오기
def fetch_setup(manager_id: int = MANAGER_ID):
    url = f"{BASE_URL}/api/devices/setup"
    try:
        resp = requests.get(url, params={"managerId": manager_id}, timeout=10)
    except Exception as e:
        print(f"[ERROR] fetch_setup() 요청 실패: {e}")
        return None

    if resp.status_code != 200:
        pretty(resp, "GET /api/devices/setup (ERROR)")
        return None

    pretty(resp, "GET /api/devices/setup (OK)")
    outer = resp.json()
    return outer.get("data")


def build_rule_maps(setup_json):
    """
    서버에서 내려준 setup_json(data)을 파싱해서
    - TEXT 라벨 -> {ruleId, chuteId, servoDeg}
    - COLOR 라벨 -> {ruleId, chuteId, servoDeg}
    """
    text_rules = {}
    color_rules = {}

    if not setup_json:
        return text_rules, color_rules

    # setup_json 은 이미 outer["data"]
    rules = setup_json.get("rules", [])

    for rule in rules:
        rid = rule.get("id")
        input_type = (rule.get("inputType") or "").upper()
        label = rule.get("inputValue")
        chutes = rule.get("chutes", [])

        if not (rid and input_type and label and chutes):
            continue

        chute = chutes[0]
        chute_id = chute.get("id")
        servo_deg = chute.get("servoDeg")

        if not chute_id:
            continue

        info = {
            "ruleId": rid,
            "chuteId": chute_id,
            "servoDeg": servo_deg,
        }

        if input_type == "TEXT":
            text_rules[label] = info
        elif input_type == "COLOR":
            color_rules[label] = info

    print("[INFO] TEXT_RULES:", text_rules)
    print("[INFO] COLOR_RULES:", color_rules)
    return text_rules, color_rules


# 2. 분류 성공 결과 보내기
def send_sorting_result(
    manager_id: int,
    rule_id: int,
    chute_id: int,
    image_path: str | None = None,
):
    url = f"{BASE_URL}/api/devices/events/sorting-result"

    payload_obj = {
        "managerId": manager_id,
        "ruleId": rule_id,
        "chuteId": chute_id,
        "processedAt": now_iso_with_offset(),
    }

    files = {
        "payload": ("payload", json.dumps(payload_obj), "application/json"),
    }

    if image_path:
        try:
            guessed = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
            files["image"] = (
                image_path.split("/")[-1],
                open(image_path, "rb"),
                guessed,
            )
        except FileNotFoundError:
            print(f"[WARN] send_sorting_result: 이미지 파일을 찾을 수 없습니다: {image_path}")

    try:
        resp = requests.post(url, files=files, timeout=30)
        pretty(resp, "POST /api/devices/events/sorting-result")
    except Exception as e:
        print(f"[ERROR] send_sorting_result 요청 실패: {e}")


# 3. 에러/예외 상황 보내기
def send_error_event(
    manager_id: int,
    error_code: str,
    rule_id: int | None = None,
    group_id: int | None = None,
    chute_id: int | None = None,
    image_path: str | None = None,
):
    url = f"{BASE_URL}/api/devices/events/error"

    payload_obj = {
        "managerId": manager_id,
        "errorCode": error_code,
        "ruleId": rule_id,
        "groupId": group_id,
        "chuteId": chute_id,
        "occurredAt": now_iso_with_offset(),
    }

    files = {
        "payload": ("payload", json.dumps(payload_obj), "application/json"),
    }

    if image_path:
        try:
            guessed = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
            files["image"] = (
                image_path.split("/")[-1],
                open(image_path, "rb"),
                guessed,
            )
        except FileNotFoundError:
            print(f"[WARN] send_error_event: 이미지 파일을 찾을 수 없습니다: {image_path}")

    try:
        resp = requests.post(url, files=files, timeout=30)
        pretty(resp, "POST /api/devices/events/error")
    except Exception as e:
        print(f"[ERROR] send_error_event 요청 실패: {e}")


# 아래는 "테스트용" 코드
def test_fetch_setup(manager_id: int):
    data = fetch_setup(manager_id)
    print("\n[TEST] fetch_setup() 반환값:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def test_sorting_result(manager_id: int, rule_id: int, chute_id: int, image_path: str):
    send_sorting_result(manager_id, rule_id, chute_id, image_path)


def test_error_event(
    manager_id: int,
    error_code: str,
    rule_id=None,
    group_id=None,
    chute_id=None,
    image_path: str | None = None,
):
    send_error_event(manager_id, error_code, rule_id, group_id, chute_id, image_path)


if __name__ == "__main__":
    MANAGER_ID = 1
    RULE_ID = 12
    CHUTE_ID = 11
    IMAGE_PATH = "/Users/skdod/Desktop/3-2/캡스톤디자인1/project/main/input_images/capture_1763388609.jpg"

    test_fetch_setup(MANAGER_ID)
    try:
        test_sorting_result(MANAGER_ID, RULE_ID, CHUTE_ID, IMAGE_PATH)
    except FileNotFoundError:
        print("\n[주의] IMAGE_PATH가 존재하지 않습니다. 이미지 경로를 지정하세요.")
    # test_error_event(...)