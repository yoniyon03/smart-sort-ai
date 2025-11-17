# 디바이스 ↔ 서버 통신 전담 모듈

import json
import mimetypes
from datetime import datetime
import requests

BASE_URL = "http://172.30.118.216:8080"

# 이 라인를 관리하는 managerId
MANAGER_ID = 1   # 예시


def now_iso_with_offset():
    """타임존 포함 ISO8601 문자열 (2025-11-17T20:40:12+09:00 형식)"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pretty(resp, title=""):
    """디버깅용 출력"""
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
    """
    서버에서 이 디바이스에 적용할 분류 규칙/슈트 설정 가져오기.
    GET /api/devices/setup?managerId=...
    """
    url = f"{BASE_URL}/api/devices/setup"
    try:
        resp = requests.get(url, params={"managerId": manager_id}, timeout=10)
    except Exception as e:
        print(f"[ERROR] fetch_setup() 요청 실패: {e}")
        return None

    if resp.status_code != 200:
        pretty(resp, "GET /api/devices/setup (ERROR)")
        return None

    data = resp.json()
    pretty(resp, "GET /api/devices/setup (OK)")
    return data


def build_rule_maps(setup_json):
    """
    서버에서 내려준 setup_json을 파싱해서
    - 텍스트 라벨 → (ruleId, chuteId)
    - 색상 라벨  → (ruleId, chuteId)
    매핑 딕셔너리 2개를 만들어 반환.

    실제 응답 구조는 백엔드에서 준 JSON 구조에 맞춰 수정 필요
    아래는 '예시' 구조:
    {
      "rules": [
        { "id": 1, "inputType": "TEXT", "keyword": "대형", "chuteId": 1 },
        { "id": 2, "inputType": "TEXT", "keyword": "중형", "chuteId": 2 },
        { "id": 3, "inputType": "COLOR", "keyword": "RED", "chuteId": 3 },
        ...
      ]
    }
    """
    text_rules = {}   # 예: text_rules["대형"] = {"ruleId": 1, "chuteId": 1}
    color_rules = {}  # 예: color_rules["RED"] = {"ruleId": 3, "chuteId": 3}

    if not setup_json:
        return text_rules, color_rules

    rules = setup_json.get("rules", [])
    for rule in rules:
        rid       = rule.get("id")
        chute_id  = rule.get("chuteId")
        rtype     = rule.get("inputType")   # "TEXT" / "COLOR" 등
        keyword   = rule.get("keyword")     # "대형", "중형", "RED" 등

        if not (rid and chute_id and keyword and rtype):
            continue

        if rtype.upper() == "TEXT":
            text_rules[keyword] = {"ruleId": rid, "chuteId": chute_id}
        elif rtype.upper() == "COLOR":
            color_rules[keyword] = {"ruleId": rid, "chuteId": chute_id}

    print("[INFO] TEXT_RULES:", text_rules)
    print("[INFO] COLOR_RULES:", color_rules)

    return text_rules, color_rules


# 2. 분류 성공 결과 보내기

def send_sorting_result(
    manager_id: int,
    rule_id: int,
    chute_id: int,
    image_path: str | None = None
):
    """
    분류 성공 시 서버에 '이번 박스를 어떤 규칙/슈트로 보냈는지' 보고
    POST /api/devices/events/sorting-result (multipart)
    """
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
            files["image"] = (image_path.split("/")[-1], open(image_path, "rb"), guessed)
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
    image_path: str | None = None
):
    """
    분류 실패 or 시스템 에러 발생 시 서버에 에러 기록
    POST /api/devices/events/error (multipart)
    """
    url = f"{BASE_URL}/api/devices/events/error"

    payload_obj = {
        "managerId": manager_id,
        "errorCode": error_code,  # 예: "OCR_LOW_CONFIDENCE", "YOLO_NO_DETECTION"
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
            files["image"] = (image_path.split("/")[-1], open(image_path, "rb"), guessed)
        except FileNotFoundError:
            print(f"[WARN] send_error_event: 이미지 파일을 찾을 수 없습니다: {image_path}")

    try:
        resp = requests.post(url, files=files, timeout=30)
        pretty(resp, "POST /api/devices/events/error")
    except Exception as e:
        print(f"[ERROR] send_error_event 요청 실패: {e}")