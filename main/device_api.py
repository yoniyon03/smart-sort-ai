import json
import mimetypes
from datetime import datetime

import requests  # pip install requests

# 서버 연결 정보
BASE_URL = "https://smartparcel-api.azurewebsites.net"
MANAGER_ID = 3  # 계정에 맞게 id 번호 수정

# 현재 시간을 ISO 8601 형식 문자열로 반환 (예: "2024-05-01T12:34:56+09:00")
def now_iso_with_offset():
    return datetime.now().astimezone().isoformat(timespec="seconds")

# 서버로부터 받은 응답을 보기 좋게 터미널에 출력하는 디버깅용 함수
# 성공 여부(Status Code)와 내용(JSON) 출력
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


# 서버에서 '분류 규칙' 데이터 가져옴
def fetch_setup(manager_id: int = MANAGER_ID):
    url = f"{BASE_URL}/api/devices/setup"
    try:
        # 서버에 GET 요청 전송
        resp = requests.get(url, params={"managerId": manager_id}, timeout=10)
    except Exception as e:
        print(f"[ERROR] fetch_setup() 요청 실패: {e}")
        return None

    # 응답 코드가 200(성공) 아니면 에러 출력
    if resp.status_code != 200:
        pretty(resp, "GET /api/devices/setup (ERROR)")
        return None

    pretty(resp, "GET /api/devices/setup (OK)")

    # JSON 데이터를 파이썬 딕셔너리로 변환해서 반환
    outer = resp.json()
    return outer.get("data")


# 서버에서 받은 복잡한 JSON 데이터를 빠르게 쓸 수 있도록 딕셔너리 형태로 정리
def build_rule_maps(setup_json):
    text_rules = {}
    color_rules = {}

    if not setup_json:
        return text_rules, color_rules

    # setup_json 안에 있는 rules 리스트
    rules = setup_json.get("rules", [])

    for rule in rules:
        rid = rule.get("id") # 규칙 ID
        input_type = (rule.get("inputType") or "").upper() # TEXT인지 COLOR인지
        label = rule.get("inputValue") # 감지할 값
        chutes = rule.get("chutes", []) # 보낼 목적지 (쓔터 정보)

        # 필수 정보 하나라도 없으면 건너뛰기
        if not (rid and input_type and label and chutes):
            continue

        # 첫 번째 슈터 정보 사용
        chute = chutes[0]
        chute_id = chute.get("id")
        servo_deg = chute.get("servoDeg")

        if not chute_id:
            continue

        # 정리된 정보 묶음
        info = {
            "ruleId": rid,
            "chuteId": chute_id,
            "servoDeg": servo_deg,
        }

        # 딕셔너리에 저장
        if input_type == "TEXT":
            text_rules[label] = info
        elif input_type == "COLOR":
            color_rules[label] = info

    print("[INFO] TEXT_RULES:", text_rules)
    print("[INFO] COLOR_RULES:", color_rules)
    return text_rules, color_rules


# 물건 분류가 '성공'했을 때, 그 결과를 서버에 보고 (사진 파일도 같이 업로드)
def send_sorting_result(
    manager_id: int,
    rule_id: int,
    chute_id: int,
    image_path: str | None = None,
):
    url = f"{BASE_URL}/api/devices/events/sorting-result"

    # 서버로 보낼 데이터 (JSON)
    payload_obj = {
        "managerId": manager_id,
        "ruleId": rule_id,
        "chuteId": chute_id,
        "processedAt": now_iso_with_offset(),
    }

    # 파일 업로드를 위한 설정
    files = {
        "payload": ("payload", json.dumps(payload_obj), "application/json"),
    }

    # 이미지 파일 있으면 열어서 files에 추가
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
        # POST 요청 전송
        resp = requests.post(url, files=files, timeout=30)
        pretty(resp, "POST /api/devices/events/sorting-result")
    except Exception as e:
        print(f"[ERROR] send_sorting_result 요청 실패: {e}")


# 분류 '실패'나 '에러' 상황을  서버에 보고
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


# device_api.py 실행 했을 때 작동하는 테스트 코드 (total_run_project.py 실행 했을 때는 사용 안 함)
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