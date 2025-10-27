# ** OCR 코드 **
from paddleocr import PaddleOCR

def run_ocr(image_path="images/필기체3.jpeg", out_dir="output"):
    ocr = PaddleOCR(
        lang='korean',  # 한글 모델
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True  # ← 글자 방향 보정(신규 권장 옵션)
    )

    # v3.x: predict 사용, cls 파라미터 없음!
    result = ocr.predict(input=image_path)

    # 결과 확인/저장
    for r in result:
        r.print()  # 콘솔에 인식 결과 출력
        r.save_to_img(out_dir)  # 결과 시각화 이미지 저장
        r.save_to_json(out_dir)  # 결과 JSON 저장

if __name__ == "__main__":
    run_ocr()