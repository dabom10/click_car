#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCV vs PaddleOCR 번호판 인식 벤치마크
사용법: python3 ocr_benchmark.py <이미지 경로> [GCV 키 경로]
  예시: python3 ocr_benchmark.py output.jpg /home/rokey/Downloads/google_vision.json
"""

import sys
import os
import cv2
import numpy as np

IMAGE_PATH   = sys.argv[1] if len(sys.argv) > 1 else "/home/rokey/Downloads/output.jpg"
GCV_KEY_PATH = sys.argv[2] if len(sys.argv) > 2 else "/home/rokey/Downloads/google_vision.json"

SEP  = "=" * 65
SEP2 = "-" * 65

# ──────────────────────────────────────────────────────────────
# 이미지 로드
# ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  대상 이미지: {IMAGE_PATH}")
img_orig = cv2.imread(IMAGE_PATH)
if img_orig is None:
    print("  ❌ 이미지 로드 실패! 경로를 확인하세요.")
    sys.exit(1)
h, w = img_orig.shape[:2]
print(f"  원본 크기 : {w} x {h} px")

# ──────────────────────────────────────────────────────────────
# 전처리 변형 목록 생성
# ──────────────────────────────────────────────────────────────
def make_variants(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """테스트할 전처리 변형 이미지 목록 반환."""
    variants = []
    h, w = img.shape[:2]

    # 원본
    variants.append(("원본", img.copy()))

    # 2x 확대
    x2 = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    variants.append(("2x 확대", x2))

    # 4x 확대
    x4 = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    variants.append(("4x 확대", x4))

    # 180° 회전
    rot = cv2.rotate(img, cv2.ROTATE_180)
    variants.append(("180° 회전", rot))

    # 180° 회전 + 2x 확대
    rot2 = cv2.resize(rot, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    variants.append(("180°+2x", rot2))

    # 그레이스케일 + Otsu 이진화 (컬러로 변환해서 입력)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw3 = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    bw3_x2 = cv2.resize(bw3, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    variants.append(("이진화+2x", bw3_x2))

    # CLAHE (대비 제한 적응형 히스토그램)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray_clahe = clahe.apply(gray)
    clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
    clahe_x2  = cv2.resize(clahe_bgr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    variants.append(("CLAHE+2x", clahe_x2))

    return variants

variants = make_variants(img_orig)
print(f"  테스트 변형 수: {len(variants)}종\n")


# ──────────────────────────────────────────────────────────────
# [1] Google Cloud Vision
# ──────────────────────────────────────────────────────────────
print(f"{SEP}")
print("  [1] Google Cloud Vision (GCV)")
print(SEP)

gcv_results = {}

try:
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GCV_KEY_PATH)
    from google.cloud import vision
    gcv_client = vision.ImageAnnotatorClient()
    print("  ✅ GCV 초기화 성공\n")

    for label, img_v in variants:
        try:
            _, buf = cv2.imencode(".jpg", img_v, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gcv_image = vision.Image(content=buf.tobytes())
            image_context = vision.ImageContext(language_hints=["ko"])

            response   = gcv_client.document_text_detection(
                image=gcv_image, image_context=image_context
            )
            annotation = response.full_text_annotation
            if response.error.message:
                result = f"API 오류: {response.error.message}"
            else:
                raw    = annotation.text.replace(" ", "").replace("\n", "")
                result = raw if raw else "(인식 없음)"
        except Exception as e:
            result = f"예외: {e}"

        gcv_results[label] = result
        mark = "✅" if result not in ("(인식 없음)", "") and "오류" not in result and "예외" not in result else "❌"
        print(f"  {mark}  [{label:12s}]  →  {result}")

except ImportError:
    print("  ❌ google-cloud-vision 미설치: pip install google-cloud-vision")
    gcv_results = {}
except Exception as e:
    print(f"  ❌ GCV 초기화 실패: {e}")
    gcv_results = {}


# ──────────────────────────────────────────────────────────────
# [2] PaddleOCR
# ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  [2] PaddleOCR")
print(SEP)

paddle_results = {}

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        lang="korean",
        use_textline_orientation=True,
        enable_mkldnn=False,
    )
    print("  ✅ PaddleOCR 초기화 성공\n")

    def paddle_ocr_run(image: np.ndarray) -> str:
        results = ocr.predict(image)
        if not isinstance(results, list):
            results = list(results)
        texts = []
        for res in results:
            # ── dict-like (OCRResult) ──
            if hasattr(res, "__getitem__") or isinstance(res, dict):
                try:
                    for t, s in zip(res["rec_texts"], res["rec_scores"]):
                        texts.append(f"{t}({s:.2f})")
                    continue
                except (KeyError, TypeError):
                    pass
            # ── get_res() ──
            if hasattr(res, "get_res"):
                d = res.get_res()
                rts = d.get("rec_texts", d.get("rec_text", []))
                rss = d.get("rec_scores", d.get("rec_score", []))
                for t, s in zip(rts, rss):
                    texts.append(f"{t}({s:.2f})")
            # ── 구버전 리스트 ──
            elif isinstance(res, list):
                for line in res:
                    try:
                        t, s = line[1][0], line[1][1]
                        texts.append(f"{t}({s:.2f})")
                    except Exception:
                        pass
        return ", ".join(texts) if texts else "(인식 없음)"

    for label, img_v in variants:
        try:
            result = paddle_ocr_run(img_v)
        except Exception as e:
            result = f"예외: {e}"
        paddle_results[label] = result
        mark = "✅" if result != "(인식 없음)" and "예외" not in result else "❌"
        print(f"  {mark}  [{label:12s}]  →  {result}")

except ImportError:
    print("  ❌ paddleocr 미설치: pip install paddleocr")
except Exception as e:
    print(f"  ❌ PaddleOCR 초기화 실패: {e}")


# ──────────────────────────────────────────────────────────────
# 최종 요약
# ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  📊 최종 요약")
print(SEP)
print(f"  {'변형':<14}  {'GCV':<25}  {'PaddleOCR'}")
print(f"  {SEP2}")
for label, _ in variants:
    gcv_r = gcv_results.get(label, "N/A")
    pad_r = paddle_results.get(label, "N/A")
    print(f"  {label:<14}  {gcv_r:<25}  {pad_r}")

print(f"\n  ※ 번호판: 09가 0228")
print(f"  ※ GCV 키 경로: {GCV_KEY_PATH}")
print(f"{SEP}\n")
