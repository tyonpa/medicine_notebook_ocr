from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path

import qrcode
import requests
import streamlit as st
from PIL import Image
from openai import OpenAI

DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "")
OCR_API_URL = os.getenv("OCR_API_URL", "http://127.0.0.1:8001")
OCR_API_TIMEOUT = float(os.getenv("OCR_API_TIMEOUT", "900"))
APP_DIR = Path(__file__).resolve().parent
PROMPT_PATH = APP_DIR / "prompt.txt"


@st.cache_resource(show_spinner=False)
def load_openai_client(base_url: str, api_key: str):
    return OpenAI(base_url=base_url, api_key=api_key)


def run_ocr_on_image(image: Image.Image, device: str) -> tuple[str, str]:
    image_bytes = image_to_png_bytes(image.convert("RGB"))
    try:
        response = requests.post(
            f"{OCR_API_URL.rstrip('/')}/ocr",
            files={"file": ("medicine_notebook.png", image_bytes, "image/png")},
            data={"device": device},
            timeout=OCR_API_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"OCR APIへの接続またはOCR実行に失敗しました: {exc}") from exc

    data = response.json()
    return str(data.get("ocr_text", "")).strip(), str(data.get("xml_text", "")).strip()


def load_extraction_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_extraction_prompt(ocr_text: str) -> str:
    prompt_template = load_extraction_prompt()
    if "{cutted_txt}" in prompt_template:
        return prompt_template.replace("{cutted_txt}", ocr_text)
    return f"{prompt_template.rstrip()}\n\n# 入力文：\n{ocr_text}"


def extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    texts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                texts.append(text)

    return "\n".join(texts).strip()


def llm_resp(system_prompt: str, user_prompt: str, api_url: str, api_key: str, model_name: str):
    client = load_openai_client(api_url, api_key)
    return client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )


def summarize_medicine_text(
    ocr_text: str,
    api_url: str,
    api_key: str,
    model_name: str,
    status_callback=None,
) -> str:
    if status_callback is not None:
        status_callback("お薬抽出中...")
    response = llm_resp(
        "あなたは指示されたフォーマットだけを返す情報抽出AIです。",
        build_extraction_prompt(ocr_text),
        api_url,
        api_key,
        model_name,
    )
    return extract_response_text(response)


def make_qr_image(text: str) -> Image.Image:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_pipeline(
    preview_image: Image.Image,
    device: str,
    api_url: str,
    api_key: str,
    model_name: str,
    status_placeholder,
):
    total_start = time.perf_counter()
    progress = {"label": "OCR中...", "result": None, "error": None}

    def worker():
        try:
            ocr_text, xml_text = run_ocr_on_image(preview_image, device)
            summary_text = summarize_medicine_text(
                ocr_text,
                api_url,
                api_key,
                model_name,
                status_callback=lambda message: progress.__setitem__("label", message),
            )
            progress["result"] = (ocr_text, xml_text, summary_text)
        except Exception as exc:  # pragma: no cover
            progress["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while thread.is_alive():
        elapsed = time.perf_counter() - total_start
        status_placeholder.info(f"{progress['label']} {elapsed:.1f}秒")
        time.sleep(0.1)

    elapsed = time.perf_counter() - total_start
    if progress["error"] is not None:
        status_placeholder.empty()
        raise progress["error"]

    status_placeholder.success(f"完了しました。処理時間: {elapsed:.1f}秒")
    return progress["result"]


def main():
    st.set_page_config(page_title="お薬手帳OCR", page_icon="💊", layout="wide")
    st.title("お薬手帳OCRアプリ")
    st.caption("画像入力 -> OCR -> お薬情報要約 -> 編集 -> QRコード生成")

    if "summary_text" not in st.session_state:
        st.session_state.summary_text = ""
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "xml_text" not in st.session_state:
        st.session_state.xml_text = ""
    if "summary_editor" not in st.session_state:
        st.session_state.summary_editor = ""
    if "llm_api_url" not in st.session_state:
        st.session_state.llm_api_url = DEFAULT_OPENAI_BASE_URL
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = DEFAULT_OPENAI_API_KEY
    if "llm_model_name" not in st.session_state:
        st.session_state.llm_model_name = DEFAULT_OPENAI_MODEL_NAME

    with st.sidebar:
        st.subheader("設定")
        device = st.selectbox("OCRデバイス", ["cpu", "cuda"], index=0)
        st.caption("`cuda` は onnxruntime-gpu と対応 GPU がある場合のみ有効です。")
        st.divider()
        st.subheader("LLM API")
        api_url = st.text_input(
            "API URL",
            key="llm_api_url",
            placeholder="例: http://127.0.0.1:8080/v1",
        )
        api_key = st.text_input(
            "API Key",
            key="llm_api_key",
            type="password",
            placeholder="OpenAI互換APIのキー",
        )
        model_name = st.text_input(
            "Model Name",
            key="llm_model_name",
            placeholder="例: qwen3.5-2B",
        )

    input_col, preview_col = st.columns([1, 1], vertical_alignment="top")

    with input_col:
        st.subheader("カメラで撮影")
        camera_image = st.camera_input("撮影", label_visibility="collapsed")
        st.subheader("画像をアップロード")
        uploaded_file = st.file_uploader(
            "画像ファイルをアップロード",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            label_visibility="collapsed",
        )

    source_file = camera_image or uploaded_file
    preview_image = None
    if source_file is not None:
        preview_image = Image.open(io.BytesIO(source_file.getvalue())).convert("RGB")

    with preview_col:
        st.subheader("入力画像")
        if preview_image is not None:
            st.image(preview_image, caption="入力画像", use_container_width=True)
        else:
            st.info("撮影またはアップロードされた画像が表示されます。")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    _, action_col, _ = st.columns([1, 2, 1])
    with action_col:
        if st.button("実行", type="primary", disabled=preview_image is None, use_container_width=True):
            missing_fields = []
            if not api_url.strip():
                missing_fields.append("API URL")
            if not api_key.strip():
                missing_fields.append("API Key")
            if not model_name.strip():
                missing_fields.append("Model Name")
            if missing_fields:
                st.error(f"{', '.join(missing_fields)} が入力されていません。")
                st.stop()

            status_placeholder = st.empty()
            st.session_state.ocr_text, st.session_state.xml_text, st.session_state.summary_text = run_pipeline(
                preview_image,
                device,
                api_url.strip(),
                api_key.strip(),
                model_name.strip(),
                status_placeholder,
            )
            st.session_state.summary_editor = st.session_state.summary_text
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("OCR結果")
        st.text_area("抽出テキスト", value=st.session_state.ocr_text, height=320, disabled=True)
    with right_col:
        st.subheader("お薬情報")
        st.text_area(
            "編集可能テキスト",
            height=320,
            key="summary_editor",
        )

    st.session_state.summary_text = st.session_state.summary_editor

    if st.session_state.summary_text.strip():
        qr_image = make_qr_image(st.session_state.summary_text.strip())
        st.subheader("QRコード")
        st.image(qr_image, caption="お薬情報のQRコード", width=280)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "QRコードをPNGで保存",
                data=image_to_png_bytes(qr_image),
                file_name="medicine_qr.png",
                mime="image/png",
            )
        with col2:
            st.download_button(
                "お薬情報テキストを保存",
                data=st.session_state.summary_text.encode("utf-8"),
                file_name="medicine_summary.txt",
                mime="text/plain",
            )

    with st.expander("詳細データ"):
        st.download_button(
            "OCR生テキストを保存",
            data=st.session_state.ocr_text.encode("utf-8"),
            file_name="ocr_text.txt",
            mime="text/plain",
        )
        st.download_button(
            "OCR XMLを保存",
            data=st.session_state.xml_text.encode("utf-8"),
            file_name="ocr_result.xml",
            mime="application/xml",
        )
        st.code(
            json.dumps(
                {
                    "model": model_name,
                    "base_url": api_url,
                    "ocr_api_url": OCR_API_URL,
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )


if __name__ == "__main__":
    main()
