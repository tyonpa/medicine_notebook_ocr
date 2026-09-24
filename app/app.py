from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import time
import unicodedata
import uuid
import warnings
from dotenv import load_dotenv
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import qrcode
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from openai import OpenAI
from qrcode.exceptions import DataOverflowError


def positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


load_dotenv()

DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "")
OCR_API_URL = os.getenv("OCR_API_URL", "http://127.0.0.1:8001")
OCR_API_TIMEOUT = positive_env_float("OCR_API_TIMEOUT", 900.0)
MAX_UPLOAD_BYTES = positive_env_int("MAX_UPLOAD_BYTES", 100 * 1024 * 1024)
MAX_SOURCE_PIXELS = positive_env_int("MAX_SOURCE_PIXELS", 60_000_000)
MAX_OCR_PIXELS = positive_env_int("MAX_OCR_PIXELS", 12_000_000)
MAX_OCR_DIMENSION = positive_env_int("MAX_OCR_DIMENSION", 3_200)
APP_DIR = Path(__file__).resolve().parent
PROMPT_PATH = APP_DIR / "prompt.txt"
LOG_DIR = Path(os.getenv("LOG_DIR") or APP_DIR.parent / "log")
LOG_WRITE_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)
INDEX_PREFIX_PATTERN = re.compile(r"^\s*\[\s*\d+\s*\]\s*")
DOSAGE_PATTERN = re.compile(
    r"^(?P<name>.+?)\s+1\s*日\s*(?P<amount>[0-9０-９]+(?:[.．][0-9０-９]+)?)\s*(?P<unit>\S.*)$"
)
NO_MEDICINE_PHRASES = ("該当なし", "ありません", "抽出できません", "見つかりません")
PERSISTED_WIDGET_KEYS = (
    ("patient_id_input", "patient_id"),
    ("include_patient_id_toggle", "include_patient_id_in_qr"),
)


class AppError(RuntimeError):
    """画面にそのまま表示できる、想定内のアプリケーションエラー。"""


class ImageInputError(AppError):
    pass


class OcrProcessingError(AppError):
    pass


class MedicineExtractionError(AppError):
    pass


class QrGenerationError(AppError):
    pass


@dataclass(frozen=True)
class PreparedImage:
    image: Image.Image
    original_size: tuple[int, int]
    resized: bool


@st.cache_resource(show_spinner=False)
def load_openai_client(base_url: str, api_key: str):
    return OpenAI(base_url=base_url, api_key=api_key)


def prepare_input_image(image_bytes: bytes) -> PreparedImage:
    if not image_bytes:
        raise ImageInputError("画像データが空です。別の画像を選択してください。")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        limit_mb = max(1, (MAX_UPLOAD_BYTES + 1024 * 1024 - 1) // (1024 * 1024))
        raise ImageInputError(f"画像ファイルが大きすぎます。{limit_mb}MB以下の画像を選択してください。")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                original_size = source.size
                source_pixels = original_size[0] * original_size[1]
                if original_size[0] <= 0 or original_size[1] <= 0:
                    raise ImageInputError("画像のサイズを確認できませんでした。")
                if source_pixels > MAX_SOURCE_PIXELS:
                    raise ImageInputError("画像の解像度が大きすぎます。解像度を下げてから再度選択してください。")
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
    except ImageInputError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ImageInputError("画像を読み込めませんでした。対応形式の画像を選択してください。") from exc

    width, height = image.size
    scale = min(
        1.0,
        MAX_OCR_DIMENSION / max(width, height),
        math.sqrt(MAX_OCR_PIXELS / (width * height)),
    )
    resized = scale < 1.0
    if resized:
        resized_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        image = image.resize(resized_size, Image.Resampling.LANCZOS)

    return PreparedImage(image=image, original_size=original_size, resized=resized)


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
    except requests.Timeout as exc:
        raise OcrProcessingError("OCR処理がタイムアウトしました。画像を小さくするか、再度実行してください。") from exc
    except requests.ConnectionError as exc:
        raise OcrProcessingError("OCRサーバーに接続できません。OCR APIが起動しているか確認してください。") from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "不明"
        raise OcrProcessingError(f"OCR処理に失敗しました。サーバー応答: {status_code}") from exc
    except requests.RequestException as exc:
        raise OcrProcessingError("OCRサーバーとの通信に失敗しました。再度実行してください。") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise OcrProcessingError("OCRサーバーから正しい応答を受け取れませんでした。") from exc
    if not isinstance(data, dict):
        raise OcrProcessingError("OCRサーバーの応答形式が正しくありません。")

    ocr_text = str(data.get("ocr_text") or "").strip()
    xml_text = str(data.get("xml_text") or "").strip()
    if not ocr_text:
        raise OcrProcessingError("画像から文字を読み取れませんでした。画像の向きや明るさを確認してください。")
    return ocr_text, xml_text


def load_extraction_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise MedicineExtractionError("お薬情報の抽出設定を読み込めませんでした。") from exc


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
    try:
        client = load_openai_client(api_url, api_key)
        return client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
    except Exception as exc:
        raise MedicineExtractionError(
            "AIによるお薬情報の抽出に失敗しました。API設定とモデルの稼働状態を確認してください。"
        ) from exc


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


def parse_medicine_text(summary_text: str) -> list[dict[str, object]]:
    """LLMの行形式を、編集UIで扱える薬データへ変換する。"""
    raw_lines = [
        line.strip()
        for line in summary_text.splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    indexed_lines = [line for line in raw_lines if INDEX_PREFIX_PATTERN.match(line)]
    target_lines = indexed_lines or [line for line in raw_lines if DOSAGE_PATTERN.match(line)]

    medicines: list[dict[str, object]] = []
    for line in target_lines:
        content = INDEX_PREFIX_PATTERN.sub("", line).strip()
        if not content or any(phrase in content for phrase in NO_MEDICINE_PHRASES):
            continue
        match = DOSAGE_PATTERN.match(content)
        if match:
            name = match.group("name").strip()
            normalized_amount = unicodedata.normalize("NFKC", match.group("amount")).replace("．", ".")
            amount = int(float(normalized_amount))
            unit = match.group("unit").strip()
        else:
            name = content
            amount = 0
            unit = "不明"

        if name:
            medicines.append({"name": name, "amount": amount, "unit": unit})

    return medicines


def format_amount(amount: int) -> str:
    return str(int(amount))


def build_medicine_text(medicines: list[dict[str, object]]) -> str:
    lines = []
    for index, medicine in enumerate(medicines, start=1):
        name = str(medicine.get("name", "")).strip() or "薬名不明"
        amount = max(0, int(medicine.get("amount", 0)))
        unit = str(medicine.get("unit", "不明")).strip() or "不明"
        lines.append(f"[{index}] {name} 1日{format_amount(amount)}{unit}")
    return "\n".join(lines)


def medicine_entries_are_valid(medicines: list[dict[str, object]]) -> bool:
    if not medicines:
        return False
    for medicine in medicines:
        try:
            amount = int(medicine.get("amount", 0))
        except (TypeError, ValueError):
            return False
        if not str(medicine.get("name", "")).strip():
            return False
        if not str(medicine.get("unit", "")).strip() or amount < 0:
            return False
    return True


def replace_medicine_entries(summary_text: str) -> None:
    for key in list(st.session_state):
        if key.startswith(("medicine_name_", "medicine_amount_", "medicine_unit_")):
            del st.session_state[key]

    entries = []
    for medicine in parse_medicine_text(summary_text):
        entry_id = st.session_state.medicine_next_id
        st.session_state.medicine_next_id += 1
        entries.append({"id": entry_id, **medicine})
    st.session_state.medicine_entries = entries


def clear_analysis_results() -> None:
    for key in list(st.session_state):
        if key.startswith(("medicine_name_", "medicine_amount_", "medicine_unit_")):
            del st.session_state[key]
    st.session_state.ocr_text = ""
    st.session_state.xml_text = ""
    st.session_state.raw_summary_text = ""
    st.session_state.summary_text = ""
    st.session_state.qr_text = ""
    st.session_state.medicine_entries = []
    st.session_state.processing_seconds = None
    st.session_state.log_record_id = ""
    st.session_state.last_logged_qr_signature = None


def append_reading_log(record: dict[str, object]) -> None:
    """読み取りログを日次のJSON Linesファイルへ1行追記する。"""
    timestamp = datetime.now().astimezone()
    entry = {"timestamp": timestamp.isoformat(timespec="seconds"), **record}
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    log_path = LOG_DIR / f"{timestamp:%Y-%m-%d}.jsonl"
    # Streamlitのセッションは同一プロセス内のスレッドで動くため、プロセス内ロックで追記を直列化する。
    with LOG_WRITE_LOCK:
        LOG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as log_file:
            log_file.write(line)


def write_reading_log(record: dict[str, object]) -> bool:
    try:
        append_reading_log(record)
    except (OSError, TypeError, ValueError):
        LOGGER.exception("Failed to write reading log")
        st.session_state.log_warning = "読み取りログを保存できませんでした。ログの保存先を確認してください。"
        return False
    return True


def get_patient_id() -> str:
    return str(st.session_state.patient_id).strip()


def restore_widget_value(widget_key: str, store_key: str) -> None:
    """ページ遷移で消えるウィジェットの値を、保存しておいた値から復元する。"""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[store_key]


def persist_widget_values() -> None:
    for widget_key, store_key in PERSISTED_WIDGET_KEYS:
        if widget_key in st.session_state:
            st.session_state[store_key] = st.session_state[widget_key]


def start_new_entry() -> None:
    clear_analysis_results()
    st.session_state.input_image_png = b""
    st.session_state.input_image_resized = False
    st.session_state.input_image_original_size = None
    st.session_state.input_image_fingerprint = ""
    # キーを変えて、画像入力ウィジェットに残っている前回の画像を破棄する。
    st.session_state.input_widget_version += 1
    for widget_key, _ in PERSISTED_WIDGET_KEYS:
        st.session_state.pop(widget_key, None)
    st.session_state.patient_id = ""
    st.session_state.include_patient_id_in_qr = False
    st.session_state.current_page = 1


def adjust_medicine_amount(entry_id: int, delta: int) -> None:
    widget_key = f"medicine_amount_{entry_id}"
    current_amount = int(st.session_state.get(widget_key, 0))
    new_amount = max(0, current_amount + delta)
    st.session_state[widget_key] = new_amount
    for medicine in st.session_state.medicine_entries:
        if medicine["id"] == entry_id:
            medicine["amount"] = new_amount
            break


def add_medicine_entry() -> None:
    entry_id = st.session_state.medicine_next_id
    st.session_state.medicine_next_id += 1
    st.session_state.medicine_entries.append(
        {"id": entry_id, "name": "", "amount": 1, "unit": "錠"}
    )


def remove_medicine_entry(entry_id: int) -> None:
    st.session_state.medicine_entries = [
        medicine for medicine in st.session_state.medicine_entries if medicine["id"] != entry_id
    ]
    for prefix in ("medicine_name_", "medicine_amount_", "medicine_unit_"):
        st.session_state.pop(f"{prefix}{entry_id}", None)


def sync_medicine_entries_from_widgets() -> None:
    for medicine in st.session_state.medicine_entries:
        entry_id = int(medicine["id"])
        medicine["name"] = st.session_state.get(
            f"medicine_name_{entry_id}", medicine["name"]
        )
        medicine["amount"] = st.session_state.get(
            f"medicine_amount_{entry_id}", medicine["amount"]
        )
        medicine["amount"] = int(medicine["amount"])
        medicine["unit"] = st.session_state.get(
            f"medicine_unit_{entry_id}", medicine["unit"]
        )
    st.session_state.summary_text = build_medicine_text(st.session_state.medicine_entries)
    st.session_state.qr_text = st.session_state.summary_text.strip()


def go_to_page(page_number: int) -> None:
    sync_medicine_entries_from_widgets()
    st.session_state.current_page = min(3, max(1, page_number))


def render_page_navigation(current_page: int, can_go_next: bool) -> None:
    with st.container(key="page_navigation"):
        previous_col, _, next_col = st.columns([1.4, 7.2, 1.4], vertical_alignment="center")
        with previous_col:
            st.button(
                "← 戻る",
                key=f"page_previous_{current_page}",
                disabled=current_page == 1,
                use_container_width=True,
                on_click=go_to_page,
                args=(current_page - 1,),
            )
        with next_col:
            if current_page == 3:
                st.button(
                    "新規",
                    key="page_new",
                    use_container_width=True,
                    on_click=start_new_entry,
                )
            else:
                st.button(
                    "次へ →",
                    key=f"page_next_{current_page}",
                    disabled=not can_go_next,
                    use_container_width=True,
                    on_click=go_to_page,
                    args=(current_page + 1,),
                )


def make_qr_image(text: str) -> Image.Image:
    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)
        return qr.make_image(fill_color="black", back_color="white").convert("RGB")
    except (DataOverflowError, ValueError) as exc:
        raise QrGenerationError("お薬情報が多すぎるためQRコードを作成できません。内容を短くしてください。") from exc


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_workflow_steps(active_step: int) -> None:
    labels = ["画像を選ぶ", "内容を確認", "QRを表示"]
    columns = st.columns(3, gap="small")
    for index, (column, label) in enumerate(zip(columns, labels), start=1):
        state_class = "is-complete" if index < active_step else "is-active" if index == active_step else ""
        with column:
            st.markdown(
                f"<div class='workflow-step {state_class}'>{label}</div>",
                unsafe_allow_html=True,
            )


def render_medicine_editor() -> list[dict[str, object]]:
    medicines = st.session_state.medicine_entries
    if not medicines:
        st.warning("お薬情報を抽出できませんでした。読み取り文章を確認し、手動で追加してください。")

    for medicine in medicines:
        entry_id = int(medicine["id"])
        with st.container(
            border=True,
            key=f"medicine_row_{entry_id}",
            horizontal=True,
            vertical_alignment="center",
            gap="small",
        ):
            name_key = f"medicine_name_{entry_id}"
            if name_key not in st.session_state:
                st.session_state[name_key] = str(medicine["name"])
            name = st.text_input(
                "お薬名",
                key=name_key,
                placeholder="お薬名を入力",
                label_visibility="collapsed",
                width="stretch",
            )
            st.button(
                "−",
                key=f"medicine_minus_{entry_id}",
                help="1減らす",
                width=40,
                on_click=adjust_medicine_amount,
                args=(entry_id, -1),
            )
            amount_key = f"medicine_amount_{entry_id}"
            st.session_state[amount_key] = int(
                st.session_state.get(amount_key, medicine["amount"])
            )
            amount = st.number_input(
                "1日の数量",
                min_value=0,
                step=1,
                key=amount_key,
                format="%d",
                label_visibility="collapsed",
                width=92,
            )
            st.button(
                "＋",
                key=f"medicine_plus_{entry_id}",
                help="1増やす",
                width=40,
                on_click=adjust_medicine_amount,
                args=(entry_id, 1),
            )
            unit_key = f"medicine_unit_{entry_id}"
            if unit_key not in st.session_state:
                st.session_state[unit_key] = str(medicine["unit"])
            unit = st.text_input(
                "単位・用法",
                key=unit_key,
                placeholder="錠、回点眼など",
                label_visibility="collapsed",
                width=150,
            )
            st.button(
                "×",
                key=f"medicine_remove_{entry_id}",
                help="このお薬を削除",
                width=40,
                on_click=remove_medicine_entry,
                args=(entry_id,),
            )

            medicine["name"] = name
            medicine["amount"] = amount
            medicine["unit"] = unit

    st.button("＋ お薬を追加", on_click=add_medicine_entry, use_container_width=True)
    return medicines


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
    if progress["result"] is None:
        status_placeholder.empty()
        raise AppError("処理結果を取得できませんでした。再度実行してください。")

    status_placeholder.success(f"完了しました。処理時間: {elapsed:.1f}秒")
    return (*progress["result"], elapsed)


def main():
    st.set_page_config(page_title="お薬手帳OCR", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #f5f8fb; }
        .block-container {
            max-width: 1240px;
            height: 100vh;
            overflow: hidden;
            position: relative;
            padding-top: 1.25rem;
            padding-bottom: .5rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        .app-hero {
            padding: .8rem 1.35rem;
            border-radius: 14px;
            color: white;
            background: linear-gradient(120deg, #135f70 0%, #168a8d 100%);
            box-shadow: 0 10px 30px rgba(19, 95, 112, 0.16);
            margin-top: .35rem;
            margin-bottom: .65rem;
        }
        .app-hero h1 { margin: 0; font-size: 1.65rem; font-weight: 750; }
        /* st.htmlにはst.markdownの見出し余白と下方向の詰め(-1rem)が付かないため、同じ見た目になるよう補う。 */
        .app-hero h1 { padding: 1.25rem 0 1rem; line-height: 1.2; }
        .app-hero {
            margin-bottom: -.35rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }
        /* 上部ヘッダーは消し、サイドバーを閉じたときはタイトル帯の設定ボタンから開く。 */
        header[data-testid="stHeader"] { display: none; }
        .app-hero-settings {
            flex: 0 0 auto;
            padding: .4rem .95rem;
            border: 1px solid rgba(255, 255, 255, .55);
            border-radius: 8px;
            color: white;
            background: rgba(255, 255, 255, .14);
            font: inherit;
            font-weight: 700;
            cursor: pointer;
        }
        .app-hero-settings:hover { background: rgba(255, 255, 255, .26); }
        body:has(section[data-testid="stSidebar"][aria-expanded="true"]) .app-hero-settings {
            display: none;
        }
        .workflow-step {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 38px;
            padding: .35rem .6rem;
            border: 0;
            border-bottom: 3px solid #d5e2e5;
            border-radius: 0;
            color: #60777d;
            background: transparent;
            font-weight: 650;
        }
        .workflow-step.is-active {
            border-color: #168a8d;
            color: #135f70;
            background: linear-gradient(180deg, transparent 45%, rgba(22, 138, 141, .08) 100%);
        }
        .workflow-step.is-complete { border-color: #135f70; color: #135f70; }
        .section-heading { margin-top: .35rem; margin-bottom: .35rem; color: #183b45; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #dbe5ea;
            background: white;
            border-radius: 14px;
            box-shadow: 0 3px 12px rgba(30, 65, 77, .04);
        }
        div[data-testid="stTextInput"] label,
        div[data-testid="stNumberInput"] label { color: #435862; font-weight: 650; }
        div[class*="st-key-medicine_row_"] div[data-testid="stHorizontalBlock"] {
            align-items: center !important;
        }
        div[class*="st-key-medicine_row_"] div[data-testid="stColumn"] {
            align-self: center !important;
        }
        div[class*="st-key-medicine_row_"] div[data-testid="stButton"] button {
            height: 40px;
            min-height: 40px;
            padding-top: 0;
            padding-bottom: 0;
        }
        div[class*="st-key-medicine_row_"] div[data-baseweb="input"] {
            min-height: 40px;
        }
        /* 下部に固定したページ移動ボタンと重ならない高さにし、はみ出した分はスクロールさせる。 */
        div[class*="st-key-app_page_"] {
            height: calc(100vh - 260px);
            flex: 0 0 auto;
            overflow-y: auto;
            padding-bottom: .5rem;
        }
        div.st-key-page_navigation {
            position: absolute;
            left: 1.5rem;
            right: auto;
            bottom: .8rem;
            width: calc(100% - 3rem) !important;
            max-width: calc(100% - 3rem) !important;
            box-sizing: border-box;
            z-index: 1000;
        }
        div.st-key-page_navigation div[data-testid="stHorizontalBlock"] {
            width: 100%;
            box-sizing: border-box;
        }
        div.st-key-page_navigation button {
            height: 46px;
            border: 0;
            color: white;
            font-weight: 700;
            background: linear-gradient(120deg, #135f70 0%, #168a8d 100%);
            box-shadow: 0 5px 16px rgba(19, 95, 112, .2);
        }
        div.st-key-page_navigation button:disabled {
            color: #7e8e96;
            background: #e3eaee;
            box-shadow: none;
        }
        .qr-panel {
            padding: .85rem 1rem;
            border-left: 4px solid #168a8d;
            border-radius: 8px;
            background: #edfafa;
            color: #174f56;
        }
        .qr-patient-id {
            width: 280px;
            line-height: 1.6;
            text-align: center;
            color: #183b45;
            font-size: 1.1rem;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        @media (max-width: 760px) {
            .workflow-step { min-height: 46px; padding: .5rem; }
            .workflow-step strong { display: none; }
            .app-hero h1 { font-size: 1.55rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.html(
        """
        <div class="app-hero">
            <h1>お薬手帳 OCR</h1>
            <button class="app-hero-settings" type="button">⚙ 設定</button>
        </div>
        <script>
        // 再描画でスクリプトが再実行されても、クリック処理は1回だけ登録する。
        if (!window.medicineOcrSettingsBound) {
            window.medicineOcrSettingsBound = true;
            document.addEventListener("click", (event) => {
                if (event.target.closest(".app-hero-settings")) {
                    document.querySelector('[data-testid="stExpandSidebarButton"]')?.click();
                }
            });
        }
        </script>
        """,
        unsafe_allow_javascript=True,
    )

    if "summary_text" not in st.session_state:
        st.session_state.summary_text = ""
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "xml_text" not in st.session_state:
        st.session_state.xml_text = ""
    if "raw_summary_text" not in st.session_state:
        st.session_state.raw_summary_text = ""
    if "medicine_entries" not in st.session_state:
        st.session_state.medicine_entries = []
    if "medicine_next_id" not in st.session_state:
        st.session_state.medicine_next_id = 0
    if "qr_text" not in st.session_state:
        st.session_state.qr_text = ""
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "processing_seconds" not in st.session_state:
        st.session_state.processing_seconds = None
    if "input_image_png" not in st.session_state:
        st.session_state.input_image_png = b""
    if "input_image_resized" not in st.session_state:
        st.session_state.input_image_resized = False
    if "input_image_original_size" not in st.session_state:
        st.session_state.input_image_original_size = None
    if "input_image_fingerprint" not in st.session_state:
        st.session_state.input_image_fingerprint = ""
    if "llm_api_url" not in st.session_state:
        st.session_state.llm_api_url = DEFAULT_OPENAI_BASE_URL
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = DEFAULT_OPENAI_API_KEY
    if "llm_model_name" not in st.session_state:
        st.session_state.llm_model_name = DEFAULT_OPENAI_MODEL_NAME
    if "input_widget_version" not in st.session_state:
        st.session_state.input_widget_version = 0
    if "patient_id" not in st.session_state:
        st.session_state.patient_id = ""
    if "include_patient_id_in_qr" not in st.session_state:
        st.session_state.include_patient_id_in_qr = False
    if "log_record_id" not in st.session_state:
        st.session_state.log_record_id = ""
    if "last_logged_qr_signature" not in st.session_state:
        st.session_state.last_logged_qr_signature = None
    if "log_warning" not in st.session_state:
        st.session_state.log_warning = ""
    persist_widget_values()

    current_page = int(st.session_state.current_page)
    render_workflow_steps(current_page)

    with st.sidebar:
        st.header("設定")
        st.subheader("OCR")
        device = st.selectbox("OCRデバイス", ["cpu", "cuda"], index=0)
        st.divider()
        st.subheader("LLM API")
        api_url = st.text_input(
            "API URL",
            key="llm_api_url",
            placeholder="http://127.0.0.1:8080/v1",
            value = st.session_state.llm_api_url
        )
        api_key = st.text_input(
            "API Key",
            key="llm_api_key",
            type="password",
            placeholder="OpenAI互換APIのキー",
            value = st.session_state.llm_api_key
        )
        model_name = st.text_input(
            "Model Name",
            key="llm_model_name",
            placeholder="モデル名",
            value = st.session_state.llm_model_name
        )
        with st.expander("詳細データ・接続情報"):
            st.download_button(
                "OCR生テキストを保存",
                data=st.session_state.ocr_text.encode("utf-8"),
                file_name="ocr_text.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.download_button(
                "OCR XMLを保存",
                data=st.session_state.xml_text.encode("utf-8"),
                file_name="ocr_result.xml",
                mime="application/xml",
                use_container_width=True,
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

    can_go_next = False
    with st.container(border=False, key=f"app_page_{current_page}"):
        if st.session_state.log_warning:
            st.warning(st.session_state.log_warning)
            st.session_state.log_warning = ""

        if current_page == 1:
            st.markdown("<h2 class='section-heading'>画像を選ぶ</h2>", unsafe_allow_html=True)
            input_col, preview_col = st.columns([1, 1], vertical_alignment="top")

            with input_col:
                with st.container(border=True):
                    input_method = st.radio(
                        "画像の入力方法",
                        ["カメラで撮影", "ファイルを選ぶ"],
                        index=1,
                        horizontal=True,
                        key="input_method",
                    )
                    camera_image = None
                    uploaded_file = None
                    input_widget_version = st.session_state.input_widget_version
                    if input_method == "カメラで撮影":
                        camera_image = st.camera_input(
                            "お薬手帳を撮影",
                            label_visibility="collapsed",
                            key=f"camera_input_{input_widget_version}",
                        )
                    else:
                        uploaded_file = st.file_uploader(
                            "お薬手帳の画像を選択",
                            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                            key=f"image_uploader_{input_widget_version}",
                        )

                with st.container(border=True):
                    restore_widget_value("patient_id_input", "patient_id")
                    st.text_input(
                        "患者ID（任意）",
                        key="patient_id_input",
                    )
                    st.session_state.patient_id = st.session_state.patient_id_input

            source_file = camera_image or uploaded_file
            preview_image = None
            image_error = None
            if source_file is not None:
                source_bytes = source_file.getvalue()
                source_fingerprint = hashlib.sha256(source_bytes).hexdigest()
                if source_fingerprint != st.session_state.input_image_fingerprint:
                    clear_analysis_results()
                try:
                    prepared_image = prepare_input_image(source_bytes)
                    preview_image = prepared_image.image
                    st.session_state.input_image_png = image_to_png_bytes(preview_image)
                    st.session_state.input_image_resized = prepared_image.resized
                    st.session_state.input_image_original_size = prepared_image.original_size
                    st.session_state.input_image_fingerprint = source_fingerprint
                except ImageInputError as exc:
                    image_error = str(exc)
                    st.session_state.input_image_png = b""
                    st.session_state.input_image_resized = False
                    st.session_state.input_image_original_size = None
                    st.session_state.input_image_fingerprint = ""
            elif st.session_state.input_image_png:
                try:
                    with Image.open(io.BytesIO(st.session_state.input_image_png)) as saved_image:
                        preview_image = saved_image.convert("RGB")
                except (UnidentifiedImageError, OSError, ValueError):
                    st.session_state.input_image_png = b""
                    st.session_state.input_image_resized = False
                    st.session_state.input_image_original_size = None
                    st.session_state.input_image_fingerprint = ""

            with preview_col:
                with st.container(border=True):
                    st.markdown("**読み取る画像**")
                    if image_error:
                        st.error(image_error)
                    elif preview_image is not None:
                        display_image = preview_image.copy()
                        display_image.thumbnail((680, 320))
                        st.image(display_image, caption="この画像を読み取ります")
                        if st.session_state.input_image_resized:
                            original_width, original_height = st.session_state.input_image_original_size
                            st.caption(
                                f"読み取り用に画像を調整しました: "
                                f"{original_width}×{original_height} → "
                                f"{preview_image.width}×{preview_image.height}"
                            )
                    else:
                        st.info("画像がここに表示されます。")

            _, action_col, _ = st.columns([1, 2.2, 1])
            with action_col:
                if st.button(
                    "OCR・AI解析を実行",
                    type="primary",
                    disabled=preview_image is None,
                    use_container_width=True,
                ):
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
                    clear_analysis_results()
                    try:
                        ocr_text, xml_text, summary_text, processing_seconds = run_pipeline(
                            preview_image,
                            device,
                            api_url.strip(),
                            api_key.strip(),
                            model_name.strip(),
                            status_placeholder,
                        )
                    except AppError as exc:
                        status_placeholder.empty()
                        st.error(str(exc))
                    except Exception:
                        LOGGER.exception("Unexpected pipeline failure")
                        status_placeholder.empty()
                        st.error("処理中に予期しないエラーが発生しました。再度実行してください。")
                    else:
                        st.session_state.ocr_text = ocr_text
                        st.session_state.xml_text = xml_text
                        st.session_state.raw_summary_text = summary_text
                        st.session_state.summary_text = summary_text
                        st.session_state.processing_seconds = processing_seconds
                        replace_medicine_entries(summary_text)
                        sync_medicine_entries_from_widgets()
                        st.session_state.log_record_id = uuid.uuid4().hex
                        write_reading_log(
                            {
                                "event": "analysis",
                                "record_id": st.session_state.log_record_id,
                                "patient_id": get_patient_id(),
                                "ocr_text": ocr_text,
                                "extracted_text_raw": summary_text,
                                "model_name": model_name.strip(),
                                "processing_seconds": round(processing_seconds, 3),
                                "image_sha256": st.session_state.input_image_fingerprint,
                            }
                        )
                        st.session_state.current_page = 2
                        st.rerun()

            can_go_next = bool(st.session_state.ocr_text or st.session_state.raw_summary_text)

        elif current_page == 2:
            st.markdown("<h2 class='section-heading'>内容を確認</h2>", unsafe_allow_html=True)
            if st.session_state.processing_seconds is not None:
                st.caption(f"解析時間: {st.session_state.processing_seconds:.1f}秒")

            with st.expander("読み取り文章全体", expanded=False):
                st.text_area(
                    "OCR読み取り結果",
                    value=st.session_state.ocr_text,
                    height=180,
                    disabled=True,
                    label_visibility="collapsed",
                )

            st.markdown("**お薬情報**")
            with st.container(
                key="medicine_column_header",
                horizontal=True,
                vertical_alignment="center",
                gap="small",
            ):
                st.caption("お薬名", width="stretch")
                st.caption("1日の数量・単位／用法", width=426)
            with st.container(height=320, border=False):
                medicines = render_medicine_editor()

            st.session_state.summary_text = build_medicine_text(medicines)
            st.session_state.qr_text = st.session_state.summary_text.strip()
            medicines_are_valid = medicine_entries_are_valid(medicines)
            if medicines and not medicines_are_valid:
                st.warning("薬名と単位・用法を入力してください。")
            can_go_next = medicines_are_valid

        else:
            sync_medicine_entries_from_widgets()
            medicine_text = st.session_state.qr_text
            patient_id = get_patient_id()
            st.markdown("<h2 class='section-heading'>QRを表示</h2>", unsafe_allow_html=True)
            if st.session_state.processing_seconds is not None:
                st.caption(f"解析時間: {st.session_state.processing_seconds:.1f}秒")

            if medicine_text:
                qr_col, detail_col = st.columns([1, 2], vertical_alignment="center")
                with detail_col:
                    # トグルの値でQRの内容が決まるため、上に置く要素は後から描画する枠だけ先に確保する。
                    status_slot = st.container()
                    content_slot = st.container()
                    restore_widget_value("include_patient_id_toggle", "include_patient_id_in_qr")
                    st.toggle(
                        "QRコードに患者IDを含める",
                        key="include_patient_id_toggle",
                        disabled=not patient_id,
                        help="オンにすると、QRコードの先頭行に患者IDを入れます。",
                    )
                    st.session_state.include_patient_id_in_qr = st.session_state.include_patient_id_toggle
                    download_slot = st.container()
                include_patient_id = bool(st.session_state.include_patient_id_in_qr and patient_id)
                current_summary = f"{patient_id}\n{medicine_text}" if include_patient_id else medicine_text

                try:
                    qr_image = make_qr_image(current_summary)
                except QrGenerationError as exc:
                    qr_image = None
                    with status_slot:
                        st.error(str(exc))
                    with download_slot:
                        st.download_button(
                            "テキストを保存",
                            data=current_summary.encode("utf-8"),
                            file_name="medicine_summary.txt",
                            mime="text/plain",
                        )
                else:
                    with qr_col:
                        st.image(qr_image, caption="お薬情報のQRコード", width=280)
                        if patient_id:
                            st.markdown(
                                f"<div class='qr-patient-id'>患者ID: {escape(patient_id)}</div>",
                                unsafe_allow_html=True,
                            )
                    with status_slot:
                        st.markdown(
                            "<div class='qr-panel'><strong>QRコードの準備ができました</strong></div>",
                            unsafe_allow_html=True,
                        )
                    with content_slot:
                        st.text_area(
                            "QRコードに含まれる内容",
                            current_summary,
                            height=150,
                            disabled=True,
                        )
                    with download_slot:
                        download_qr_col, download_text_col = st.columns(2)
                        with download_qr_col:
                            st.download_button(
                                "QRコードを保存",
                                data=image_to_png_bytes(qr_image),
                                file_name="medicine_qr.png",
                                mime="image/png",
                                use_container_width=True,
                            )
                        with download_text_col:
                            st.download_button(
                                "テキストを保存",
                                data=current_summary.encode("utf-8"),
                                file_name="medicine_summary.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )

                # 再描画のたびに記録しないよう、内容が変わったときだけ記録する。
                qr_log_signature = (st.session_state.log_record_id, patient_id, current_summary)
                if qr_log_signature != st.session_state.last_logged_qr_signature:
                    if write_reading_log(
                        {
                            "event": "qr",
                            "record_id": st.session_state.log_record_id,
                            "patient_id": patient_id,
                            "extracted_text_edited": medicine_text,
                            "qr_text": current_summary,
                            "patient_id_in_qr": include_patient_id,
                            "qr_generated": qr_image is not None,
                        }
                    ):
                        st.session_state.last_logged_qr_signature = qr_log_signature
                    elif st.session_state.log_warning:
                        st.warning(st.session_state.log_warning)
                        st.session_state.log_warning = ""

    render_page_navigation(current_page, can_go_next)


if __name__ == "__main__":
    main()
