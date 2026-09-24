# お薬手帳OCRアプリ

`ndlocr-lite` を subprocess として実行する FastAPI OCR サーバーと、OpenAI 互換 API に対する1段階抽出で薬名と1日の服用量を整形する Streamlit アプリです。
低スペックなノートパソコンなどのローカル環境での動作を目的としています。

## 起動

※venv環境で、同ディレクトリにgit cloneした場合を想定しています。

依存関係をインストールします。

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

ターミナル1で OCR API を起動します。

```bash
source .venv/bin/activate
uvicorn app.ocr_api:app --host 127.0.0.1 --port 8001
```

ターミナル2で Streamlit を起動します。

```bash
source .venv/bin/activate
streamlit run app/app.py
```

## 前提

- OpenAI 互換 API が利用できること
- Streamlit のサイドバーで `API URL`, `API Key`, `Model Name` を入力すること
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL_NAME` を設定すると、サイドバー入力欄の初期値として使われる
- OCR API のデフォルト接続先は `OCR_API_URL=http://127.0.0.1:8001`
- `ndlocr-lite` の配置先を変える場合は OCR API 側で `NDLOCR_LITE_DIR` を設定すること
- Streamlit はリポジトリ直下から起動すること（`.streamlit/config.toml` は起動したディレクトリから読み込まれる）

## システム構成

```mermaid
flowchart TB
    U["利用者の端末（ブラウザ）<br/>カメラ撮影 / 画像選択・患者ID入力<br/>お薬情報の編集・QR表示 / 保存"]

    subgraph Host["ローカルPC"]
        direction TB
        subgraph ST["Streamlit アプリ：app/app.py（既定 :8501）"]
            direction LR
            PRE["画像前処理<br/>EXIF回転補正・縮小（Pillow）"]
            UI["3ページUI<br/>画像を選ぶ → 内容を確認 → QRを表示"]
            QR["QRコード生成<br/>（qrcode）"]
            LOGW["読み取りログ書き込み"]
        end
        CFG[(".env / .streamlit/config.toml<br/>app/prompt.txt")]
        LOG[("log/YYYY-MM-DD.jsonl<br/>権限 0600")]
        subgraph API["OCR API：app/ocr_api.py（FastAPI + uvicorn、127.0.0.1:8001）"]
            EP["POST /ocr ・ GET /health"]
        end
        TMP[("一時ディレクトリ<br/>入力PNG・txt / xml / json")]
        subgraph NDL["NDLOCR-Lite：ndlocr-lite/src/ocr.py（subprocess）"]
            direction LR
            L1["レイアウト認識<br/>DEIMv2（ONNX）"] --> L2["文字列認識<br/>PARSeq（ONNX）"] --> L3["読み順整序<br/>xy_cut"]
        end
    end

    LLM["OpenAI互換 LLM API<br/>Responses API（OPENAI_BASE_URL）"]

    U <-->|"HTTP / WebSocket"| UI
    CFG -.->|設定の読み込み| ST
    UI --> PRE
    PRE -->|"PNG + device（multipart）"| EP
    EP -->|"ocr_text, xml_text"| UI
    EP -->|"python ocr.py --sourceimg ... --device cpu/cuda"| NDL
    EP --> TMP
    NDL --> TMP
    UI <-->|"prompt.txt + OCRテキスト ⇄ お薬抽出テキスト"| LLM
    UI --> QR
    UI --> LOGW --> LOG
```

```mermaid
sequenceDiagram
    autonumber
    actor User as 利用者（ブラウザ）
    participant App as Streamlit（app.py）
    participant OCR as OCR API（ocr_api.py）
    participant NDL as NDLOCR-Lite（ocr.py）
    participant LLM as OpenAI互換 LLM API
    participant Log as log/*.jsonl

    User->>App: 画像選択・患者ID入力
    App->>App: 画像前処理（EXIF補正・画素数上限で縮小）
    User->>App: 「OCR・AI解析を実行」
    App->>OCR: POST /ocr（PNG, device）
    OCR->>NDL: subprocess 実行（タイムアウト 900秒）
    NDL-->>OCR: txt / xml / json（一時ディレクトリ）
    OCR-->>App: ocr_text, xml_text
    App->>LLM: responses.create（prompt.txt + OCRテキスト、temperature=0）
    LLM-->>App: お薬抽出テキスト（[n] 薬名 1日量）
    App->>Log: event="analysis"（OCRテキスト・LLMの生出力 等）
    User->>App: 内容を確認・編集
    User->>App: 「次へ」
    App->>App: QRコード生成（トグルがオンなら先頭行に患者ID）
    App->>Log: event="qr"（編集後テキスト・QRの内容。内容が変わったときのみ）
    App-->>User: QR表示・PNG/テキストのダウンロード
    User->>App: 「新規」→ 状態を初期化して1ページ目へ
```

## 読み取りログ

読み取り結果は `log/YYYY-MM-DD.jsonl`（`LOG_DIR` で変更可）に1行1レコードの JSON Lines で追記されます。患者IDと服薬情報を含むため、ファイル権限は 0600 で作成し、Git の管理対象外にしています。

- `event: "analysis"`: OCR・AI解析の完了時。`timestamp`, `record_id`, `patient_id`, `ocr_text`, `extracted_text_raw`（LLMの生出力）, `model_name`, `processing_seconds`, `image_sha256`
- `event: "qr"`: QR表示時（内容が変わるたびに1行）。`timestamp`, `record_id`, `patient_id`, `extracted_text_edited`（編集後のお薬情報）, `qr_text`, `patient_id_in_qr`, `qr_generated`

同じ読み取りの2種類のレコードは `record_id` で結合できます（例: `pandas.read_json(path, lines=True)`, `jsonlite::stream_in(file(path))`）。

## ライセンス

本アプリはOCR処理にNDLOCR-Liteを利用しています。

NDLOCR-Lite: https://github.com/ndl-lab/ndlocr-lite

- License: CC BY 4.0
- Copyright: National Diet Library / ndl-lab
