from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ROOT_DIR = Path(__file__).resolve().parents[1]
NDLOCR_LITE_DIR = Path(os.getenv("NDLOCR_LITE_DIR", str(ROOT_DIR / "ndlocr-lite"))).resolve()
NDLOCR_SRC_DIR = NDLOCR_LITE_DIR / "src"
NDLOCR_PYTHON = os.getenv("NDLOCR_PYTHON", sys.executable)


def positive_env_number(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


NDLOCR_TIMEOUT = positive_env_number("NDLOCR_TIMEOUT", 900.0)
MAX_OCR_UPLOAD_BYTES = int(positive_env_number("MAX_OCR_UPLOAD_BYTES", 50 * 1024 * 1024))
UPLOAD_CHUNK_SIZE = 1024 * 1024

app = FastAPI(title="NDLOCR-Lite subprocess API")


@app.get("/health")
def health() -> dict[str, str]:
    ocr_script = NDLOCR_SRC_DIR / "ocr.py"
    return {
        "status": "ok" if ocr_script.exists() else "degraded",
        "ocr_script": "available" if ocr_script.exists() else "missing",
    }


def save_upload_file(upload: UploadFile, destination: Path) -> int:
    total_size = 0
    try:
        with destination.open("wb") as output:
            while chunk := upload.file.read(UPLOAD_CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_OCR_UPLOAD_BYTES:
                    limit_mb = max(
                        1,
                        (MAX_OCR_UPLOAD_BYTES + 1024 * 1024 - 1) // (1024 * 1024),
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"uploaded image exceeds {limit_mb}MB",
                    )
                output.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail="failed to save uploaded image") from exc

    if total_size == 0:
        raise HTTPException(status_code=400, detail="uploaded image is empty")
    return total_size


@app.post("/ocr")
def ocr_image(
    file: UploadFile = File(...),
    device: str = Form("cpu"),
) -> dict[str, Any]:
    if device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be 'cpu' or 'cuda'")
    if not (NDLOCR_SRC_DIR / "ocr.py").exists():
        raise HTTPException(status_code=500, detail=f"ocr.py not found: {NDLOCR_SRC_DIR / 'ocr.py'}")

    with tempfile.TemporaryDirectory(prefix="ndlocr_api_") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "medicine_notebook.png"
        output_dir = temp_path / "output"
        output_dir.mkdir()

        save_upload_file(file, input_path)

        command = [
            NDLOCR_PYTHON,
            "ocr.py",
            "--sourceimg",
            str(input_path),
            "--output",
            str(output_dir),
            "--device",
            device,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(NDLOCR_SRC_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=NDLOCR_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail=f"NDLOCR-Lite timed out after {NDLOCR_TIMEOUT}s") from exc

        if completed.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "NDLOCR-Lite subprocess failed",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )

        stem = input_path.stem
        txt_path = output_dir / f"{stem}.txt"
        xml_path = output_dir / f"{stem}.xml"
        json_path = output_dir / f"{stem}.json"

        if not txt_path.exists():
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "NDLOCR-Lite did not produce text output",
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )

        json_data: dict[str, Any] | None = None
        if json_path.exists():
            try:
                parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="NDLOCR-Lite produced invalid JSON output",
                ) from exc
            if isinstance(parsed_json, dict):
                json_data = parsed_json

        try:
            ocr_text = txt_path.read_text(encoding="utf-8", errors="replace")
            xml_text = (
                xml_path.read_text(encoding="utf-8", errors="replace")
                if xml_path.exists()
                else ""
            )
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="failed to read NDLOCR-Lite output",
            ) from exc

        return {
            "ocr_text": ocr_text,
            "xml_text": xml_text,
            "json": json_data,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
