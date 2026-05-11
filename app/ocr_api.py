from __future__ import annotations

import json
import os
import shutil
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
NDLOCR_TIMEOUT = float(os.getenv("NDLOCR_TIMEOUT", "900"))

app = FastAPI(title="NDLOCR-Lite subprocess API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_image(
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

        with input_path.open("wb") as wf:
            shutil.copyfileobj(file.file, wf)

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
            json_data = json.loads(json_path.read_text(encoding="utf-8"))

        return {
            "ocr_text": txt_path.read_text(encoding="utf-8"),
            "xml_text": xml_path.read_text(encoding="utf-8") if xml_path.exists() else "",
            "json": json_data,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
