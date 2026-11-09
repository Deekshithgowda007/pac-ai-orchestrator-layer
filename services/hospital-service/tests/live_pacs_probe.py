"""Live probe: POST /upload against a RUNNING dcm4chee, then ask the archive what it holds.

Not collected by pytest (needs an archive). Run it from services/hospital-service:

    PACS_HOST=arc PACS_RS=http://arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs python tests/live_pacs_probe.py

MAIN_PY may point at another copy of main.py (for example the pre-fix file) to compare.
For every fixture it prints what /upload answered and, if the archive stored the instance,
whether the stored copy has the original TransferSyntaxUID and decodes to the original pixels.
Exit 0 only if EVERY fixture is stored with its original TransferSyntaxUID and pixels.
Exit 1 if a stored instance differs from what was uploaded; exit 2 if nothing differs but
some fixture was not stored at all (a refusal is safe, but it is not a pass).
"""
import importlib.util
import io
import json
import os
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pydicom
import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dicom_fixtures import FIXTURES, build, pixel_sha256  # noqa: E402

RS = os.environ["PACS_RS"].rstrip("/")
MAIN_PY = os.environ.get("MAIN_PY", str(HERE.parent / "main.py"))

fake = types.ModuleType("psycopg2")
fake.connect = lambda **_: MagicMock()
extras = types.ModuleType("psycopg2.extras")
extras.Json = lambda value: value
fake.extras = extras
sys.modules["psycopg2"], sys.modules["psycopg2.extras"] = fake, extras

spec = importlib.util.spec_from_file_location("main_under_probe", MAIN_PY)
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
main.PACS_HOST = os.environ["PACS_HOST"]
if hasattr(main, "PACS_STOW_URL"):  # absent in a pre-fix main.py given via MAIN_PY
    main.PACS_STOW_URL = RS + "/studies"

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)


def stored_copy(ds):
    url = (f"{RS}/studies/{ds.StudyInstanceUID}/series/{ds.SeriesInstanceUID}"
           f"/instances/{ds.SOPInstanceUID}")
    # transfer-syntax=* : without it the archive transcodes to Explicit VR LE on retrieve
    response = requests.get(
        url, headers={"Accept": 'multipart/related; type="application/dicom"; transfer-syntax=*'},
        timeout=60)
    if response.status_code != 200:
        return None
    boundary = [part.split("=", 1)[1].strip('"') for part in response.headers["Content-Type"].split(";")
                if part.strip().startswith("boundary=")][0]
    payload = response.content.split(b"--" + boundary.encode())[1].split(b"\r\n\r\n", 1)[1]
    return pydicom.dcmread(io.BytesIO(payload[:-2] if payload.endswith(b"\r\n") else payload))


rows, damaged, refused = [], False, []
for name in sorted(FIXTURES):
    blob, _ = build(name)
    original = pydicom.dcmread(io.BytesIO(blob))
    answer = client.post("/upload", files={"file": (f"{name}.dcm", blob, "application/dicom")}).json()
    row = {"fixture": name, "uploaded_ts": str(original.file_meta.TransferSyntaxUID),
           "upload_answer": {key: str(value)[:160] for key, value in answer.items()}}
    stored = stored_copy(original)
    row["stored_in_archive"] = stored is not None
    if stored is None:
        refused.append(name)
    if stored is not None:
        row["stored_ts"] = str(stored.file_meta.TransferSyntaxUID)
        try:
            row["pixels_identical"] = pixel_sha256(stored) == pixel_sha256(original)
        except Exception as exc:  # an undecodable stored copy is the corruption itself
            row["pixels_identical"] = False
            row["decode_error"] = f"{type(exc).__name__}: {exc}"[:200]
        row["ts_identical"] = row["stored_ts"] == row["uploaded_ts"]
        damaged = damaged or not (row["pixels_identical"] and row["ts_identical"])
    rows.append(row)

print(json.dumps(rows, indent=2))
if not damaged and refused:
    print("RESULT: not stored:", ", ".join(refused))
    sys.exit(2)
print("RESULT:", "a stored instance DIFFERS from what was uploaded" if damaged
      else "every stored instance matches what was uploaded")
sys.exit(1 if damaged else 0)
