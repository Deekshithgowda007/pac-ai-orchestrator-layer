from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import Json
import uuid
import time
import os
import pydicom
import requests

from pynetdicom import AE

app = FastAPI()

# -----------------------
# PACS CONFIG
# -----------------------

PACS_HOST = "dcm4chee-arc"
PACS_PORT = 11112
PACS_AET = "DCM4CHEE"
# STOW-RS endpoint of the same archive: used only when the archive does not accept the
# file's own transfer syntax over C-STORE (dcm4chee's stock C-STORE capability omits, for
# example, Deflated and Explicit VR Big Endian, while its STOW-RS stores both as received).
PACS_STOW_URL = os.getenv(
    "PACS_STOW_URL", "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs/studies")


def stow_original_bytes(payload: bytes):
    """STOW-RS the uploaded file exactly as received. Returns (stored, detail).

    The bytes are the multipart part verbatim: nothing is parsed, re-encoded or relabelled.
    Only HTTP 200 counts as stored. 202 means the archive stored with warnings or coerced
    something, and 409 means it refused; neither is reported as success.
    """
    boundary = "scansure-" + uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Type: application/dicom\r\n\r\n".encode()
            + payload + f"\r\n--{boundary}--\r\n".encode())
    try:
        response = requests.post(
            PACS_STOW_URL, data=body, timeout=120,
            headers={"Content-Type": f'multipart/related; type="application/dicom"; boundary={boundary}',
                     "Accept": "application/dicom+json"})
    except requests.RequestException as exc:
        return False, f"STOW-RS request failed: {type(exc).__name__}"
    return response.status_code == 200, f"STOW-RS HTTP {response.status_code}"

# -----------------------
# CORS
# -----------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# WAIT FOR POSTGRES
# -----------------------

while True:
    try:
        conn = psycopg2.connect(
            host="scansure-postgres",
            database="scansure",
            user="postgres",
            password="postgres"
        )
        print("Connected to ScanSure DB")
        break
    except Exception as e:
        print("Waiting for postgres...", e)
        time.sleep(3)

cursor = conn.cursor()

# -----------------------
# ONBOARD
# -----------------------

@app.post("/onboard")
def onboard(data: dict):

    hospital_id = str(uuid.uuid4())

    cursor.execute(
        """
        INSERT INTO hospitals
        (id,name,city,email,password,phone,modalities)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            hospital_id,
            data["name"],
            data["city"],
            data["email"],
            data["password"],
            data["phone"],
            data["modalities"]
        )
    )

    conn.commit()

    return {"hospital_id": hospital_id}


# -----------------------
# LOGIN
# -----------------------

@app.post("/login")
def login(data: dict):

    cursor.execute(
        "SELECT id FROM hospitals WHERE email=%s AND password=%s",
        (data["email"], data["password"])
    )

    result = cursor.fetchone()

    if result:
        return {"hospital_id": result[0]}

    return {"error": "invalid login"}


# -----------------------
# UPLOAD DICOM
# -----------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    temp_path = f"/tmp/{file.filename}"

    try:

        # Save uploaded file
        payload = await file.read()
        with open(temp_path, "wb") as f:
            f.write(payload)

        ds = pydicom.dcmread(temp_path)

        print("DICOM Modality:", getattr(ds, "Modality", "Unknown"))
        print("StudyInstanceUID:", getattr(ds, "StudyInstanceUID", "Unknown"))

        # Offer the archive exactly one presentation context: the file's own SOP class in
        # the file's own transfer syntax. With no alternative on the table the C-STORE
        # either stores the object as it was received or is refused; it can never be
        # converted on the way (offering the default syntaxes let the archive pick
        # Implicit VR LE, and compressed files had no context at all).
        sop_class = getattr(ds, "SOPClassUID", None)
        transfer_syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        if not sop_class or not transfer_syntax:
            os.remove(temp_path)
            return {"error": "DICOM file has no SOPClassUID or no TransferSyntaxUID; "
                             "refusing to guess how to send it"}

        ae = AE(ae_title="SCANSURE")
        ae.add_requested_context(sop_class, [transfer_syntax])

        print(f"Connecting to PACS {PACS_HOST}:{PACS_PORT} ...")

        assoc = ae.associate(
    PACS_HOST,
    PACS_PORT,
    ae_title=PACS_AET
)

        if assoc.is_established:

            print("Association established with PACS")

            status = assoc.send_c_store(ds)

            print("CSTORE STATUS:", status)

            assoc.release()

            os.remove(temp_path)

            return {
                "message": "DICOM successfully sent to PACS",
                "status": str(status)
            }

        else:
            print("Association failed")

            if os.path.exists(temp_path):
                os.remove(temp_path)

            # No association: the archive is unreachable, or it does not accept this
            # transfer syntax over C-STORE. Offering a different syntax would convert the
            # file, so the original bytes go by STOW-RS instead, or nothing is stored.
            stored, detail = stow_original_bytes(payload)
            if stored:
                return {
                    "message": "DICOM successfully sent to PACS",
                    "status": detail,
                    "transport": "stow-rs",
                }

            return {"error": "PACS did not accept an association offering only "
                             f"{sop_class.name} in {transfer_syntax.name} ({transfer_syntax}), "
                             f"and the STOW-RS fallback did not store it ({detail}). "
                             "Nothing was stored; a file is never converted on upload."}

    except Exception as e:

        print("Upload error:", str(e))

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {"error": str(e)}


@app.post("/results")
def store_result(data: dict):
    report_id = data.get("id") or str(uuid.uuid4())

    cursor.execute(
        """
        INSERT INTO ai_reports
        (id, study_uid, series_uid, sop_uid, filename, modality, body_part, status, model_id, summary, report, dicom_metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            report_id,
            data.get("study_uid"),
            data.get("series_uid"),
            data.get("sop_uid"),
            data.get("filename"),
            data.get("modality"),
            data.get("body_part"),
            data.get("status"),
            data.get("model_id"),
            data.get("summary"),
            Json(data.get("report", {})),
            Json(data.get("metadata", {})),
        )
    )
    conn.commit()
    return {"ok": True, "id": report_id}


@app.get("/results/{study_uid}")
def get_results(study_uid: str):
    cursor.execute(
        """
        SELECT id, study_uid, series_uid, sop_uid, filename, modality, body_part, status, model_id, summary, report, dicom_metadata, created_at
        FROM ai_reports
        WHERE study_uid = %s
        ORDER BY created_at DESC
        """,
        (study_uid,)
    )
    rows = cursor.fetchall()
    return {
        "study_uid": study_uid,
        "results": [
            {
                "id": row[0],
                "study_uid": row[1],
                "series_uid": row[2],
                "sop_uid": row[3],
                "filename": row[4],
                "modality": row[5],
                "body_part": row[6],
                "status": row[7],
                "model_id": row[8],
                "summary": row[9],
                "report": row[10],
                "metadata": row[11],
                "created_at": row[12],
            }
            for row in rows
        ]
    }
