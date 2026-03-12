from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import uuid
import time
import os
import pydicom

from pynetdicom import AE, StoragePresentationContexts
from pydicom.uid import ExplicitVRLittleEndian

app = FastAPI()

# -----------------------
# PACS CONFIG
# -----------------------

PACS_HOST = "dcm4chee-arc"
PACS_PORT = 11112
PACS_AET = "DCM4CHEE"

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
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        ds = pydicom.dcmread(temp_path)

        print("DICOM Modality:", getattr(ds, "Modality", "Unknown"))
        print("StudyInstanceUID:", getattr(ds, "StudyInstanceUID", "Unknown"))

        # Convert to compatible transfer syntax
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ae = AE(ae_title="SCANSURE")

        for context in StoragePresentationContexts:
            ae.add_requested_context(context.abstract_syntax)

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

            return {"error": "PACS association failed"}

    except Exception as e:

        print("Upload error:", str(e))

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {"error": str(e)}