from pacs_client import PACSClient
from dicom_utils import extract_series_info
from ai_router import route_to_ai_models

def process_study(message):
    study_uid = message["study_uid"]

    pacs = PACSClient(
        base_url=PACS_BASE_URL,
        aet=PACS_AET,
        user=PACS_USER,
        password=PACS_PASS
    )

    study_metadata = pacs.get_study_metadata(study_uid)
    series_info = extract_series_info(study_metadata)

    for series in series_info:
        models = route_to_ai_models(series["modality"])

        print(
            f"🧠 Study {study_uid} | "
            f"Series {series['series_uid']} | "
            f"Modality {series['modality']} | "
            f"Models {models}"
        )

        # Store routing decision (DB / Supabase)
