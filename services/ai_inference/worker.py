# services/ai_inference/worker.py
import os, time, io, json, uuid, threading, traceback
from dotenv import load_dotenv
from supabase import create_client
import requests
import numpy as np
import pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import highdicom as hd
from skimage.filters import threshold_otsu
from PIL import Image

# Optional model APIs
import openai
from transformers import pipeline

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL/SUPABASE_KEY required")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

HF_API_KEY = os.getenv("HF_API_KEY")

PACS_QIDO = os.getenv("PACS_QIDO_URL","http://dcm4chee:8080/dcm4chee-arc/aets/DCM4CHEE/rs")
PACS_WADO = os.getenv("PACS_WADO_URL","http://dcm4chee:8080/dcm4chee-arc/aets/DCM4CHEE/rs")
PACS_STOW = os.getenv("PACS_STOW_URL","http://dcm4chee:8080/dcm4chee-arc/aets/DCM4CHEE/rs")

POLL_INTERVAL = int(os.getenv("AI_POLL_INTERVAL_SECONDS", "5"))
CONCURRENCY_PER_ENGINE = int(os.getenv("AI_CONCURRENCY_PER_ENGINE", "1"))

# parse engine instances configuration (ENV): "MRI:Head=2,CT:Chest=1"
ENGINE_INSTANCES_CONFIG = os.getenv("ENGINE_INSTANCES_CONFIG","")
_engine_counts = {}
if ENGINE_INSTANCES_CONFIG:
    for spec in ENGINE_INSTANCES_CONFIG.split(","):
        try:
            lhs, n = spec.split("=")
            modality, body_part = lhs.split(":")
            _engine_counts[(modality.strip(), body_part.strip())] = int(n)
        except Exception:
            continue

# helper HTTP functions to PACS DICOMweb
def qido_series(study_uid):
    url = f"{PACS_QIDO}/studies/{study_uid}/series"
    r = requests.get(url, headers={"Accept":"application/json"})
    r.raise_for_status()
    return r.json()

def qido_instances(study_uid, series_uid):
    url = f"{PACS_QIDO}/studies/{study_uid}/series/{series_uid}/instances"
    r = requests.get(url, headers={"Accept":"application/json"})
    r.raise_for_status()
    return r.json()

def wado_instance(study_uid, series_uid, sop_uid):
    url = f"{PACS_WADO}/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}"
    r = requests.get(url, headers={"Accept":"application/dicom"})
    r.raise_for_status()
    return pydicom.dcmread(io.BytesIO(r.content))

def stow_rs(ds):
    from email.generator import _make_boundary as make_boundary
    boundary = make_boundary()
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b"Content-Type: application/dicom\r\n\r\n")
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, write_like_original=False)
    body.write(buf.getvalue())
    body.write(f"\r\n--{boundary}--\r\n".encode())
    headers = {"Content-Type": f"multipart/related; type=application/dicom; boundary={boundary}"}
    url = f"{PACS_STOW}/studies"
    r = requests.post(url, headers=headers, data=body.getvalue())
    r.raise_for_status()
    return True

# Build a small local HF segmentation pipeline if available; else will fallback to Otsu
def hf_segment_slice(image_pil):
    if HF_API_KEY:
        try:
            seg_pipe = pipeline("image-segmentation", model="facebook/detr-resnet-50", use_auth_token=HF_API_KEY)
            return seg_pipe(image_pil)
        except Exception:
            return None
    return None

# Simple Otsu baseline segmentation (3D)
def simple_otsu_seg(volume: np.ndarray):
    vol = volume.astype(np.float32)
    if vol.max() > 0:
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    t = threshold_otsu(vol)
    mask = (vol >= t).astype(np.uint8)
    return mask

def make_dicom_seg(mask3d, ref_slices, algorithm_name="Auto"):
    first = ref_slices[0]
    seg_uid = generate_uid()
    seg_series_uid = generate_uid()
    seg = hd.seg.Segmentation(
        source_images=ref_slices,
        pixel_array=mask3d[np.newaxis, ...],  # (segments, z, y, x)
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[
            hd.seg.SegmentDescription(
                segment_number=1,
                segment_label="AI_Segment",
                segmented_property_category=hd.seg.codes.SCT.AnatomicalStructure,
                segmented_property_type=hd.seg.codes.SCT.Organ,
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_name=algorithm_name
            )
        ],
        series_instance_uid=seg_series_uid,
        series_number=999,
        sop_instance_uid=seg_uid,
        instance_number=1,
        manufacturer="KasettiTech",
        transfer_syntax_uid=ExplicitVRLittleEndian
    )
    seg.PatientName = getattr(first, "PatientName", "ANON")
    seg.PatientID = getattr(first, "PatientID", "ANON")
    seg.StudyInstanceUID = first.StudyInstanceUID
    seg.SeriesInstanceUID = seg_series_uid
    seg.FrameOfReferenceUID = first.FrameOfReferenceUID
    return seg

def generate_summary_openai(modality, body_part, brief_findings_text):
    if not OPENAI_API_KEY:
        return f"(no OPENAI_API_KEY) Summary for {modality}/{body_part}: {brief_findings_text}"
    try:
        prompt = f"""You are a radiology assistant. Given brief findings text: {brief_findings_text}
Return a concise radiologist-style impression (< 120 words). Modality: {modality}. Body part: {body_part}."""
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if os.getenv("USE_GPT4O") else "gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=250
        )
        return resp.choices[0].message['content'].strip()
    except Exception as e:
        return f"(openai error) {str(e)}"

def build_volume_and_series(study_uid, series_uid):
    insts = qido_instances(study_uid, series_uid)
    slices = []
    for it in insts:
        sop = it["00080018"]["Value"][0]
        ds = wado_instance(study_uid, series_uid, sop)
        if hasattr(ds, "PixelData"):
            slices.append(ds)
    def zpos(ds):
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            return float(getattr(ds, "InstanceNumber", 0))
    slices.sort(key=zpos)
    if not slices:
        raise RuntimeError("No pixel data")
    vol = np.stack([s.pixel_array for s in slices], axis=0)
    return slices, vol

def process_job(job):
    job_id = job["id"]
    print(f"[worker] processing job {job_id}")
    try:
        # mark processing and assign a dummy engine (we can choose best-fit engine)
        sb.table("inference_queue").update({"status":"processing","updated_at":"now()"}).eq("id", job_id).execute()

        # fetch candidate series (choose first series)
        series_list = qido_series(job["study_uid"])
        if not series_list:
            raise RuntimeError("No series found in PACS")

        target = series_list[0]
        series_uid = target["0020000E"]["Value"][0]
        ref_slices, volume = build_volume_and_series(job["study_uid"], series_uid)

        # 1) segmentation step
        # Try HF on middle slice, otherwise do full 3D Otsu
        algorithm_used = "Otsu3D"
        try:
            hf_result = None
            try:
                mid = volume.shape[0] // 2
                pil = Image.fromarray((volume[mid] / (volume[mid].max()+1e-6) * 255).astype('uint8')).convert('RGB')
                hf_result = hf_segment_slice(pil)
            except Exception:
                hf_result = None

            if hf_result:
                # For demo we create a simple binary mask from intensity heuristics
                mask3d = simple_otsu_seg(volume)  # converting HF result heavy -> fall back to Otsu mapping
                algorithm_used = "HuggingFace-demo->Otsu3D"
            else:
                mask3d = simple_otsu_seg(volume)
        except Exception as inner:
            print("segmentation failed:", inner)
            mask3d = simple_otsu_seg(volume)
            algorithm_used = "Otsu3D"

        seg = make_dicom_seg(mask3d, ref_slices, algorithm_name=algorithm_used)
        # POST SEG to PACS
        stow_rs(seg)

        # 2) text summary: produce a brief findings text and ask OpenAI for a radiology impression
        brief = f"Auto segmentation generated with algorithm {algorithm_used}; volume shape {volume.shape}"
        openai_summary = generate_summary_openai(job.get("modality",""), job.get("body_part",""), brief)

        # store result record
        res = {
            "queue_id": job_id,
            "ai_model_id": job.get("ai_model_id"),
            "summary": openai_summary,
            "seg_sop_instance_uid": seg.SOPInstanceUID,
            "seg_series_instance_uid": seg.SeriesInstanceUID
        }
        sb.table("inference_results").insert(res).execute()

        # update queue row to completed
        sb.table("inference_queue").update({"status":"completed","updated_at":"now()"}).eq("id", job_id).execute()

        # update inference_logs if present
        sb.table("inference_logs").update({"status":"completed", "error_message": None}).eq("study_uid", job["study_uid"]).execute()

        print(f"[worker] job {job_id} completed")
    except Exception as e:
        traceback.print_exc()
        # mark failed
        sb.table("inference_queue").update({"status":"failed", "error": str(e), "updated_at":"now()"}).eq("id", job_id).execute()
        sb.table("inference_logs").update({"status":"failed", "error_message": str(e)}).eq("study_uid", job["study_uid"]).execute()

# Poll loop: pick highest priority pending job and claim it (atomic-ish)
def claim_and_process():
    # Use a simple approach: select first pending by priority then attempt update to processing with filter to ensure not raced.
    sel = sb.table("inference_queue").select("*").eq("status","pending").order("priority", {"ascending": True}).order("created_at", {"ascending": True}).limit(1).execute()
    if not sel.data:
        return None
    job = sel.data[0]
    # attempt to claim: update status to processing only if still pending
    updated = sb.table("inference_queue").update({"status":"processing","updated_at":"now()"}).eq("id", job["id"]).eq("status","pending").execute()
    # if update success (1 row), proceed
    # supabase client returns .data which may be [] if no rows updated
    if updated.data:
        process_job(job)
        return True
    return None

def worker_thread_loop(stop_event):
    while not stop_event.is_set():
        try:
            claimed = claim_and_process()
            if not claimed:
                time.sleep(POLL_INTERVAL)
            # otherwise continue immediately (process_job blocks)
        except Exception as e:
            print("worker loop exception:", e)
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    # Spawn a pool: number of threads = sum of configured engine instance * concurrency per engine
    total_instances = 0
    for key, n in _engine_counts.items():
        total_instances += n
    if total_instances == 0:
        # default single worker
        total_instances = 1

    threads = []
    stop_event = threading.Event()
    for i in range(total_instances * CONCURRENCY_PER_ENGINE):
        t = threading.Thread(target=worker_thread_loop, args=(stop_event,), daemon=True)
        t.start()
        threads.append(t)
        print(f"[worker] started thread {i+1}")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        stop_event.set()
        print("shutting down workers...")
        for t in threads:
            t.join()
