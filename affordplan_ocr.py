"""
================================================================================
 AFFORDPLAN — CLAUDE OCR SERVICE
 Production-ready · Haiku-first → Sonnet for handwritten
================================================================================

 WHAT IT DOES
   Reads any medical document image, detects its type, and extracts all fields
   needed by the four Affordplan flows:
     1. Health Record Management
     2. Medicine Order
     3. Lab Test Home Collection Booking
     4. Maternity Onboarding (EDD)

 ROUTING
   Printed document    → Claude Haiku   (fast, cheap  ~₹0.27/doc)
   Handwritten doc     → Claude Sonnet  (accurate     ~₹1.27/doc)
   Parse error/unknown → Claude Sonnet  (fallback)

 SETUP
   pip install flask flask-cors anthropic gunicorn
   set ANTHROPIC_API_KEY = sk-ant-...   (environment variable, never in code)

 RUN
   Local      : python affordplan_ocr.py       → http://localhost:5002
   Production : gunicorn affordplan_ocr:app --bind 0.0.0.0:$PORT

 SINGLE ENDPOINT
   POST /compare
   Content-Type : multipart/form-data
   Field        : image  (JPG / PNG / WEBP, max ~10MB)
   Response     : JSON   (see RESPONSE SHAPE below)

================================================================================
 RESPONSE SHAPE
================================================================================

 {
   "claude": {

     // ── COMMON (all document types) ───────────────────────────────────────
     "document_type"   : "prescription",   // see DOCUMENT TYPES below
     "type_confidence" : "high",           // high / medium / low
     "is_handwritten"  : true,             // true = handwritten, false = printed
     "patient_name"    : "Ramesh Kumar",   // null if not found
     "patient_uhid"    : "123456789",      // Unique Health ID / ABHA number
     "doctor_name"     : "Dr. A. Sharma",  // null if not found
     "hospital_name"   : "City Care Clinic",
     "date"            : "12 May 2026",
     "summary"         : "Prescription for acute bronchitis with 3 medicines.",
     "raw_text"        : "full text read from the document, line by line",
     "warnings"        : [],

     // ── FLOW 1: HEALTH RECORD MANAGEMENT ─────────────────────────────────
     // Always present. Use document_type to decide which tab to open.
     // document_type values:
     //   prescription | lab_report | bill | imaging | discharge | other

     // ── FLOW 2: MEDICINE ORDER ────────────────────────────────────────────
     // Present when document_type == "prescription"
     "diagnosis"  : "Acute bronchitis",
     "medicines"  : [
       {
         "name"       : "Azithromycin",
         "strength"   : "500 mg",
         "form"       : "Tablet",
         "frequency"  : "1-0-0",
         "duration"   : "3 days",
         "quantity"   : 3,           // number of units to order
         "confidence" : "high"       // high / medium / low
       }
     ],
     "tests_advised" : [             // lab tests doctor has advised
       { "name": "CBC", "confidence": "high" }
     ],

     // ── FLOW 3: LAB TEST HOME COLLECTION BOOKING ─────────────────────────
     // Present when document_type == "lab_report"
     "lab_tests" : [
       {
         "name"            : "Hemoglobin",
         "value"           : "13.2",
         "unit"            : "g/dL",
         "reference_range" : "13.0–17.0",
         "flag"            : "NORMAL",    // NORMAL / HIGH / LOW / null
         "confidence"      : "high"
       }
     ],

     // ── FLOW 4: MATERNITY ONBOARDING ─────────────────────────────────────
     // Present on any document that mentions EDD/LMP (usually imaging/USG)
     "edd" : "2026-11-15",   // Expected Date of Delivery (ISO YYYY-MM-DD or as written)
     "lmp" : "2026-02-08",   // Last Menstrual Period

     // ── OTHER DOCUMENT TYPE FIELDS ────────────────────────────────────────
     // imaging
     "scan_type"  : "Ultrasound",
     "body_part"  : "Abdomen",
     "findings"   : "Single live intrauterine fetus...",
     "impression" : "Normal fetal growth.",

     // discharge
     "admission_date"  : "01 May 2026",
     "discharge_date"  : "05 May 2026",
     "procedures"      : ["Appendectomy"],
     "instructions"    : "Rest for 2 weeks. Follow up in OPD.",

     // bill
     "bill_items"   : [
       { "description": "Consultation", "amount": "500", "confidence": "high" }
     ],
     "total_amount" : "4820",
     "gst"          : "0",

     // ── ROUTING META (for logging / debugging) ────────────────────────────
     "_meta" : {
       "route"         : "haiku",       // "haiku" or "sonnet"
       "models_used"   : ["claude-haiku-4-5-20251001"],
       "upgraded"      : false,
       "reason"        : "printed_haiku_sufficient",
       "total_elapsed" : 4.21,          // seconds
       "haiku_call"    : { "model":"...", "elapsed":4.21,
                           "input_tokens":1480, "output_tokens":520 },
       "sonnet_call"   : null           // populated only if upgraded
     }
   }
 }

================================================================================
 HOW EACH FLOW USES THE RESPONSE
================================================================================

 // FLOW 1 — Health Record Management
 const d = response.claude;
 d.document_type    // to pick the right record folder
 d.patient_name     // to match / assign to a family member
 d.patient_uhid     // UHID match
 d.doctor_name
 d.hospital_name
 d.date

 // FLOW 2 — Medicine Order
 if (d.document_type === 'prescription') {
   d.medicines        // array — show in cart, each item has quantity to order
   d.diagnosis
   d.tests_advised    // show "your doctor also advised these tests"
 }

 // FLOW 3 — Lab Test Home Collection Booking
 if (d.document_type === 'lab_report') {
   d.lab_tests        // array — show tests, values, flags for booking
 }

 // FLOW 4 — Maternity Onboarding
 if (d.edd) {
   d.edd              // pre-fill EDD in onboarding form
   d.lmp              // pre-fill LMP
 }

================================================================================
"""

import base64
import json
import os
import re
import time
from pathlib import Path

import anthropic
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────────────────────
# Config  —  change models here if needed
# ─────────────────────────────────────────────────────────────────────────────
MODEL_HAIKU  = "claude-haiku-4-5-20251001"   # printed docs   ~₹0.27/doc
MODEL_SONNET = "claude-sonnet-4-6"           # handwritten    ~₹1.27/doc
MAX_TOKENS   = 2048

ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
MIMES   = {".jpg":"image/jpeg", ".jpeg":"image/jpeg",
           ".png":"image/png",  ".webp":"image/webp"}

app    = Flask(__name__)
CORS(app)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

HTML_PAGE = Path(__file__).parent / "ocr_test_page.html"


# ─────────────────────────────────────────────────────────────────────────────
# Extraction Prompt
# ─────────────────────────────────────────────────────────────────────────────
PROMPT = """
You are a medical document reader for Affordplan, a healthcare platform in India.

Look at this document image and do TWO things:

STEP 1 — Classify the document into exactly ONE type:
  prescription | lab_report | bill | imaging | discharge | other

STEP 2 — Extract every piece of information visible on it.

Return ONLY valid JSON (no markdown, no extra text) in this exact shape:

{
  "document_type"   : "prescription",
  "type_confidence" : "high",
  "is_handwritten"  : true,

  "patient_name"    : null,
  "patient_uhid"    : null,
  "doctor_name"     : null,
  "hospital_name"   : null,
  "date"            : null,

  "edd"             : null,
  "lmp"             : null,

  "summary"         : "one short line describing the document",
  "warnings"        : [],
  "raw_text"        : "all visible text, line by line, exactly as written",

  "fields": {

    // IF document_type == "prescription"
    "diagnosis"    : null,
    "medicines"    : [
      {
        "name"      : null,
        "strength"  : null,
        "form"      : null,
        "frequency" : null,
        "duration"  : null,
        "quantity"  : null,
        "confidence": "high"
      }
    ],
    "tests_advised": [
      { "name": null, "confidence": "high" }
    ],

    // IF document_type == "lab_report"
    "lab_tests": [
      {
        "name"            : null,
        "value"           : null,
        "unit"            : null,
        "reference_range" : null,
        "flag"            : null,
        "confidence"      : "high"
      }
    ],

    // IF document_type == "imaging"
    "scan_type"  : null,
    "body_part"  : null,
    "findings"   : null,
    "impression" : null,

    // IF document_type == "discharge"
    "admission_date" : null,
    "discharge_date" : null,
    "diagnosis"      : null,
    "procedures"     : [],
    "instructions"   : null,

    // IF document_type == "bill"
    "bill_items"   : [
      { "description": null, "amount": null, "confidence": "high" }
    ],
    "total_amount" : null,
    "gst"          : null,

    // IF document_type == "other"
    "notes": null
  }
}

RULES:
1. type_confidence  : high / medium / low
2. is_handwritten   : true if the main content is handwritten, false if printed
3. patient_uhid     : extract UHID / ABHA / UHN / patient ID number if present
4. medicines.quantity : calculate from frequency × duration
                        e.g. 1-0-1 × 5 days = 10 tablets. Integer only.
5. medicines.frequency : use Indian format  e.g. 1-0-1  1-1-1  0-0-1
6. lab_tests.flag   : NORMAL / HIGH / LOW / ABNORMAL — copy exactly as printed,
                      or derive from value vs reference_range if not printed
7. edd / lmp        : extract whenever present (prescriptions, USG, any doc).
                      Use ISO YYYY-MM-DD when unambiguous, else copy as written.
8. If a value is unclear write null + "confidence":"low". NEVER guess or invent.
9. raw_text         : copy every visible word, line by line.
10. Only include fields relevant to the detected document_type in "fields".
    Leave irrelevant sections out entirely — do not send empty arrays for them.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Single Claude call
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(model, image_bytes, mime):
    start = time.time()
    b64   = base64.standard_b64encode(image_bytes).decode("utf-8")
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        elapsed = round(time.time() - start, 2)
        text    = msg.content[0].text.strip()
        text    = re.sub(r"^```json\s*", "", text)
        text    = re.sub(r"\s*```$",     "", text)
        try:
            data     = json.loads(text)
            parse_ok = True
        except json.JSONDecodeError:
            data     = {"error": "invalid_json", "raw_response": text[:400]}
            parse_ok = False

        data["_call"] = {
            "model":         model,
            "elapsed":       elapsed,
            "parse_ok":      parse_ok,
            "input_tokens":  getattr(msg.usage, "input_tokens",  None),
            "output_tokens": getattr(msg.usage, "output_tokens", None),
        }
        return data

    except Exception as e:
        return {
            "error":  str(e),
            "_call":  {"model": model, "elapsed": round(time.time() - start, 2),
                       "parse_ok": False},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Routing: Haiku first, Sonnet only if handwritten
# ─────────────────────────────────────────────────────────────────────────────
def needs_upgrade(data):
    """
    Returns (True, reason) if Sonnet should be used.

    RULE: Handwritten → Sonnet. Everything else → keep Haiku.
    """
    # Parse failed — try Sonnet as last resort
    if data.get("error") or not data.get("_call", {}).get("parse_ok", False):
        return True, "haiku_unparseable_or_error"

    # ONLY trigger: handwritten document
    if data.get("is_handwritten") is True:
        return True, "handwritten_routed_to_sonnet"

    return False, "printed_haiku_sufficient"


# ─────────────────────────────────────────────────────────────────────────────
# Flatten fields.* to top level
# ─────────────────────────────────────────────────────────────────────────────
def flatten(data, route_meta):
    fields = data.get("fields") or {}
    flat   = {**data, **fields}

    # Ensure top-level keys the app always reads
    for key in ("document_type", "type_confidence", "is_handwritten",
                "patient_name", "patient_uhid", "doctor_name",
                "hospital_name", "date", "edd", "lmp",
                "summary", "raw_text", "warnings"):
        flat.setdefault(key, None)

    flat["_meta"] = route_meta
    flat.pop("_call",   None)
    flat.pop("fields",  None)
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Main extract function
# ─────────────────────────────────────────────────────────────────────────────
def extract(image_bytes, mime):
    t0    = time.time()
    haiku = call_claude(MODEL_HAIKU, image_bytes, mime)
    upgrade, reason = needs_upgrade(haiku)

    if not upgrade:
        meta = {
            "route":        "haiku",
            "models_used":  [MODEL_HAIKU],
            "upgraded":     False,
            "reason":       reason,
            "total_elapsed": round(time.time() - t0, 2),
            "haiku_call":   haiku.get("_call"),
            "sonnet_call":  None,
        }
        return flatten(haiku, meta)

    # Handwritten or error — escalate to Sonnet
    sonnet = call_claude(MODEL_SONNET, image_bytes, mime)

    # If Sonnet also fails but Haiku had something, keep Haiku
    if (sonnet.get("error") or not sonnet.get("_call", {}).get("parse_ok")) \
            and not haiku.get("error"):
        chosen      = haiku
        route_label = "haiku_fallback_sonnet_also_failed"
    else:
        chosen      = sonnet
        route_label = "sonnet"

    meta = {
        "route":         route_label,
        "models_used":   [MODEL_HAIKU, MODEL_SONNET],
        "upgraded":      True,
        "reason":        reason,
        "total_elapsed": round(time.time() - t0, 2),
        "haiku_call":    haiku.get("_call"),
        "sonnet_call":   sonnet.get("_call"),
    }
    return flatten(chosen, meta)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    if HTML_PAGE.exists():
        return HTML_PAGE.read_text(), 200, {"Content-Type": "text/html"}
    return ("<h2>Affordplan OCR Service</h2>"
            "<p>POST /compare — form field: image (JPG/PNG/WEBP)</p>"), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "service": "affordplan-ocr",
        "fast":    MODEL_HAIKU,
        "strong":  MODEL_SONNET,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Sarvam proxy routes  —  avoids CORS issues when calling from browser
# All Sarvam API calls go server-to-server through here
# ─────────────────────────────────────────────────────────────────────────────
import requests as req_lib   # pip install requests (already a common dependency)

SARVAM_BASE = "https://api.sarvam.ai/doc-digitization/job/v1"
SARVAM_CHAT = "https://api.sarvam.ai/v1/chat/completions"

def sarvam_headers(api_key):
    return {"Content-Type": "application/json", "api-subscription-key": api_key}


@app.route("/sarvam/create-job", methods=["POST"])
def sarvam_create_job():
    """Step 1 — Create a Sarvam doc digitization job"""
    api_key = request.headers.get("X-Sarvam-Key", "")
    if not api_key:
        return jsonify({"error": "Missing X-Sarvam-Key header"}), 400
    body = request.get_json(force=True) or {}
    r = req_lib.post(SARVAM_BASE, headers=sarvam_headers(api_key), json=body, timeout=30)
    return jsonify(r.json()), r.status_code


@app.route("/sarvam/register-files", methods=["POST"])
def sarvam_register_files():
    """Step 2 — Register files with the job"""
    api_key = request.headers.get("X-Sarvam-Key", "")
    if not api_key:
        return jsonify({"error": "Missing X-Sarvam-Key header"}), 400
    body = request.get_json(force=True) or {}
    r = req_lib.post(f"{SARVAM_BASE}/upload-files", headers=sarvam_headers(api_key),
                     json=body, timeout=30)
    return jsonify(r.json()), r.status_code


@app.route("/sarvam/upload-file", methods=["POST"])
def sarvam_upload_file():
    """Step 3 — Upload file bytes to Azure blob (server-side, no CORS issues)"""
    upload_url = request.headers.get("X-Upload-Url", "")
    if not upload_url:
        return jsonify({"error": "Missing X-Upload-Url header"}), 400
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["image"]
    file_bytes = f.read()
    mime = f.content_type or "application/octet-stream"
    r = req_lib.put(upload_url, headers={
        "x-ms-blob-type": "BlockBlob",
        "Content-Type": mime,
    }, data=file_bytes, timeout=60)
    if r.status_code in (200, 201):
        return jsonify({"status": "uploaded"}), 200
    return jsonify({"error": f"Upload failed: {r.status_code}", "detail": r.text[:300]}), r.status_code


@app.route("/sarvam/start-job/<job_id>", methods=["POST"])
def sarvam_start_job(job_id):
    """Step 4 — Start the job"""
    api_key = request.headers.get("X-Sarvam-Key", "")
    if not api_key:
        return jsonify({"error": "Missing X-Sarvam-Key header"}), 400
    hdrs = {**sarvam_headers(api_key), "X-Dashboard": "true"}
    r = req_lib.post(f"{SARVAM_BASE}/{job_id}/start", headers=hdrs, timeout=30)
    try:
        return jsonify(r.json()), r.status_code
    except Exception:
        return jsonify({"status": "started", "raw": r.text[:200]}), r.status_code


@app.route("/sarvam/job-status/<job_id>", methods=["GET"])
def sarvam_job_status(job_id):
    """Step 5 — Poll job status"""
    api_key = request.headers.get("X-Sarvam-Key", "")
    if not api_key:
        return jsonify({"error": "Missing X-Sarvam-Key header"}), 400
    r = req_lib.get(f"{SARVAM_BASE}/{job_id}/status",
                    headers={"api-subscription-key": api_key}, timeout=30)
    return jsonify(r.json()), r.status_code


@app.route("/sarvam/download-files/<job_id>", methods=["POST"])
def sarvam_download_files(job_id):
    """Step 6a — Get download URLs"""
    api_key = request.headers.get("X-Sarvam-Key", "")
    if not api_key:
        return jsonify({"error": "Missing X-Sarvam-Key header"}), 400
    body = request.get_json(force=True) or {}
    r = req_lib.post(f"{SARVAM_BASE}/{job_id}/download-files",
                     headers=sarvam_headers(api_key), json=body, timeout=30)
    return jsonify(r.json()), r.status_code


@app.route("/sarvam/fetch-output", methods=["POST"])
def sarvam_fetch_output():
    """Step 6b — Download the actual output file content from blob URL"""
    body = request.get_json(force=True) or {}
    dl_url = body.get("url", "")
    if not dl_url:
        return jsonify({"error": "Missing url"}), 400
    r = req_lib.get(dl_url, timeout=60)
    if r.status_code == 200:
        return jsonify({"text": r.text}), 200
    return jsonify({"error": f"Download failed: {r.status_code}"}), r.status_code


@app.route("/sarvam/extract", methods=["POST"])
def sarvam_extract():
    """Step 6c — Apply targeted prompt on extracted text using Sarvam chat model"""
    api_key = request.headers.get("X-Sarvam-Key", "")
    if not api_key:
        return jsonify({"error": "Missing X-Sarvam-Key header"}), 400
    body = request.get_json(force=True) or {}
    prompt  = body.get("prompt", "")
    text    = body.get("text", "")
    r = req_lib.post(SARVAM_CHAT, headers=sarvam_headers(api_key), json={
        "model": "sarvam-m",
        "messages": [
            {"role": "system", "content": "You are a medical data extractor. Return ONLY valid JSON."},
            {"role": "user",   "content": prompt + "\n\nDOCUMENT TEXT:\n" + text}
        ],
        "max_tokens": 1500,
        "temperature": 0
    }, timeout=60)
    return jsonify(r.json()), r.status_code


@app.route("/compare", methods=["POST"])
def compare():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Send form field 'image'."}), 400

    f      = request.files["image"]
    suffix = Path(f.filename or "").suffix.lower()

    if suffix not in ALLOWED:
        return jsonify({"error": f"Unsupported type '{suffix}'. Use JPG, PNG or WEBP."}), 400

    result = extract(f.read(), MIMES[suffix])

    # Always returned under "claude" — keeps the app call stable
    return jsonify({"claude": result})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set.")
    port = int(os.environ.get("PORT", "5002"))
    print("=" * 60)
    print("  Affordplan OCR Service")
    print(f"  Printed     → {MODEL_HAIKU}")
    print(f"  Handwritten → {MODEL_SONNET}")
    print(f"  http://localhost:{port}  |  POST /compare")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=port)
