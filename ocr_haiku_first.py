"""
=============================================================================
 Medical Records / Medicine Delivery - OCR Service (Haiku-first -> Sonnet)
=============================================================================
 Reads a medical document, detects WHICH type it is, and extracts its fields.

 STRATEGY (cost-optimised, no separate "check" call)
   1) Send the image to Claude HAIKU first  (cheapest).
   2) Look at Haiku's OWN confidence signals in the JSON it returns.
   3) If those signals say the read is weak  ->  retry the SAME image on SONNET.
   4) Return the better result, tagged with which model(s) were used.

   Typical cost: Haiku-only on clear/printed docs; Haiku+Sonnet only on the
   hard ones (messy handwriting, low type confidence, unreadable fields).

 WHAT TRIGGERS THE SONNET UPGRADE  (see needs_upgrade() - tune there)
   - Haiku returned invalid / unparseable JSON, OR
   - document_type is missing or type_confidence == "low", OR
   - too many extracted items are low-confidence (>= UPGRADE_LOW_RATIO), OR
   - the doc is handwritten AND type confidence is not high.

 DOCUMENT TYPES DETECTED
   prescription | lab_report | bill | imaging | discharge | other
   (the app maps these to: Prescription, Lab Tests, Medical Bills,
    Imaging, Discharge Letters, Others)

 INTEGRATION (3 steps)
   1) pip install flask flask-cors anthropic gunicorn
   2) set  ANTHROPIC_API_KEY = sk-ant-...
   3) run  python ocr_haiku_first.py     (local: http://localhost:5002)
           gunicorn ocr_haiku_first:app  (production; Procfile entry)

 APP CALL
   const fd = new FormData(); fd.append('image', file);
   const r = await fetch(OCR_SERVER + '/compare', { method:'POST', body: fd });
   const data = await r.json();          // -> data.claude  (flattened, see _meta.route)
=============================================================================
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_FAST   = "claude-haiku-4-5-20251001"   # tried first  (cheapest)
MODEL_STRONG = "claude-sonnet-4-6"           # fallback     (more accurate)
MAX_TOKENS   = 2048

# Upgrade tuning: if this fraction (or more) of extracted items are
# low-confidence, escalate Haiku -> Sonnet. 0.34 = "about a third or more".
UPGRADE_LOW_RATIO = 0.34

ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
MIMES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".png": "image/png", ".webp": "image/webp"}

app = Flask(__name__)
CORS(app)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Prompt  (classify + extract in ONE call)
# ---------------------------------------------------------------------------
PROMPT = """You are a medical document reader. Look at this image and do TWO things.

STEP 1 - Classify the document into exactly ONE type:
  prescription | lab_report | bill | imaging | discharge | other

STEP 2 - Extract everything written/printed on it.

Return ONLY valid JSON (no markdown) in this exact shape:
{
  "document_type": "prescription",
  "type_confidence": "high",
  "is_handwritten": true,
  "patient_name": null,
  "doctor_name": null,
  "hospital_name": null,
  "date": null,
  "edd": null,
  "lmp": null,
  "summary": "one short line describing the document",
  "fields": { ... type-specific data, see below ... },
  "raw_text": "all visible text, line by line, exactly as written",
  "warnings": []
}

Type-specific "fields":
- prescription: { "diagnosis": null, "medicines": [{"name","strength","form","frequency","duration","quantity","confidence"}], "tests": [{"name","confidence"}] }
- lab_report:   { "tests": [{"name","value","unit","reference_range","flag","confidence"}] }
- bill:         { "items": [{"description","amount","confidence"}], "total_amount": null, "gst": null }
- imaging:      { "scan_type": null, "body_part": null, "findings": null, "impression": null }
- discharge:    { "admission_date": null, "discharge_date": null, "diagnosis": null, "procedures": [], "instructions": null }
- other:        { "notes": null }

Rules:
- type_confidence: high / medium / low
- is_handwritten: true if the main content is handwritten, false if printed
- If a value is unclear, use null + "confidence":"low". NEVER guess.
- Always fill "raw_text" with everything you can read.
- MATERNITY: if the document mentions an Expected Date of Delivery (EDD / EDC /
  "delivery date" / "due date") put it in top-level "edd". If it mentions LMP
  (Last Menstrual Period) put it in "lmp". Use ISO format YYYY-MM-DD when the
  date is unambiguous, otherwise copy it exactly as written. Leave null if absent.
  (These often appear on ultrasound / sonography reports, which are type "imaging".)
"""


# ---------------------------------------------------------------------------
# One Claude call
# ---------------------------------------------------------------------------
def call_claude(model, image_bytes, mime):
    start = time.time()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        elapsed = round(time.time() - start, 2)
        text = msg.content[0].text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            parse_ok = True
        except json.JSONDecodeError:
            data = {"error": "invalid_json", "raw": text[:300]}
            parse_ok = False
        data["_call"] = {"model": model, "elapsed": elapsed, "parse_ok": parse_ok,
                         "input_tokens": getattr(msg.usage, "input_tokens", None),
                         "output_tokens": getattr(msg.usage, "output_tokens", None)}
        return data
    except Exception as e:
        return {"error": str(e), "_call": {"model": model, "elapsed": round(time.time() - start, 2),
                                           "parse_ok": False}}


# ---------------------------------------------------------------------------
# Decide whether to escalate Haiku -> Sonnet
# RULE: Handwritten -> Sonnet. Everything else -> Haiku.
# ---------------------------------------------------------------------------
def _iter_items(data):
    fields = data.get("fields") or {}
    for key in ("medicines", "tests", "items"):
        for it in (fields.get(key) or []):
            if isinstance(it, dict):
                yield it

def needs_upgrade(data):
    # Unparseable / error -> Sonnet as last resort
    if data.get("error") or not data.get("_call", {}).get("parse_ok", False):
        return True, "haiku_unparseable_or_error"

    # ONLY trigger: handwritten -> Sonnet
    if data.get("is_handwritten") is True:
        return True, "handwritten_routed_to_sonnet"

    # Everything else (printed, mixed, unknown) -> keep Haiku result
    return False, "haiku_sufficient"


# ---------------------------------------------------------------------------
# Flatten fields.* to top level so the app reads them directly
# ---------------------------------------------------------------------------
def flatten(data, route_meta):
    fields = data.get("fields") or {}
    flat = dict(data)
    flat.update(fields)
    flat["document_type"] = data.get("document_type")
    flat["is_handwritten"] = data.get("is_handwritten")
    flat["raw_text"] = data.get("raw_text", "")
    flat["_meta"] = route_meta
    flat.pop("_call", None)
    return flat


# ---------------------------------------------------------------------------
# Routing: Haiku first, Sonnet only if needed
# ---------------------------------------------------------------------------
def extract(image_bytes, mime):
    t0 = time.time()

    haiku = call_claude(MODEL_FAST, image_bytes, mime)
    upgrade, reason = needs_upgrade(haiku)

    if not upgrade:
        meta = {
            "route": "haiku",
            "models_used": [MODEL_FAST],
            "upgraded": False,
            "reason": reason,
            "total_elapsed": round(time.time() - t0, 2),
            "haiku_call": haiku.get("_call"),
        }
        return flatten(haiku, meta)

    # escalate to Sonnet on the same image
    sonnet = call_claude(MODEL_STRONG, image_bytes, mime)
    # if Sonnet also failed but Haiku had parsed something, keep the better one
    chosen = sonnet
    if (sonnet.get("error") or not sonnet.get("_call", {}).get("parse_ok")) and not haiku.get("error"):
        chosen = haiku
        chosen_name = "haiku_fallback_sonnet_failed"
    else:
        chosen_name = "sonnet"

    meta = {
        "route": chosen_name,
        "models_used": [MODEL_FAST, MODEL_STRONG],
        "upgraded": True,
        "reason": reason,
        "total_elapsed": round(time.time() - t0, 2),
        "haiku_call": haiku.get("_call"),
        "sonnet_call": sonnet.get("_call"),
    }
    return flatten(chosen, meta)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
import pathlib

HTML_PAGE = pathlib.Path(__file__).parent / "ocr_test_page.html"

@app.route("/", methods=["GET"])
def index():
    if HTML_PAGE.exists():
        return HTML_PAGE.read_text(), 200, {"Content-Type": "text/html"}
    return "<h2>Claude OCR Service</h2><p>POST /compare with form field 'image'</p>", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "fast": MODEL_FAST, "strong": MODEL_STRONG})


@app.route("/compare", methods=["POST"])
def compare():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded (expected form field 'image')"}), 400

    f = request.files["image"]
    suffix = Path(f.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        return jsonify({"error": f"Unsupported file type '{suffix}'. Use JPG, PNG or WEBP."}), 400

    result = extract(f.read(), MIMES[suffix])
    # Returned under "claude" so the app keeps reading response.claude unchanged.
    return jsonify({"claude": result})


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set.")
    port = int(os.environ.get("PORT", "5002"))
    print("=" * 60)
    print(f" OCR  Haiku-first -> Sonnet fallback")
    print(f"   fast   : {MODEL_FAST}")
    print(f"   strong : {MODEL_STRONG}")
    print(f" Listening on 0.0.0.0:{port}  |  POST /compare")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=port)
