from fastapi import FastAPI, Form, Response
import uvicorn
from pymongo import MongoClient
import boto3
from botocore.client import Config
import logging
import re  
from datetime import datetime
from typing import Optional
import os
import ssl
import certifi
import requests  
import zipfile  
import uuid     
from io import BytesIO

# --- LOGGER CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webhook")

# --- LIGHTWEIGHT SECRETS LOADER ---
def load_streamlit_secrets():
    """Manually parses secrets.toml to bypass loading Streamlit's engine."""
    secrets = {}
    current_section = None
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    
    if not os.path.exists(secrets_path):
        logger.error(f"Secrets file not found at: {secrets_path}")
        return None
        
    try:
        with open(secrets_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip()
                    secrets[current_section] = {}
                elif "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if current_section:
                        secrets[current_section][key] = val
        return secrets
    except Exception as e:
        logger.error(f"Failed parsing secrets file: {e}")
        return None

# --- LOAD CONFIGURATION ---
secrets = load_streamlit_secrets()
if not secrets or "mongo" not in secrets or "r2" not in secrets:
    logger.critical("🚨 CRITICAL ERROR: Could not find [mongo] and [r2] credentials in secrets.toml!")
    raise SystemExit()

MONGO_URI = secrets["mongo"]["uri"]
MONGO_DBNAME = secrets["mongo"]["dbname"]
R2_CONFIG = secrets["r2"]

# Force SSL bypass parameters directly inside the URI query string
if "tlsAllowInvalidCertificates" not in MONGO_URI:
    separator = "&" if "?" in MONGO_URI else "?"
    MONGO_URI = f"{MONGO_URI}{separator}tls=true&tlsAllowInvalidCertificates=true"

# --- DATABASE & STORAGE CONNECTIONS ---
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DBNAME]
    
    s3_client = boto3.client(
        's3',
        endpoint_url=f"https://{R2_CONFIG['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_CONFIG['access_key_id'],
        aws_secret_access_key=R2_CONFIG['secret_access_key'],
        region_name="auto",
        config=Config(
            signature_version='s3v4',
            region_name="auto"
        )
    )
    logger.info("✅ Database & Cloudflare R2 Clients Successfully Initialized!")
except Exception as e:
    logger.critical(f"Connection Initialization Failed: {e}")
    raise SystemExit()

# --- FASTAPI APPLICATION ---
app = FastAPI(title="CapitUp India WhatsApp Webhook Receiver")

def xml_response(message_text: str) -> Response:
    """Generates a standard Twilio TwiML XML response, safely escaping raw XML characters exactly once."""
    # Negative lookahead to only replace raw '&' that are not already part of safe entities (&amp;, &lt;, &gt;) [3]
    safe_text = re.sub(r"&(?!amp;|lt;|gt;)", "&amp;", message_text)
    safe_text = safe_text.replace("<", "&lt;").replace(">", "&gt;")
    
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Message>
            <Body>{safe_text}</Body>
        </Message>
    </Response>"""
    return Response(content=xml_content, media_type="application/xml")

def shorten_url(long_url: str) -> str:
    """Shortens a long pre-signed R2 URL using the free TinyURL API."""
    try:
        r = requests.get(f"https://tinyurl.com/api-create.php?url={long_url}", timeout=5)
        if r.status_code == 200:
            return r.text.strip()
    except Exception as e:
        logger.error(f"TinyURL shortening failed: {e}")
    return long_url  # Fallback to the long URL if the API is offline or times out

# --- CONVERSATIONAL DUAL-MODE QUERY ANALYZER ---
def process_single_segment(segment_text: str):
    """
    Parses a single segment of a query (e.g. 'FETCH 101 102 103 OF POLICY').
    Returns a tuple of (list of matched MongoDB records, list of requested employee IDs).
    """
    segment_text = segment_text.strip().upper()
    if not segment_text.startswith("FETCH"):
        segment_text = "FETCH " + segment_text
        
    parts = segment_text.split()
    if len(parts) < 2:
        return [], []
        
    # Detect conversational connectors
    connector_idx = None
    for idx, part in enumerate(parts):
        if idx > 1 and part in ["OF", "FOR", "FROM", "UNDER"]:
            connector_idx = idx
            break
            
    # Extract space-separated IDs and target Policy/Company name
    if connector_idx:
        emp_ids = [pid.strip() for pid in parts[1:connector_idx] if pid.strip()]
        policy_no = " ".join(parts[connector_idx + 1:]).strip()
    else:
        # Fallback if no connector word was specified
        if len(parts) > 2:
            emp_ids = [parts[1].strip()]
            policy_no = " ".join(parts[2:]).strip()
        else:
            emp_ids = [parts[1].strip()]
            policy_no = None
            
    records = []
    for emp_id in emp_ids:
        query = {"emp_id": emp_id}
        if policy_no:
            # Format-resilient loose regex matching [1]
            normalized_pattern = re.sub(r"[\/\-\_\s]+", ".*", policy_no)
            query["$or"] = [
                {"policy_no": {"$regex": f"^{normalized_pattern}", "$options": "i"}},
                {"company_name": {"$regex": f"^{normalized_pattern}", "$options": "i"}}
            ]
            
        logger.info(f"Querying Atlas DB: {query}")
        ecard_record = db.ecards.find_one(query)
        if ecard_record:
            records.append(ecard_record)
            
    return records, emp_ids

@app.post("/whatsapp")
async def handle_incoming_whatsapp(
    From: Optional[str] = Form(None),  # Twilio format: 'whatsapp:+91XXXXXXXXXX'
    Body: str = Form(...)              # Message text
):
    # Handle status callbacks gracefully [2]
    if not From or not Body:
        return Response(content="OK", media_type="text/plain")

    sender_phone = From.strip().lower()
    raw_body = Body.strip().upper()
    
    # ✨ FIX: Detect and strip out the keyword 'ZIP' first so it never pollutes the policy name parsing [1]
    is_zip_forced = False
    if "ZIP" in raw_body:
        is_zip_forced = True
        raw_body = re.sub(r"\bZIP\b", "", raw_body).strip()
        
    logger.info(f"Incoming message from {sender_phone}: {raw_body} (Forced ZIP: {is_zip_forced})")
    
    # 1. SECURITY GATE: Validate if sender is an authorized support agent
    agent = db.authorized_agents.find_one({"phone_number": sender_phone, "status": "active"})
    if not agent:
        logger.warning(f"Unauthorized access block from: {sender_phone}")
        return xml_response("🔒 ACCESS DENIED: Your mobile number is not registered as an authorized CapitUp support agent.")
        
    # 2. SEGMENT MULTIPLE QUERIES (Split by comma or newline)
    raw_segments = re.split(r'[\n,]+', raw_body)
    has_fetch_start = raw_body.startswith("FETCH")
    agent_name = agent.get("name", "Unknown")
    
    matched_records = []
    total_requested_ids = []
    
    for segment in raw_segments:
        segment = segment.strip()
        if not segment:
            continue
            
        # Propagate command prefix to subsequent segments if omitted [1]
        if has_fetch_start and not segment.startswith("FETCH"):
            segment = "FETCH " + segment
            
        segment_matches, segment_req_ids = process_single_segment(segment)
        matched_records.extend(segment_matches)
        total_requested_ids.extend(segment_req_ids)
        
    # Deduplicate lists
    total_requested_ids = list(set(total_requested_ids))
    seen = set()
    unique_records = []
    for r in matched_records:
        r_id = f"{r['emp_id']}_{r['policy_no']}"
        if r_id not in seen:
            seen.add(r_id)
            unique_records.append(r)
            
    # 3. IF NO RECOGNIZED RECORDS ARE FOUND
    if not unique_records:
        return xml_response("❌ No matching e-cards could be located in the database for your query.")
        
    # 4. CHOOSE DISPATCH MODE (Auto-ZIP vs. Individual)
    # ✨ FIX: Decisions are now made based on how many IDs were REQUESTED (not found), or if ZIP is forced [1]
    is_zip_request = len(total_requested_ids) > 3 or is_zip_forced
    
    if is_zip_request:
        # --- MODE A: AUTOMATED ZIP BUNDLING ENGINE --- [1]
        logger.info(f"Triggering Automated ZIP Packaging for {len(unique_records)} located records...")
        zip_buffer = BytesIO()
        
        try:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for record in unique_records:
                    # Download card PDF bytes directly from Cloudflare R2 into RAM [1, 2]
                    response = s3_client.get_object(Bucket=R2_CONFIG["bucket_name"], Key=record["r2_key"])
                    pdf_bytes = response["Body"].read()
                    
                    filename = f"HealthCard_{record['emp_id']}_{record['card_type']}.pdf"
                    zf.writestr(filename, pdf_bytes)
                    
            zip_bytes = zip_buffer.getvalue()
            
            # Upload the compiled ZIP file back to a temporary Cloudflare folder [1]
            temp_zip_id = str(uuid.uuid4())[:8]
            temp_zip_key = f"ecards/temp_zips/{temp_zip_id}.zip"
            
            s3_client.put_object(
                Bucket=R2_CONFIG["bucket_name"],
                Key=temp_zip_key,
                Body=zip_bytes,
                ContentType="application/zip"
            )
            
            # Generate pre-signed URL for the temporary ZIP archive (5 mins expiry) [1]
            presigned_zip_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': R2_CONFIG["bucket_name"], 'Key': temp_zip_key},
                ExpiresIn=300
            )
            
            # Shorten the pre-signed ZIP link using TinyURL [1]
            short_zip_url = shorten_url(presigned_zip_url)
            
            # Identify which queried IDs were missing from the database to alert the agent
            located_ids = [r["emp_id"] for r in unique_records]
            missing_ids = [rid for rid in total_requested_ids if rid not in located_ids]
            
            missing_warning_text = ""
            if missing_ids:
                missing_warning_text = f"\n\n⚠️ *Missing from DB:* {', '.join(missing_ids)}"
            
            # Log successful batch run to audit logs
            db.whatsapp_audit_logs.insert_one({
                "agent_phone": sender_phone,
                "agent_name": agent_name,
                "batch_count": len(unique_records),
                "query_policy_no": unique_records[0]["policy_no"],
                "timestamp": datetime.utcnow()
            })
            
            reply_msg = (
                f"📦 *Bulk Campaign ZIP Generated!*\n"
                f"Compiled **{len(unique_records)}** located e-card PDFs cleanly inside a temporary archive.{missing_warning_text}\n\n"
                f"🔗 *Download ZIP* (Expires in 5 minutes):\n"
                f"{short_zip_url}"
            )
            return xml_response(reply_msg)
            
        except Exception as e:
            logger.error(f"Failed to generate compiled ZIP archive: {e}")
            return xml_response("❌ Failed to bundle and package your batch e-cards into a ZIP file.")
            
    else:
        # --- MODE B: INDIVIDUAL DISPATCH MODE (1 TO 3 CARDS) --- [1]
        response_lines = []
        for record in unique_records:
            try:
                # Generate individual pre-signed URL (5 mins expiry) [1]
                presigned_download_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': R2_CONFIG["bucket_name"], 'Key': record["r2_key"]},
                    ExpiresIn=300
                )
                
                # Shorten the URL [1]
                short_url = shorten_url(presigned_download_url)
                
                # Log successful query
                db.whatsapp_audit_logs.insert_one({
                    "agent_phone": sender_phone,
                    "agent_name": agent_name,
                    "query_emp_id": record["emp_id"],
                    "query_policy_no": record["policy_no"],
                    "timestamp": datetime.utcnow()
                })
                
                response_lines.append(
                    f"✅ *Employee ID:* {record['emp_id']}\n"
                    f"Plan: {record['policy_no']}\n"
                    f"🔗 *Secure Download Link* (Expires in 5 mins):\n"
                    f"{short_url}\n"
                )
            except Exception as e:
                logger.error(f"Failed to generate individual pre-signed URL for {record['emp_id']}: {e}")
                response_lines.append(f"❌ Failed to retrieve card for ID *{record['emp_id']}*.")
                
        final_reply_msg = "\n".join(response_lines).strip()
        return xml_response(final_reply_msg)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)