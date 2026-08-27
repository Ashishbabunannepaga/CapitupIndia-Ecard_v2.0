# app.py
import streamlit as st
import pymupdf as fitz  # Updated PyMuPDF syntax
from PIL import Image
import io
import json
import re
import pandas as pd

from key_rotator import GeminiKeyRotator
from cloudflare_db import CloudflareD1
from agent_engine import VisionExtractionAgent, DisambiguationAgent, QualityAuditAgent


# ==============================================================================
# HELPER FUNCTIONS & MULTI-ENCODING CSV / EXCEL PARSER
# ==============================================================================
def clean_phone_number(val) -> str:
    """Fixes Excel scientific notation (e.g. 9.25E+09 -> 9250000000) and removes dirty labels."""
    if pd.isna(val) or not str(val).strip() or str(val).lower() == "nan":
        return ""
    
    val_str = str(val).strip()
    
    # Repair exponential scientific notation
    if re.match(r"^\d+(\.\d+)?[eE]\+\d+$", val_str):
        try:
            val_str = f"{int(float(val_str)):d}"
        except Exception:
            pass

    return val_str


def parse_phone_numbers(phone_raw_str: str) -> list:
    """Unpacks multi-number strings like '040-23096784/5412' or '9848218200/23193033'."""
    if not phone_raw_str:
        return []
        
    cleaned_str = clean_phone_number(phone_raw_str)
    raw_tokens = re.split(r"[,;&]|\s+and\s+", cleaned_str, flags=re.IGNORECASE)
    phone_list = []

    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue

        if "/" in token:
            parts = token.split("/")
            base_number = re.sub(r"[^\d]", "", parts[0])
            if base_number:
                phone_list.append(parts[0].strip())
                for ext in parts[1:]:
                    ext_digits = re.sub(r"[^\d]", "", ext)
                    if len(ext_digits) < len(base_number) and len(base_number) >= len(ext_digits):
                        prefix = base_number[:-len(ext_digits)]
                        phone_list.append(prefix + ext_digits)
                    else:
                        phone_list.append(ext.strip())
        else:
            num_only = re.sub(r"[^\d\+\-\s\(\)]", "", token).strip()
            if num_only and len(re.sub(r"\D", "", num_only)) >= 6:
                phone_list.append(num_only)

    return list(dict.fromkeys(phone_list))


def parse_representatives(contact_str: str, designation_str: str, mobile_str: str) -> list:
    """Splits multi-representative strings and extracts embedded phone numbers."""
    if not contact_str or contact_str.lower() == "nan":
        return []

    reps = []
    names = re.split(r"&|,|\b\s+and\s+\b", contact_str, flags=re.IGNORECASE)
    clean_mob = clean_phone_number(mobile_str)

    for name_raw in names:
        name_clean = name_raw.strip()
        rep_mobile = clean_mob

        phone_match = re.search(r"[\-\:\s]+(\d{10}|\d{8,11})\b", name_clean)
        if phone_match:
            rep_mobile = phone_match.group(1)
            name_clean = name_clean.replace(phone_match.group(0), "").strip()

        if name_clean:
            reps.append({
                "name": name_clean,
                "designation": designation_str if designation_str.lower() != "nan" else "",
                "mobile": rep_mobile
            })

    return reps


def normalize_json_record(entry: dict) -> dict:
    """Normalizes JSON keys across all variations into Cloudflare D1 internal schema."""
    panel_no = entry.get("panel_no") or entry.get("record_id") or 0
    raw_name = str(entry.get("raw_name") or entry.get("organization_name") or "").strip()
    address = str(entry.get("address") or entry.get("full_address") or "").strip()
    pincode = str(entry.get("pincode") or entry.get("postal_code") or "").strip()
    
    phones = entry.get("phones") if "phones" in entry else entry.get("contact_numbers", [])
    emails = entry.get("emails", [])
    website = str(entry.get("website", "")).strip()
    aliases = entry.get("aliases", [])
    nature_of_business = str(entry.get("nature_of_business") or entry.get("business_details") or "").strip()
    
    reps = entry.get("representatives", [])

    # Bleed cleanup for Company Name -> Address
    suffix_pattern = r"^(.*?\b(PVT\s*\.?\s*LTD\s*\.?|LTD\s*\.?|CO\s*\.?\s*LTD\s*\.?|INC\s*\.?|CORP\s*\.?))\s*(.*)$"
    match = re.match(suffix_pattern, raw_name, re.IGNORECASE)
    
    if match:
        clean_name = match.group(1).strip()
        bleed_text = match.group(3).strip()
        if bleed_text:
            raw_name = clean_name
            address = f"{bleed_text} {address}".strip()

    try:
        panel_no = int(float(str(panel_no)))
    except ValueError:
        panel_no = 0

    return {
        "panel_no": panel_no,
        "raw_name": raw_name,
        "address": address,
        "pincode": pincode,
        "phones": phones if isinstance(phones, list) else [str(phones)],
        "emails": emails if isinstance(emails, list) else [str(emails)],
        "website": website,
        "aliases": aliases if isinstance(aliases, list) else [],
        "representatives": reps if isinstance(reps, list) else [],
        "nature_of_business": nature_of_business
    }


def dual_format_smart_csv_parser(uploaded_file) -> list:
    """
    Unified Engine with Multi-Encoding Support (UTF-8, CP1252, Latin1).
    Handles Windows Excel exported CSV files without unicode errors.
    """
    filename = uploaded_file.name.lower()
    df = None

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file, dtype=str)
    else:
        # Multi-encoding fallback list for Windows CSV files
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
        for enc in encodings_to_try:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, dtype=str, encoding=enc, on_bad_lines="skip")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if df is None:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, dtype=str, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")

    # Detect title header banner row offset if present
    header_offset = None
    for i in range(min(5, len(df))):
        row_str_values = [str(v).lower() for v in df.iloc[i].values]
        if any(k in row_str_values for k in ["unitname", "company name", "companyname", "sl.no", "s.no", "s no"]):
            header_offset = i
            break

    if header_offset is not None and header_offset > 0:
        new_headers = [str(c).strip() for c in df.iloc[header_offset].values]
        df = df.iloc[header_offset + 1:].copy()
        df.columns = new_headers

    records = []
    cols_lower = [str(c).strip().lower() for c in df.columns]
    
    # Auto-detect format type
    is_sheet_type_1 = any("unitname" in c or "lineofactivity" in c for c in cols_lower)

    for idx, row in df.iterrows():
        r = {str(k).strip(): (str(v).strip() if pd.notna(v) and str(v).lower() != "nan" else "") for k, v in row.items()}
        
        if not any(r.values()):
            continue

        panel_no = idx + 1
        raw_name = ""
        full_address = ""
        pincode = ""
        phones = []
        emails = []
        website = ""
        representatives = []
        nature_parts = []

        if is_sheet_type_1:
            # === SHEET TYPE 1 (UnitName, District, lineofActivity, InvestmentInCrores, NoofEmployees) ===
            panel_no = r.get("S No", r.get("S.No", r.get("S. No", r.get("S No\n", idx + 1))))
            raw_name = r.get("UnitName", r.get("Unit Name", ""))
            
            comm_addr = r.get("COMMUNICATIONADDRESS", "")
            loc = r.get("LOCATION", "")
            dist = r.get("District", "")
            full_address = ", ".join([c for c in [comm_addr, loc, dist] if c])

            pin_m = re.search(r"\b\d{6}\b", full_address)
            if pin_m:
                pincode = pin_m.group(0)

            # Route Emails & Websites
            email_raw = r.get("Email", r.get("Email id", ""))
            if email_raw:
                for item in re.split(r"[,;]", email_raw):
                    item = item.strip()
                    if "@" in item:
                        emails.append(item)
                    elif "www." in item or item.endswith(".com") or item.endswith(".in") or item.endswith(".org"):
                        website = item

            activity = r.get("lineofActivity", r.get("Line of Activity", ""))
            invest = r.get("InvestmentInCrores", "")
            emp = r.get("NoofEmployees", r.get("NoofEmployees\ns", ""))

            if activity: nature_parts.append(f"Activity: {activity}")
            if invest: nature_parts.append(f"Investment: ₹{invest} Cr")
            if emp: nature_parts.append(f"Employees: {emp}")

        else:
            # === SHEET TYPE 2 (Company Name, Address, Contact person, Designation, Mobile no, Email id, Telephone, Category) ===
            panel_no = r.get("SL.No", r.get("SL. No", r.get("S.No", idx + 1)))
            raw_name = r.get("Company Name", r.get("CompanyName", ""))
            full_address = r.get("Address", "")

            pin_m = re.search(r"\b\d{6}\b", full_address)
            if pin_m:
                pincode = pin_m.group(0)

            c_person = r.get("Contact person", "")
            c_desig = r.get("Designation", "")
            c_mobile = r.get("Mobile no", r.get("Mobile No", ""))
            representatives = parse_representatives(c_person, c_desig, c_mobile)

            tele_raw = r.get("Telephone", "")
            phones = parse_phone_numbers(f"{c_mobile}, {tele_raw}")

            email_field = r.get("Email id", r.get("Email", ""))
            if email_field:
                for token in re.split(r"[,;]", email_field):
                    token = token.strip()
                    if "@" in token:
                        emails.append(token)
                    elif "www." in token or token.startswith("http") or re.search(r"\.(in|com|net|org|co)$", token):
                        website = token

            category = r.get("Category", "")
            if category: nature_parts.append(f"Category: {category}")

        records.append(normalize_json_record({
            "panel_no": panel_no,
            "raw_name": raw_name,
            "address": full_address,
            "pincode": pincode,
            "phones": phones,
            "emails": emails,
            "website": website,
            "representatives": representatives,
            "nature_of_business": " | ".join(nature_parts)
        }))

    return records


# ==============================================================================
# 1. PAGE SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Directory Contact Portal", page_icon="🏢", layout="wide")
st.title("🏢 Contact Portal & Agentic Entity Disambiguation Engine")

# Fetch credentials from Streamlit Secrets
GEMINI_KEYS = st.secrets.get("GEMINI_API_KEYS", [])
CF_ACCOUNT_ID = st.secrets.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_DATABASE_ID = st.secrets.get("CLOUDFLARE_DATABASE_ID", "")
CF_API_TOKEN = st.secrets.get("CLOUDFLARE_API_TOKEN", "")

if not GEMINI_KEYS or not CF_ACCOUNT_ID or not CF_DATABASE_ID or not CF_API_TOKEN:
    st.error("⚠️ Secrets missing! Please configure GEMINI_API_KEYS and Cloudflare details in `.streamlit/secrets.toml`.")
    st.stop()

# Initialize core clients in session state
if "rotator" not in st.session_state:
    st.session_state.rotator = GeminiKeyRotator(GEMINI_KEYS)

rotator = st.session_state.rotator
db = CloudflareD1(CF_ACCOUNT_ID, CF_DATABASE_ID, CF_API_TOKEN)
vision_agent = VisionExtractionAgent(rotator)

# ==============================================================================
# 2. UI TABS
# ==============================================================================
tab_search, tab_ingest, tab_json_upload = st.tabs([
    "🔍 Search & Resolve Entities", 
    "📤 Ingest Directory PDF",
    "📁 Import CSV / Excel / JSON File"
])

# --- TAB 1: SEARCH PORTAL ---
with tab_search:
    st.subheader("Disambiguation Search")
    st.caption("Search for any full company name, acronym (e.g. 'XYZ'), or keyword:")

    search_query = st.text_input("Enter search phrase:", placeholder="e.g. XYZ, Automotive, or Executive Name")

    if search_query:
        with st.spinner("Executing Agentic Search on Cloudflare D1..."):
            results = db.search_companies_agentic(search_query)

        if results:
            st.success(f"Found {len(results)} matching entity/entities:")
            for item in results:
                canonical_name = item.get("canonical_name", "Unknown Entity")
                panel_no = item.get("panel_no", "N/A")
                
                with st.expander(f"🏢 **{canonical_name}** (Panel #{panel_no})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Contact Details:**")
                        st.write(f"📍 **Address:** {item.get('address') or 'N/A'}")
                        st.write(f"📮 **Pincode:** {item.get('pincode') or 'N/A'}")
                        st.write(f"🌐 **Website:** {item.get('website') or 'N/A'}")
                        
                        emails = json.loads(item.get("emails") or "[]")
                        phones = json.loads(item.get("phones") or "[]")
                        st.write(f"✉️ **Emails:** {', '.join(emails) if emails else 'N/A'}")
                        st.write(f"📞 **Phones:** {', '.join(phones) if phones else 'N/A'}")
                        
                        aliases = json.loads(item.get("aliases") or "[]")
                        if aliases:
                            st.write(f"🏷️ **Aliases / Acronyms:** {', '.join(aliases)}")

                    with col2:
                        st.markdown("**Key Executives / Representatives:**")
                        reps = json.loads(item.get("representatives") or "[]")
                        if reps and isinstance(reps, list):
                            for r in reps:
                                name = r.get('name', 'N/A')
                                desig = r.get('designation', '')
                                mob = r.get('mobile', '')
                                st.write(f"- 👤 **{name}** {f'({desig})' if desig else ''} {f'| Mob: {mob}' if mob else ''}")
                        else:
                            st.write("No executives listed.")
                            
                        st.markdown("**Nature of Business (NB):**")
                        st.info(item.get("nature_of_business") or "No business description provided.")
        else:
            st.warning("No records matched your search query.")

# --- TAB 2: PDF INGESTION PIPELINE ---
with tab_ingest:
    st.subheader("Agentic PDF Ingestion Pipeline")
    uploaded_pdf = st.file_uploader("Upload Directory PDF", type=["pdf"])

    if uploaded_pdf and st.button("🚀 Process PDF Directory"):
        doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        total_pages = len(doc)
        
        progress_bar = st.progress(0)
        status_box = st.empty()
        
        total_saved = 0

        for page_idx in range(total_pages):
            status_box.text(f"Processing Page {page_idx + 1} of {total_pages} using Gemini 3.6 Flash...")
            
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            try:
                raw_companies = vision_agent.extract_from_page(img)
                processed_batch = []
                
                for raw_comp in raw_companies:
                    normalized = normalize_json_record(raw_comp)
                    processed_comp = DisambiguationAgent.process_entity(normalized)
                    audited_comp = QualityAuditAgent.audit(processed_comp)
                    processed_batch.append(audited_comp)

                if processed_batch:
                    db.bulk_insert_companies(processed_batch)
                    total_saved += len(processed_batch)

            except Exception as e:
                st.error(f"Page {page_idx + 1} processing failed: {e}")

            progress_bar.progress((page_idx + 1) / total_pages)

        status_box.empty()
        st.balloons()
        st.success(f"Successfully processed {total_pages} page(s) and committed {total_saved} companies to Cloudflare D1!")

# --- TAB 3: SMART DUAL-FORMAT CSV / EXCEL / JSON IMPORTER ---
with tab_json_upload:
    st.subheader("📁 Universal Lossless Directory Importer")
    st.caption("Supports both Sheet formats (`.csv` or `.xlsx`) and `.json`. Automatically detects encoding (CP1252/UTF-8), fixes exponential numbers, routes URLs, and parses multi-contacts.")

    col_file, col_text = st.columns(2)
    with col_file:
        file_upload = st.file_uploader("Upload Directory File (`.csv`, `.xlsx`, `.json`)", type=["csv", "xlsx", "xls", "json"])
    with col_text:
        pasted_json_text = st.text_area("Or Paste Raw Directory JSON Array:", height=180, placeholder="[...] ")

    if st.button("📥 Validate, Repair & Ingest to DB"):
        parsed_data = None

        if file_upload:
            filename = file_upload.name.lower()
            try:
                if filename.endswith(".json"):
                    parsed_data = json.loads(file_upload.read().decode("utf-8"))
                else:
                    parsed_data = dual_format_smart_csv_parser(file_upload)
                    st.info(f"📊 Successfully parsed {len(parsed_data)} company records from file.")
            except Exception as e:
                st.error(f"Failed to parse uploaded file: {e}")

        elif pasted_json_text.strip():
            try:
                parsed_data = json.loads(pasted_json_text.strip())
            except Exception as e:
                st.error(f"Failed to parse pasted text: {e}")

        if parsed_data:
            if isinstance(parsed_data, list):
                cleaned_batch = []
                
                with st.spinner("Applying Disambiguation & Quality Audit Agents..."):
                    for raw_entry in parsed_data:
                        normalized_entry = normalize_json_record(raw_entry)
                        processed_entry = DisambiguationAgent.process_entity(normalized_entry)
                        audited_entry = QualityAuditAgent.audit(processed_entry)
                        cleaned_batch.append(audited_entry)

                with st.spinner(f"Writing {len(cleaned_batch)} cleaned records to Cloudflare D1..."):
                    db.bulk_insert_companies(cleaned_batch)
                    
                st.balloons()
                st.success(f"🎉 Success! Perfectly parsed and imported {len(cleaned_batch)} records into Cloudflare D1 database.")
            else:
                st.error("Invalid data format. Expected an array of company records.")
