import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
import logging
import base64
import zipfile              
from io import BytesIO 
from datetime import datetime, timedelta
import os
import warnings
import difflib  
import gc  
import time  
import ssl
import certifi
import requests  
import re  
from werkzeug.security import generate_password_hash, check_password_hash
from paddleocr import PaddleOCR
import boto3
from botocore.client import Config
from pymongo import MongoClient
import pymongo

# --- EMAIL DEPENDENCIES ---
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# --- SUPPRESS AI & C++ NOISE ---
os.environ["GLOG_minloglevel"] = "3"   
os.environ["KMP_WARNINGS"] = "0"       
warnings.filterwarnings("ignore")      

from parser_worker import extract_metadata_from_text, CardMetadata

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="CapitupIndia E-Card Portal", page_icon="🪪", layout="wide")

# --- LOCAL STORAGE VOLUME CONFIGURATION ---
LOCAL_STORAGE_DIR = "./local_ecards"
os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)

# Static file management paths (Asset Vault)
CLAIM_FORM_PATH = "claim_form.pdf"
POSTER_PATH = "poster.png"
LOGO_PATH = "logo.png"

# Default live Google Apps Script API endpoint
DEFAULT_GAS_URL = "https://script.google.com/macros/s/AKfycbwexxFRlk43f3-SP6fH5VsgSeGpf-cDQXkETNlUT8OJ06AlOGirJ39ivP44HszMMNpAFg/exec"

# --- GLOBAL UTILITY & HELPERS ---
def guess_column(columns, keywords, index_fallback=0):
    for col in columns:
        for kw in keywords:
            if kw.upper() in str(col).upper(): return col
    return columns[index_fallback]

def parse_int_safe(val):
    if pd.isna(val):
        return None
    val_str = str(val).split('.')[0].strip()
    try:
        return int(val_str)
    except ValueError:
        return None

def clean_and_align_dataframe(df):
    if df.empty:
        return df
    first_col_name = str(df.columns[0]).upper()
    if "TOTAL RECORD" in first_col_name or "RECORD COUNT" in first_col_name:
        header_row_idx = None
        for idx in range(min(5, len(df))):
            row_vals = [str(x).strip().upper() for x in df.iloc[idx].values]
            if any("POLICY" in val for val in row_vals):
                header_row_idx = idx
                break
        if header_row_idx is not None:
            df.columns = [str(x).strip() for x in df.iloc[header_row_idx].values]
            df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def robust_guess_column(columns, primary_keywords, fallback_keywords=None):
    columns_upper = [str(c).strip().upper() for c in columns]
    for kw in primary_keywords:
        kw_up = kw.strip().upper()
        if kw_up in columns_upper:
            return columns[columns_upper.index(kw_up)]
    if fallback_keywords:
        for kw in fallback_keywords:
            kw_up = kw.strip().upper()
            if kw_up in columns_upper:
                return columns[columns_upper.index(kw_up)]
    for idx, col in enumerate(columns_upper):
        for kw in primary_keywords:
            kw_up = kw.strip().upper()
            if len(kw_up) > 1 and kw_up in col:
                return columns[idx]
    if fallback_keywords:
        for idx, col in enumerate(columns_upper):
            for kw in fallback_keywords:
                kw_up = kw.strip().upper()
                if len(kw_up) > 1 and kw_up in col:
                    return columns[idx]
    return None

# --- HYBRID CREDENTIALS INITIALIZATION ---
try:
    MONGO_URI = st.secrets["mongo"]["uri"]
    MONGO_DBNAME = st.secrets["mongo"]["dbname"]
except KeyError:
    st.error("🚨 CRITICAL ERROR: Could not find MongoDB Atlas [mongo] credentials in secrets!")
    st.stop()

# CLOUDFLARE R2 SETUP
R2_ENABLED = False
if "r2" in st.secrets:
    R2_CONFIG = dict(st.secrets["r2"])
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
    R2_ENABLED = True
else:
    st.warning("⚠️ Cloudflare R2 credentials ['r2'] not found in secrets. Cloudflare uploads will be skipped.")

@st.cache_resource(show_spinner="Loading AI Vision Engine... (First load takes a few seconds)")
def load_ocr_engine():
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    return PaddleOCR(use_textline_orientation=True, lang='en')

# --- MONGO DATABASE CONNECTIONS ---
@st.cache_resource
def get_mongo_client():
    return MongoClient(
        MONGO_URI,
        tls=True,
        tlsAllowInvalidCertificates=True
    )

def get_db():
    client = get_mongo_client()
    return client[MONGO_DBNAME]

def init_db():
    db = get_db()
    db.users.create_index("username", unique=True)
    db.ecards.create_index(
        [("policy_no", pymongo.ASCENDING), ("emp_id", pymongo.ASCENDING), ("card_type", pymongo.ASCENDING)],
        unique=True,
        name="unique_emp_card_type"
    )
    db.card_members.create_index("emp_id")
    db.directory.create_index([("emp_id", pymongo.ASCENDING), ("policy_no", pymongo.ASCENDING)], unique=True)

def authenticate_user(username, password):
    db = get_db()
    user = db.users.find_one({"username": username})
    return user and check_password_hash(user['password_hash'], password)

def create_user(username, password):
    db = get_db()
    try:
        db.users.insert_one({
            "username": username,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.utcnow()
        })
        return True
    except pymongo.errors.DuplicateKeyError:
        return False

# Hierarchical Campaign Folder Router
def save_card_to_db(emp_id, pdf_bytes, username, family_members, policy_no="UNKNOWN", card_type="BASE", company_name=None):
    clean_emp_id = str(emp_id).strip().upper()
    clean_policy_no = str(policy_no).strip().upper()
    clean_company_name = str(company_name).strip().upper() if company_name else None
    
    # Windows-forbidden characters regex sanitizer: \ / : * ? " < > | [1]
    illegal_chars = r'[\\/*?:"<>|]'
    
    sanitized_policy = re.sub(illegal_chars, "", clean_policy_no).strip().replace(" ", "_")
    if not sanitized_policy:
        sanitized_policy = "UNKNOWN_POLICY"
    
    if clean_company_name:
        sanitized_company = re.sub(illegal_chars, "", clean_company_name).strip().replace(" ", "_")
        if not sanitized_company:
            sanitized_company = "UNKNOWN_COMPANY"
        # Structure: Company Name / Policy Number / Card Type / Employee ID.pdf [1]
        sub_folder = os.path.join(sanitized_company, sanitized_policy, card_type)
        file_key = f"ecards/{sanitized_company}/{sanitized_policy}/{card_type}/{clean_emp_id}.pdf"
    else:
        # Structure: Policy Number / Card Type / Employee ID.pdf (Fallback if Company is missing) [1]
        sub_folder = os.path.join(sanitized_policy, card_type)
        file_key = f"ecards/{sanitized_policy}/{card_type}/{clean_emp_id}.pdf"
    
    # 1. LOCAL STORAGE BACKUP
    card_folder = os.path.join(LOCAL_STORAGE_DIR, sub_folder)
    os.makedirs(card_folder, exist_ok=True)
    local_file_path = os.path.join(card_folder, f"{clean_emp_id}.pdf")
    with open(local_file_path, "wb") as f:
        f.write(pdf_bytes)
        
    # 2. CLOUDFLARE R2 HYBRID STORAGE
    if R2_ENABLED:
        try:
            s3_client.put_object(
                Bucket=R2_CONFIG["bucket_name"],
                Key=file_key,
                Body=pdf_bytes,
                ContentType="application/pdf"
            )
        except Exception as e:
            logging.error(f"Cloudflare upload failed for {clean_emp_id} (Policy: {clean_policy_no}): {e}")

    # 3. MONGODB METADATA SYNCHRONIZATION
    db = get_db()
    db.ecards.update_one(
        {"emp_id": clean_emp_id, "policy_no": clean_policy_no, "card_type": card_type},
        {"$set": {
            "emp_id": clean_emp_id,
            "policy_no": clean_policy_no,
            "company_name": clean_company_name,
            "card_type": card_type,
            "file_path": local_file_path,
            "r2_key": file_key,
            "uploaded_by": username,
            "upload_date": datetime.utcnow(),
            "email_sent": False  
        }},
        upsert=True
    )
               
    db.card_members.delete_many({"emp_id": clean_emp_id, "policy_no": clean_policy_no, "policy_type": card_type})
    member_docs = []
    for member in family_members:
        member_docs.append({
            "emp_id": clean_emp_id,
            "name": member.name,
            "policy_no": clean_policy_no,
            "policy_type": card_type,
            "card_no": member.card_no,
            "relationship": member.relationship,
            "age": member.age,
            "valid_up_to": member.valid_up_to
        })
    if member_docs:
        db.card_members.insert_many(member_docs)

def save_employee_to_directory(emp_id, name, email, policy_no):
    db = get_db()
    db.directory.update_one(
        {"emp_id": str(emp_id).strip().upper(), "policy_no": str(policy_no).strip().upper()},
        {"$set": {
            "emp_id": str(emp_id).strip().upper(),
            "name": str(name).strip(),
            "email": str(email).strip().lower(),
            "policy_no": str(policy_no).strip().upper(),
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )

def get_cards_from_db(emp_id, policy_no=None):
    db = get_db()
    query = {"emp_id": str(emp_id).strip().upper()}
    if policy_no:
        query["policy_no"] = str(policy_no).strip().upper()
        
    db_results = list(db.ecards.find(query))
    
    cards_list = []
    for row in db_results:
        local_path = row.get("file_path")
        pdf_data = None
        
        if local_path and os.path.exists(local_path):
            try:
                with open(local_path, "rb") as f:
                    pdf_data = f.read()
            except Exception as e:
                logging.error(f"Error reading local file {local_path}: {e}")
        
        if not pdf_data and R2_ENABLED and "r2_key" in row:
            try:
                response = s3_client.get_object(Bucket=R2_CONFIG["bucket_name"], Key=row["r2_key"])
                pdf_data = response["Body"].read()
                
                if local_path:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(pdf_data)
            except Exception as e:
                logging.error(f"Cloudflare R2 download fallback failed for {emp_id}: {e}")
                
        if pdf_data:
            cards_list.append({
                "card_type": row["card_type"],
                "policy_no": row["policy_no"],
                "pdf_data": pdf_data,
                "uploaded_by": row.get("uploaded_by", "UNKNOWN"),
                "upload_date": row.get("upload_date", datetime.utcnow())
            })
    return cards_list

def get_card_from_db(emp_id):
    cards = get_cards_from_db(emp_id)
    return cards[0] if cards else None

def get_members_from_db(emp_id=None):
    db = get_db()
    if emp_id:
        cursor = db.card_members.find({"emp_id": emp_id}).sort("relationship", -1)
    else:
        cursor = db.card_members.find().sort("emp_id", 1)
        
    results = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        results.append(doc)
    return results

def get_bulk_cards_from_db(emp_ids, policy_no=None):
    if not emp_ids:
        return []
    db = get_db()
    query = {"emp_id": {"$in": emp_ids}}
    if policy_no:
        query["policy_no"] = str(policy_no).strip().upper()
        
    db_results = list(db.ecards.find(query))
    
    results = []
    for row in db_results:
        local_path = row.get("file_path")
        pdf_data = None
        
        if local_path and os.path.exists(local_path):
            try:
                with open(local_path, "rb") as f:
                    pdf_data = f.read()
            except Exception as e:
                logging.error(f"Failed loading local file {local_path}: {e}")
                
        if not pdf_data and R2_ENABLED and "r2_key" in row:
            try:
                response = s3_client.get_object(Bucket=R2_CONFIG["bucket_name"], Key=row["r2_key"])
                pdf_data = response["Body"].read()
                
                if local_path:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(pdf_data)
            except Exception as e:
                logging.error(f"Failed loading cloud backup for bulk {row['emp_id']}: {e}")
                
        if pdf_data:
            results.append({
                "emp_id": row["emp_id"],
                "card_type": row["card_type"],
                "policy_no": row["policy_no"],
                "pdf_data": pdf_data
            })
    return results

# --- SMTP EMAIL DISPATCH ENGINE ---
def send_multi_ecard_email(recipient_email, subject, body_html, cards_list):
    try:
        SMTP_CONFIG = st.secrets["smtp"]
        
        msg = MIMEMultipart('mixed')
        msg['From'] = SMTP_CONFIG["sender_email"]
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        msg_related = MIMEMultipart('related')
        msg.attach(msg_related)
        
        if os.path.exists(POSTER_PATH):
            poster_tag = """
            <tr>
              <td align="center" style="padding: 0 40px 30px 40px;">
                <img src="cid:poster_image" alt="Mediclaim Summary Poster" style="width: 100%; max-width: 520px; height: auto; border-radius: 6px; display: block;" />
              </td>
            </tr>
            """
            body_html = body_html.replace("<!-- FOOTER -->", poster_tag + "\n<!-- FOOTER -->")
            
        # HTML text MIMEText must always be attached FIRST inside the related container
        msg_related.attach(MIMEText(body_html, 'html'))
        
        if os.path.exists(LOGO_PATH):
            from email.mime.image import MIMEImage
            with open(LOGO_PATH, "rb") as f:
                logo_data = f.read()
            msg_logo = MIMEImage(logo_data)
            msg_logo.add_header('Content-ID', '<logo_image>')
            msg_logo.add_header('Content-Disposition', 'inline', filename="logo.png")
            msg_related.attach(msg_logo)
        
        if os.path.exists(POSTER_PATH):
            from email.mime.image import MIMEImage
            with open(POSTER_PATH, "rb") as f:
                img_data = f.read()
            msg_img = MIMEImage(img_data)
            msg_img.add_header('Content-ID', '<poster_image>')
            msg_img.add_header('Content-Disposition', 'inline', filename="poster.png")
            msg_related.attach(msg_img)
            
        for card in cards_list:
            pdf_bytes = bytes(card["pdf_data"])
            card_label = card["card_type"]
            
            attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
            attachment.add_header('Content-Disposition', 'attachment', filename=f"HealthCard_{card_label}.pdf")
            msg.attach(attachment)
            
        if os.path.exists(CLAIM_FORM_PATH):
            with open(CLAIM_FORM_PATH, "rb") as f:
                claim_bytes = f.read()
            claim_attachment = MIMEApplication(claim_bytes, _subtype="pdf")
            claim_attachment.add_header('Content-Disposition', 'attachment', filename="Reimbursement_Claim_Form.pdf")
            msg.attach(claim_attachment)
            
        port = int(SMTP_CONFIG["port"])
        if port == 465:
            with smtplib.SMTP_SSL(SMTP_CONFIG["server"], port, timeout=120) as server:
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(SMTP_CONFIG["username"], recipient_email, msg.as_string())
        else:
            with smtplib.SMTP(SMTP_CONFIG["server"], port, timeout=120) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(SMTP_CONFIG["username"], recipient_email, msg.as_string())
        return True
    except Exception as e:
        logging.error(f"Failed to send multi-attachment email to {recipient_email}: {e}")
        return False

# --- EXTRACTION BOUNDARY LOGIC ---
def detect_card_boundaries(page):
    try:
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
        row_sums = np.sum(thresh, axis=1)
        gaps = np.where(row_sums < (thresh.shape[1] * 0.05))[0] 

        height = pix.h
        split_y = [0]
        for i in range(1, len(gaps)):
            if gaps[i] - gaps[i-1] > 20: 
                split_y.append(gaps[i])
        split_y.append(height)

        rects = []
        for i in range(len(split_y)-1):
            y1, y2 = split_y[i], split_y[i+1]
            if y2 - y1 > 150:  
                rects.append(fitz.Rect(0, y1 * (page.rect.height / height), page.rect.width, y2 * (page.rect.height / height)))
        if not rects: raise ValueError()
        return rects
    except:
        h3 = page.rect.height / 3
        return [fitz.Rect(0, 0, page.rect.width, h3), fitz.Rect(0, h3, page.rect.width, h3*2), fitz.Rect(0, h3*2, page.rect.width, page.rect.height)]

# --- GOOGLE FORM CONTROLLER UTILITIES (Web App API Wrapper) ---
def get_form_status(api_url):
    try:
        r = requests.get(api_url + "?action=status", timeout=10)
        return r.text.strip().upper()
    except Exception as e:
        logging.error(f"Failed to fetch Google Form status: {e}")
        return "UNKNOWN / DISCONNECTED"

def set_form_status(api_url, action):
    try:
        r = requests.get(api_url + f"?action=status", timeout=10)
        return r.text.strip().upper()
    except Exception as e:
        logging.error(f"Failed to set Google Form status to {action}: {e}")
        return None

def schedule_form_close(api_url, hours):
    try:
        r = requests.get(api_url + f"?action=schedule&hours={hours}", timeout=10)
        return r.text.strip()
    except Exception as e:
        logging.error(f"Failed to schedule Google Form closure: {e}")
        return None

def get_gas_url_from_db():
    db = get_db()
    setting = db.settings.find_one({"key": "gas_url"})
    if setting:
        return setting["value"]
    return DEFAULT_GAS_URL

def save_gas_url_to_db(url):
    db = get_db()
    db.settings.update_one(
        {"key": "gas_url"},
        {"$set": {"value": url.strip()}},
        upsert=True
    )

def get_deadline_from_db(policy_no):
    db = get_db()
    setting = db.settings.find_one({"key": f"deadline_{policy_no}"})
    if setting:
        return setting["value"]
    return "Not Set"

def save_deadline_to_db(policy_no, deadline_str):
    db = get_db()
    db.settings.update_one(
        {"key": f"deadline_{policy_no}"},
        {"$set": {"value": deadline_str}},
        upsert=True
    )

# --- INITIALIZE HYBRID DATABASE SCHEMA ---
init_db()

if "failed_emails" not in st.session_state:
    st.session_state.failed_emails = []

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 E-Card System Portal")
        st.markdown("Please log in or register to access the database.")
        st.markdown("---")
        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register New User"])
        
        with tab_login:
            with st.form("login_form"):
                user_input = st.text_input("Username")
                pass_input = st.text_input("Password", type="password")
                if st.form_submit_button("Login", width="stretch", type="primary"):
                    if authenticate_user(user_input, pass_input):
                        st.session_state.logged_in = True
                        st.session_state.username = user_input
                        st.rerun()
                    else:
                        st.error("❌ Invalid Credentials.")
                        
        with tab_register:
            with st.form("register_form"):
                new_user = st.text_input("Choose a Username")
                new_pass = st.text_input("Choose a Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Register", width="stretch", type="primary"):
                    if new_pass != confirm_pass: st.error("❌ Passwords do not match!")
                    elif create_user(new_user, new_pass): st.success("✅ Account created!")
                    else: st.error("⚠️ Username already exists.")
    st.stop() 

st.sidebar.title(f"👤 Welcome, {st.session_state.username}")
if st.sidebar.button("Logout", type="primary", width="stretch"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.title("🪪 CapitupIndia E-Card Database Portal")
if R2_ENABLED:
    st.success("✅ Cloudflare R2 Connected! Cards will automatically sync to the global CDN.")

main_tab1, main_tab4, main_tab2, main_tab3, main_tab5, main_tab6 = st.tabs([
    "📥 Ingest E-Cards", "📥 Bulk Retrieval", "📊 Global Directory", 
    "📤 Upload Master (V1)", "🔍 Search Individual", "✉️ Email Distribution"
])

ocr_engine = load_ocr_engine()

# --- TAB 1: MODULAR INGESTION SYSTEM ---
with main_tab1:
    col_base_module, col_topup_module = st.columns(2)
    with col_base_module:
        st.markdown("<div style='border: 1px solid #0d6efd; padding: 15px; border-radius: 8px; background-color: #f8f9fa;'>", unsafe_allow_html=True)
        st.subheader("🟦 Base Policy Ingestion Module")
        base_excel = st.file_uploader("1. Upload Base Member List (Excel/CSV)", type=["xlsx", "xls", "csv"], key="bu_base")
        base_pdfs = st.file_uploader("2. Drop Base E-Card PDFs (Select files from Base folder)", type=["pdf"], accept_multiple_files=True, key="pdf_base")
        
        if st.button("🚀 Process & Ingest Base Policies", type="primary", width="stretch", key="btn_base"):
            if not (base_excel and base_pdfs):
                st.error("Please provide both the Base Member List and the matching Base PDFs.")
            else:
                with st.spinner("Processing Base folder cards..."):
                    if base_excel.name.endswith('.csv'): df_base = pd.read_csv(base_excel)
                    else: df_base = pd.read_excel(base_excel)
                    
                    df_base = clean_and_align_dataframe(df_base)
                    df_base_cols = list(df_base.columns)
                    
                    emp_col = "Hat" if "Hat" in df_base_cols else ("Co" if "Co" in df_base_cols else guess_column(df_base_cols, ["EMP", "ID"], index_fallback=3))
                    g_policy = robust_guess_column(df_base_cols, ["POLICY NO", "POLICY", "POL"])
                    g_name = robust_guess_column(df_base_cols, ["MEMBER NAME", "NAME", "INSURED"])
                    g_card = robust_guess_column(df_base_cols, ["ID CARD NO", "CARD NO", "CARD NUMBER"])
                    g_relation = robust_guess_column(df_base_cols, ["RELATION", "RELATIONSHIP", "RELATI", "REL"])
                    g_age = robust_guess_column(df_base_cols, ["AGE", "A"]) 
                    g_expiry = robust_guess_column(df_base_cols, ["RISK EXPIRY DATE", "EXPIRY", "VALID", "EXPIR"])
                    email_col = robust_guess_column(df_base_cols, ["EMAIL", "ACCESS", "MAIL"])
                    
                    # Track Company Name column inside Base sheet
                    g_company = robust_guess_column(df_base_cols, ["COMPANY NAME", "COMPANY", "CORPORATE", "CLIENT"])
                    base_company_name = str(df_base.iloc[0][g_company]).strip().upper() if g_company else None
                    
                    base_policy_no_rule = str(df_base.iloc[0][g_policy]).strip().upper() if g_policy else "UNKNOWN"
                    
                    base_count = 0
                    for pdf_file in base_pdfs:
                        emp_id = os.path.splitext(pdf_file.name)[0].strip().upper()
                        if emp_col and emp_col in df_base_cols:
                            matching_rows = df_base[df_base[emp_col].astype(str).str.strip() == emp_id]
                            if not matching_rows.empty:
                                family_members = []
                                
                                # Prioritize 'SELF' row for core welcome directory
                                primary_row = None
                                if g_relation:
                                    for _, r in matching_rows.iterrows():
                                        if str(r[g_relation]).strip().upper() in ["SELF", "PRIMARY", "EMPLOYEE", "PROPOSER"]:
                                            primary_row = r
                                            break
                                if primary_row is None:
                                    primary_row = matching_rows.iloc[0]
                                    
                                email_val = str(primary_row[email_col]).strip() if email_col else ""
                                save_employee_to_directory(emp_id, str(primary_row[g_name]).strip(), email_val, base_policy_no_rule)
                                
                                for _, row in matching_rows.iterrows():
                                    parsed_age = parse_int_safe(row[g_age]) if g_age else None
                                    family_members.append(CardMetadata(
                                        emp_id=emp_id,
                                        name=str(row[g_name]).strip() if g_name else "UNKNOWN",
                                        policy_no=base_policy_no_rule,
                                        policy_type="BASE",
                                        card_no=str(row[g_card]).strip() if g_card else "UNKNOWN",
                                        relationship=str(row[g_relation]).strip() if g_relation else "SELF",
                                        age=parsed_age,
                                        valid_up_to=str(row[g_expiry]).strip() if g_expiry else "UNKNOWN"
                                    ))
                                
                                pdf_bytes = pdf_file.getvalue()
                                save_card_to_db(emp_id, pdf_bytes, st.session_state.username, family_members, base_policy_no_rule, "BASE", base_company_name)
                                base_count += 1
                    
                    st.success(f"✅ Ingested **{base_count}** Base policy bundles securely.")
                    gc.collect()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_topup_module:
        st.markdown("<div style='border: 1px solid #fd7e14; padding: 15px; border-radius: 8px; background-color: #f8f9fa;'>", unsafe_allow_html=True)
        st.subheader("🟧 Top Up Policy Ingestion Module")
        topup_excel = st.file_uploader("1. Upload Top Up Member List (Excel/CSV)", type=["xlsx", "xls", "csv"], key="bu_topup")
        topup_pdfs = st.file_uploader("2. Drop Top Up E-Card PDFs (Select files from Topup folder)", type=["pdf"], accept_multiple_files=True, key="pdf_topup")
        
        if st.button("🚀 Process & Ingest Top Up Policies", type="primary", width="stretch", key="btn_topup"):
            if not (topup_excel and topup_pdfs):
                st.error("Please provide both the Top Up Member List and the matching Top Up PDFs.")
            else:
                with st.spinner("Processing Topup folder cards..."):
                    if topup_excel.name.endswith('.csv'): df_top = pd.read_csv(topup_excel)
                    else: df_top = pd.read_excel(topup_excel)
                    
                    df_top = clean_and_align_dataframe(df_top)
                    df_top_cols = list(df_top.columns)
                    
                    emp_col_top = "Hat" if "Hat" in df_top_cols else ("Co" if "Co" in df_top_cols else robust_guess_column(df_top_cols, ["EMP", "ID"], ["CODE", "CO"]))
                    g_policy_top = robust_guess_column(df_top_cols, ["POLICY NO", "POLICY", "POL"])
                    g_name_top = robust_guess_column(df_top_cols, ["MEMBER NAME", "NAME", "INSURED"])
                    g_card_top = robust_guess_column(df_top_cols, ["ID CARD NO", "CARD NO", "CARD NUMBER"])
                    g_relation_top = robust_guess_column(df_top_cols, ["RELATION", "RELATIONSHIP", "RELATI", "REL"])
                    g_age_top = robust_guess_column(df_top_cols, ["AGE", "A"]) 
                    g_expiry_top = robust_guess_column(df_top_cols, ["RISK EXPIRY DATE", "EXPIRY", "VALID", "EXPIR"])
                    email_col_top = robust_guess_column(df_top_cols, ["EMAIL", "ACCESS", "MAIL"])
                    
                    # Capture dynamic Company Name from sheet
                    g_company_top = robust_guess_column(df_top_cols, ["COMPANY NAME", "COMPANY", "CORPORATE", "CLIENT"])
                    topup_company_name = str(df_top.iloc[0][g_company_top]).strip().upper() if g_company_top else None
                    
                    topup_policy_no_rule = str(df_top.iloc[0][g_policy_top]).strip().upper() if g_policy_top else "UNKNOWN"
                    
                    topup_count = 0
                    for pdf_file in topup_pdfs:
                        emp_id = os.path.splitext(pdf_file.name)[0].strip().upper()
                        if emp_col_top and emp_col_top in df_top_cols:
                            matching_rows = df_top[df_top[emp_col_top].astype(str).str.strip() == emp_id]
                            if not matching_rows.empty:
                                family_members = []
                                
                                # Prioritize 'SELF' row for core welcome directory
                                primary_row_top = None
                                if g_relation_top:
                                    for _, r in matching_rows.iterrows():
                                        if str(r[g_relation_top]).strip().upper() in ["SELF", "PRIMARY", "EMPLOYEE", "PROPOSER"]:
                                            primary_row_top = r
                                            break
                                if primary_row_top is None:
                                    primary_row_top = matching_rows.iloc[0]
                                    
                                email_val = str(primary_row_top[email_col_top]).strip() if email_col_top else ""
                                save_employee_to_directory(emp_id, str(primary_row_top[g_name_top]).strip(), email_val, topup_policy_no_rule)
                                
                                for _, row in matching_rows.iterrows():
                                    parsed_age_top = parse_int_safe(row[g_age_top]) if g_age_top else None
                                    family_members.append(CardMetadata(
                                        emp_id=emp_id,
                                        name=str(row[g_name_top]).strip() if g_name_top else "UNKNOWN",
                                        policy_no=topup_policy_no_rule,
                                        policy_type="TOPUP",
                                        card_no=str(row[g_card_top]).strip() if g_card_top else "UNKNOWN",
                                        relationship=str(row[g_relation_top]).strip() if g_relation_top else "SELF",
                                        age=parsed_age_top,
                                        valid_up_to=str(row[g_expiry_top]).strip() if g_expiry_top else "UNKNOWN"
                                    ))
                                
                                pdf_bytes = pdf_file.getvalue()
                                save_card_to_db(emp_id, pdf_bytes, st.session_state.username, family_members, topup_policy_no_rule, "TOPUP", topup_company_name)
                                topup_count += 1
                            
                    st.success(f"✅ Ingested **{topup_count}** Top Up policy bundles securely.")
                    gc.collect()
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: BULK RETRIEVAL ---
with main_tab4:
    st.markdown("### 📥 Bulk E-Card Retrieval")
    bulk_policy_filter = st.text_input("Target Policy Number (Optional - helps isolate overlapping employee IDs):", placeholder="e.g. 11022026", key="bulk_policy_no")
    bulk_input = st.text_area("List of Employee IDs (comma or space separated):", height=150)
    
    if st.button("📦 Fetch Cards & Build ZIP", type="primary", width="stretch"):
        if bulk_input.strip():
            with st.spinner("Fetching hybrid database filepaths..."):
                clean_ids = list(set([i.strip().upper() for i in bulk_input.replace(',', ' ').split() if i.strip()]))
                
                # Tenant-isolated bulk retrieval
                found_cards = get_bulk_cards_from_db(clean_ids, policy_no=bulk_policy_filter)
                
                if found_cards:
                    bulk_zip_buffer = BytesIO()
                    with zipfile.ZipFile(bulk_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for card in found_cards:
                            # Prepend policy to filename if there is any overlap risk
                            zf.writestr(f"{card['policy_no']}_{card['emp_id']}_{card['card_type']}_ECard.pdf", bytes(card['pdf_data']))
                    st.session_state.bulk_zip_data = bulk_zip_buffer.getvalue()
                    
                    missing_ids = [i for i in clean_ids if i not in [c['emp_id'] for c in found_cards]]
                    msg = f"✅ Successfully packaged **{len(found_cards)}** files."
                    if missing_ids: msg += f"\n\n⚠️ **Missing:** {', '.join(missing_ids)}"
                    st.info(msg)
                else:
                    st.error("❌ None of the requested IDs were found in DB.")

    if st.session_state.get('bulk_zip_data'):
        st.download_button("📥 Download Batch ZIP", data=st.session_state.bulk_zip_data, file_name="Bulk_ECards.zip", mime="application/zip", type="primary", width="stretch")

# --- TAB 3: GLOBAL DIRECTORY ---
with main_tab2:
    st.markdown("### 📊 Active Employee Directory")
    all_members = get_members_from_db()
    if all_members:
        df_all = pd.DataFrame(all_members).drop(columns=['_id', 'id'], errors='ignore')
        search_term = st.text_input("🔍 Search by Name, Emp Code, or Policy Number:")
        if search_term:
            df_all = df_all[df_all['name'].str.contains(search_term, case=False, na=False) | 
                            df_all['emp_id'].str.contains(search_term, case=False, na=False) |
                            df_all['policy_no'].str.contains(search_term, case=False, na=False)]
        st.dataframe(df_all, hide_index=True, width="stretch")

# --- TAB 4: UPLOAD V1 MASTER / SMART BULK INGESTION ---
with main_tab3:
    st.markdown("Upload a **V1 Master PDF** or **Multiple Pre-Split PDFs** (IDs will be parsed from cards or filenames).")
    
    # ✨ FIX: Enabled accept_multiple_files=True to allow bulk individual uploads
    pdf_files = st.file_uploader(
        "Upload E-Card PDF(s)", 
        type=["pdf"], 
        accept_multiple_files=True, 
        key="v1upload"
    )

    if pdf_files:
        # Determine if we are splitting a single master PDF or uploading pre-split cards in bulk
        is_master = len(pdf_files) == 1 and st.checkbox(
            "This is a single Master PDF containing multiple cards (Auto-split required)", 
            value=True, 
            key="is_master_toggle"
        )
        
        if st.button("Process Uploaded PDF(s)", type="primary", width="stretch"):
            progress_bar = st.progress(0)
            
            # --- MODE A: PROCESS SINGLE MASTER PDF ---
            if is_master:
                with st.spinner("Splitting and processing master PDF..."):
                    pdf_file = pdf_files[0]
                    doc = fitz.open(stream=pdf_file.getbuffer(), filetype="pdf")
                    employee_data = {}       
                    employee_metadata = {}   
                    
                    for page_num in range(len(doc)):
                        progress_bar.progress(page_num / len(doc))
                        page = doc[page_num]
                        
                        for rect in detect_card_boundaries(page):
                            raw_text = page.get_text("text", clip=rect)
                            if "Emp" not in raw_text:
                                try:
                                    pix = page.get_pixmap(clip=rect, dpi=200)
                                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                                    res = ocr_engine.ocr(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cls=False)
                                    if res and res[0]: raw_text += " \n " + " ".join([line[1][0] for line in res[0]])
                                except: pass
                            
                            parsed_data = extract_metadata_from_text(raw_text)
                            emp_id = parsed_data.emp_id
                            
                            if emp_id:  
                                if emp_id not in employee_data: 
                                    employee_data[emp_id] = []
                                    employee_metadata[emp_id] = []
                                employee_data[emp_id].append((page_num, rect))
                                employee_metadata[emp_id].append(parsed_data)

                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for emp_id, locations in employee_data.items():
                            out_pdf = fitz.open()
                            for (page_num, rect) in locations:
                                out_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                                out_pdf[-1].set_cropbox(rect)
                            
                            card_type = "BASE"
                            test_text = " ".join([m.name + " " + m.card_no for m in employee_metadata[emp_id]]).lower()
                            if any(kw in test_text for kw in ["topup", "top up", "top-up", "super top"]):
                                card_type = "TOPUP"
                            
                            pdf_bytes = out_pdf.tobytes(garbage=4, deflate=True)
                            p_no = employee_metadata[emp_id][0].policy_no if employee_metadata[emp_id] else "UNKNOWN"
                            comp_name = employee_metadata[emp_id][0].company_name if employee_metadata[emp_id] else None
                            
                            # Save with corporate hierarchical directory structures
                            save_card_to_db(emp_id, pdf_bytes, st.session_state.username, employee_metadata[emp_id], p_no, card_type, comp_name)
                            zip_file.writestr(f"{emp_id}_ECard.pdf", pdf_bytes)
                            out_pdf.close()
                            
                    st.session_state.zip_data = zip_buffer.getvalue()
                    doc.close()
                    gc.collect()
                    progress_bar.progress(1.0)
                    st.success(f"✅ Extracted V1 data for {len(employee_data)} Employees!")

            # --- MODE B: PROCESS MULTIPLE PRE-SPLIT INDIVIDUAL PDFs ---
            else:
                total_files = len(pdf_files)
                processed_count = 0
                mismatches = []
                
                with st.spinner(f"Ingesting {total_files} individual e-cards..."):
                    for idx, pdf_file in enumerate(pdf_files):
                        progress_bar.progress((idx + 1) / total_files)
                        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                        
                        # Process first page of the split PDF
                        if len(doc) > 0:
                            page = doc[0]
                            rect = page.rect
                            raw_text = page.get_text("text")
                            
                            # Fallback to OCR if page has scanned/flattened image text
                            if "Emp" not in raw_text:
                                try:
                                    pix = page.get_pixmap(dpi=200)
                                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                                    res = ocr_engine.ocr(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cls=False)
                                    if res and res[0]: raw_text += " \n " + " ".join([line[1][0] for line in res[0]])
                                except: pass
                                
                            parsed_data = extract_metadata_from_text(raw_text)
                            emp_id = parsed_data.emp_id
                            
                            # 🛡️ Fallback: If OCR fails to parse the ID on the card, use the PDF's filename!
                            if not emp_id or emp_id == "UNKNOWN":
                                raw_filename = os.path.splitext(pdf_file.name)[0].strip().upper()
                                # Clean common suffixes like _ecards or _family_ecards
                                emp_id = re.sub(r"(_ECARDS|_ECARD|_FAMILY_ECARDS|_FAMILY_ECARD|_FAMILY|_CARDS|_CARD)$", "", raw_filename)
                                
                            if emp_id:
                                card_type = "BASE"
                                test_text = raw_text.lower()
                                if any(kw in test_text for kw in ["topup", "top up", "top-up", "super top"]):
                                    card_type = "TOPUP"
                                    
                                p_no = parsed_data.policy_no if parsed_data.policy_no else "UNKNOWN"
                                comp_name = parsed_data.company_name
                                pdf_bytes = pdf_file.getvalue()
                                
                                # Store the card directly to R2 under the company hierarchical directory folder
                                save_card_to_db(emp_id, pdf_bytes, st.session_state.username, [parsed_data], p_no, card_type, comp_name)
                                processed_count += 1
                            else:
                                mismatches.append(f"File {pdf_file.name}: Could not parse ID from card or filename.")
                        doc.close()
                        
                    gc.collect()
                    progress_bar.progress(1.0)
                    st.success(f"✅ Successfully ingested and mapped **{processed_count}** individual e-cards to Cloudflare R2!")
                    if mismatches:
                        with st.expander(f"⚠️ View Unmapped Files ({len(mismatches)} warnings)"):
                            for err in mismatches: st.text(err)

    if st.session_state.get('zip_data'):
        st.download_button("📥 Download Master Extracted PDFs", data=st.session_state.zip_data, file_name="V1_Master_ECards.zip", mime="application/zip", type="primary", width="stretch")

# --- TAB 5: SEARCH INDIVIDUAL ---
with main_tab5:
    col_search, col_btn = st.columns([3, 1])
    search_id = col_search.text_input("Enter Employee ID (Hat / Co):", label_visibility="collapsed", placeholder="e.g. 101")
    
    # Optional search isolating filter
    search_policy_no = st.text_input("Specific Client Policy (Optional):", placeholder="e.g. 11022026", key="search_policy_filter")
    
    if col_btn.button("🔍 Search", width="stretch") and search_id:
        cards = get_cards_from_db(search_id.strip().upper(), policy_no=search_policy_no)
        members = get_members_from_db(search_id.strip().upper())
        
        if cards:
            st.success(f"✅ Found **{len(cards)}** associated policy bundle(s) locally.")
            if members:
                # Filter directory display by active query policy
                display_members = pd.DataFrame(members).drop(columns=['_id', 'id', 'emp_id'], errors='ignore')
                if search_policy_no:
                    display_members = display_members[display_members["policy_no"].str.upper() == search_policy_no.strip().upper()]
                st.dataframe(display_members, hide_index=True, width="stretch")
            
            for card in cards:
                c_type = card["card_type"]
                p_no = card["policy_no"]
                pdf_bytes = bytes(card['pdf_data'])
                st.download_button(label=f"📥 Download {c_type} Card ({p_no})", data=pdf_bytes, file_name=f"CapitupIndia_{search_id.upper()}_{p_no}_{c_type}.pdf", mime="application/pdf")
                preview_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_num in range(len(preview_doc)):
                    st.image(preview_doc[page_num].get_pixmap(dpi=150).tobytes("png"), width="stretch") # ✨ Updated: use width='stretch'
                preview_doc.close()
        else:
            st.error("No E-Card found.")

# --- TAB 6: EMAIL DISTRIBUTION AGENT (UPDATED ACCORDING TO IMAGE 2) ---
with main_tab6:
    db = get_db()
    
    st.markdown("### ✉️ Email Dispatch Center")
    st.markdown("Configure corporate assets and automate welcome email dispatch for active employees.")
    
    # Get distinct policies registered in system to act as client selectors
    policies_registered = db.ecards.distinct("policy_no")
    if not policies_registered:
        policies_registered = ["No Clients Ingested"]
        
    col_p_select, col_empty = st.columns([1, 2])
    selected_client_policy = col_p_select.selectbox("Select Active Client Campaign", policies_registered, key="t6_client_policy")

    st.divider()

    col_left_layout, col_right_layout = st.columns([1.2, 1])

    # --- LEFT COLUMN: ASSET VAULT & ARCHITECT ---
    with col_left_layout:
        st.subheader("⚙️ Insurer Asset Vault")
        col_vault1, col_vault2, col_vault3 = st.columns(3) # ✨ Refactored: Render 3 boxes side-by-side
        
        # Reimbursement Claim Form File Box
        with col_vault1:
            st.markdown("<div style='border: 1px solid #ddd; padding: 10px; border-radius: 6px; text-align: center; background-color: #fafafa;'>", unsafe_allow_html=True)
            st.markdown("📄 **Claim Form**")
            if os.path.exists(CLAIM_FORM_PATH):
                st.success("Active")
                if st.button("Delete Form", key="del_form"):
                    os.remove(CLAIM_FORM_PATH)
                    st.rerun()
            else:
                st.warning("Missing")
                uploaded_claim_doc = st.file_uploader("Upload Claim Form", type=["pdf"], label_visibility="collapsed", key="v_claim")
                if uploaded_claim_doc:
                    with open(CLAIM_FORM_PATH, "wb") as f:
                        f.write(uploaded_claim_doc.getbuffer())
                    st.success("Saved!")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Corporate Welcome Poster File Box
        with col_vault2:
            st.markdown("<div style='border: 1px solid #ddd; padding: 10px; border-radius: 6px; text-align: center; background-color: #fafafa;'>", unsafe_allow_html=True)
            st.markdown("🖼️ **Welcome Poster**")
            if os.path.exists(POSTER_PATH):
                st.success("Active")
                if st.button("Delete Poster", key="del_poster"):
                    os.remove(POSTER_PATH)
                    st.rerun()
            else:
                st.warning("Missing")
                uploaded_poster_doc = st.file_uploader("Upload Poster", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="v_poster")
                if uploaded_poster_doc:
                    with open(POSTER_PATH, "wb") as f:
                        f.write(uploaded_poster_doc.getbuffer())
                    st.success("Saved!")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # ✨ Corporate Logo File Box (Asset Vault 3) ✨
        with col_vault3:
            st.markdown("<div style='border: 1px solid #ddd; padding: 10px; border-radius: 6px; text-align: center; background-color: #fafafa;'>", unsafe_allow_html=True)
            st.markdown("🛡️ **Corporate Logo**")
            if os.path.exists(LOGO_PATH):
                st.success("Active")
                if st.button("Delete Logo", key="del_logo"):
                    os.remove(LOGO_PATH)
                    st.rerun()
            else:
                st.warning("Missing")
                uploaded_logo_doc = st.file_uploader("Upload Logo", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="v_logo")
                if uploaded_logo_doc:
                    with open(LOGO_PATH, "wb") as f:
                        f.write(uploaded_logo_doc.getbuffer())
                    st.success("Saved!")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Email Architect Area
        st.subheader("✉️ Email Architect")
        subject_line_input = st.text_input("SUBJECT LINE", value="Your Health Insurance E-Card & Welcome Kit")
        
        # Load active deadline from database to display to users [1]
        active_deadline_text = get_deadline_from_db(selected_client_policy)
        
        # Build Branded Header Logo Component dynamically
        logo_tag_component = ""
        if os.path.exists(LOGO_PATH):
            logo_tag_component = """
            <div style="text-align: center; margin-bottom: 15px;">
              <img src="cid:logo_image" alt="CapitUp India Logo" style="height: 60px; width: auto; display: inline-block;" />
            </div>
            """
        
        # Extracted Palette Corporate Design: Primary Dark (#0B1E30), Luxurious Gold (#C29B38), and Accent Teal (#23C2A9)
        # Added dynamic {{deadline}} variable component inside the string template [1]
        brand_html_template = f"""<div style="font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; max-width: 650px; margin: 0 auto; border: 1px solid #C29B38; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); background-color: #ffffff;">
  <!-- CapitUp India Official Branded Header -->
  <div style="background-color: #0B1E30; padding: 28px 24px; text-align: center; border-bottom: 3px solid #C29B38; position: relative;">
    {logo_tag_component}
    <h2 style="color: #ffffff; margin: 0; font-size: 22px; letter-spacing: 1px; font-weight: 800; text-transform: uppercase;">CAPITUP INDIA</h2>
    <p style="color: #C29B38; margin: 5px 0 0 0; font-size: 11px; font-weight: bold; letter-spacing: 2px;">YOUR SECURE EMPLOYEE BENEFITS PARTNER</p>
  </div>
  
  <!-- Email Content Area -->
  <div style="padding: 32px 24px;">
    <p style="font-size: 15px; margin-top: 0;">Dear <strong>{{{{name}}}}</strong>,</p>
    <p style="font-size: 14px; font-style: italic; color: #555;">Greetings..!</p>
    <p style="font-size: 14px;">We are pleased to inform you that the Group Mediclaim Policy has been renewed with <strong>Bajaj Allianz General Insurance Company</strong> for the period <strong>26-May-2026 to 25-May-2027</strong>.</p>
    <p style="font-size: 14px;">Please find attached your Health Cards / E-Cards and the policy coverage details for your reference.</p>
    <p style="font-size: 14px; font-weight: 500; color: #0B1E30;">The login credentials for accessing the Bajaj Allianz portal will be shared shortly.</p>
    
    <div style="background-color: #F4F6F8; border-left: 4px solid #C29B38; padding: 14px; margin: 20px 0; border-radius: 4px;">
      <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <tr>
          <td style="width: 40%; font-weight: bold; color: #0B1E30; padding: 3px 0;">Employee ID:</td>
          <td style="color: #333; padding: 3px 0;">{{{{emp_id}}}}</td>
        </tr>
        <tr>
          <td style="font-weight: bold; color: #0B1E30; padding: 3px 0;">Policy Number:</td>
          <td style="color: #333; padding: 3px 0;">{{{{policy_no}}}}</td>
        </tr>
      </table>
    </div>

    <!-- Cashless Hospitalization Process Section -->
    <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 24px 0; background-color: #ffffff;">
      <h3 style="margin-top: 0; color: #0B1E30; font-size: 15px; border-bottom: 2px solid #23C2A9; padding-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🏥 Cashless Hospitalization Process</h3>
      <p style="font-size: 13px; margin: 8px 0;">In case of planned or emergency hospitalization, kindly follow the steps below:</p>
      <ol style="font-size: 13px; padding-left: 20px; margin: 10px 0; color: #555;">
        <li style="margin-bottom: 8px;">Identify a network hospital from the official locator list available here: <br/>
          <a href="https://www.bajajallianz.com/branch-locator.html" target="_blank" style="color: #23C2A9; font-weight: bold; text-decoration: none;">Bajaj Allianz Hospital Locator</a>
        </li>
        <li style="margin-bottom: 8px;">At the hospital insurance desk, please provide the following verifications:
          <ul style="padding-left: 15px; margin-top: 4px; list-style-type: circle;">
            <li>Health Card / E-Card</li>
            <li>Aadhaar Card</li>
            <li>Employee ID Card</li>
          </ul>
        </li>
        <li>The hospital desk will coordinate directly with Bajaj Allianz to initiate the cashless authorization process.</li>
      </ol>
    </div>

    <!-- Reimbursement Claim Documents Section -->
    <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 24px 0; background-color: #ffffff;">
      <h3 style="margin-top: 0; color: #0B1E30; font-size: 15px; border-bottom: 2px solid #C29B38; padding-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">📋 Reimbursement Claim Documents</h3>
      <p style="font-size: 13px; margin: 8px 0;">In case of reimbursement claims, kindly upload the documents through the Bajaj Allianz portal or share a single compiled PDF file (under 10 MB) with our team at <a href="mailto:syamala.g@capitupindia.com" style="color: #23C2A9; font-weight: bold; text-decoration: none;">syamala.g@capitupindia.com</a> or <a href="mailto:claims@capitupindia.com" style="color: #23C2A9; font-weight: bold; text-decoration: none;">claims@capitupindia.com</a>.</p>
      <p style="font-size: 13px; font-weight: bold; margin-bottom: 6px; color: #0B1E30;">Please ensure that the following documents are submitted:</p>
      <ul style="font-size: 12px; padding-left: 20px; margin: 0; color: #555; line-height: 1.5;">
        <li style="margin-bottom: 4px;"><strong>Claim Form Part-A (attached):</strong> Checklist duly filled and signed by the employee.</li>
        <li style="margin-bottom: 4px;"><strong>Claim Form Part-B (attached):</strong> Duly completed with hospital stamp and authorized signature.</li>
        <li style="margin-bottom: 4px;">Original detailed discharge summary with hospital stamp and signature (including date and time).</li>
        <li style="margin-bottom: 4px;">Original final bill with hospital stamp and signature (including date and time).</li>
        <li style="margin-bottom: 4px;">Original payment receipts corresponding to the final bill.</li>
        <li style="margin-bottom: 4px;">Original pharmacy bills with stamp and signature.</li>
        <li style="margin-bottom: 4px;">Original diagnostic/laboratory reports and X-Ray/Scan reports with payment receipts, if applicable.</li>
        <li style="margin-bottom: 4px;">Original prescription of the first consultation and previous consultation records, if any.</li>
        <li style="margin-bottom: 4px;">Copy of Patient Health ID Card & Aadhaar Card.</li>
        <li style="margin-bottom: 4px;">Copy of Employee PAN Card & Employee ID Card.</li>
        <li style="margin-bottom: 4px;">Copy of Employee's cancelled cheque leaf (with printed Name, Account Number and IFSC) or the first page of the bank passbook.</li>
        <li style="margin-bottom: 4px;">Employee contact details (mobile number, email ID, and address).</li>
      </ul>
    </div>

    <!-- Call to Actions & dynamic Correction Google Form Pre-fill Link -->
    <div style="text-align: center; margin: 32px 0 16px 0;">
      <a href="https://docs.google.com/forms/d/e/1FAIpQLSfMZ0SHY4pr9NVfZwHQRhU6Jmy-vN2K8INePRdkYQarVA_EMw/viewform?usp=pp_url&entry.877007954={{{{name}}}}&entry.863990631={{{{emp_id}}}}&entry.1115400795={{{{policy_no}}}}" 
         style="background-color: #23C2A9; color: #ffffff; padding: 14px 28px; text-decoration: none; font-size: 13px; font-weight: bold; border-radius: 6px; display: inline-block; box-shadow: 0 4px 10px rgba(35, 194, 169, 0.25); border: 1px solid #1fa895; transition: background-color 0.2s;">
        📝 Request E-Card Correction
      </a>
      <p style="color: #888; font-size: 10px; margin-top: 10px;">If you detect any spelling or coverage discrepancies, click the button above to request corrections.</p>
      <p style="color: #C29B38; font-size: 11px; font-weight: bold; margin-top: 6px;">⏱️ Correction Form Window Closes On: {{{{deadline}}}}</p>
    </div>
  </div>

  <!-- Dynamic Poster Attachment Hook -->
  <!-- FOOTER -->

  <!-- CapitUp India Corporate Footer with Logo Signature Mark -->
  <div style="background-color: #F4F6F8; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
    <p style="margin: 0; font-size: 12px; color: #0B1E30; font-weight: bold;">Thank you for being part of the CapitUp Family</p>
    <p style="margin: 4px 0 0 0; font-size: 10px; color: #888;">CapitUp India Pvt. Ltd. | 4th Floor, HUDA Techno Enclave, HITEC City, Hyderabad-500081</p>
    <div style="margin-top: 15px; font-size: 9px; color: #C29B38; font-weight: bold; letter-spacing: 1px;">
      ⚡ SECURED BY CAPITUP INDIA WATERMARK SYSTEM
    </div>
  </div>
</div>"""
        
        # Save the brand template strictly containing the un-evaluated {{deadline}} variable parameter [1]
        html_body_input = st.text_area("HTML BODY TEMPLATE (VARIABLES: {{name}}, {{emp_id}}, {{policy_no}})", value=brand_html_template, height=250)
        
        # Live HTML Preview (Evaluate the live database deadline value to the UI previews) [1]
        if st.checkbox("👁️ Toggle Live Preview", key="live_prev"):
            st.markdown("#### Live Preview Frame")
            preview_rendered = html_body_input.replace("{{name}}", st.session_state.username).replace("{{emp_id}}", "MOCK-101").replace("{{policy_no}}", selected_client_policy).replace("{{deadline}}", active_deadline_text)
            
            # Temporary local layout preview
            st.markdown("##### Logo & Header Card Preview")
            if os.path.exists(LOGO_PATH):
                st.image(LOGO_PATH, caption="Vault Logo Asset Preview", width="stretch") # ✨ Updated: use width='stretch'
            
            st.markdown("##### Email Content Preview")
            st.components.v1.html(preview_rendered, height=500, scrolling=True)

    # --- RIGHT COLUMN: PENDING ENROLLMENT & PROCESS QUEUE ---
    with col_right_layout:
        st.subheader("👥 Pending Enrollment")
        st.markdown("Active users missing a welcome email.")

        # Real-time Directory Mapping Sheet Upload Box with Idempotent Session State Guard to break loop
        st.markdown("<div style='background-color:#f0f2f6; padding:15px; border-radius:6px; border:1px solid #ddd;'>", unsafe_allow_html=True)
        t6_mapping_file = st.file_uploader(
            "📥 Upload Client Mapping Directory (CSV/Excel) to map names & emails:",
            type=["csv", "xlsx", "xls"],
            key="t6_mapping_uploader"
        )
        
        # Unique session state key check to prevent processing loops
        if t6_mapping_file:
            file_id_key = f"processed_{t6_mapping_file.name}_{t6_mapping_file.size}_{selected_client_policy}"
            if not st.session_state.get(file_id_key, False):
                with st.spinner("Processing mapping directory sheet..."):
                    try:
                        if t6_mapping_file.name.endswith('.csv'):
                            df_map = pd.read_csv(t6_mapping_file)
                        else:
                            df_map = pd.read_excel(t6_mapping_file)
                        
                        df_map = clean_and_align_dataframe(df_map)
                        df_map_cols = list(df_map.columns)
                        
                        # Robust Guessing to strictly target headers
                        emp_col_map = robust_guess_column(df_map_cols, ["EMP ID", "EMPLOYEE ID", "ID", "HAT", "CO"])
                        name_col_map = robust_guess_column(df_map_cols, ["MEMBER NAME", "NAME", "EMPLOYEE NAME", "INSURED"])
                        email_col_map = robust_guess_column(df_map_cols, ["EMAIL", "EMAIL ADDRESS", "E CARDS ACCESS CODE", "ACCESS", "MAIL"])
                        rel_col_map = robust_guess_column(df_map_cols, ["RELATION", "RELATIONSHIP", "RELATI", "REL"])
                        
                        if not emp_col_map or not name_col_map or not email_col_map:
                            st.error("🚨 Column mapping failed. Please check your spreadsheet headers (Emp ID, Name, Email).")
                        else:
                            added_records = 0
                            parsed_employees = {}
                            
                            # Parse and filter out spouses/dependents from overwriting the primary insured
                            for _, row in df_map.iterrows():
                                raw_emp_id = str(row[emp_col_map]).strip().upper()
                                raw_name = str(row[name_col_map]).strip()
                                raw_email = str(row[email_col_map]).strip().lower()
                                raw_rel = str(row[rel_col_map]).strip().upper() if (rel_col_map and rel_col_map in row) else "SELF"
                                
                                if not raw_emp_id or raw_emp_id in ["NAN", ""]:
                                    continue
                                    
                                # Always prefer the Proposer/Primary Employee (Relationship = SELF)
                                if raw_emp_id not in parsed_employees or raw_rel in ["SELF", "PRIMARY", "EMPLOYEE", "PROPOSER"]:
                                    parsed_employees[raw_emp_id] = {
                                        "name": raw_name,
                                        "email": raw_email
                                    }
                            
                            # Bulk upsert resolved primary records to database directory
                            for emp_id_key, detail in parsed_employees.items():
                                save_employee_to_directory(emp_id_key, detail["name"], detail["email"], selected_client_policy)
                                added_records += 1
                            
                            # Mark this file as processed in session state to break the loop
                            st.session_state[file_id_key] = True
                            st.success(f"✅ Successfully loaded and synced **{added_records}** primary employees to the database.")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error parsing mapping sheet: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Identify all ecards under this policy that have NOT been emailed yet
        pending_ecards = list(db.ecards.find({
            "policy_no": selected_client_policy, 
            "email_sent": {"$ne": True}
        }))
        
        pending_display_list = []
        for ecard in pending_ecards:
            # Match employee email and name from the directory collection
            directory_record = db.directory.find_one({
                "emp_id": ecard["emp_id"], 
                "policy_no": selected_client_policy
            })
            if directory_record:
                pending_display_list.append({
                    "EMP ID": ecard["emp_id"],
                    "Name": directory_record.get("name", "UNKNOWN"),
                    "Email": directory_record.get("email", ""),
                    "Card Type": ecard["card_type"]
                })
            else:
                pending_display_list.append({
                    "EMP ID": ecard["emp_id"],
                    "Name": "UNKNOWN (Incomplete Directory Metadata)",
                    "Email": "",
                    "Card Type": ecard["card_type"]
                })

        # Render Pending Enrollment Table (Matches image 2 layout)
        if pending_display_list:
            df_pending = pd.DataFrame(pending_display_list)
            st.dataframe(df_pending[["EMP ID", "Name", "Email", "Card Type"]], hide_index=True, use_container_width=True)
        else:
            st.info("No pending enrollments. All emails for this policy have been sent.")

        st.divider()

        st.subheader("✉️ Process Mail Queue")
        st.markdown("Dispatches SMTP mail & fetches E-Cards.")
        
        batch_limit = st.number_input("Batch Run Limit", min_value=1, max_value=500, value=20, step=1)
        
        # Generate clean dynamic action button text based on pending count
        jobs_to_process = min(len(pending_display_list), batch_limit)
        
        if st.button(f"▶️ Process {jobs_to_process} Jobs", type="primary", use_container_width=True, disabled=(jobs_to_process == 0)):
            sent_success_count = 0
            progress_bar = st.progress(0)
            status_update = st.empty()
            
            for idx, job in enumerate(pending_display_list[:jobs_to_process]):
                emp_id = job["EMP ID"]
                recipient_email = job["Email"]
                emp_name = job["Name"]
                
                if not recipient_email or "@" not in recipient_email:
                    logging.warning(f"Skipping job for {emp_id} due to invalid or missing email.")
                    continue
                
                status_update.text(f"Fetching cards and sending email to {emp_name} ({emp_id})...")
                
                # Fetch e-cards dynamically from database (and fall back to R2 automatically if missing)
                cards = get_cards_from_db(emp_id, policy_no=selected_client_policy)
                
                if cards:
                    # Dynamically inject the user's variables into the HTML editor template [1]
                    # Dynamically replace the {{deadline}} variable in the generated email right before dispatching [1]
                    customized_html = html_body_input.replace("{{name}}", emp_name).replace("{{emp_id}}", emp_id).replace("{{policy_no}}", selected_client_policy).replace("{{deadline}}", active_deadline_text)
                    
                    # Dispatch secure email with correct e-cards attached
                    mail_sent = send_multi_ecard_email(recipient_email, subject_line_input, customized_html, cards)
                    
                    if mail_sent:
                        # Mark e-card as sent in MongoDB to remove them from future queues
                        db.ecards.update_many(
                            {"emp_id": emp_id, "policy_no": selected_client_policy},
                            {"$set": {"email_sent": True}}
                        )
                        sent_success_count += 1
                        
                progress_bar.progress((idx + 1) / jobs_to_process)
                
            status_update.empty()
            progress_bar.empty()
            st.success(f"Successfully processed queue! Sent **{sent_success_count}** welcome emails.")
            time.sleep(1)
            st.rerun()

        # Testing & Admin Queue Controller Box (Re-queue Utility)
        st.divider()
        with st.expander("🛠️ Admin Testing & Queue Controls", expanded=False):
            st.markdown("Use these utility tools to reset email-sent flags and run simulations.")
            
            # --- Google Form Response Controller (Web App API Wrapper) ---
            st.markdown("#### 📝 Google Form Status Controller")
            
            # Retrieve GAS URL securely from DB settings to avoid copy-pasting every time
            gas_setting = db.settings.find_one({"key": "gas_url"})
            stored_gas_url = gas_setting["value"] if gas_setting else ""
            
            gas_url_input = st.text_input("Google Apps Script Web App URL:", value=stored_gas_url, placeholder="https://script.google.com/macros/s/.../exec")
            
            if gas_url_input != stored_gas_url:
                db.settings.update_one({"key": "gas_url"}, {"$set": {"key": "gas_url", "value": gas_url_input}}, upsert=True)
                st.success("Google Apps Script URL securely saved to database!")
                time.sleep(0.5)
                st.rerun()
                
            if gas_url_input:
                # Ping the API to get current live status [1]
                current_form_status = get_form_status(gas_url_input)
                
                # Dynamic Sync: If Google Form has auto-closed, sync the database status immediately [1, 2]
                current_deadline_db_value = get_deadline_from_db(selected_client_policy)
                if current_form_status == "CLOSED" and current_deadline_db_value not in ["Form Closed", "Expired / Closed"]:
                    save_deadline_to_db(selected_client_policy, "Expired / Closed")
                    st.rerun()
                
                if current_form_status == "OPEN":
                    st.markdown(f"Live Form Status: **🟢 ACTIVE (Accepting Responses)**")
                elif current_form_status == "CLOSED":
                    st.markdown(f"Live Form Status: **🔴 INACTIVE (Form Shut)**")
                else:
                    st.markdown(f"Live Form Status: **⚠️ DISCONNECTED ({current_form_status})**")
                
                # Dynamic Response Window Deadline Selector
                st.markdown("##### ⏱️ Schedule Response Window Deadline")
                deadline_option = st.selectbox(
                    "Set Duration Window (Closes form automatically):",
                    ["Manual (No Timer)", "1 Day", "3 Days", "1 Week (7 Days)", "10 Days", "2 Weeks (14 Days)"],
                    key="deadline_timer_select"
                )
                
                col_open_form, col_close_form = st.columns(2)
                
                # Execute form opening & schedule triggers concurrently
                if col_open_form.button("🟢 Open Form & Start Timer", use_container_width=True):
                    if deadline_option == "Manual (No Timer)":
                        res = set_form_status(gas_url_input, "open")
                        if res:
                            save_deadline_to_db(selected_client_policy, "Manual Close Required")
                            st.success("Google Form is now OPEN to corrections (Manual closing required)!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        # Convert selected readable string to actual hours
                        duration_mapping = {
                            "1 Day": 24.0,
                            "3 Days": 72.0,
                            "1 Week (7 Days)": 168.0,
                            "10 Days": 240.0,
                            "2 Weeks (14 Days)": 336.0
                        }
                        target_hours = duration_mapping[deadline_option]
                        
                        # Ping API to register Google's cloud server-side timer trigger
                        response_text = schedule_form_close(gas_url_input, target_hours)
                        
                        if response_text and "SCHEDULED_FOR_" in response_text:
                            # Parse UTC string returned from Google's server
                            iso_str = response_text.replace("SCHEDULED_FOR_", "")
                            utc_datetime = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
                            
                            # Convert to local visual output string (IST timezone offset)
                            local_datetime = utc_datetime + timedelta(hours=5, minutes=30)
                            formatted_deadline_str = local_datetime.strftime("%B %d, %Y at %I:%M %p (IST)")
                            
                            # Persist selected deadline in database settings
                            save_deadline_to_db(selected_client_policy, formatted_deadline_str)
                            st.success(f"Google Form opened and scheduled to close on {formatted_deadline_str}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed to communicate timer trigger with Google Web App. API response: {response_text}")
                        
                if col_close_form.button("🔴 Close Google Form", use_container_width=True):
                    res = set_form_status(gas_url_input, "close")
                    if res:
                        save_deadline_to_db(selected_client_policy, "Form Closed")
                        st.warning("Google Form is now CLOSED to corrections!")
                        time.sleep(1)
                        st.rerun()
            
            st.divider()
            st.markdown("#### 🔄 Reset Queue Items")
            
            # 1. Reset Specific Employee
            col_admin_input, col_admin_btn = st.columns([2, 1])
            reset_target_id = col_admin_input.text_input("Target Employee ID to Re-queue:", placeholder="e.g. TEST101", key="admin_reset_id_val")
            if col_admin_btn.button("🔄 Re-queue Employee", use_container_width=True) and reset_target_id:
                clean_target = reset_target_id.strip().upper()
                db.ecards.update_many(
                    {"emp_id": clean_target, "policy_no": selected_client_policy},
                    {"$set": {"email_sent": False}}
                )
                st.success(f"Employee {clean_target} successfully re-queued for {selected_client_policy}!")
                time.sleep(1)
                st.rerun()
                
            # 2. Reset Entire Campaign
            if st.button("🚨 Re-queue All Users in Selected Campaign", type="secondary", use_container_width=True, key="admin_reset_all_btn"):
                db.ecards.update_many(
                    {"policy_no": selected_client_policy},
                    {"$set": {"email_sent": False}}
                )
                st.success("All e-cards successfully re-queued for sending!")
                time.sleep(1)
                st.rerun()
