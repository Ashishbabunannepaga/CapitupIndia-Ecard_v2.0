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
import concurrent.futures 

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

import pillow_heif
pillow_heif.register_heif_opener()

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

@st.cache_resource(show_spinner="Loading AI Vision Engine... (First load takes a few seconds)")
def load_ocr_engine():
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    return PaddleOCR(use_textline_orientation=True, lang='en')

# --- MONGO DATABASE CONNECTIONS ---
@st.cache_resource
def get_mongo_client():
    return MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)

def get_db():
    client = get_mongo_client()
    return client[MONGO_DBNAME]

def init_db():
    db = get_db()
    db.users.create_index("username", unique=True)
    db.ecards.create_index(
        [("policy_no", pymongo.ASCENDING), ("emp_id", pymongo.ASCENDING), ("card_type", pymongo.ASCENDING)],
        unique=True, name="unique_emp_card_type"
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
    
    illegal_chars = r'[\\/*?:"<>|]'
    sanitized_policy = re.sub(illegal_chars, "", clean_policy_no).strip().replace(" ", "_")
    if not sanitized_policy: sanitized_policy = "UNKNOWN_POLICY"
    
    if clean_company_name:
        sanitized_company = re.sub(illegal_chars, "", clean_company_name).strip().replace(" ", "_")
        if not sanitized_company: sanitized_company = "UNKNOWN_COMPANY"
        sub_folder = os.path.join(sanitized_company, sanitized_policy, card_type)
        file_key = f"ecards/{sanitized_company}/{sanitized_policy}/{card_type}/{clean_emp_id}.pdf"
    else:
        sub_folder = os.path.join(sanitized_policy, card_type)
        file_key = f"ecards/{sanitized_policy}/{card_type}/{clean_emp_id}.pdf"
    
    card_folder = os.path.join(LOCAL_STORAGE_DIR, sub_folder)
    os.makedirs(card_folder, exist_ok=True)
    local_file_path = os.path.join(card_folder, f"{clean_emp_id}.pdf")
    with open(local_file_path, "wb") as f: f.write(pdf_bytes)
        
    if R2_ENABLED:
        try:
            s3_client.put_object(Bucket=R2_CONFIG["bucket_name"], Key=file_key, Body=pdf_bytes, ContentType="application/pdf")
        except Exception as e:
            logging.error(f"Cloudflare upload failed: {e}")

    db = get_db()
    db.ecards.update_one(
        {"emp_id": clean_emp_id, "policy_no": clean_policy_no, "card_type": card_type},
        {"$set": {
            "emp_id": clean_emp_id, "policy_no": clean_policy_no, "company_name": clean_company_name,
            "card_type": card_type, "file_path": local_file_path, "r2_key": file_key,
            "uploaded_by": username, "upload_date": datetime.utcnow(), "email_sent": False  
        }}, upsert=True
    )
               
    db.card_members.delete_many({"emp_id": clean_emp_id, "policy_no": clean_policy_no, "policy_type": card_type})
    member_docs = []
    for member in family_members:
        member_docs.append({
            "emp_id": clean_emp_id, "name": member.name, "policy_no": clean_policy_no, "policy_type": card_type,
            "card_no": member.card_no, "relationship": member.relationship, "age": member.age, "valid_up_to": member.valid_up_to
        })
    if member_docs: db.card_members.insert_many(member_docs)

def save_employee_to_directory(emp_id, name, email, policy_no):
    db = get_db()
    db.directory.update_one(
        {"emp_id": str(emp_id).strip().upper(), "policy_no": str(policy_no).strip().upper()},
        {"$set": {"emp_id": str(emp_id).strip().upper(), "name": str(name).strip(), "email": str(email).strip().lower(), "policy_no": str(policy_no).strip().upper(), "updated_at": datetime.utcnow()}},
        upsert=True
    )

def get_cards_from_db(emp_id, policy_no=None):
    db = get_db()
    query = {"emp_id": str(emp_id).strip().upper()}
    if policy_no: query["policy_no"] = str(policy_no).strip().upper()
    db_results = list(db.ecards.find(query))
    
    cards_list = []
    for row in db_results:
        local_path = row.get("file_path")
        pdf_data = None
        if local_path and os.path.exists(local_path):
            try:
                with open(local_path, "rb") as f: pdf_data = f.read()
            except Exception: pass
        if not pdf_data and R2_ENABLED and "r2_key" in row:
            try:
                response = s3_client.get_object(Bucket=R2_CONFIG["bucket_name"], Key=row["r2_key"])
                pdf_data = response["Body"].read()
                if local_path:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "wb") as f: f.write(pdf_data)
            except Exception: pass
        if pdf_data:
            cards_list.append({
                "card_type": row["card_type"], "policy_no": row["policy_no"], "pdf_data": pdf_data,
                "uploaded_by": row.get("uploaded_by", "UNKNOWN"), "upload_date": row.get("upload_date", datetime.utcnow())
            })
    return cards_list

def get_members_from_db(emp_id=None):
    db = get_db()
    if emp_id: cursor = db.card_members.find({"emp_id": emp_id}).sort("relationship", -1)
    else: cursor = db.card_members.find().sort("emp_id", 1)
    results = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        results.append(doc)
    return results

def get_bulk_cards_from_db(emp_ids, policy_no=None):
    if not emp_ids: return []
    db = get_db()
    query = {"emp_id": {"$in": emp_ids}}
    if policy_no: query["policy_no"] = str(policy_no).strip().upper()
    db_results = list(db.ecards.find(query))
    results = []
    for row in db_results:
        local_path = row.get("file_path")
        pdf_data = None
        if local_path and os.path.exists(local_path):
            try:
                with open(local_path, "rb") as f: pdf_data = f.read()
            except Exception: pass
        if not pdf_data and R2_ENABLED and "r2_key" in row:
            try:
                response = s3_client.get_object(Bucket=R2_CONFIG["bucket_name"], Key=row["r2_key"])
                pdf_data = response["Body"].read()
            except Exception: pass
        if pdf_data:
            results.append({"emp_id": row["emp_id"], "card_type": row["card_type"], "policy_no": row["policy_no"], "pdf_data": pdf_data})
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
            poster_tag = """<tr><td align="center" style="padding: 0 40px 30px 40px;"><img src="cid:poster_image" alt="Mediclaim Summary Poster" style="width: 100%; max-width: 520px; height: auto; border-radius: 6px; display: block;" /></td></tr>"""
            body_html = body_html.replace("<!-- FOOTER -->", poster_tag + "\n<!-- FOOTER -->")
        msg_related.attach(MIMEText(body_html, 'html'))
        if os.path.exists(LOGO_PATH):
            from email.mime.image import MIMEImage
            with open(LOGO_PATH, "rb") as f: logo_data = f.read()
            msg_logo = MIMEImage(logo_data)
            msg_logo.add_header('Content-ID', '<logo_image>')
            msg_logo.add_header('Content-Disposition', 'inline', filename="logo.png")
            msg_related.attach(msg_logo)
        if os.path.exists(POSTER_PATH):
            from email.mime.image import MIMEImage
            with open(POSTER_PATH, "rb") as f: img_data = f.read()
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
            with open(CLAIM_FORM_PATH, "rb") as f: claim_bytes = f.read()
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
                server.ehlo(); server.starttls(); server.ehlo()
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(SMTP_CONFIG["username"], recipient_email, msg.as_string())
        return True
    except Exception as e:
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
            if gaps[i] - gaps[i-1] > 20: split_y.append(gaps[i])
        split_y.append(height)

        rects = []
        for i in range(len(split_y)-1):
            y1, y2 = split_y[i], split_y[i+1]
            if y2 - y1 > 150: rects.append(fitz.Rect(0, y1 * (page.rect.height / height), page.rect.width, y2 * (page.rect.height / height)))
        if not rects: raise ValueError()
        return rects
    except:
        h3 = page.rect.height / 3
        return [fitz.Rect(0, 0, page.rect.width, h3), fitz.Rect(0, h3, page.rect.width, h3*2), fitz.Rect(0, h3*2, page.rect.width, page.rect.height)]

# --- GOOGLE FORM CONTROLLER UTILITIES ---
def get_form_status(api_url):
    try: return requests.get(api_url + "?action=status", timeout=10).text.strip().upper()
    except: return "UNKNOWN / DISCONNECTED"

def set_form_status(api_url, action):
    try: return requests.get(api_url + f"?action=status", timeout=10).text.strip().upper()
    except: return None

def schedule_form_close(api_url, hours):
    try: return requests.get(api_url + f"?action=schedule&hours={hours}", timeout=10).text.strip()
    except: return None

def get_gas_url_from_db():
    setting = get_db().settings.find_one({"key": "gas_url"})
    return setting["value"] if setting else DEFAULT_GAS_URL

def save_gas_url_to_db(url):
    get_db().settings.update_one({"key": "gas_url"}, {"$set": {"value": url.strip()}}, upsert=True)

def get_deadline_from_db(policy_no):
    setting = get_db().settings.find_one({"key": f"deadline_{policy_no}"})
    return setting["value"] if setting else "Not Set"

def save_deadline_to_db(policy_no, deadline_str):
    get_db().settings.update_one({"key": f"deadline_{policy_no}"}, {"$set": {"value": deadline_str}}, upsert=True)

# --- INITIALIZE HYBRID DATABASE SCHEMA ---
init_db()

if "failed_emails" not in st.session_state: st.session_state.failed_emails = []
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ""

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
                    else: st.error("❌ Invalid Credentials.")
                        
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

# Fetch default keys globally
secret_keys = []
if "GEMINI_API_KEYS" in st.secrets:
    sk = st.secrets["GEMINI_API_KEYS"]
    if isinstance(sk, list): secret_keys = [str(k).strip() for k in sk if str(k).strip()]
    elif isinstance(sk, str): secret_keys = [k.strip() for k in sk.split(',') if k.strip()]

st.sidebar.title(f"👤 Welcome, {st.session_state.username}")
if st.sidebar.button("Logout", type="primary", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.title("🪪 CapitupIndia E-Card Database Portal")
if R2_ENABLED:
    st.success("✅ Cloudflare R2 Connected! Cards will automatically sync to the global CDN.")

main_tab1, main_tab4, main_tab2, main_tab3, main_tab5, main_tab6, main_tab7, main_tab8 = st.tabs([
    "📥 Ingest E-Cards", "📥 Bulk Retrieval", "📊 Global Directory", 
    "📤 Upload Master (V1)", "🔍 Search Individual", "✉️ Email Distribution", "🧬 Familyfication",
    "🔍 Coverage Gap Finder"
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
        
        if st.button("🚀 Process & Ingest Base Policies", type="primary", use_container_width=True, key="btn_base"):
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
        
        if st.button("🚀 Process & Ingest Top Up Policies", type="primary", use_container_width=True, key="btn_topup"):
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
    bulk_policy_filter = st.text_input("Target Policy Number (Optional):", placeholder="e.g. 11022026", key="bulk_policy_no")
    bulk_input = st.text_area("List of Employee IDs (comma or space separated):", height=150)
    
    if st.button("📦 Fetch Cards & Build ZIP", type="primary", use_container_width=True):
        if bulk_input.strip():
            with st.spinner("Fetching hybrid database filepaths..."):
                clean_ids = list(set([i.strip().upper() for i in bulk_input.replace(',', ' ').split() if i.strip()]))
                found_cards = get_bulk_cards_from_db(clean_ids, policy_no=bulk_policy_filter)
                
                if found_cards:
                    bulk_zip_buffer = BytesIO()
                    with zipfile.ZipFile(bulk_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for card in found_cards:
                            zf.writestr(f"{card['policy_no']}_{card['emp_id']}_{card['card_type']}_ECard.pdf", bytes(card['pdf_data']))
                    st.session_state.bulk_zip_data = bulk_zip_buffer.getvalue()
                    
                    missing_ids = [i for i in clean_ids if i not in [c['emp_id'] for c in found_cards]]
                    msg = f"✅ Successfully packaged **{len(found_cards)}** files."
                    if missing_ids: msg += f"\n\n⚠️ **Missing:** {', '.join(missing_ids)}"
                    st.info(msg)
                else:
                    st.error("❌ None of the requested IDs were found in DB.")

    if st.session_state.get('bulk_zip_data'):
        st.download_button("📥 Download Batch ZIP", data=st.session_state.bulk_zip_data, file_name="Bulk_ECards.zip", mime="application/zip", type="primary", use_container_width=True)

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
        st.dataframe(df_all, hide_index=True, use_container_width=True)

# --- TAB 4: UPLOAD V1 MASTER / SMART BULK INGESTION ---
# --- TAB 4: UNIVERSAL UPLOAD ENGINE ---
with main_tab3:
    st.markdown("### 🛠️ Universal E-Card Ingestion Engine")
    st.markdown("Upload **Master PDFs** or **Individual Cards**. Mix and match the processing logic below to handle any insurer format.")
    
    pdf_files = st.file_uploader("Upload E-Card PDF(s)", type=["pdf"], accept_multiple_files=True, key="v1upload")

    # --- NEW: DYNAMIC PROCESSING CONTROL PANEL ---
    st.markdown("#### ⚙️ Processing Logic Controls")
    st.markdown("<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        opt_master = st.checkbox("✂️ **Split Master PDF**", help="Check this if you uploaded a large PDF containing multiple ID cards per page.", value=False)
    with c2:
        opt_merge = st.checkbox("🧬 **Group & Merge by Family**", help="Combines separate PDFs sharing the same Employee ID into one single family file.", value=True)
    with c3:
        opt_rename = st.checkbox("🔄 **Smart Rename Files**", help="Renames the final output files automatically using the extracted Employee ID.", value=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.text("") # spacer

    if pdf_files and st.button("🚀 Process & Execute Logic", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        total_files = len(pdf_files)
        
        extracted_cards = [] 
        
        # ==========================================
        # PHASE 1: EXTRACTION (Split or Individual)
        # ==========================================
        with st.spinner("Extracting text and scanning PDFs..."):
            for idx, pdf_file in enumerate(pdf_files):
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                
                if opt_master:
                    # Logic 1: Iterate pages and slice out bounding boxes
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        for rect in detect_card_boundaries(page):
                            raw_text = page.get_text("text", clip=rect)
                            
                            if "EMPLOYEE" not in raw_text.upper() and "EMP" not in raw_text.upper():
                                try:
                                    pix = page.get_pixmap(clip=rect, dpi=200)
                                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                                    res = ocr_engine.ocr(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cls=False)
                                    if res and res[0]: raw_text += " \n " + " ".join([line[1][0] for line in res[0]])
                                except: pass
                                
                            parsed_data = extract_metadata_from_text(raw_text)
                            
                            if not parsed_data.emp_id:
                                emp_match = re.search(r"(?:EMPLOYEE\s*CODE|EMP\s*ID)\s*[:\-]?\s*([A-Za-z0-9]+)", raw_text, re.IGNORECASE)
                                if emp_match: parsed_data.emp_id = emp_match.group(1).strip().upper()
                            if not parsed_data.policy_no:
                                pol_match = re.search(r"POLICY\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)", raw_text, re.IGNORECASE)
                                if pol_match: parsed_data.policy_no = pol_match.group(1).strip().upper()
                                
                            emp_id = parsed_data.emp_id
                            
                            # Slice out the specific card bytes
                            temp_doc = fitz.open()
                            temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                            temp_doc[-1].set_cropbox(rect)
                            card_bytes = temp_doc.tobytes(garbage=4, deflate=True)
                            temp_doc.close()
                            
                            extracted_cards.append({
                                "emp_id": emp_id if emp_id else "UNKNOWN",
                                "metadata": parsed_data,
                                "bytes": card_bytes,
                                "raw_text": raw_text,
                                "original_name": f"MasterPage_{page_num}.pdf"
                            })
                else:
                    # Logic 2: Treat as Individual whole PDF
                    if len(doc) > 0:
                        page = doc[0]
                        raw_text = page.get_text("text")
                        
                        if "EMPLOYEE" not in raw_text.upper() and "EMP" not in raw_text.upper():
                            try:
                                pix = page.get_pixmap(dpi=200)
                                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                                res = ocr_engine.ocr(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cls=False)
                                if res and res[0]: raw_text += " \n " + " ".join([line[1][0] for line in res[0]])
                            except: pass
                            
                        parsed_data = extract_metadata_from_text(raw_text)
                        
                        if not parsed_data.emp_id:
                            emp_match = re.search(r"(?:EMPLOYEE\s*CODE|EMP\s*ID)\s*[:\-]?\s*([A-Za-z0-9]+)", raw_text, re.IGNORECASE)
                            if emp_match: parsed_data.emp_id = emp_match.group(1).strip().upper()
                        if not parsed_data.policy_no:
                            pol_match = re.search(r"POLICY\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)", raw_text, re.IGNORECASE)
                            if pol_match: parsed_data.policy_no = pol_match.group(1).strip().upper()
                            
                        final_emp_id = parsed_data.emp_id
                        if not final_emp_id or final_emp_id == "UNKNOWN":
                            raw_filename = os.path.splitext(pdf_file.name)[0].strip().upper()
                            final_emp_id = re.sub(r"(_ECARDS|_ECARD|_FAMILY_ECARDS|_FAMILY_ECARD|_FAMILY|_CARDS|_CARD)$", "", raw_filename)
                            
                        extracted_cards.append({
                            "emp_id": final_emp_id if final_emp_id else "UNKNOWN",
                            "metadata": parsed_data,
                            "bytes": pdf_file.getvalue(),
                            "raw_text": raw_text,
                            "original_name": pdf_file.name
                        })
                doc.close()
                progress_bar.progress((idx + 1) / total_files)

        # ==========================================
        # PHASE 2: GROUPING, RENAMING & SAVING
        # ==========================================
        with st.spinner("Applying requested logic and pushing to Database..."):
            zip_buffer = BytesIO()
            processed_count = 0
            mismatches = []
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                
                if opt_merge:
                    # Logic 3A: Group by Family
                    family_groups = {}
                    for card in extracted_cards:
                        eid = card["emp_id"]
                        if eid == "UNKNOWN":
                            mismatches.append(f"Could not parse ID for {card['original_name']}")
                            continue
                        if eid not in family_groups:
                            family_groups[eid] = {"bytes": [], "metadata": [], "raw_text_concat": ""}
                        family_groups[eid]["bytes"].append(card["bytes"])
                        family_groups[eid]["metadata"].append(card["metadata"])
                        family_groups[eid]["raw_text_concat"] += " " + card["raw_text"]
                    
                    for eid, group in family_groups.items():
                        merged_pdf = fitz.open()
                        for b in group["bytes"]:
                            t_doc = fitz.open(stream=b, filetype="pdf")
                            merged_pdf.insert_pdf(t_doc)
                            t_doc.close()
                        
                        merged_bytes = merged_pdf.tobytes(garbage=4, deflate=True)
                        merged_pdf.close()
                        
                        card_type = "BASE"
                        if any(kw in group["raw_text_concat"].lower() for kw in ["topup", "top up", "top-up", "super top"]): card_type = "TOPUP"
                        p_no = group["metadata"][0].policy_no if group["metadata"][0].policy_no else "UNKNOWN"
                        comp_name = getattr(group["metadata"][0], 'company_name', None)
                        
                        save_card_to_db(eid, merged_bytes, st.session_state.username, group["metadata"], p_no, card_type, comp_name)
                        
                        save_name = f"{eid}ECard.pdf" if opt_rename else f"{eid}.pdf"
                        zip_file.writestr(save_name, merged_bytes)
                        processed_count += 1
                        
                else:
                    # Logic 3B: Process as Strictly Individual Cards
                    for idx_card, card in enumerate(extracted_cards):
                        eid = card["emp_id"]
                        if eid == "UNKNOWN":
                            mismatches.append(f"Could not parse ID for {card['original_name']}")
                            continue
                        
                        card_type = "BASE"
                        if any(kw in card["raw_text"].lower() for kw in ["topup", "top up", "top-up", "super top"]): card_type = "TOPUP"
                        p_no = card["metadata"].policy_no if card["metadata"].policy_no else "UNKNOWN"
                        comp_name = getattr(card["metadata"], 'company_name', None)
                        
                        save_card_to_db(eid, card["bytes"], st.session_state.username, [card["metadata"]], p_no, card_type, comp_name)
                        
                        if opt_rename:
                            safe_name = re.sub(r'[^A-Za-z0-9]', '', str(card["metadata"].name)) if card["metadata"].name else str(idx_card)
                            save_name = f"{eid}_ECard.pdf"
                        else:
                            save_name = card["original_name"]
                            
                        zip_file.writestr(save_name, card["bytes"])
                        processed_count += 1
                        
            st.session_state.zip_data = zip_buffer.getvalue()
            gc.collect(); progress_bar.progress(1.0)
            
            mode_text = "Families Grouped" if opt_merge else "Individual Cards Processed"
            st.success(f"✅ Executed Logic! Successfully completed **{processed_count}** {mode_text}.")
            if mismatches:
                with st.expander(f"⚠️ View Unmapped Files ({len(mismatches)} warnings)"):
                    for err in mismatches: st.text(err)

    if st.session_state.get('zip_data'):
        st.download_button("📥 Download Final PDF Output (.zip)", data=st.session_state.zip_data, file_name="Processed_ECards_Archive.zip", mime="application/zip", type="primary", use_container_width=True)
# --- TAB 5: SEARCH INDIVIDUAL ---
with main_tab5:
    col_search, col_btn = st.columns([3, 1])
    search_id = col_search.text_input("Enter Employee ID (Hat / Co):", label_visibility="collapsed", placeholder="e.g. 101")
    search_policy_no = st.text_input("Specific Client Policy (Optional):", placeholder="e.g. 11022026", key="search_policy_filter")
    
    if col_btn.button("🔍 Search", use_container_width=True) and search_id:
        cards = get_cards_from_db(search_id.strip().upper(), policy_no=search_policy_no)
        members = get_members_from_db(search_id.strip().upper())
        
        if cards:
            st.success(f"✅ Found **{len(cards)}** associated policy bundle(s) locally.")
            if members:
                display_members = pd.DataFrame(members).drop(columns=['_id', 'id', 'emp_id'], errors='ignore')
                if search_policy_no:
                    display_members = display_members[display_members["policy_no"].str.upper() == search_policy_no.strip().upper()]
                st.dataframe(display_members, hide_index=True, use_container_width=True)
            
            for card in cards:
                c_type = card["card_type"]
                p_no = card["policy_no"]
                pdf_bytes = bytes(card['pdf_data'])
                st.download_button(label=f"📥 Download {c_type} Card ({p_no})", data=pdf_bytes, file_name=f"CapitupIndia_{search_id.upper()}_{p_no}_{c_type}.pdf", mime="application/pdf")
                preview_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_num in range(len(preview_doc)):
                    st.image(preview_doc[page_num].get_pixmap(dpi=150).tobytes("png"), use_container_width=True) 
                preview_doc.close()
        else: st.error("No E-Card found.")

# --- TAB 6: EMAIL DISTRIBUTION AGENT ---
with main_tab6:
    db = get_db()
    
    st.markdown("### ✉️ Email Dispatch Center")
    st.markdown("Configure corporate assets and automate welcome email dispatch for active employees.")
    
    policies_registered = db.ecards.distinct("policy_no")
    if not policies_registered: policies_registered = ["No Clients Ingested"]
        
    col_p_select, col_empty = st.columns([1, 2])
    selected_client_policy = col_p_select.selectbox("Select Active Client Campaign", policies_registered, key="t6_client_policy")

    st.divider()
    col_left_layout, col_right_layout = st.columns([1.2, 1])

    with col_left_layout:
        st.subheader("⚙️ Insurer Asset Vault")
        col_vault1, col_vault2, col_vault3 = st.columns(3) 
        
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
                    with open(CLAIM_FORM_PATH, "wb") as f: f.write(uploaded_claim_doc.getbuffer())
                    st.success("Saved!"); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

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
                    with open(POSTER_PATH, "wb") as f: f.write(uploaded_poster_doc.getbuffer())
                    st.success("Saved!"); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

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
                    with open(LOGO_PATH, "wb") as f: f.write(uploaded_logo_doc.getbuffer())
                    st.success("Saved!"); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("✉️ Email Architect")
        subject_line_input = st.text_input("SUBJECT LINE", value="Your Health Insurance E-Card & Welcome Kit")
        active_deadline_text = get_deadline_from_db(selected_client_policy)
        
        logo_tag_component = ""
        if os.path.exists(LOGO_PATH):
            logo_tag_component = """<div style="text-align: center; margin-bottom: 15px;"><img src="cid:logo_image" alt="CapitUp India Logo" style="height: 60px; width: auto; display: inline-block;" /></div>"""
        
        brand_html_template = f"""<div style="font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; max-width: 650px; margin: 0 auto; border: 1px solid #C29B38; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); background-color: #ffffff;">
  <div style="background-color: #0B1E30; padding: 28px 24px; text-align: center; border-bottom: 3px solid #C29B38; position: relative;">
    {logo_tag_component}
    <h2 style="color: #ffffff; margin: 0; font-size: 22px; letter-spacing: 1px; font-weight: 800; text-transform: uppercase;">CAPITUP INDIA</h2>
    <p style="color: #C29B38; margin: 5px 0 0 0; font-size: 11px; font-weight: bold; letter-spacing: 2px;">YOUR SECURE EMPLOYEE BENEFITS PARTNER</p>
  </div>
  
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

    <div style="text-align: center; margin: 32px 0 16px 0;">
      <a href="https://docs.google.com/forms/d/e/1FAIpQLSfMZ0SHY4pr9NVfZwHQRhU6Jmy-vN2K8INePRdkYQarVA_EMw/viewform?usp=pp_url&entry.877007954={{{{name}}}}&entry.863990631={{{{emp_id}}}}&entry.1115400795={{{{policy_no}}}}" 
         style="background-color: #23C2A9; color: #ffffff; padding: 14px 28px; text-decoration: none; font-size: 13px; font-weight: bold; border-radius: 6px; display: inline-block; box-shadow: 0 4px 10px rgba(35, 194, 169, 0.25); border: 1px solid #1fa895; transition: background-color 0.2s;">
        📝 Request E-Card Correction
      </a>
      <p style="color: #888; font-size: 10px; margin-top: 10px;">If you detect any spelling or coverage discrepancies, click the button above to request corrections.</p>
      <p style="color: #C29B38; font-size: 11px; font-weight: bold; margin-top: 6px;">⏱️ Correction Form Window Closes On: {{{{deadline}}}}</p>
    </div>
  </div>
  <!-- FOOTER -->
  <div style="background-color: #F4F6F8; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
    <p style="margin: 0; font-size: 12px; color: #0B1E30; font-weight: bold;">Thank you for being part of the CapitUp Family</p>
    <p style="margin: 4px 0 0 0; font-size: 10px; color: #888;">CapitUp India Pvt. Ltd. | 4th Floor, HUDA Techno Enclave, HITEC City, Hyderabad-500081</p>
    <div style="margin-top: 15px; font-size: 9px; color: #C29B38; font-weight: bold; letter-spacing: 1px;">
      ⚡ SECURED BY CAPITUP INDIA WATERMARK SYSTEM
    </div>
  </div>
</div>"""
        
        html_body_input = st.text_area("HTML BODY TEMPLATE (VARIABLES: {{name}}, {{emp_id}}, {{policy_no}})", value=brand_html_template, height=250)
        
        if st.checkbox("👁️ Toggle Live Preview", key="live_prev"):
            st.markdown("#### Live Preview Frame")
            preview_rendered = html_body_input.replace("{{name}}", st.session_state.username).replace("{{emp_id}}", "MOCK-101").replace("{{policy_no}}", selected_client_policy).replace("{{deadline}}", active_deadline_text)
            st.markdown("##### Logo & Header Card Preview")
            if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, caption="Vault Logo Asset Preview", use_container_width=True) 
            st.markdown("##### Email Content Preview")
            st.components.v1.html(preview_rendered, height=500, scrolling=True)

    with col_right_layout:
        st.subheader("👥 Pending Enrollment")
        st.markdown("Active users missing a welcome email.")

        st.markdown("<div style='background-color:#f0f2f6; padding:15px; border-radius:6px; border:1px solid #ddd;'>", unsafe_allow_html=True)
        t6_mapping_file = st.file_uploader("📥 Upload Client Mapping Directory (CSV/Excel) to map names & emails:", type=["csv", "xlsx", "xls"], key="t6_mapping_uploader")
        
        if t6_mapping_file:
            file_id_key = f"processed_{t6_mapping_file.name}_{t6_mapping_file.size}_{selected_client_policy}"
            if not st.session_state.get(file_id_key, False):
                with st.spinner("Processing mapping directory sheet..."):
                    try:
                        if t6_mapping_file.name.endswith('.csv'): df_map = pd.read_csv(t6_mapping_file)
                        else: df_map = pd.read_excel(t6_mapping_file)
                        df_map = clean_and_align_dataframe(df_map)
                        df_map_cols = list(df_map.columns)
                        
                        emp_col_map = robust_guess_column(df_map_cols, ["EMP ID", "EMPLOYEE ID", "ID", "HAT", "CO"])
                        name_col_map = robust_guess_column(df_map_cols, ["MEMBER NAME", "NAME", "EMPLOYEE NAME", "INSURED"])
                        email_col_map = robust_guess_column(df_map_cols, ["EMAIL", "EMAIL ADDRESS", "E CARDS ACCESS CODE", "ACCESS", "MAIL"])
                        rel_col_map = robust_guess_column(df_map_cols, ["RELATION", "RELATIONSHIP", "RELATI", "REL"])
                        
                        if not emp_col_map or not name_col_map or not email_col_map:
                            st.error("🚨 Column mapping failed. Please check your spreadsheet headers (Emp ID, Name, Email).")
                        else:
                            added_records = 0
                            parsed_employees = {}
                            for _, row in df_map.iterrows():
                                raw_emp_id = str(row[emp_col_map]).strip().upper()
                                raw_name = str(row[name_col_map]).strip()
                                raw_email = str(row[email_col_map]).strip().lower()
                                raw_rel = str(row[rel_col_map]).strip().upper() if (rel_col_map and rel_col_map in row) else "SELF"
                                
                                if not raw_emp_id or raw_emp_id in ["NAN", ""]: continue
                                if raw_emp_id not in parsed_employees or raw_rel in ["SELF", "PRIMARY", "EMPLOYEE", "PROPOSER"]:
                                    parsed_employees[raw_emp_id] = {"name": raw_name, "email": raw_email}
                            
                            for emp_id_key, detail in parsed_employees.items():
                                save_employee_to_directory(emp_id_key, detail["name"], detail["email"], selected_client_policy)
                                added_records += 1
                            
                            st.session_state[file_id_key] = True
                            st.success(f"✅ Successfully loaded and synced **{added_records}** primary employees to the database.")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e: st.error(f"Error parsing mapping sheet: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        pending_ecards = list(db.ecards.find({"policy_no": selected_client_policy, "email_sent": {"$ne": True}}))
        
        pending_display_list = []
        for ecard in pending_ecards:
            directory_record = db.directory.find_one({"emp_id": ecard["emp_id"], "policy_no": selected_client_policy})
            if directory_record:
                pending_display_list.append({"EMP ID": ecard["emp_id"], "Name": directory_record.get("name", "UNKNOWN"), "Email": directory_record.get("email", ""), "Card Type": ecard["card_type"]})
            else:
                pending_display_list.append({"EMP ID": ecard["emp_id"], "Name": "UNKNOWN (Incomplete Directory Metadata)", "Email": "", "Card Type": ecard["card_type"]})

        if pending_display_list:
            df_pending = pd.DataFrame(pending_display_list)
            st.dataframe(df_pending[["EMP ID", "Name", "Email", "Card Type"]], hide_index=True, use_container_width=True)
        else:
            st.info("No pending enrollments. All emails for this policy have been sent.")

        st.divider()
        st.subheader("✉️ Process Mail Queue")
        st.markdown("Dispatches SMTP mail & fetches E-Cards.")
        
        batch_limit = st.number_input("Batch Run Limit", min_value=1, max_value=500, value=20, step=1)
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
                cards = get_cards_from_db(emp_id, policy_no=selected_client_policy)
                
                if cards:
                    customized_html = html_body_input.replace("{{name}}", emp_name).replace("{{emp_id}}", emp_id).replace("{{policy_no}}", selected_client_policy).replace("{{deadline}}", active_deadline_text)
                    mail_sent = send_multi_ecard_email(recipient_email, subject_line_input, customized_html, cards)
                    
                    if mail_sent:
                        db.ecards.update_many({"emp_id": emp_id, "policy_no": selected_client_policy}, {"$set": {"email_sent": True}})
                        sent_success_count += 1
                        
                progress_bar.progress((idx + 1) / jobs_to_process)
                
            status_update.empty()
            progress_bar.empty()
            st.success(f"Successfully processed queue! Sent **{sent_success_count}** welcome emails.")
            time.sleep(1)
            st.rerun()

        st.divider()
        with st.expander("🛠️ Admin Testing & Queue Controls", expanded=False):
            st.markdown("Use these utility tools to reset email-sent flags and run simulations.")
            st.markdown("#### 📝 Google Form Status Controller")
            
            gas_setting = db.settings.find_one({"key": "gas_url"})
            stored_gas_url = gas_setting["value"] if gas_setting else ""
            gas_url_input = st.text_input("Google Apps Script Web App URL:", value=stored_gas_url, placeholder="https://script.google.com/macros/s/.../exec")
            
            if gas_url_input != stored_gas_url:
                db.settings.update_one({"key": "gas_url"}, {"$set": {"key": "gas_url", "value": gas_url_input}}, upsert=True)
                st.success("Google Apps Script URL securely saved to database!")
                time.sleep(0.5)
                st.rerun()
                
            if gas_url_input:
                current_form_status = get_form_status(gas_url_input)
                current_deadline_db_value = get_deadline_from_db(selected_client_policy)
                
                if current_form_status == "CLOSED" and current_deadline_db_value not in ["Form Closed", "Expired / Closed"]:
                    save_deadline_to_db(selected_client_policy, "Expired / Closed")
                    st.rerun()
                
                if current_form_status == "OPEN": st.markdown(f"Live Form Status: **🟢 ACTIVE (Accepting Responses)**")
                elif current_form_status == "CLOSED": st.markdown(f"Live Form Status: **🔴 INACTIVE (Form Shut)**")
                else: st.markdown(f"Live Form Status: **⚠️ DISCONNECTED ({current_form_status})**")
                
                st.markdown("##### ⏱️ Schedule Response Window Deadline")
                deadline_option = st.selectbox(
                    "Set Duration Window (Closes form automatically):",
                    ["Manual (No Timer)", "1 Day", "3 Days", "1 Week (7 Days)", "10 Days", "2 Weeks (14 Days)"],
                    key="deadline_timer_select"
                )
                
                col_open_form, col_close_form = st.columns(2)
                
                if col_open_form.button("🟢 Open Form & Start Timer", use_container_width=True):
                    if deadline_option == "Manual (No Timer)":
                        res = set_form_status(gas_url_input, "open")
                        if res:
                            save_deadline_to_db(selected_client_policy, "Manual Close Required")
                            st.success("Google Form is now OPEN to corrections (Manual closing required)!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        duration_mapping = {"1 Day": 24.0, "3 Days": 72.0, "1 Week (7 Days)": 168.0, "10 Days": 240.0, "2 Weeks (14 Days)": 336.0}
                        target_hours = duration_mapping[deadline_option]
                        response_text = schedule_form_close(gas_url_input, target_hours)
                        
                        if response_text and "SCHEDULED_FOR_" in response_text:
                            iso_str = response_text.replace("SCHEDULED_FOR_", "")
                            utc_datetime = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
                            local_datetime = utc_datetime + timedelta(hours=5, minutes=30)
                            formatted_deadline_str = local_datetime.strftime("%B %d, %Y at %I:%M %p (IST)")
                            
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
            
            col_admin_input, col_admin_btn = st.columns([2, 1])
            reset_target_id = col_admin_input.text_input("Target Employee ID to Re-queue:", placeholder="e.g. TEST101", key="admin_reset_id_val")
            if col_admin_btn.button("🔄 Re-queue Employee", use_container_width=True) and reset_target_id:
                clean_target = reset_target_id.strip().upper()
                db.ecards.update_many({"emp_id": clean_target, "policy_no": selected_client_policy}, {"$set": {"email_sent": False}})
                st.success(f"Employee {clean_target} successfully re-queued for {selected_client_policy}!")
                time.sleep(1)
                st.rerun()
                
            if st.button("🚨 Re-queue All Users in Selected Campaign", type="secondary", use_container_width=True, key="admin_reset_all_btn"):
                db.ecards.update_many({"policy_no": selected_client_policy}, {"$set": {"email_sent": False}})
                st.success("All e-cards successfully re-queued for sending!")
                time.sleep(1)
                st.rerun()


# --- TAB 7: LIGHTNING-FAST FILENAME-BASED FAMILYFICATION ---
with main_tab7:
    st.markdown("### 👨‍👩‍👧‍👦 Lightning-Fast E-Card Familyfication")
    st.markdown("This module uses **Filename Matching** (Card Number ➡️ Employee ID) to instantly merge family packets without using AI. It guarantees mathematical accuracy and processes hundreds of cards in seconds.")
    
    master_file = st.file_uploader("1. Upload Active List Master Tracker (Excel/CSV)", type=["xlsx", "xls", "csv"], key="fam_master_v3")
    
    if master_file:
        try:
            if master_file.name.endswith(".csv"):
                df_fam = pd.read_csv(master_file)
            else:
                df_fam = pd.read_excel(master_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            df_fam = pd.DataFrame()
            
        if not df_fam.empty:
            df_fam = clean_and_align_dataframe(df_fam)
            
            st.info("Map the columns from your Master List:")
            col1, col2 = st.columns(2)
            with col1:
                # E.g. "UHID" or "OLD UHID"
                uhid_col = st.selectbox("Select the **Card Number / UHID** Column:", df_fam.columns, key="fam_uhid_col")
            with col2:
                # E.g. "EMPLOYEE ID"
                emp_id_col = st.selectbox("Select the **Employee ID** Column:", df_fam.columns, key="fam_emp_col_v3")
                
            st.divider()
            fam_pdf_files = st.file_uploader("2. Upload Individual E-Card PDFs", type=["pdf"], accept_multiple_files=True, key="fam_pdfs_v3")
            
            if fam_pdf_files and st.button("🧬 Fast Merge by Card Number", type="primary", use_container_width=True):
                
                # --- STEP 1: CREATE UHID -> EMP_ID MAPPING ---
                df_clean = df_fam.dropna(subset=[uhid_col, emp_id_col])
                uhid_to_emp = {}
                
                for _, row in df_clean.iterrows():
                    # Clean the UHID (strip whitespaces, remove .0 if excel parsed it as a float)
                    clean_uhid = str(row[uhid_col]).strip().upper()
                    if clean_uhid.endswith('.0'):
                        clean_uhid = clean_uhid[:-2]
                    
                    clean_emp = str(row[emp_id_col]).strip().upper()
                    
                    if clean_uhid:
                        uhid_to_emp[clean_uhid] = clean_emp
                
                # --- STEP 2: PROCESS THE PDFS INSTANTLY ---
                family_groups = {} 
                unmatched_pdfs = []
                
                progress_bar = st.progress(0)
                
                for i, pdf in enumerate(fam_pdf_files):
                    # 1. Clean the filename to isolate the UHID (e.g. IL0916163245000)
                    raw_name = pdf.name.upper()
                    # Strip .PDF and common trailing strings like _ECARD
                    clean_filename = re.sub(r'(\.PDF|_ECARD|_CARD|_FAMILY).*$', '', raw_name).strip()
                    
                    # 2. Look up the Employee ID
                    matched_emp_id = uhid_to_emp.get(clean_filename)
                    
                    # 3. Fallback: Check if the filename contains the UHID (Partial Match)
                    if not matched_emp_id:
                        for u_key in uhid_to_emp.keys():
                            if u_key in clean_filename or clean_filename in u_key:
                                matched_emp_id = uhid_to_emp[u_key]
                                break
                    
                    # 4. Group Them
                    if matched_emp_id:
                        if matched_emp_id not in family_groups:
                            family_groups[matched_emp_id] = []
                        family_groups[matched_emp_id].append((pdf.name, pdf.getvalue()))
                    else:
                        unmatched_pdfs.append((pdf.name, pdf.getvalue()))
                        
                    progress_bar.progress((i + 1) / len(fam_pdf_files))
                    
                # --- STEP 3: MERGE INTO FAMILIES USING FITZ ---
                st.success("✅ Mapping complete! Merging PDFs into Family packets...")
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Merge matched families
                    for emp_id, pdf_list in family_groups.items():
                        merged_pdf = fitz.open()
                        for (p_name, p_bytes) in pdf_list:
                            try:
                                temp_pdf = fitz.open(stream=p_bytes, filetype="pdf")
                                merged_pdf.insert_pdf(temp_pdf)
                                temp_pdf.close()
                            except Exception as e:
                                st.warning(f"Could not merge {p_name}: {str(e)}")
                                
                        merged_pdf_bytes = merged_pdf.tobytes(garbage=4, deflate=True)
                        zf.writestr(f"Family_Packets/{emp_id}.pdf", merged_pdf_bytes)
                        merged_pdf.close()
                        
                    # Save unmatched cards
                    for (p_name, p_bytes) in unmatched_pdfs:
                        zf.writestr(f"Unmatched_Cards/{p_name}", p_bytes)
                        
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Cards Processed", len(fam_pdf_files))
                m2.metric("Families Consolidated", len(family_groups))
                m3.metric("Unmatched / Orphan Cards", len(unmatched_pdfs))
                
                if len(unmatched_pdfs) > 0:
                    st.warning("⚠️ Some cards could not be mapped. Make sure the filename exactly matches the Card Number/UHID in the Excel sheet.")

                st.download_button(
                    label="📥 Download Familyfied E-Cards (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="CapitUp_Family_Ecards.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )


# --- TAB 8: COVERAGE GAP FINDER ---
with main_tab8:
    st.markdown("### 🔍 Active List vs. ZIP E-Card Coverage Analyzer")
    st.markdown("This utility evaluates your Active Member tracker against a ZIP archive of processed E-Cards to pinpoint coverage gaps.")

    col_gap_1, col_gap_2 = st.columns(2)
    with col_gap_1:
        st.markdown("#### 1. Upload Active List")
        gap_excel_file = st.file_uploader("Upload Active List (Excel/CSV)", type=["xlsx", "xls", "csv"], key="gap_analysis_excel")
    with col_gap_2:
        st.markdown("#### 2. Upload E-Cards Archive")
        gap_zip_file = st.file_uploader("Upload E-Cards ZIP Archive", type=["zip"], key="gap_analysis_zip")

    if gap_excel_file:
        try:
            if gap_excel_file.name.endswith(".csv"):
                df_gap = pd.read_csv(gap_excel_file)
            else:
                df_gap = pd.read_excel(gap_excel_file)
        except Exception as e:
            st.error(f"Error reading the active list file: {e}")
            df_gap = pd.DataFrame()

        if not df_gap.empty:
            df_gap = clean_and_align_dataframe(df_gap)
            
            st.markdown("---")
            st.markdown("#### ⚙️ Column & Verification Settings")
            col_sel1, col_sel2 = st.columns(2)
            
            with col_sel1:
                gap_emp_col = st.selectbox(
                    "Primary Employee ID Column (used to match with ZIP filenames):", 
                    df_gap.columns, 
                    key="gap_analysis_emp_col"
                )
            with col_sel2:
                gap_alt_col = st.selectbox(
                    "Secondary Match Column (e.g. Card No/UHID, optional):", 
                    ["None"] + list(df_gap.columns), 
                    key="gap_analysis_alt_col"
                )

            if gap_zip_file:
                if st.button("🔍 Run Coverage Analyzer", type="primary", use_container_width=True, key="btn_run_gap_analysis"):
                    with st.spinner("Analyzing coverage gaps..."):
                        
                        # --- STEP 1: READ ZIP ARCHIVE FILENAMES ---
                        zip_contents = []
                        try:
                            with zipfile.ZipFile(gap_zip_file) as z:
                                for file_info in z.infolist():
                                    if not file_info.is_dir():
                                        base_name = os.path.basename(file_info.filename).upper()
                                        name_no_ext, _ = os.path.splitext(base_name)
                                        # Normalize filename identifiers
                                        clean_name = re.sub(r"(_ECARDS|_ECARD|_FAMILY_ECARDS|_FAMILY_ECARD|_FAMILY|_CARDS|_CARD)$", "", name_no_ext).strip()
                                        zip_contents.append({
                                            "original_path": file_info.filename,
                                            "clean_name": clean_name
                                        })
                        except Exception as e:
                            st.error(f"Error parsing ZIP file: {e}")
                            st.stop()

                        zip_filenames = [item["clean_name"] for item in zip_contents]

                        # --- STEP 2: SAFE BOUNDARY-MATCHING HELPER ---
                        def is_match(target_val, z_filename):
                            target_val = str(target_val).strip().upper()
                            z_filename = str(z_filename).strip().upper()
                            if not target_val or target_val in ["NAN", "NONE", ""]:
                                return False
                            if target_val == z_filename:
                                return True
                            
                            # Boundary check to prevent false substring matches (e.g., '101' matching '1015' or '1')
                            pattern = r'\b' + re.escape(target_val) + r'\b'
                            if re.search(pattern, z_filename):
                                return True
                            
                            normalized_filename = z_filename.replace('_', ' ').replace('-', ' ')
                            if target_val in normalized_filename.split():
                                return True
                            return False

                        # --- STEP 3: EXECUTE GAP IDENTIFICATION ---
                        has_card_list = []
                        for idx, row in df_gap.iterrows():
                            emp_id = str(row[gap_emp_col]).strip().upper()
                            if emp_id.endswith(".0"):
                                emp_id = emp_id[:-2]

                            match_found = False
                            for z_fn in zip_filenames:
                                if is_match(emp_id, z_fn):
                                    match_found = True
                                    break

                            if not match_found and gap_alt_col != "None":
                                alt_val = str(row[gap_alt_col]).strip().upper()
                                if alt_val.endswith(".0"):
                                    alt_val = alt_val[:-2]
                                if alt_val and alt_val not in ["NAN", "NONE", ""]:
                                    for z_fn in zip_filenames:
                                        if is_match(alt_val, z_fn):
                                            match_found = True
                                            break

                            has_card_list.append(match_found)

                        df_gap["E_Card_Status_Matched"] = has_card_list
                        
                        df_matched = df_gap[df_gap["E_Card_Status_Matched"] == True].copy()
                        df_missing = df_gap[df_gap["E_Card_Status_Matched"] == False].copy()

                        # Clean up status flags from output
                        df_missing.drop(columns=["E_Card_Status_Matched"], errors="ignore", inplace=True)
                        df_matched.drop(columns=["E_Card_Status_Matched"], errors="ignore", inplace=True)

                        st.divider()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Members in Tracker", len(df_gap))
                        m2.metric("Matched E-Cards Found", len(df_matched))
                        m3.metric("Missing E-Cards (Gaps)", len(df_missing))

                        if not df_missing.empty:
                            st.warning(f"Identified {len(df_missing)} active member(s) lacking corresponding matches in the ZIP file archive.")
                            st.dataframe(df_missing, hide_index=True, use_container_width=True)

                            # --- STEP 4: EXPORT MISSING ENTRIES AS EXCEL ---
                            try:
                                output_buffer = BytesIO()
                                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                                    df_missing.to_excel(writer, index=False, sheet_name="Missing E-Cards")
                                excel_bytes = output_buffer.getvalue()

                                st.download_button(
                                    label="📥 Download Missing E-Cards Report (.xlsx)",
                                    data=excel_bytes,
                                    file_name="Missing_ECards_Report.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    type="primary",
                                    use_container_width=True
                                )
                            except Exception as write_err:
                                st.error(f"Failed to generate Excel report file: {write_err}")
                        else:
                            st.success("All tracker entries matched corresponding files in the ZIP folder.")
