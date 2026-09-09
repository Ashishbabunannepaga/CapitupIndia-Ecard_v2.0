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
import secrets  
import urllib.parse  
from werkzeug.security import generate_password_hash, check_password_hash
from paddleocr import PaddleOCR
import boto3
from botocore.client import Config
from pymongo import MongoClient
import pymongo
from bson.binary import Binary

# --- EMAIL DEPENDENCIES ---
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage

# --- SUPPRESS AI & C++ NOISE ---
os.environ["GLOG_minloglevel"] = "3"   
os.environ["KMP_WARNINGS"] = "0"       
warnings.filterwarnings("ignore")      

from parser_worker import extract_metadata_from_text, CardMetadata

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="CapitUp Dual-POV Benefits Portal", page_icon="🪪", layout="wide")

import pillow_heif
pillow_heif.register_heif_opener()

DEFAULT_GAS_URL = "https://script.google.com/macros/s/AKfycbwexxFRlk43f3-SP6fH5VsgSeGpf-cDQXkETNlUT8OJ06AlOGirJ39ivP44HszMMNpAFg/exec"
ALLOWED_DOMAINS = ["capitupindia.com", "capitup.com"]

# --- GLOBAL UTILITY & HELPERS ---
def guess_column(columns, keywords, index_fallback=0):
    for col in columns:
        for kw in keywords:
            if kw.upper() in str(col).upper(): return col
    return columns[index_fallback]

def parse_int_safe(val):
    if pd.isna(val): return None
    val_str = str(val).split('.')[0].strip()
    try: return int(val_str)
    except ValueError: return None

def clean_and_align_dataframe(df, forced_header_row=None):
    if df.empty: return df
    KEYWORD_BANK = [
        "EMP", "EMPLOYEE", "ID", "MEMBER", "NAME", "INSURED", "EMAIL", "MAIL", 
        "RELATION", "RELATIONSHIP", "POLICY", "CARD", "DOB", "AGE", "GENDER", 
        "GHI", "SUM", "DOJ", "CO", "HAT", "ADDRESS", "UHID", "CODE", "STATUS", "CORP"
    ]
    if forced_header_row is not None and 0 <= forced_header_row < len(df):
        raw_headers = df.iloc[forced_header_row].values
        clean_headers = []
        seen = {}
        for i, h in enumerate(raw_headers):
            h_clean = str(h).strip() if pd.notna(h) and str(h).strip() not in ["", "nan", "None"] else f"Column_{i+1}"
            if h_clean in seen:
                seen[h_clean] += 1
                h_clean = f"{h_clean}_{seen[h_clean]}"
            else: seen[h_clean] = 0
            clean_headers.append(h_clean)
        df.columns = clean_headers
        df = df.iloc[forced_header_row + 1:].reset_index(drop=True)
        return df.dropna(how="all").reset_index(drop=True)

    cols_str = [str(c).strip().upper() for c in df.columns]
    unnamed_count = sum(1 for c in cols_str if "UNNAMED" in c or c in ["", "NAN", "NONE"])
    needs_header_search = (unnamed_count / len(cols_str)) > 0.3 or any(kw in cols_str[0] for kw in ["TOTAL RECORD", "RECORD COUNT", "REPORT", "CLIENT", "LIST"])
    best_header_idx = None
    max_score = 0
    scan_limit = min(15, len(df))
    for r_idx in range(scan_limit):
        row_vals = [str(x).strip().upper() for x in df.iloc[r_idx].values if pd.notna(x)]
        score = 0
        for val in row_vals:
            for kw in KEYWORD_BANK:
                if kw in val: score += 1; break
        if score > max_score and score >= 2:
            max_score = score
            best_header_idx = r_idx
    if best_header_idx is not None and (needs_header_search or max_score >= 3):
        raw_headers = df.iloc[best_header_idx].values
        clean_headers = []
        seen = {}
        for i, h in enumerate(raw_headers):
            h_clean = str(h).strip() if pd.notna(h) and str(h).strip() not in ["", "nan", "None"] else f"Column_{i+1}"
            if h_clean in seen:
                seen[h_clean] += 1
                h_clean = f"{h_clean}_{seen[h_clean]}"
            else: seen[h_clean] = 0
            clean_headers.append(h_clean)
        df.columns = clean_headers
        df = df.iloc[best_header_idx + 1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all").reset_index(drop=True)

def robust_guess_column(columns, primary_keywords, fallback_keywords=None):
    columns_upper = [str(c).strip().upper() for c in columns]
    for kw in primary_keywords:
        kw_up = kw.strip().upper()
        if kw_up in columns_upper: return columns[columns_upper.index(kw_up)]
    if fallback_keywords:
        for kw in fallback_keywords:
            kw_up = kw.strip().upper()
            if kw_up in columns_upper: return columns[columns_upper.index(kw_up)]
    for idx, col in enumerate(columns_upper):
        for kw in primary_keywords:
            kw_up = kw.strip().upper()
            if len(kw_up) > 1 and kw_up in col: return columns[idx]
    if fallback_keywords:
        for idx, col in enumerate(columns_upper):
            for kw in fallback_keywords:
                kw_up = kw.strip().upper()
                if len(kw_up) > 1 and kw_up in col: return columns[idx]
    return None

# --- DATABASE & R2 CREDENTIALS ---
try:
    MONGO_URI = st.secrets["mongo"]["uri"]
    MONGO_DBNAME = st.secrets["mongo"]["dbname"]
except KeyError:
    st.error("🚨 CRITICAL ERROR: Could not find MongoDB Atlas [mongo] credentials in secrets!")
    st.stop()

R2_ENABLED = False
if "r2" in st.secrets:
    R2_CONFIG = dict(st.secrets["r2"])
    s3_client = boto3.client(
        's3',
        endpoint_url=f"https://{R2_CONFIG['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_CONFIG['access_key_id'],
        aws_secret_access_key=R2_CONFIG['secret_access_key'],
        region_name="auto", 
        config=Config(signature_version='s3v4', region_name="auto")
    )
    R2_ENABLED = True

@st.cache_resource(show_spinner=False)
def load_ocr_engine():
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    return PaddleOCR(use_textline_orientation=True, lang='en')

@st.cache_resource
def get_mongo_client():
    return MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)

def get_db():
    return get_mongo_client()[MONGO_DBNAME]

# --- GOOGLE FORM UTILITIES ---
def get_form_status(api_url):
    if not api_url or not str(api_url).startswith("http"): return "DISCONNECTED"
    try: return requests.get(api_url + "?action=status", timeout=4).text.strip().upper()
    except Exception: return "DISCONNECTED"

def set_form_status(api_url, action):
    if not api_url or not str(api_url).startswith("http"): return None
    try: return requests.get(api_url + f"?action={action}", timeout=5).text.strip().upper()
    except Exception: return None

def schedule_form_close(api_url, hours):
    if not api_url or not str(api_url).startswith("http"): return None
    try: return requests.get(api_url + f"?action=schedule&hours={hours}", timeout=5).text.strip()
    except Exception: return None

def get_deadline_from_db(policy_no):
    try:
        db = get_db()
        setting = db.settings.find_one({"key": f"deadline_{policy_no}"})
        return setting["value"] if setting else "Not Set"
    except Exception: return "Not Set"

def save_deadline_to_db(policy_no, deadline_str):
    try:
        db = get_db()
        db.settings.update_one({"key": f"deadline_{policy_no}"}, {"$set": {"value": str(deadline_str)}}, upsert=True)
    except Exception as e: logging.error(f"Failed to save deadline: {e}")

def init_db():
    db = get_db()
    db.users.create_index("username", unique=True)
    db.users.create_index("email", unique=True, sparse=True)
    db.ecards.create_index(
        [("policy_no", pymongo.ASCENDING), ("emp_id", pymongo.ASCENDING), ("card_type", pymongo.ASCENDING)],
        unique=True, name="unique_emp_card_type"
    )
    db.card_members.create_index("emp_id")
    db.directory.create_index([("emp_id", pymongo.ASCENDING), ("policy_no", pymongo.ASCENDING)], unique=True)
    db.assets.create_index("name", unique=True)
    db.email_logs.create_index("timestamp", expireAfterSeconds=604800)
    db.email_logs.create_index([("policy_no", pymongo.ASCENDING), ("status", pymongo.ASCENDING)])

# --- HIGH-SPEED CACHED R2 & MONGO DISCOVERY ---
@st.cache_data(ttl=60, show_spinner=False)
def get_live_tenants_and_policies():
    discovered = {}
    if R2_ENABLED:
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=R2_CONFIG["bucket_name"], Prefix="ecards/", Delimiter="/"):
                for cp in page.get('CommonPrefixes', []):
                    comp_prefix = cp['Prefix']
                    parts = comp_prefix.strip('/').split('/')
                    if len(parts) >= 2:
                        comp_name = parts[1].replace('_', ' ').strip().upper()
                        if comp_name not in discovered: discovered[comp_name] = []
                        p_page = s3_client.list_objects_v2(Bucket=R2_CONFIG["bucket_name"], Prefix=comp_prefix, Delimiter="/")
                        for pp in p_page.get('CommonPrefixes', []):
                            pol_parts = pp['Prefix'].strip('/').split('/')
                            if len(pol_parts) >= 3:
                                pol_no = pol_parts[2].replace('_', '-').strip().upper()
                                if pol_no not in discovered[comp_name] and pol_no != "UNKNOWN-POLICY":
                                    discovered[comp_name].append(pol_no)
        except Exception as e: logging.error(f"R2 Tenant discovery error: {e}")

    try:
        db = get_db()
        mongo_records = db.ecards.aggregate([
            {"$group": {"_id": {"company": "$company_name", "policy": "$policy_no"}}}
        ])
        for rec in mongo_records:
            c = rec["_id"].get("company")
            p = rec["_id"].get("policy")
            if c and c not in ["UNKNOWN_COMPANY", "GENERAL_CORP"]:
                c_clean = c.replace('_', ' ').strip().upper()
                if c_clean not in discovered: discovered[c_clean] = []
                if p and p not in ["UNKNOWN_POLICY", "UNKNOWN"] and p not in discovered[c_clean]:
                    discovered[c_clean].append(p.strip().upper())
    except Exception as e: logging.error(f"MongoDB discovery error: {e}")

    return discovered

def authenticate_user(username_or_email, password):
    db = get_db()
    clean_id = username_or_email.strip()
    user = db.users.find_one({"$or": [{"username": clean_id}, {"email": clean_id.lower()}]})
    return user and check_password_hash(user['password_hash'], password)

def create_user(username, email, password):
    db = get_db()
    try:
        db.users.insert_one({
            "username": username.strip(), "email": email.strip().lower(),
            "password_hash": generate_password_hash(password), "role": "ADMIN", "is_verified": True, "created_at": datetime.utcnow()
        })
        return True
    except pymongo.errors.DuplicateKeyError: return False

# --- STORAGE & DATA HELPERS ---
def save_card_to_db(emp_id, pdf_bytes, username, family_members, policy_no="UNKNOWN", card_type="BASE", company_name=None):
    clean_emp_id = str(emp_id).strip().upper()
    clean_policy_no = str(policy_no).strip().upper()
    clean_company_name = str(company_name).strip().upper() if company_name else "UNKNOWN_COMPANY"
    
    illegal_chars = r'[\\/*?:"<>|]'
    sanitized_policy = re.sub(illegal_chars, "", clean_policy_no).strip().replace(" ", "_")
    if not sanitized_policy: sanitized_policy = "UNKNOWN_POLICY"
    sanitized_company = re.sub(illegal_chars, "", clean_company_name).strip().replace(" ", "_")
    
    file_key = f"ecards/{sanitized_company}/{sanitized_policy}/{card_type}/{clean_emp_id}.pdf"
    update_payload = {
        "emp_id": clean_emp_id, "policy_no": clean_policy_no, "company_name": clean_company_name,
        "card_type": card_type, "uploaded_by": username, "upload_date": datetime.utcnow(), "email_sent": False  
    }
    if R2_ENABLED:
        try:
            s3_client.put_object(Bucket=R2_CONFIG["bucket_name"], Key=file_key, Body=pdf_bytes, ContentType="application/pdf")
            update_payload["r2_key"] = file_key
        except Exception as e: logging.error(f"R2 upload failed: {e}")
    else: update_payload["pdf_binary"] = Binary(pdf_bytes)

    db = get_db()
    db.ecards.update_one({"emp_id": clean_emp_id, "policy_no": clean_policy_no, "card_type": card_type}, {"$set": update_payload}, upsert=True)
    db.card_members.delete_many({"emp_id": clean_emp_id, "policy_no": clean_policy_no, "policy_type": card_type})
    member_docs = []
    for member in family_members:
        member_docs.append({
            "emp_id": clean_emp_id, "name": member.name, "policy_no": clean_policy_no, "policy_type": card_type,
            "card_no": member.card_no, "relationship": member.relationship, "age": member.age, "valid_up_to": member.valid_up_to
        })
    if member_docs: db.card_members.insert_many(member_docs)

def save_employee_to_directory(emp_id, name, email, policy_no, company_name=None, role="EMPLOYEE"):
    db = get_db()
    db.directory.update_one(
        {"emp_id": str(emp_id).strip().upper(), "policy_no": str(policy_no).strip().upper()},
        {"$set": {
            "emp_id": str(emp_id).strip().upper(), "name": str(name).strip(), 
            "email": str(email).strip().lower(), "policy_no": str(policy_no).strip().upper(),
            "company_name": str(company_name).strip().upper() if company_name else "UNKNOWN_CORP",
            "role": role,
            "updated_at": datetime.utcnow()
        }}, upsert=True
    )

def get_cards_from_db(emp_id, policy_no=None):
    db = get_db()
    query = {"emp_id": str(emp_id).strip().upper()}
    if policy_no: query["policy_no"] = str(policy_no).strip().upper()
    db_results = list(db.ecards.find(query))
    cards_list = []
    for row in db_results:
        pdf_data = None
        if R2_ENABLED and "r2_key" in row:
            try:
                response = s3_client.get_object(Bucket=R2_CONFIG["bucket_name"], Key=row["r2_key"])
                pdf_data = response["Body"].read()
            except Exception: pass
        elif "pdf_binary" in row: pdf_data = row["pdf_binary"]
            
        if pdf_data:
            cards_list.append({
                "card_type": row["card_type"], "policy_no": row["policy_no"], "pdf_data": pdf_data,
                "company_name": row.get("company_name", "UNKNOWN"), "upload_date": row.get("upload_date", datetime.utcnow())
            })
    return cards_list

def get_members_from_db(emp_id=None):
    db = get_db()
    if emp_id: cursor = db.card_members.find({"emp_id": str(emp_id).strip().upper()}).sort("relationship", -1)
    else: cursor = db.card_members.find().sort("emp_id", 1).limit(100)
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
        pdf_data = None
        if R2_ENABLED and "r2_key" in row:
            try:
                response = s3_client.get_object(Bucket=R2_CONFIG["bucket_name"], Key=row["r2_key"])
                pdf_data = response["Body"].read()
            except Exception: pass
        elif "pdf_binary" in row: pdf_data = row["pdf_binary"]
        if pdf_data:
            results.append({"emp_id": row["emp_id"], "card_type": row["card_type"], "policy_no": row["policy_no"], "pdf_data": pdf_data})
    return results

# --- CACHED ASSETS ---
@st.cache_data(ttl=60, show_spinner=False)
def get_asset(asset_name):
    doc = get_db().assets.find_one({"name": asset_name})
    return doc["data"] if doc else None

def save_asset(asset_name, binary_data):
    get_db().assets.update_one({"name": asset_name}, {"$set": {"data": Binary(binary_data)}}, upsert=True)
    st.cache_data.clear()

def delete_asset(asset_name):
    get_db().assets.delete_one({"name": asset_name})
    st.cache_data.clear()

def log_email_dispatch(emp_id, name, recipient_email, policy_no, status, error_reason=None, campaign_type="WELCOME_KIT"):
    try:
        db = get_db()
        db.email_logs.insert_one({
            "emp_id": str(emp_id).strip().upper(), "name": str(name).strip(),
            "recipient_email": str(recipient_email).strip().lower(), "policy_no": str(policy_no).strip().upper(),
            "status": status, "campaign_type": campaign_type, "error_reason": str(error_reason) if error_reason else None,
            "dispatched_by": st.session_state.get("username", "SYSTEM"), "timestamp": datetime.utcnow()
        })
    except Exception as e: logging.error(f"Log dispatch error: {e}")

# --- SMTP DISPATCH ENGINES ---
def send_multi_ecard_email(recipient_email, subject, body_html, cards_list):
    try:
        SMTP_CONFIG = st.secrets["smtp"]
        msg = MIMEMultipart('mixed')
        msg['From'] = SMTP_CONFIG["sender_email"]
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg_related = MIMEMultipart('related')
        msg.attach(msg_related)
        
        poster_bytes = get_asset("poster")
        if poster_bytes:
            poster_tag = """<tr><td align="center" style="padding: 0 40px 30px 40px;"><img src="cid:poster_image" alt="Mediclaim Summary Poster" style="width: 100%; max-width: 520px; height: auto; border-radius: 6px; display: block;" /></td></tr>"""
            body_html = body_html.replace("<!-- FOOTER -->", poster_tag + "\n<!-- FOOTER -->")
            
        msg_related.attach(MIMEText(body_html, 'html'))
        logo_bytes = get_asset("logo")
        if logo_bytes:
            msg_logo = MIMEImage(logo_bytes)
            msg_logo.add_header('Content-ID', '<logo_image>')
            msg_logo.add_header('Content-Disposition', 'inline', filename="logo.png")
            msg_related.attach(msg_logo)
            
        if poster_bytes:
            msg_img = MIMEImage(poster_bytes)
            msg_img.add_header('Content-ID', '<poster_image>')
            msg_img.add_header('Content-Disposition', 'inline', filename="poster.png")
            msg_related.attach(msg_img)
            
        for card in cards_list:
            pdf_bytes = bytes(card["pdf_data"])
            card_label = card["card_type"]
            attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
            attachment.add_header('Content-Disposition', 'attachment', filename=f"HealthCard_{card_label}.pdf")
            msg.attach(attachment)
            
        claim_bytes = get_asset("claim_form")
        if claim_bytes:
            claim_attachment = MIMEApplication(claim_bytes, _subtype="pdf")
            claim_attachment.add_header('Content-Disposition', 'attachment', filename="Reimbursement_Claim_Form.pdf")
            msg.attach(claim_attachment)
            
        port = int(SMTP_CONFIG["port"])
        if port == 465:
            with smtplib.SMTP_SSL(SMTP_CONFIG["server"], port, timeout=25) as server:
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(SMTP_CONFIG["username"], recipient_email, msg.as_string())
        else:
            with smtplib.SMTP(SMTP_CONFIG["server"], port, timeout=25) as server:
                server.ehlo(); server.starttls(); server.ehlo()
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(SMTP_CONFIG["username"], recipient_email, msg.as_string())
        return True, None
    except Exception as e: return False, str(e)

def send_launch_email(recipient_email, subject, body_html, guide_asset_key=None, banner_asset_key=None):
    try:
        SMTP_CONFIG = st.secrets["smtp"]
        msg = MIMEMultipart('mixed')
        msg['From'] = SMTP_CONFIG["sender_email"]
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg_related = MIMEMultipart('related')
        msg.attach(msg_related)
        
        banner_bytes = get_asset(banner_asset_key) if banner_asset_key else None
        if banner_bytes:
            banner_tag = """<tr><td align="center" style="padding: 0 40px 20px 40px;"><img src="cid:launch_banner" alt="Portal Overview" style="width: 100%; max-width: 540px; height: auto; border-radius: 8px; display: block;" /></td></tr>"""
            body_html = body_html.replace("<!-- BANNER -->", banner_tag + "\n<!-- BANNER -->")
            
        msg_related.attach(MIMEText(body_html, 'html'))
        logo_bytes = get_asset("logo")
        if logo_bytes:
            msg_logo = MIMEImage(logo_bytes)
            msg_logo.add_header('Content-ID', '<logo_image>')
            msg_logo.add_header('Content-Disposition', 'inline', filename="logo.png")
            msg_related.attach(msg_logo)
            
        if banner_bytes:
            msg_b = MIMEImage(banner_bytes)
            msg_b.add_header('Content-ID', '<launch_banner>')
            msg_b.add_header('Content-Disposition', 'inline', filename="launch_banner.png")
            msg_related.attach(msg_b)
            
        guide_bytes = get_asset(guide_asset_key) if guide_asset_key else None
        if guide_bytes:
            guide_att = MIMEApplication(guide_bytes, _subtype="pdf")
            guide_att.add_header('Content-Disposition', 'attachment', filename="CapitUp_Portal_Guide.pdf")
            msg.attach(guide_att)
            
        port = int(SMTP_CONFIG["port"])
        if port == 465:
            with smtplib.SMTP_SSL(SMTP_CONFIG["server"], port, timeout=25) as server:
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(SMTP_CONFIG["username"], recipient_email, msg.as_string())
        else:
            with smtplib.SMTP(SMTP_CONFIG["server"], port, timeout=25) as server:
                server.ehlo(); server.starttls(); server.ehlo()
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(SMTP_CONFIG["username"], recipient_email, msg.as_string())
        return True, None
    except Exception as e: return False, str(e)

# --- AUTH HELPERS ---
def is_corporate_email(email):
    if not email or "@" not in email: return False
    domain = email.strip().split("@")[-1].lower()
    return not ALLOWED_DOMAINS or domain in ALLOWED_DOMAINS

def send_registration_otp(recipient_email, otp_code):
    try:
        SMTP_CONFIG = st.secrets["smtp"]
        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_CONFIG["sender_email"]
        msg['To'] = recipient_email
        msg['Subject'] = "🔐 CapitUp Portal - Registration Verification Code"
        body_html = f"""<div style="font-family: Arial, sans-serif; padding: 20px; border: 1px solid #C29B38; border-radius: 8px; max-width: 500px;"><div style="background-color: #0B1E30; padding: 15px; text-align: center;"><h2 style="color: #ffffff; margin: 0;">CAPITUP PORTAL VERIFICATION</h2></div><div style="padding: 20px; text-align: center;"><p>Your one-time security code is:</p><div style="background-color: #F4F6F8; padding: 12px; font-size: 26px; font-weight: bold; letter-spacing: 6px; color: #0B1E30; border-radius: 4px;">{otp_code}</div><p style="color: #666; font-size: 11px; margin-top: 15px;">Valid for 5 minutes.</p></div></div>"""
        msg.attach(MIMEText(body_html, 'html'))
        port = int(SMTP_CONFIG["port"])
        if port == 465:
            with smtplib.SMTP_SSL(SMTP_CONFIG["server"], port, timeout=20) as server:
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(SMTP_CONFIG["username"], recipient_email, msg.as_string())
        else:
            with smtplib.SMTP(SMTP_CONFIG["server"], port, timeout=20) as server:
                server.ehlo(); server.starttls(); server.ehlo()
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(SMTP_CONFIG["username"], recipient_email, msg.as_string())
        return True
    except Exception as e: return False

init_db()

# --- AUTH STATE ---
if "failed_emails" not in st.session_state: st.session_state.failed_emails = []
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ""
if 'reg_step' not in st.session_state: st.session_state.reg_step = 1
if 'reg_payload' not in st.session_state: st.session_state.reg_payload = {}
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# --- LOGIN & REGISTRATION GATE ---
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 CapitUp Portal Login")
        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Verified Registration"])
        with tab_login:
            with st.form("login_form"):
                user_input = st.text_input("Username or Corporate Email")
                pass_input = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True, type="primary"):
                    if authenticate_user(user_input.strip(), pass_input):
                        st.session_state.logged_in = True
                        st.session_state.username = user_input.strip()
                        st.rerun()
                    else: st.error("❌ Invalid credentials.")
        with tab_register:
            if st.session_state.reg_step == 1:
                with st.form("reg_step1"):
                    reg_email = st.text_input("Corporate Email*", placeholder="name@capitupindia.com")
                    reg_user = st.text_input("Choose a Username*")
                    reg_pass = st.text_input("Choose Password (Min 8 chars)*", type="password")
                    reg_confirm = st.text_input("Confirm Password*", type="password")
                    if st.form_submit_button("📩 Send Verification OTP", use_container_width=True, type="primary"):
                        clean_email, clean_user = reg_email.strip().lower(), reg_user.strip()
                        if not clean_email or not clean_user or not reg_pass: st.warning("⚠️ Fill all fields.")
                        elif not is_corporate_email(clean_email): st.error(f"🚨 Only {', '.join(ALLOWED_DOMAINS)} allowed.")
                        elif len(reg_pass) < 8 or reg_pass != reg_confirm: st.error("❌ Password error (min 8 chars / match).")
                        elif get_db().users.find_one({"$or": [{"username": clean_user}, {"email": clean_email}]}): st.error("⚠️ Account exists.")
                        else:
                            otp = str(secrets.randbelow(900000) + 100000)
                            if send_registration_otp(clean_email, otp):
                                st.session_state.reg_payload = {"username": clean_user, "email": clean_email, "password": reg_pass, "otp": otp, "expires_at": time.time() + 300}
                                st.session_state.reg_step = 2; st.success("✅ OTP sent!"); time.sleep(1); st.rerun()
                            else: st.error("❌ Failed to send OTP.")
            elif st.session_state.reg_step == 2:
                with st.form("reg_step2"):
                    ent_otp = st.text_input("Enter 6-Digit OTP", max_chars=6)
                    c_v1, c_v2 = st.columns(2)
                    if c_v1.form_submit_button("✅ Verify & Register", use_container_width=True, type="primary"):
                        payload = st.session_state.reg_payload
                        if time.time() > payload.get("expires_at", 0): st.error("⏰ OTP Expired."); st.session_state.reg_step = 1
                        elif ent_otp.strip() == payload.get("otp"):
                            if create_user(payload["username"], payload["email"], payload["password"]):
                                st.success("🎉 Account created!"); st.session_state.reg_step = 1; time.sleep(1.5); st.rerun()
                        else: st.error("❌ Incorrect OTP.")
                    if c_v2.form_submit_button("🔄 Cancel", use_container_width=True):
                        st.session_state.reg_step = 1; st.rerun()
    st.stop()

# --- SIDEBAR PERSONA SWITCHER ---
st.sidebar.title(f"👤 {st.session_state.username}")
persona_mode = st.sidebar.radio(
    "🎯 Select Portal Perspective:", 
    ["🏢 HR / Corporate Admin Hub", "👤 User (Employee) Self-Service"]
)

if st.sidebar.button("Logout", type="secondary", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

db = get_db()

# ==============================================================================
# 👤 PERSPECTIVE 1: USER (EMPLOYEE) SELF-SERVICE & 24/7 AI COPILOT
# ==============================================================================
if persona_mode == "👤 User (Employee) Self-Service":
    st.title("🪪 Employee Digital Health Wallet & Benefits Copilot")
    st.caption("Access instant multi-page family E-Cards, review coverage limits, and chat 24/7 with your policy.")

    col_u_search, col_u_btn = st.columns([3, 1])
    user_emp_id = col_u_search.text_input("Enter your Employee ID or Corporate Email:", placeholder="e.g. 771461 or 800042", key="u_emp_search_key")
    
    if user_emp_id:
        clean_uid = user_emp_id.strip().upper()
        dir_entry = db.directory.find_one({"$or": [{"emp_id": clean_uid}, {"email": clean_uid.lower()}]})
        real_emp_id = dir_entry["emp_id"] if dir_entry else clean_uid
        
        cards = get_cards_from_db(real_emp_id)
        members = get_members_from_db(real_emp_id)
        
        if cards or members:
            company_name = dir_entry.get("company_name", "Corporate Group Mediclaim") if dir_entry else (cards[0].get("company_name") if cards else "Corporate Insurance")
            emp_name = dir_entry.get("name", "Employee") if dir_entry else (members[0].get("name") if members else "Employee")
            
            st.success(f"Welcome, **{emp_name}** ({company_name})")
            
            u_tab_wallet, u_tab_chat = st.tabs(["🪪 Digital E-Card Wallet", "🤖 24/7 Policy AI Copilot"])
            
            # WALLET
            with u_tab_wallet:
                st.subheader("👨‍👩‍👧‍👦 Family Coverage & Digital Health Cards")
                if members:
                    df_m = pd.DataFrame(members).drop(columns=['_id', 'id', 'emp_id'], errors='ignore')
                    st.dataframe(df_m, hide_index=True, use_container_width=True)
                
                if cards:
                    st.markdown("#### 📥 Instant Download & Previews")
                    for c_idx, c in enumerate(cards):
                        c_type = c["card_type"]
                        p_no = c["policy_no"]
                        pdf_b = bytes(c["pdf_data"])
                        
                        col_d1, col_d2 = st.columns([1.5, 3])
                        with col_d1:
                            st.markdown(f"**{c_type} Card** | Policy: `{p_no}`")
                            st.download_button(
                                label=f"📥 Download {c_type} Family Card (.pdf)",
                                data=pdf_b,
                                file_name=f"CapitUp_{real_emp_id}_{c_type}_ECard.pdf",
                                mime="application/pdf",
                                type="primary",
                                key=f"btn_dl_{c_idx}"
                            )
                        with col_d2:
                            with st.expander(f"👁️ View {c_type} Card Preview", expanded=(c_idx==0)):
                                pdoc = fitz.open(stream=pdf_b, filetype="pdf")
                                for pnum in range(len(pdoc)):
                                    st.image(pdoc[pnum].get_pixmap(dpi=150).tobytes("png"), use_container_width=True)
                                pdoc.close()
                else:
                    st.warning("E-Card PDF is being generated. Please check back shortly.")
                    
            # 24/7 AI POLICY COPILOT
            with u_tab_chat:
                st.subheader("🤖 Ask Anything About Your Health Policy")
                st.caption("Instant answers on Room Rent limits, Maternity rules, Daycare surgeries, and Claim processes.")
                
                col_q1, col_q2, col_q3 = st.columns(3)
                if col_q1.button("🏥 Room Rent Limit", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "What is my room rent limit?"})
                    st.session_state.chat_history.append({"role": "assistant", "content": "Under your company's Group Health Insurance with Bajaj Allianz, **Room Rent is capped at 1% of Sum Insured per day** for Normal Rooms, and **2% for ICU charges**. If you choose a higher room category, proportionate deductions will apply."})
                if col_q2.button("🤱 Maternity Coverage", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Is maternity covered?"})
                    st.session_state.chat_history.append({"role": "assistant", "content": "Yes! Maternity is covered up to **₹50,000 for Normal Delivery** and **₹75,000 for C-Section**. The standard 9-month waiting period is waived under your corporate group policy."})
                if col_q3.button("📋 Reimbursement Steps", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "How do I file a reimbursement claim?"})
                    st.session_state.chat_history.append({"role": "assistant", "content": "To file a reimbursement claim:\n1. Collect Claim Form Part A (Employee) & Part B (Hospital).\n2. Compile original final bill, payment receipts, discharge summary, and pharmacy bills.\n3. Submit a single PDF under 10MB to claims@capitupindia.com within 30 days of discharge."})
                
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                        
                user_query = st.chat_input("Ask a question about your coverage, exclusions, or claims...")
                if user_query:
                    with st.chat_message("user"): st.markdown(user_query)
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    
                    uq_lower = user_query.lower()
                    if "room" in uq_lower or "rent" in uq_lower or "icu" in uq_lower:
                        ans = "Your **Room Rent limit is 1% of Sum Insured/day** (e.g., ₹4,000/day on a 4 Lakh policy) and **ICU is capped at 2% of Sum Insured/day**. Proportionate deductions apply to doctor fees if a higher category room is chosen."
                    elif "maternity" in uq_lower or "baby" in uq_lower or "delivery" in uq_lower or "pregnancy" in uq_lower:
                        ans = "**Maternity Coverage:** Covered up to ₹50,000 (Normal) and ₹75,000 (C-Section) for up to 2 children. New-born baby cover is included from Day 1 within the overall family floater sum insured."
                    elif "claim" in uq_lower or "reimbursement" in uq_lower or "bill" in uq_lower:
                        ans = "For **Reimbursement Claims**, ensure you submit:\n• Duly filled Claim Forms (Part A & B)\n• Original Discharge Summary & Detailed Final Bill\n• Payment Receipts & Diagnostic Reports\n• Cancelled Cheque & ID proofs to `claims@capitupindia.com`."
                    elif "pre-existing" in uq_lower or "ped" in uq_lower or "waiting" in uq_lower:
                        ans = "Under your Corporate Group Policy, **Pre-Existing Diseases (PED) are covered from Day 1** with 0 waiting period!"
                    elif "cashless" in uq_lower or "hospital" in uq_lower or "admission" in uq_lower:
                        ans = "For **Cashless Hospitalization**, show your CapitUp Digital Health Card along with your Aadhaar Card at the hospital's TPA/Insurance Helpdesk 48 hours prior for planned admissions, or within 24 hours for emergencies."
                    else:
                        ans = f"Under your policy `{cards[0]['policy_no'] if cards else 'Active Policy'}`, your family is covered for 24-hour hospitalization, 30-day pre-hospitalization, and 60-day post-hospitalization medical expenses. For specific claim approvals, reach out to support@capitupindia.com."
                        
                    with st.chat_message("assistant"): st.markdown(ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
        else:
            st.error("❌ No active coverage or E-Card located for this Employee ID/Email.")
    st.stop()


# ==============================================================================
# 🏢 PERSPECTIVE 2: HR / CORPORATE ADMIN HUB (MULTI-TENANT)
# ==============================================================================
st.title("🏢 Corporate HR Administration & Ingestion Hub")

tab_universal, tab_modular, tab_bulk, tab_directory, tab_search, tab_email, tab_launch, tab_family, tab_gap = st.tabs([
    "📤 Universal Processing", 
    "📥 Ingest E-Cards", 
    "📥 Bulk Retrieval", 
    "📊 Global Directory", 
    "🔍 Search Individual", 
    "✉️ E-Card Welcome Kit", 
    "🚀 Portal Launch & Feedback",
    "🧬 Familyfication",
    "🔍 Coverage Gap Finder"
])

# --- TAB 1: UNIVERSAL PROCESSING & LIVE R2 TENANT ROUTER ---
with tab_universal:
    st.markdown("### 🛠️ Universal Ingestion Engine with Live Cloud Discovery")
    
    live_tenants_map = get_live_tenants_and_policies()
    discovered_comp_list = sorted(list(live_tenants_map.keys()))
    
    st.markdown("#### 🏢 Target Corporate Tenant & Policy Setup")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        tenant_opts = ["➕ Custom / New Tenant Name"] + discovered_comp_list
        sel_comp_choice = st.selectbox("Select Client Company (Scanned from Cloud):", tenant_opts, key="u_comp_select")
        
        if sel_comp_choice == "➕ Custom / New Tenant Name":
            target_company = st.text_input("Enter Tenant / Company Name:", placeholder="e.g. STRATEGIC SYSTEMS IT SOLUTIONS", key="u_comp_input_custom")
        else:
            target_company = st.text_input("Company Name Override (Optional):", value=sel_comp_choice, key="u_comp_input_override")

    with col_c2:
        avail_pols = live_tenants_map.get(sel_comp_choice, []) if sel_comp_choice in live_tenants_map else []
        pol_opts = ["➕ Custom / New Policy Number"] + avail_pols
        sel_pol_choice = st.selectbox("Select Policy Number (Scanned from Cloud):", pol_opts, key="u_pol_select")
        
        if sel_pol_choice == "➕ Custom / New Policy Number":
            target_policy_override = st.text_input("Enter Policy Number:", placeholder="e.g. OG-27-1801-8403-00000112", key="u_pol_input_custom")
        else:
            target_policy_override = st.text_input("Policy Number Override (Optional):", value=sel_pol_choice, key="u_pol_input_override")

    clean_comp_preview = re.sub(r'[\\/*?:"<>|]', "", target_company).strip().replace(" ", "_").upper() if target_company else "AUTO_DETECT"
    clean_pol_preview = re.sub(r'[\\/*?:"<>|]', "", target_policy_override).strip().replace(" ", "_").upper() if target_policy_override else "AUTO_DETECT"
    st.info(f"📁 **R2 Destination Path:** `ecards / {clean_comp_preview} / {clean_pol_preview} / [CARD_TYPE] / [EMP_ID].pdf`")

    st.divider()
    pdf_files = st.file_uploader("Upload E-Card PDF(s)", type=["pdf"], accept_multiple_files=True, key="v1upload")

    st.markdown("<div style='background-color: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #dee2e6;'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: opt_master = st.checkbox("✂️ **Split Master PDF**", value=False)
    with c2: opt_merge = st.checkbox("🧬 **Group & Merge Families**", value=True)
    with c3: opt_rename = st.checkbox("🔄 **Smart Rename Files**", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if pdf_files and st.button("🚀 Process & Ingest E-Cards", type="primary", use_container_width=True):
        ocr_engine = load_ocr_engine()
        progress_bar = st.progress(0)
        extracted_cards = [] 
        for idx, pdf_file in enumerate(pdf_files):
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            if opt_master:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    for rect in detect_card_boundaries(page):
                        raw_text = page.get_text("text", clip=rect)
                        parsed_data = extract_metadata_from_text(raw_text)
                        if not parsed_data.emp_id:
                            em = re.search(r"(?:EMPLOYEE\s*CODE|EMP\s*ID)\s*[:\-]?\s*([A-Za-z0-9]+)", raw_text, re.IGNORECASE)
                            if em: parsed_data.emp_id = em.group(1).strip().upper()
                        if not parsed_data.policy_no:
                            pm = re.search(r"POLICY\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)", raw_text, re.IGNORECASE)
                            if pm: parsed_data.policy_no = pm.group(1).strip().upper()
                        
                        temp_doc = fitz.open()
                        temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                        temp_doc[-1].set_cropbox(rect)
                        card_bytes = temp_doc.tobytes(garbage=4, deflate=True)
                        temp_doc.close()
                        extracted_cards.append({"emp_id": parsed_data.emp_id or "UNKNOWN", "metadata": parsed_data, "bytes": card_bytes, "raw_text": raw_text, "original_name": f"P{page_num}.pdf"})
            else:
                if len(doc) > 0:
                    raw_text = doc[0].get_text("text")
                    parsed_data = extract_metadata_from_text(raw_text)
                    if not parsed_data.emp_id:
                        em = re.search(r"(?:EMPLOYEE\s*CODE|EMP\s*ID)\s*[:\-]?\s*([A-Za-z0-9]+)", raw_text, re.IGNORECASE)
                        if em: parsed_data.emp_id = em.group(1).strip().upper()
                    if not parsed_data.policy_no:
                        pm = re.search(r"POLICY\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)", raw_text, re.IGNORECASE)
                        if pm: parsed_data.policy_no = pm.group(1).strip().upper()
                    final_emp_id = parsed_data.emp_id or re.sub(r"(_ECARDS|_ECARD|_FAMILY).*$", "", os.path.splitext(pdf_file.name)[0].strip().upper())
                    extracted_cards.append({"emp_id": final_emp_id or "UNKNOWN", "metadata": parsed_data, "bytes": pdf_file.getvalue(), "raw_text": raw_text, "original_name": pdf_file.name})
            doc.close()
            progress_bar.progress((idx + 1) / len(pdf_files))

        zip_buffer = BytesIO()
        processed_count = 0
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            if opt_merge:
                family_groups = {}
                for card in extracted_cards:
                    eid = card["emp_id"]
                    if eid == "UNKNOWN": continue
                    if eid not in family_groups: family_groups[eid] = {"bytes": [], "metadata": [], "raw_text": ""}
                    family_groups[eid]["bytes"].append(card["bytes"])
                    family_groups[eid]["metadata"].append(card["metadata"])
                    family_groups[eid]["raw_text"] += " " + card["raw_text"]
                
                for eid, group in family_groups.items():
                    merged_pdf = fitz.open()
                    for b in group["bytes"]:
                        tdoc = fitz.open(stream=b, filetype="pdf"); merged_pdf.insert_pdf(tdoc); tdoc.close()
                    merged_bytes = merged_pdf.tobytes(garbage=4, deflate=True)
                    merged_pdf.close()
                    
                    card_type = "TOPUP" if any(kw in group["raw_text"].lower() for kw in ["topup", "top up", "super top"]) else "BASE"
                    p_no = target_policy_override.strip().upper() if target_policy_override else (group["metadata"][0].policy_no or "UNKNOWN_POLICY")
                    comp_name = target_company.strip().upper() if target_company else (getattr(group["metadata"][0], 'company_name', None) or "GENERAL_CORP")
                    
                    save_card_to_db(eid, merged_bytes, st.session_state.username, group["metadata"], p_no, card_type, comp_name)
                    save_name = f"{eid}_ECard.pdf" if opt_rename else f"Family_{eid}.pdf"
                    zip_file.writestr(save_name, merged_bytes)
                    processed_count += 1
            else:
                for idx_c, card in enumerate(extracted_cards):
                    eid = card["emp_id"]
                    if eid == "UNKNOWN": continue
                    card_type = "TOPUP" if any(kw in card["raw_text"].lower() for kw in ["topup", "top up", "super top"]) else "BASE"
                    p_no = target_policy_override.strip().upper() if target_policy_override else (card["metadata"].policy_no or "UNKNOWN_POLICY")
                    comp_name = target_company.strip().upper() if target_company else (getattr(card["metadata"], 'company_name', None) or "GENERAL_CORP")
                    save_card_to_db(eid, card["bytes"], st.session_state.username, [card["metadata"]], p_no, card_type, comp_name)
                    save_name = f"{eid}_{idx_c}_ECard.pdf" if opt_rename else card["original_name"]
                    zip_file.writestr(save_name, card["bytes"])
                    processed_count += 1

        st.session_state.zip_data = zip_buffer.getvalue()
        gc.collect(); progress_bar.progress(1.0)
        st.cache_data.clear() # Clear cache to show new client immediately
        st.success(f"✅ Ingestion Complete! Saved **{processed_count}** files to Tenant: **{clean_comp_preview}**.")
        if st.session_state.get('zip_data'):
            st.download_button("📥 Download Output ZIP", data=st.session_state.zip_data, file_name=f"{clean_comp_preview}_ECards.zip", mime="application/zip", type="primary", use_container_width=True)

# --- TAB 2: MODULAR INGESTION ---
with tab_modular:
    col_bm, col_tm = st.columns(2)
    with col_bm:
        st.subheader("🟦 Base Policy Module")
        b_excel = st.file_uploader("Upload Member List", type=["xlsx", "xls", "csv"], key="m_base_xl")
        b_pdfs = st.file_uploader("Upload Base Cards", type=["pdf"], accept_multiple_files=True, key="m_base_pdf")
        if st.button("Ingest Base", type="primary", use_container_width=True) and b_excel and b_pdfs:
            df = clean_and_align_dataframe(pd.read_csv(b_excel) if b_excel.name.endswith('.csv') else pd.read_excel(b_excel))
            cols = list(df.columns)
            emp_c = guess_column(cols, ["EMP", "ID", "HAT", "CO"])
            name_c = robust_guess_column(cols, ["NAME", "MEMBER", "INSURED"])
            comp_c = robust_guess_column(cols, ["COMPANY", "CORPORATE", "CLIENT"])
            pol_c = robust_guess_column(cols, ["POLICY", "POL"])
            c_name = str(df.iloc[0][comp_c]).strip().upper() if comp_c else "BASE_CORP"
            p_no = str(df.iloc[0][pol_c]).strip().upper() if pol_c else "UNKNOWN"
            count = 0
            for pfile in b_pdfs:
                eid = os.path.splitext(pfile.name)[0].strip().upper()
                m_rows = df[df[emp_c].astype(str).str.strip() == eid]
                if not m_rows.empty:
                    save_employee_to_directory(eid, str(m_rows.iloc[0][name_c]), "", p_no, c_name)
                    save_card_to_db(eid, pfile.getvalue(), st.session_state.username, [], p_no, "BASE", c_name)
                    count += 1
            st.cache_data.clear()
            st.success(f"Ingested {count} Base Cards.")

    with col_tm:
        st.subheader("🟧 Top-Up Policy Module")
        t_excel = st.file_uploader("Upload Topup List", type=["xlsx", "xls", "csv"], key="m_top_xl")
        t_pdfs = st.file_uploader("Upload Topup Cards", type=["pdf"], accept_multiple_files=True, key="m_top_pdf")
        if st.button("Ingest Topup", type="primary", use_container_width=True) and t_excel and t_pdfs:
            df = clean_and_align_dataframe(pd.read_csv(t_excel) if t_excel.name.endswith('.csv') else pd.read_excel(t_excel))
            cols = list(df.columns)
            emp_c = guess_column(cols, ["EMP", "ID", "HAT", "CO"])
            name_c = robust_guess_column(cols, ["NAME", "MEMBER", "INSURED"])
            comp_c = robust_guess_column(cols, ["COMPANY", "CORPORATE", "CLIENT"])
            pol_c = robust_guess_column(cols, ["POLICY", "POL"])
            c_name = str(df.iloc[0][comp_c]).strip().upper() if comp_c else "TOPUP_CORP"
            p_no = str(df.iloc[0][pol_c]).strip().upper() if pol_c else "UNKNOWN"
            count = 0
            for pfile in t_pdfs:
                eid = os.path.splitext(pfile.name)[0].strip().upper()
                m_rows = df[df[emp_c].astype(str).str.strip() == eid]
                if not m_rows.empty:
                    save_employee_to_directory(eid, str(m_rows.iloc[0][name_c]), "", p_no, c_name)
                    save_card_to_db(eid, pfile.getvalue(), st.session_state.username, [], p_no, "TOPUP", c_name)
                    count += 1
            st.cache_data.clear()
            st.success(f"Ingested {count} Topup Cards.")

# --- TAB 3: BULK RETRIEVAL ---
with tab_bulk:
    st.markdown("### 📥 Bulk E-Card Retrieval")
    b_pol = st.text_input("Filter Policy (Optional):", placeholder="e.g. OG-27-1801-8403-00000112")
    b_input = st.text_area("Employee IDs (comma/space separated):")
    if st.button("📦 Fetch & Package ZIP", type="primary", use_container_width=True) and b_input.strip():
        clean_ids = list(set([i.strip().upper() for i in b_input.replace(',', ' ').split() if i.strip()]))
        found = get_bulk_cards_from_db(clean_ids, policy_no=b_pol)
        if found:
            zbuf = BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for c in found: zf.writestr(f"{c['policy_no']}_{c['emp_id']}_{c['card_type']}.pdf", bytes(c['pdf_data']))
            st.download_button("📥 Download Batch ZIP", data=zbuf.getvalue(), file_name="Bulk_ECards.zip", mime="application/zip", type="primary", use_container_width=True)
            st.success(f"Packaged {len(found)} cards.")
        else: st.error("No matching cards found.")

# --- TAB 4: DIRECTORY ---
with tab_directory:
    st.markdown("### 📊 Active Employee Directory")
    members = get_members_from_db()
    if members:
        df_all = pd.DataFrame(members).drop(columns=['_id', 'id'], errors='ignore')
        st_term = st.text_input("🔍 Search Directory:")
        if st_term: df_all = df_all[df_all['name'].str.contains(st_term, case=False, na=False) | df_all['emp_id'].str.contains(st_term, case=False, na=False)]
        st.dataframe(df_all, hide_index=True, use_container_width=True)

# --- TAB 5: SEARCH INDIVIDUAL ---
with tab_search:
    col_s1, col_s2 = st.columns([3, 1])
    s_id = col_s1.text_input("Enter Employee ID:", placeholder="e.g. 771461")
    s_pol = st.text_input("Policy Number (Optional):")
    if col_s2.button("🔍 Search", use_container_width=True) and s_id:
        cards = get_cards_from_db(s_id, policy_no=s_pol)
        members = get_members_from_db(s_id)
        if cards:
            st.success(f"Found {len(cards)} card(s).")
            if members: st.dataframe(pd.DataFrame(members).drop(columns=['_id', 'id', 'emp_id'], errors='ignore'), hide_index=True, use_container_width=True)
            for c in cards:
                pdf_b = bytes(c["pdf_data"])
                st.download_button(f"📥 Download {c['card_type']} ({c['policy_no']})", data=pdf_b, file_name=f"{s_id}_{c['card_type']}.pdf", mime="application/pdf")
                pdoc = fitz.open(stream=pdf_b, filetype="pdf")
                for pnum in range(len(pdoc)): st.image(pdoc[pnum].get_pixmap(dpi=150).tobytes("png"), use_container_width=True)
                pdoc.close()
        else: st.error("No card found.")

# --- TAB 6: EMAIL DISTRIBUTION (WELCOME KIT & OVERRIDES) ---
with tab_email:
    st.markdown("### ✉️ Welcome Kit & E-Card Distribution Center")
    
    live_tenants_map = get_live_tenants_and_policies()
    discovered_comp_list = sorted(list(live_tenants_map.keys()))
    
    col_camp1, col_camp2 = st.columns(2)
    with col_camp1:
        camp_comp_opts = ["🔍 Auto-Detect / All"] + discovered_comp_list + ["➕ Custom Company Name"]
        sel_camp_comp = st.selectbox("Select Client Company (Live Cloud Scanned):", camp_comp_opts, key="t6_camp_comp")
        if sel_camp_comp == "➕ Custom Company Name":
            t6_custom_company = st.text_input("Enter Company Name Override:", placeholder="e.g. STRATEGIC SYSTEMS IT SOLUTIONS", key="t6_comp_override_manual")
        else:
            t6_custom_company = st.text_input("Company Name Override (Optional):", value=sel_camp_comp if sel_camp_comp != "🔍 Auto-Detect / All" else "", key="t6_comp_override")

    with col_camp2:
        avail_camp_pols = live_tenants_map.get(sel_camp_comp, []) if sel_camp_comp in live_tenants_map else []
        all_db_pols = db.ecards.distinct("policy_no")
        combined_pols = sorted(list(set(avail_camp_pols + all_db_pols)))
        camp_pol_opts = ["🔍 All Policies"] + combined_pols + ["➕ Custom Policy Number"]
        sel_camp_pol = st.selectbox("Select Target Policy Campaign:", camp_pol_opts, key="t6_camp_pol")
        if sel_camp_pol == "➕ Custom Policy Number":
            t6_custom_policy = st.text_input("Enter Policy Number Override:", placeholder="e.g. OG-27-1801-8403-00000112", key="t6_pol_override_manual")
        else:
            t6_custom_policy = st.text_input("Policy Number Override (Optional):", value=sel_camp_pol if sel_camp_pol != "🔍 All Policies" else "", key="t6_pol_override")

    active_scope_policy = t6_custom_policy.strip().upper() if t6_custom_policy else (sel_camp_pol if sel_camp_pol not in ["🔍 All Policies", "➕ Custom Policy Number"] else None)
    active_display_company = t6_custom_company.strip().upper() if t6_custom_company else (sel_camp_comp if sel_camp_comp != "🔍 Auto-Detect / All" else "Corporate Group Mediclaim")
    
    st.divider()
    col_l1, col_l2 = st.columns([1.2, 1])
    
    with col_l1:
        st.subheader("⚙️ Asset Vault & Google Form Controller")
        c_v1, c_v2, c_v3 = st.columns(3)
        with c_v1:
            if get_asset("claim_form"):
                st.success("Claim Form Active"); 
                if st.button("Delete Form", key="d_cf"): delete_asset("claim_form"); st.rerun()
            else:
                up_cf = st.file_uploader("Upload Claim Form", type=["pdf"], key="up_cf")
                if up_cf: save_asset("claim_form", up_cf.getvalue()); st.rerun()
        with c_v2:
            if get_asset("poster"):
                st.success("Poster Active"); 
                if st.button("Delete Poster", key="d_pos"): delete_asset("poster"); st.rerun()
            else:
                up_pos = st.file_uploader("Upload Poster", type=["png", "jpg"], key="up_pos")
                if up_pos: save_asset("poster", up_pos.getvalue()); st.rerun()
        with c_v3:
            if get_asset("logo"):
                st.success("Logo Active"); 
                if st.button("Delete Logo", key="d_logo"): delete_asset("logo"); st.rerun()
            else:
                up_log = st.file_uploader("Upload Logo", type=["png", "jpg"], key="up_log")
                if up_log: save_asset("logo", up_log.getvalue()); st.rerun()

        st.markdown("---")
        gas_url = db.settings.find_one({"key": "gas_url"})["value"] if db.settings.find_one({"key": "gas_url"}) else DEFAULT_GAS_URL
        f_stat = get_form_status(gas_url)
        dline = get_deadline_from_db(active_scope_policy or "DEFAULT")
        st.markdown(f"Live Form Status: **{'🟢 OPEN' if f_stat=='OPEN' else '🔴 CLOSED'}** | Active Closes: **{dline}**")
        d_opt = st.selectbox("Set Response Window:", ["1 Day (24 hrs)", "3 Days (72 hrs)", "1 Week (7 Days)", "Manual Open"], key="t6_d_opt")
        c_o1, c_o2 = st.columns(2)
        if c_o1.button("🟢 Open Form", use_container_width=True, type="primary"):
            res = schedule_form_close(gas_url, 24.0 if "1 Day" in d_opt else 72.0)
            formatted_dl = (datetime.utcnow() + timedelta(hours=5, minutes=30, days=1 if "1 Day" in d_opt else 3)).strftime("%d-%b-%Y at %I:%M %p (IST)")
            save_deadline_to_db(active_scope_policy or "DEFAULT", formatted_dl)
            st.success("Form Opened!"); time.sleep(1); st.rerun()
        if c_o2.button("🔴 Close Form", use_container_width=True):
            set_form_status(gas_url, "close")
            save_deadline_to_db(active_scope_policy or "DEFAULT", "Form Closed"); st.warning("Form Closed!"); time.sleep(1); st.rerun()
            
        subj_in = st.text_input("SUBJECT LINE", value="Your Health Insurance E-Card & Welcome Kit", key="t6_subj")
        
        logo_tag_component = """<div style="text-align: center; margin-bottom: 15px;"><img src="cid:logo_image" alt="CapitUp India Logo" style="height: 60px; width: auto; display: inline-block;" /></div>""" if get_asset("logo") else ""
        
        brand_html_template = f"""<div style="font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; max-width: 650px; margin: 0 auto; border: 1px solid #C29B38; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); background-color: #ffffff;">
  <div style="background-color: #0B1E30; padding: 28px 24px; text-align: center; border-bottom: 3px solid #C29B38;">
    {logo_tag_component}
    <h2 style="color: #ffffff; margin: 0; font-size: 22px; letter-spacing: 1px; font-weight: 800; text-transform: uppercase;">CAPITUP INDIA</h2>
    <p style="color: #C29B38; margin: 5px 0 0 0; font-size: 11px; font-weight: bold; letter-spacing: 2px;">YOUR SECURE EMPLOYEE BENEFITS PARTNER</p>
  </div>
  <div style="padding: 32px 24px;">
    <p style="font-size: 15px; margin-top: 0;">Dear <strong>{{{{name}}}}</strong>,</p>
    <p style="font-size: 14px;">We are pleased to welcome you to the <strong>CapitUp India</strong> ecosystem. Your Group Health Insurance policy with <strong>Bajaj Allianz General Insurance Company</strong> is active for the period <strong>26-May-2026 to 25-May-2027</strong>.</p>
    <p style="font-size: 14px;">Please find attached your Health Cards / E-Cards and policy coverage details for your reference.</p>
    <div style="background-color: #F4F6F8; border-left: 4px solid #C29B38; padding: 14px; margin: 20px 0; border-radius: 4px;">
      <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <tr><td style="width: 40%; font-weight: bold; color: #0B1E30; padding: 3px 0;">Employee ID:</td><td style="color: #333;">{{{{emp_id}}}}</td></tr>
        <tr><td style="font-weight: bold; color: #0B1E30; padding: 3px 0;">Company:</td><td style="color: #333;">{{{{company_name}}}}</td></tr>
        <tr><td style="font-weight: bold; color: #0B1E30; padding: 3px 0;">Policy Number:</td><td style="color: #333;">{{{{policy_no}}}}</td></tr>
      </table>
    </div>
    <div style="text-align: center; margin: 30px 0;">
      <p style="color: #C29B38; font-size: 11px; font-weight: bold;">⏱️ Correction Form Window Closes On: {{{{deadline}}}}</p>
    </div>
  </div>
  <!-- FOOTER -->
  <div style="background-color: #F4F6F8; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
    <p style="margin: 0; font-size: 12px; color: #0B1E30; font-weight: bold;">Thank you for being part of the CapitUp Family</p>
    <p style="margin: 4px 0 0 0; font-size: 10px; color: #888;">CapitUp India Pvt. Ltd. | HITEC City, Hyderabad</p>
  </div>
</div>"""
        html_in = st.text_area("HTML BODY TEMPLATE", value=brand_html_template, height=200, key="t6_html_tmpl")
        
        if st.checkbox("👁️ Live Preview", key="prev_t6"):
            rendered_t6 = html_in.replace("{{name}}", st.session_state.username).replace("{{emp_id}}", "MOCK-101").replace("{{company_name}}", active_display_company).replace("{{policy_no}}", active_scope_policy or "OG-27-1801-8403-00000112").replace("{{deadline}}", dline)
            st.components.v1.html(rendered_t6, height=450, scrolling=True)

    with col_l2:
        st.subheader("👥 Directory Sync & Mail Queue")
        up_map = st.file_uploader("Upload Client Mapping Sheet (CSV/Excel)", type=["csv", "xlsx"], key="t6_map_up")
        if up_map:
            df_m = clean_and_align_dataframe(pd.read_csv(up_map) if up_map.name.endswith('.csv') else pd.read_excel(up_map))
            cols_m = list(df_m.columns)
            c_e1, c_e2 = st.columns(2)
            e_col = c_e1.selectbox("Emp ID Column", cols_m)
            em_col = c_e1.selectbox("Email Column", cols_m)
            n_col = c_e2.selectbox("Name Column", cols_m)
            rel_col = c_e2.selectbox("Relation Column (Optional)", ["None (All Primary)"] + cols_m)
            
            if st.button("🚀 Sync Directory to Campaign", type="primary", use_container_width=True):
                synced_c = 0
                has_rel = rel_col != "None (All Primary)"
                for _, r in df_m.iterrows():
                    em_val = str(r[em_col]).strip().lower()
                    if em_val in ["nan", "none", "null", "undefined"]: em_val = ""
                    rel_val = str(r[rel_col]).strip().upper() if has_rel else "SELF"
                    if rel_val in ["SELF", "PRIMARY", "EMPLOYEE", "PROPOSER"]:
                        save_employee_to_directory(str(r[e_col]).strip().upper(), str(r[n_col]).strip(), em_val, active_scope_policy or "DEFAULT", active_display_company)
                        synced_c += 1
                st.success(f"Synced {synced_c} primary employees!"); time.sleep(1); st.rerun()

        # HIGH-SPEED BATCH QUERY: Eliminated N+1 MongoDB Network Calls
        q_filter = {"email_sent": {"$ne": True}}
        if active_scope_policy: q_filter["policy_no"] = active_scope_policy
            
        pending_cards = list(db.ecards.find(q_filter))
        ready_jobs = []
        missing_jobs = []
        
        if pending_cards:
            all_eids = list(set(e["emp_id"] for e in pending_cards))
            dir_lookup = {d["emp_id"]: d for d in db.directory.find({"emp_id": {"$in": all_eids}})}
            
            for e in pending_cards:
                drec = dir_lookup.get(e["emp_id"])
                em = drec.get("email", "") if drec else ""
                ename = drec.get("name", "Employee") if drec else "Employee"
                if em in ["nan", "none", "null", "undefined"]: em = ""
                
                item = {"EMP ID": e["emp_id"], "Name": ename, "Email": em if em else "⚠️ Missing Email (nan)", "Policy": e["policy_no"]}
                if em and "@" in em: ready_jobs.append(item)
                else: missing_jobs.append(item)

        m_c1, m_c2 = st.columns(2)
        m_c1.metric("🟢 Ready to Dispatch", len(ready_jobs))
        m_c2.metric("⚠️ Missing Email (Skipped)", len(missing_jobs))

        if ready_jobs or missing_jobs:
            st.dataframe(pd.DataFrame(ready_jobs + missing_jobs), hide_index=True, use_container_width=True)

        st.markdown("---")
        b_lim = st.number_input("Batch Limit", min_value=1, max_value=500, value=min(20, max(1, len(ready_jobs))), key="t6_batch_lim")
        
        c_proc, c_clear = st.columns([1.5, 1])
        with c_proc:
            if st.button(f"▶️ Process {min(len(ready_jobs), b_lim)} Ready Jobs", type="primary", use_container_width=True, disabled=(len(ready_jobs)==0)):
                sent = 0
                for j in ready_jobs[:b_lim]:
                    cards = get_cards_from_db(j["EMP ID"], policy_no=j["Policy"])
                    if cards:
                        body = html_in.replace("{{name}}", j["Name"]).replace("{{emp_id}}", j["EMP ID"]).replace("{{company_name}}", active_display_company).replace("{{policy_no}}", j["Policy"]).replace("{{deadline}}", dline)
                        ok, err = send_multi_ecard_email(j["Email"], subj_in, body, cards)
                        if ok:
                            db.ecards.update_many({"emp_id": j["EMP ID"], "policy_no": j["Policy"]}, {"$set": {"email_sent": True}})
                            log_email_dispatch(j["EMP ID"], j["Name"], j["Email"], j["Policy"], "DELIVERED", campaign_type="WELCOME_KIT")
                            sent += 1
                        else:
                            log_email_dispatch(j["EMP ID"], j["Name"], j["Email"], j["Policy"], "FAILED", err, campaign_type="WELCOME_KIT")
                st.success(f"Dispatched {sent} emails!"); time.sleep(1.5); st.rerun()

        with c_clear:
            if st.button("🗑️ Clear Pending Queue", use_container_width=True, help="Dismisses remaining incomplete/resigned employees from queue."):
                db.ecards.update_many(q_filter, {"$set": {"email_sent": True}})
                st.warning("Pending queue cleared! Counter reset to 0.")
                time.sleep(1); st.rerun()

        # 7-Day History and 1-Click Retry
        st.markdown("---")
        st.markdown("##### 📊 7-Day Dispatch Audit & Retry Hub")
        logs = list(db.email_logs.find({"campaign_type": "WELCOME_KIT"}).sort("timestamp", -1).limit(50))
        failed_l = [l for l in logs if l["status"] == "FAILED"]
        
        if failed_l:
            if st.button(f"🔄 Re-queue All {len(failed_l)} Failed Emails (1-Click)", type="primary", use_container_width=True, key="t6_retry_fail"):
                f_ids = list(set([doc["emp_id"] for doc in failed_l]))
                db.ecards.update_many({"emp_id": {"$in": f_ids}}, {"$set": {"email_sent": False}})
                db.email_logs.delete_many({"status": "FAILED", "campaign_type": "WELCOME_KIT"})
                st.success(f"Re-queued {len(f_ids)} failed users!"); time.sleep(1); st.rerun()

        if logs:
            st.dataframe(pd.DataFrame([{
                "Time (IST)": (l["timestamp"] + timedelta(hours=5, minutes=30)).strftime("%d-%b %I:%M %p"),
                "Emp ID": l["emp_id"], "Email": l["recipient_email"], "Status": "✅ " + l["status"] if l["status"]=="DELIVERED" else "❌ " + l["status"],
                "Error": l.get("error_reason") or "Delivered"
            } for l in logs]), hide_index=True, use_container_width=True)

# ==============================================================================
# --- TAB 7: 🚀 PORTAL LAUNCH & BROADCAST (DUAL HR & USER SUB-MODULES) ---
# ==============================================================================
with tab_launch:
    st.markdown("### 🚀 Portal Launch & Feedback Broadcast Center")
    
    live_tenants_map = get_live_tenants_and_policies()
    discovered_comp_list = sorted(list(live_tenants_map.keys()))
    
    col_lt1, col_lt2 = st.columns(2)
    with col_lt1:
        camp_launch_opts = ["🌐 All Corporate Clients"] + discovered_comp_list + ["➕ Custom Company Name"]
        sel_l_comp = st.selectbox("Select Target Client Company:", camp_launch_opts, key="t7_launch_comp")
        if sel_l_comp == "➕ Custom Company Name":
            t7_custom_comp = st.text_input("Company Name Override:", placeholder="e.g. STRATEGIC SYSTEMS", key="t7_comp_man")
        else:
            t7_custom_comp = st.text_input("Company Name Override (Optional):", value=sel_l_comp if sel_l_comp != "🌐 All Corporate Clients" else "", key="t7_comp_auto")

    with col_lt2:
        avail_l_pols = live_tenants_map.get(sel_l_comp, []) if sel_l_comp in live_tenants_map else []
        all_dir_pols = db.directory.distinct("policy_no")
        combined_l_pols = sorted(list(set(avail_l_pols + all_dir_pols)))
        l_pol_opts = ["🌐 All Policies"] + combined_l_pols + ["➕ Custom Policy Number"]
        sel_l_pol = st.selectbox("Select Target Policy Scope:", l_pol_opts, key="t7_launch_pol")
        if sel_l_pol == "➕ Custom Policy Number":
            t7_custom_pol = st.text_input("Policy Number Override:", placeholder="e.g. OG-27-1801-8403-00000112", key="t7_pol_man")
        else:
            t7_custom_pol = st.text_input("Policy Number Override (Optional):", value=sel_l_pol if sel_l_pol != "🌐 All Policies" else "", key="t7_pol_auto")

    active_launch_comp = t7_custom_comp.strip().upper() if t7_custom_comp else (sel_l_comp if sel_l_comp != "🌐 All Corporate Clients" else "Your Company")
    active_launch_pol = t7_custom_pol.strip().upper() if t7_custom_pol else (sel_l_pol if sel_l_pol != "🌐 All Policies" else None)

    st.divider()
    subtab_user_launch, subtab_hr_launch = st.tabs(["👤 Employee (User) Launch Broadcast", "🏢 Tenant HR Admin Broadcast"])

    # SUB-TAB 1: USER LAUNCH BROADCAST
    with subtab_user_launch:
        col_u_left, col_u_right = st.columns([1.2, 1])
        with col_u_left:
            st.subheader("⚙️ User Launch Assets & Links")
            c_ug1, c_ug2 = st.columns(2)
            with c_ug1:
                if get_asset("user_portal_guide"):
                    st.success("User Guide Active")
                    if st.button("Delete User Guide", key="d_upg"): delete_asset("user_portal_guide"); st.rerun()
                else:
                    up_ug = st.file_uploader("Upload User Guide PDF", type=["pdf"], key="up_upg")
                    if up_ug: save_asset("user_portal_guide", up_ug.getvalue()); st.rerun()
            with c_ug2:
                if get_asset("user_launch_banner"):
                    st.success("User Banner Active")
                    if st.button("Delete User Banner", key="d_upb"): delete_asset("user_launch_banner"); st.rerun()
                else:
                    up_ub = st.file_uploader("Upload User Banner Image", type=["png", "jpg"], key="up_upb")
                    if up_ub: save_asset("user_launch_banner", up_ub.getvalue()); st.rerun()
                    
            p_url_u = st.text_input("User Portal URL:", value="https://portal.capitupindia.com", key="t7_u_purl")
            fb_url_base_u = "https://docs.google.com/forms/d/e/1FAIpQLSdpJ-_GbT1AGeD1tIVEMbvF0DtNexO7fz_0nJE1mdKBu6rrag/viewform?usp=pp_url"
            fb_url_u = st.text_input("User Feedback Form Base URL:", value=fb_url_base_u, key="t7_u_fburl")
            l_subj_u = st.text_input("User Subject Line:", value="🚀 Welcome to the All-New CapitUp Benefits Portal (Beta)!", key="t7_u_subj")
            
            logo_tag_u = """<div style="text-align: center; margin-bottom: 15px;"><img src="cid:logo_image" alt="CapitUp India Logo" style="height: 60px; width: auto; display: inline-block;" /></div>""" if get_asset("logo") else ""
            
            user_launch_html = f"""<div style="font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; max-width: 650px; margin: 0 auto; border: 1px solid #C29B38; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); background-color: #ffffff;">
  <div style="background-color: #0B1E30; padding: 30px 24px; text-align: center; border-bottom: 3px solid #C29B38;">
    {logo_tag_u}
    <h1 style="color: #ffffff; margin: 0; font-size: 22px; letter-spacing: 1px; font-weight: 800; text-transform: uppercase;">WELCOME TO NEXT-GEN HEALTHCARE ACCESS</h1>
    <p style="color: #C29B38; margin: 6px 0 0 0; font-size: 11px; font-weight: bold; letter-spacing: 2px;">CAPITUP EMPLOYEE BENEFITS PORTAL (BETA)</p>
  </div>
  <!-- BANNER -->
  <div style="padding: 32px 24px;">
    <p style="font-size: 15px; margin-top: 0;">Dear <strong>{{{{name}}}}</strong>,</p>
    <p style="font-size: 14px; color: #444;">We are thrilled to unveil the <strong>CapitUp Employee Health & Benefits Portal (Beta)</strong>—crafted specifically for you and your family at <strong>{{{{company_name}}}}</strong>!</p>
    <div style="background-color: #F4F6F8; border-left: 4px solid #23C2A9; padding: 18px; margin: 24px 0; border-radius: 6px;">
      <h3 style="margin-top: 0; color: #0B1E30; font-size: 14px; text-transform: uppercase;">✨ Your Digital Health & Benefits Hub</h3>
      <ul style="font-size: 13px; padding-left: 20px; margin: 8px 0; color: #444; line-height: 1.6;">
        <li><strong>⚡ Instant E-Card Wallet:</strong> View & download unified family health cards in seconds.</li>
        <li><strong>🤖 24/7 AI Policy Copilot:</strong> Ask instant questions about room rent limits, maternity rules, and claim steps.</li>
        <li><strong>👨‍👩‍👧‍👦 Family Coverage View:</strong> Review covered dependents and Sum Insured limits with total transparency.</li>
      </ul>
    </div>
    <div style="text-align: center; margin: 30px 0;">
      <a href="{{{{portal_url}}}}" style="background-color: #0B1E30; color: #ffffff; padding: 15px 36px; text-decoration: none; font-size: 14px; font-weight: bold; border-radius: 6px; display: inline-block; border: 2px solid #C29B38;">🌐 Explore the CapitUp Portal</a>
    </div>
    <div style="border: 1px solid #C29B38; background-color: #FCF9F2; border-radius: 8px; padding: 20px; margin: 28px 0;">
      <h3 style="margin-top: 0; color: #0B1E30; font-size: 14px;">🌟 Let's Build This Together (Your Beta Feedback Matters!)</h3>
      <p style="font-size: 13px; margin: 8px 0; color: #444;">Did you find your cards easily? Tell us what features you would love to see next!</p>
      <div style="text-align: center; margin-top: 14px;">
        <a href="{{{{feedback_url}}}}" style="background-color: #23C2A9; color: #ffffff; padding: 12px 26px; text-decoration: none; font-size: 13px; font-weight: bold; border-radius: 6px; display: inline-block;">📝 Share Your Thoughts & Feedback</a>
      </div>
    </div>
  </div>
  <div style="background-color: #F4F6F8; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
    <p style="margin: 0; font-size: 12px; color: #0B1E30; font-weight: bold;">CapitUp India Pvt. Ltd. | HITEC City, Hyderabad</p>
  </div>
</div>"""
            u_html = st.text_area("USER EMAIL HTML", value=user_launch_html, height=180, key="t7_u_html")
            
            if st.checkbox("👁️ Preview User Email", key="prev_t7_u"):
                dummy_fb = f"{fb_url_u}&entry.1752786264=MOCK101&entry.1314062294=Employee&entry.2060790789={urllib.parse.quote(active_launch_comp)}"
                rendered_u = u_html.replace("{{name}}", st.session_state.username).replace("{{emp_id}}", "MOCK-101").replace("{{company_name}}", active_launch_comp).replace("{{portal_url}}", p_url_u).replace("{{feedback_url}}", dummy_fb)
                st.components.v1.html(rendered_u, height=450, scrolling=True)

        with col_u_right:
            st.subheader("📢 User Audience & Dispatch")
            up_u_active = st.file_uploader("Upload Active Employee List (CSV/Excel)", type=["csv", "xlsx"], key="t7_u_active_up")
            if up_u_active:
                df_ua = clean_and_align_dataframe(pd.read_csv(up_u_active) if up_u_active.name.endswith('.csv') else pd.read_excel(up_u_active))
                cols_ua = list(df_ua.columns)
                c_ua1, c_ua2 = st.columns(2)
                e_col_u = c_ua1.selectbox("Emp ID Column", cols_ua, key="t7_u_ecol")
                em_col_u = c_ua1.selectbox("Email Column", cols_ua, key="t7_u_emcol")
                n_col_u = c_ua2.selectbox("Name Column", cols_ua, key="t7_u_ncol")
                if st.button("🚀 Sync Employee Roster", type="primary", use_container_width=True, key="btn_sync_t7_u"):
                    synced_u = 0
                    for _, r in df_ua.iterrows():
                        em_val = str(r[em_col_u]).strip().lower()
                        if em_val in ["nan", "none", "null", "undefined"]: em_val = ""
                        save_employee_to_directory(str(r[e_col_u]).strip().upper(), str(r[n_col_u]).strip(), em_val, active_launch_pol or "DEFAULT", active_launch_comp, role="EMPLOYEE")
                        synced_u += 1
                    st.success(f"Synced {synced_u} active employees!"); time.sleep(1); st.rerun()

            q_u = {"role": "EMPLOYEE"}
            if active_launch_pol: q_u["policy_no"] = active_launch_pol
            users_u = list(db.directory.find(q_u))
            ready_u = [u for u in users_u if not u.get("launch_announced") and u.get("email") and "@" in u.get("email")]
            st.metric("🟢 Ready to Invite (Employees)", len(ready_u))

            b_lim_u = st.number_input("Batch Limit", min_value=1, max_value=500, value=min(20, max(1, len(ready_u))), key="t7_u_blim")
            
            c_up1, c_up2 = st.columns([1.5, 1])
            with c_up1:
                if st.button(f"🚀 Send {min(len(ready_u), b_lim_u)} User Invites", type="primary", use_container_width=True, disabled=(len(ready_u)==0), key="btn_send_u_launch"):
                    sent_u = 0
                    for u in ready_u[:b_lim_u]:
                        encoded_n = urllib.parse.quote(str(u.get("name", "Employee")).strip())
                        encoded_eid = urllib.parse.quote(str(u["emp_id"]).strip())
                        encoded_c = urllib.parse.quote(str(active_launch_comp).strip())
                        dynamic_fb_link = f"{fb_url_u}&entry.1752786264={encoded_eid}&entry.1314062294={encoded_n}&entry.2060790789={encoded_c}"
                        
                        body = u_html.replace("{{name}}", u.get("name", "Employee"))\
                                     .replace("{{emp_id}}", u["emp_id"])\
                                     .replace("{{company_name}}", active_launch_comp)\
                                     .replace("{{portal_url}}", p_url_u)\
                                     .replace("{{feedback_url}}", dynamic_fb_link)
                                     
                        ok, err = send_launch_email(u["email"], l_subj_u, body, guide_asset_key="user_portal_guide", banner_asset_key="user_launch_banner")
                        if ok:
                            db.directory.update_one({"_id": u["_id"]}, {"$set": {"launch_announced": True}})
                            log_email_dispatch(u["emp_id"], u.get("name", "Employee"), u["email"], u.get("policy_no", "UNKNOWN"), "DELIVERED", campaign_type="USER_PORTAL_LAUNCH")
                            sent_u += 1
                        else:
                            log_email_dispatch(u["emp_id"], u.get("name", "Employee"), u["email"], u.get("policy_no", "UNKNOWN"), "FAILED", err, campaign_type="USER_PORTAL_LAUNCH")
                    st.success(f"Broadcasted to {sent_u} employees!"); time.sleep(1.5); st.rerun()

            with c_up2:
                if st.button("🗑️ Clear Queue", use_container_width=True, key="btn_clear_u_q"):
                    db.directory.update_many(q_u, {"$set": {"launch_announced": True}})
                    st.warning("User queue cleared!"); time.sleep(1); st.rerun()

    # SUB-TAB 2: TENANT HR ADMIN BROADCAST
    with subtab_hr_launch:
        col_hr_left, col_hr_right = st.columns([1.2, 1])
        with col_hr_left:
            st.subheader("⚙️ HR Admin Launch Assets & Endpoints")
            c_hrg1, c_hrg2 = st.columns(2)
            with c_hrg1:
                if get_asset("hr_portal_guide"):
                    st.success("HR Guide Active")
                    if st.button("Delete HR Guide", key="d_hrpg"): delete_asset("hr_portal_guide"); st.rerun()
                else:
                    up_hrg = st.file_uploader("Upload HR Admin Guide PDF", type=["pdf"], key="up_hrpg")
                    if up_hrg: save_asset("hr_portal_guide", up_hrg.getvalue()); st.rerun()
            with c_hrg2:
                if get_asset("hr_launch_banner"):
                    st.success("HR Banner Active")
                    if st.button("Delete HR Banner", key="d_hrpb"): delete_asset("hr_launch_banner"); st.rerun()
                else:
                    up_hrb = st.file_uploader("Upload HR Banner Image", type=["png", "jpg"], key="up_hrpb")
                    if up_hrb: save_asset("hr_launch_banner", up_hrb.getvalue()); st.rerun()
                    
            p_url_hr = st.text_input("HR Admin Portal URL:", value="https://admin.capitupindia.com", key="t7_hr_purl")
            fb_url_base_hr = "https://docs.google.com/forms/d/e/1FAIpQLSdpJ-_GbT1AGeD1tIVEMbvF0DtNexO7fz_0nJE1mdKBu6rrag/viewform?usp=pp_url"
            fb_url_hr = st.text_input("HR Feedback Form Base URL:", value=fb_url_base_hr, key="t7_hr_fburl")
            l_subj_hr = st.text_input("HR Subject Line:", value="🏢 Introducing Your CapitUp Corporate Benefits Management Workspace", key="t7_hr_subj")
            
            logo_tag_hr = """<div style="text-align: center; margin-bottom: 15px;"><img src="cid:logo_image" alt="CapitUp India Logo" style="height: 60px; width: auto; display: inline-block;" /></div>""" if get_asset("logo") else ""
            
            hr_launch_html = f"""<div style="font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; max-width: 650px; margin: 0 auto; border: 1px solid #C29B38; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); background-color: #ffffff;">
  <div style="background-color: #0B1E30; padding: 30px 24px; text-align: center; border-bottom: 3px solid #C29B38;">
    {logo_tag_hr}
    <h1 style="color: #ffffff; margin: 0; font-size: 22px; letter-spacing: 1px; font-weight: 800; text-transform: uppercase;">WELCOME TO YOUR HR BENEFITS WORKSPACE</h1>
    <p style="color: #C29B38; margin: 6px 0 0 0; font-size: 11px; font-weight: bold; letter-spacing: 2px;">POWERED BY CAPITUP CORPORATE INSURTECH</p>
  </div>
  <!-- BANNER -->
  <div style="padding: 32px 24px;">
    <p style="font-size: 15px; margin-top: 0;">Dear <strong>{{{{name}}}}</strong> (HR / People Ops Team),</p>
    <p style="font-size: 14px; color: #444;">We are pleased to introduce the all-new <strong>CapitUp Tenant HR Administration Suite (Beta)</strong> for <strong>{{{{company_name}}}}</strong>.</p>
    <div style="background-color: #F4F6F8; border-left: 4px solid #0B1E30; padding: 18px; margin: 24px 0; border-radius: 6px;">
      <h3 style="margin-top: 0; color: #0B1E30; font-size: 14px; text-transform: uppercase;">⚡ Enterprise Capabilities at Your Fingertips</h3>
      <ul style="font-size: 13px; padding-left: 20px; margin: 8px 0; color: #444; line-height: 1.6;">
        <li><strong>📊 Executive Overview:</strong> Real-time tenant analytics, coverage summaries, and live health metrics.</li>
        <li><strong>👥 Employee Directory & DPDP:</strong> Total control over active employee rosters and data privacy compliance.</li>
        <li><strong>🔄 Endorsement Engine:</strong> Automated addition/deletion sync with instant pro-rata premium raters.</li>
        <li><strong>📈 Claims Analytics & MIS Dumps:</strong> Download raw claim MIS sheets and CMO-level AI risk insights.</li>
        <li><strong>❓ Live Request Queue:</strong> Track employee support tickets and endorsement status in real-time.</li>
      </ul>
    </div>
    <div style="text-align: center; margin: 30px 0;">
      <a href="{{{{portal_url}}}}" style="background-color: #0B1E30; color: #ffffff; padding: 15px 36px; text-decoration: none; font-size: 14px; font-weight: bold; border-radius: 6px; display: inline-block; border: 2px solid #C29B38;">🏢 Access Your HR Hub</a>
    </div>
    <div style="border: 1px solid #C29B38; background-color: #FCF9F2; border-radius: 8px; padding: 20px; margin: 28px 0;">
      <h3 style="margin-top: 0; color: #0B1E30; font-size: 14px;">🌟 Shape Your Corporate Benefits Experience (HR Beta Feedback)</h3>
      <p style="font-size: 13px; margin: 8px 0; color: #444;">As our trusted HR partner, your perspective is crucial. Tell us how we can make your benefits management even simpler.</p>
      <div style="text-align: center; margin-top: 14px;">
        <a href="{{{{feedback_url}}}}" style="background-color: #23C2A9; color: #ffffff; padding: 12px 26px; text-decoration: none; font-size: 13px; font-weight: bold; border-radius: 6px; display: inline-block;">📝 Share HR Executive Feedback</a>
      </div>
    </div>
  </div>
  <div style="background-color: #F4F6F8; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
    <p style="margin: 0; font-size: 12px; color: #0B1E30; font-weight: bold;">CapitUp Corporate Enterprise Support | Dedicated Partner Desk</p>
  </div>
</div>"""
            hr_html = st.text_area("HR EMAIL HTML", value=hr_launch_html, height=180, key="t7_hr_html")
            
            if st.checkbox("👁️ Preview HR Email", key="prev_t7_hr"):
                dummy_fb_hr = f"{fb_url_hr}&entry.1752786264=HR101&entry.1314062294=HR%20Partner&entry.2060790789={urllib.parse.quote(active_launch_comp)}"
                rendered_hr = hr_html.replace("{{name}}", st.session_state.username).replace("{{company_name}}", active_launch_comp).replace("{{portal_url}}", p_url_hr).replace("{{feedback_url}}", dummy_fb_hr)
                st.components.v1.html(rendered_hr, height=450, scrolling=True)

        with col_hr_right:
            st.subheader("📢 HR Admin Audience & Dispatch")
            up_hr_active = st.file_uploader("Upload HR Team Contacts (CSV/Excel)", type=["csv", "xlsx"], key="t7_hr_active_up")
            if up_hr_active:
                df_hra = clean_and_align_dataframe(pd.read_csv(up_hr_active) if up_hr_active.name.endswith('.csv') else pd.read_excel(up_hr_active))
                cols_hra = list(df_hra.columns)
                c_hra1, c_hra2 = st.columns(2)
                e_col_hr = c_hra1.selectbox("HR ID / Code", cols_hra, key="t7_hr_ecol")
                em_col_hr = c_hra1.selectbox("HR Email Column", cols_hra, key="t7_hr_emcol")
                n_col_hr = c_hra2.selectbox("HR Name Column", cols_hra, key="t7_hr_ncol")
                if st.button("🚀 Sync HR Admin Contacts", type="primary", use_container_width=True, key="btn_sync_t7_hr"):
                    synced_hr = 0
                    for _, r in df_hra.iterrows():
                        em_val = str(r[em_col_hr]).strip().lower()
                        if em_val in ["nan", "none", "null", "undefined"]: em_val = ""
                        save_employee_to_directory(str(r[e_col_hr]).strip().upper(), str(r[n_col_hr]).strip(), em_val, active_launch_pol or "DEFAULT", active_launch_comp, role="HR_ADMIN")
                        synced_hr += 1
                    st.success(f"Synced {synced_hr} HR contacts!"); time.sleep(1); st.rerun()

            q_hr = {"role": "HR_ADMIN"}
            if active_launch_pol: q_hr["policy_no"] = active_launch_pol
            users_hr = list(db.directory.find(q_hr))
            ready_hr = [u for u in users_hr if not u.get("launch_announced") and u.get("email") and "@" in u.get("email")]
            st.metric("🟢 Ready to Invite (HR Admins)", len(ready_hr))

            b_lim_hr = st.number_input("Batch Limit", min_value=1, max_value=100, value=min(20, max(1, len(ready_hr))), key="t7_hr_blim")
            
            c_hrp1, c_hrp2 = st.columns([1.5, 1])
            with c_hrp1:
                if st.button(f"🚀 Send {min(len(ready_hr), b_lim_hr)} HR Invites", type="primary", use_container_width=True, disabled=(len(ready_hr)==0), key="btn_send_hr_launch"):
                    sent_hr = 0
                    for u in ready_hr[:b_lim_hr]:
                        encoded_n = urllib.parse.quote(str(u.get("name", "HR Partner")).strip())
                        encoded_eid = urllib.parse.quote(str(u.get("emp_id", "HR")).strip())
                        encoded_c = urllib.parse.quote(str(active_launch_comp).strip())
                        dynamic_fb_link_hr = f"{fb_url_hr}&entry.1752786264={encoded_eid}&entry.1314062294={encoded_n}&entry.2060790789={encoded_c}"
                        
                        body = hr_html.replace("{{name}}", u.get("name", "HR Partner"))\
                                       .replace("{{emp_id}}", u.get("emp_id", "HR"))\
                                       .replace("{{company_name}}", active_launch_comp)\
                                       .replace("{{portal_url}}", p_url_hr)\
                                       .replace("{{feedback_url}}", dynamic_fb_link_hr)
                                       
                        ok, err = send_launch_email(u["email"], l_subj_hr, body, guide_asset_key="hr_portal_guide", banner_asset_key="hr_launch_banner")
                        if ok:
                            db.directory.update_one({"_id": u["_id"]}, {"$set": {"launch_announced": True}})
                            log_email_dispatch(u["emp_id"], u.get("name", "HR Partner"), u["email"], u.get("policy_no", "UNKNOWN"), "DELIVERED", campaign_type="HR_PORTAL_LAUNCH")
                            sent_hr += 1
                        else:
                            log_email_dispatch(u["emp_id"], u.get("name", "HR Partner"), u["email"], u.get("policy_no", "UNKNOWN"), "FAILED", err, campaign_type="HR_PORTAL_LAUNCH")
                    st.success(f"Broadcasted to {sent_hr} HR leaders!"); time.sleep(1.5); st.rerun()

            with c_hrp2:
                if st.button("🗑️ Clear HR Queue", use_container_width=True, key="btn_clear_hr_q"):
                    db.directory.update_many(q_hr, {"$set": {"launch_announced": True}})
                    st.warning("HR queue cleared!"); time.sleep(1); st.rerun()

# --- TAB 8: FAMILYFICATION ---
with tab_family:
    st.markdown("### 👨‍👩‍👧‍👦 Lightning-Fast E-Card Familyfication")
    mf = st.file_uploader("Upload Active Tracker", type=["xlsx", "xls", "csv"], key="fam_mf")
    if mf:
        df_f = clean_and_align_dataframe(pd.read_csv(mf) if mf.name.endswith('.csv') else pd.read_excel(mf))
        c_u1, c_u2 = st.columns(2)
        ucol = c_u1.selectbox("UHID / Card No Column", df_f.columns)
        ecol = c_u2.selectbox("Employee ID Column", df_f.columns)
        f_pdfs = st.file_uploader("Upload Cards", type=["pdf"], accept_multiple_files=True, key="fam_pdfs")
        if f_pdfs and st.button("🧬 Consolidate Families", type="primary", use_container_width=True):
            u_to_e = {str(r[ucol]).strip().upper().replace('.0', ''): str(r[ecol]).strip().upper() for _, r in df_f.dropna(subset=[ucol, ecol]).iterrows()}
            fgroups = {}
            for pdf in f_pdfs:
                cname = re.sub(r'(\.PDF|_ECARD|_CARD).*$', '', pdf.name.upper()).strip()
                meid = u_to_e.get(cname) or next((u_to_e[k] for k in u_to_e if k in cname), None)
                if meid:
                    if meid not in fgroups: fgroups[meid] = []
                    fgroups[meid].append(pdf.getvalue())
            zbuf = BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for eid, bytes_list in fgroups.items():
                    mpdf = fitz.open()
                    for b in bytes_list:
                        td = fitz.open(stream=b, filetype="pdf"); mpdf.insert_pdf(td); td.close()
                    zf.writestr(f"{eid}_Family_ECard.pdf", mpdf.tobytes(garbage=4, deflate=True))
                    mpdf.close()
            st.download_button("📥 Download Family Packets (.zip)", data=zbuf.getvalue(), file_name="Family_ECards.zip", mime="application/zip", type="primary", use_container_width=True)

# --- TAB 9: GAP FINDER ---
with tab_gap:
    st.markdown("### 🔍 Coverage Gap Finder")
    c_g1, c_g2 = st.columns(2)
    g_xl = c_g1.file_uploader("Upload Active Tracker", type=["xlsx", "xls", "csv"], key="gap_xl")
    g_zp = c_g2.file_uploader("Upload E-Cards ZIP", type=["zip"], key="gap_zp")
    if g_xl and g_zp:
        df_g = clean_and_align_dataframe(pd.read_csv(g_xl) if g_xl.name.endswith('.csv') else pd.read_excel(g_xl))
        g_ecol = st.selectbox("Primary Emp ID Column", df_g.columns, key="gap_ecol")
        if st.button("🔍 Run Analyzer", type="primary", use_container_width=True):
            with zipfile.ZipFile(g_zp) as z:
                znames = [re.sub(r'(\.PDF|_ECARD|_CARD|_FAMILY).*$', '', os.path.basename(f.filename).upper()).strip() for f in z.infolist() if not f.is_dir()]
            matched = df_g[df_g[g_ecol].astype(str).str.strip().str.upper().isin(znames)]
            missing = df_g[~df_g[g_ecol].astype(str).str.strip().str.upper().isin(znames)]
            m1, m2, m3 = st.columns(3)
            m1.metric("Total in Tracker", len(df_g))
            m2.metric("Matched E-Cards", len(matched))
            m3.metric("Missing Gaps", len(missing))
            if not missing.empty:
                st.warning(f"Missing {len(missing)} members.")
                out = BytesIO()
                with pd.ExcelWriter(out, engine='openpyxl') as w: missing.to_excel(w, index=False)
                st.download_button("📥 Download Missing Report (.xlsx)", data=out.getvalue(), file_name="Missing_ECards.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)
            else: st.success("All members matched!")
