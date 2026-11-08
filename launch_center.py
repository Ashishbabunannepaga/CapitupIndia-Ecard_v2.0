"""
Portal Launch & Feedback Center (Tab 7) and the in-portal Beta Feedback widget.

Kept in its own module so app.py stays the Streamlit entry point and only
needs two calls:
    render_launch_center(db, helpers)        -> inside `with tab_launch:`
    render_feedback_widget(db, username, ...) -> in the sidebar / employee view
"""
import base64
import html
import logging
import os
import re
import smtplib
import time
import urllib.parse
from datetime import datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, make_msgid

import pandas as pd
import pymongo
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(APP_DIR, "launch_assets")

# Bundled defaults, used when nothing has been uploaded to the asset vault.
BUNDLED = {
    "logo": "capitup_logo.png",
    "diwali_card": "diwali_greeting_card.jpg",
    "hr_portal_guide": "dhAIrya_HR_Setup_Guide.pdf",
    "video_thumb": "launch_video_thumbnail.jpg",
}

CAMPAIGN_HR = "HR_PORTAL_LAUNCH"
CAMPAIGN_USER = "USER_PORTAL_LAUNCH"
TEMPLATE_VERSION = "2026-11-dhairya-v2"

DEFAULT_CONFIG = {
    "portal_name": "dhAIrya",
    "portal_tagline": "CapitUp India's group health benefits portal",
    "hr_portal_url": "https://portal.capitupindia.com/portal-login",
    "user_portal_url": "https://portal.capitupindia.com/",
    "feedback_form_url": "https://docs.google.com/forms/d/e/1FAIpQLSdpJ-_GbT1AGeD1tIVEMbvF0DtNexO7fz_0nJE1mdKBu6rrag/viewform?usp=pp_url",
    # Fallback when a company has no line in company_data_dates.
    "data_cutoff": "see Recent batches in the portal",
    # One "Company name = date" per line: date of the latest endorsement file we hold.
    "company_data_dates": "\n".join([
        "ZoomRx = 06 Nov 2026", "Alphamed = 04 Nov 2026", "RandomTrees = 04 Nov 2026",
        "Graviti = 30 Oct 2026", "Technogen = 29 Oct 2026", "Polmor = 28 Oct 2026",
        "ASA Bhanu = 23 Oct 2026", "Kloudgin = 21 Oct 2026", "Intellativ = 14 Oct 2026",
        "Stratsol = 05 Oct 2026", "Purview = 28 Sep 2026", "Ecoled = 17 Sep 2026",
        "Siemon = 15 Sep 2026", "Biochemical = 14 Aug 2026", "Intonation = 13 Aug 2026",
    ]),
    "support_email": "info@capitupindia.com",
    "video_url": "",
    "festive_greeting": True,
    "hr_subject": "\U0001fa94 Happy Diwali from CapitUp: dhAIrya is live for your HR team (beta)",
    "user_subject": "\U0001fa94 Happy Diwali! Your health cover is now on dhAIrya",
    "sender_name": "Team CapitUp India",
}

# Google Form "entry." IDs used to pre-fill the feedback form.
FB_ENTRY_ID = "entry.1752786264"
FB_ENTRY_NAME = "entry.1314062294"
FB_ENTRY_COMPANY = "entry.2060790789"

FEEDBACK_CATEGORIES = ["General", "E-Card download", "Login / access", "Data looks wrong",
                       "Feature request", "Bug / error", "Speed"]
FEEDBACK_STATUSES = ["NEW", "REVIEWED", "PLANNED", "DONE"]

BLANKS = {"", "nan", "none", "null", "undefined", "nat"}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _clean(val):
    s = "" if val is None else str(val).strip()
    if s.lower() in BLANKS:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _valid_email(e):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e or ""))


def _bundled_bytes(key):
    path = os.path.join(ASSET_DIR, BUNDLED.get(key, ""))
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _data_uri(b, mime):
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"


def _html_to_text(body_html):
    t = re.sub(r"(?is)<(script|style).*?</\1>", "", body_html)
    t = re.sub(r"(?i)<br\s*/?>|</p>|</tr>|</h\d>|</li>", "\n", t)
    t = re.sub(r"(?i)<li[^>]*>", "  - ", t)
    t = re.sub(r'(?is)<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>', r"\2 (\1)", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def load_config(db):
    cfg = dict(DEFAULT_CONFIG)
    try:
        doc = db.settings.find_one({"key": "launch_config"})
        if doc and isinstance(doc.get("value"), dict):
            cfg.update({k: v for k, v in doc["value"].items() if k in DEFAULT_CONFIG})
    except Exception as e:
        logging.error(f"launch_config load failed: {e}")
    return cfg


def save_config(db, cfg):
    db.settings.update_one({"key": "launch_config"},
                           {"$set": {"value": cfg, "updated_at": datetime.utcnow()}}, upsert=True)


def ensure_indexes(db):
    """Cheap and idempotent; cached per process so it doesn't run every rerun."""
    if st.session_state.get("_launch_idx_ok"):
        return
    try:
        db.hr_contacts.create_index("email", unique=True)
        db.hr_contacts.create_index("company_name")
        db.portal_feedback.create_index([("created_at", pymongo.DESCENDING)])
        st.session_state["_launch_idx_ok"] = True
    except Exception as e:
        logging.error(f"launch index creation failed: {e}")


# ----------------------------------------------------------------------------
# Email templates
# ----------------------------------------------------------------------------
_HEAD = """<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#F3EFE6;padding:24px 0;font-family:'Segoe UI',Arial,Helvetica,sans-serif;">
<tr><td align="center">
<table role="presentation" width="640" cellpadding="0" cellspacing="0" style="max-width:640px;width:100%;background:#ffffff;border-radius:14px;overflow:hidden;border:1px solid #E8D9B0;">
  <tr><td style="background:#ffffff;padding:20px 28px 14px 28px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
      <td valign="middle"><img src="cid:logo_image" alt="CapitUp India" height="56" style="height:56px;width:auto;display:block;border:0;" /></td>
      <td valign="middle" align="right" style="font-family:'Segoe UI',Arial,sans-serif;">
        <span style="font-size:26px;font-weight:800;color:#1c2430;letter-spacing:-0.5px;">dh<span style="color:#c9963f;">AI</span>rya</span><br/>
        <span style="font-size:10px;font-weight:700;letter-spacing:2px;color:#8d5f29;">BY CAPITUP INDIA</span>
      </td>
    </tr></table>
  </td></tr>
  <tr><td style="height:4px;background:#c9963f;background-image:linear-gradient(90deg,#8d5f29,#ecc976,#1aab93);font-size:0;line-height:0;">&nbsp;</td></tr>
"""

_FOOT = """  <tr><td style="background:#1c2430;padding:26px 28px;text-align:center;">
    <p style="margin:0;color:#ecc976;font-size:15px;font-weight:800;letter-spacing:2px;">CAPITUP INDIA &middot; WE BACK YOU UP</p>
    <p style="margin:8px 0 0 0;color:#C9D3DD;font-size:12px;line-height:1.6;">CapitUp India Pvt. Ltd. &nbsp;|&nbsp; HITEC City, Hyderabad<br/>
    Questions? Write to <a href="mailto:{{support_email}}" style="color:#1aab93;text-decoration:none;">{{support_email}}</a> or simply reply to this email.</p>
    <p style="margin:10px 0 0 0;color:#7C8B99;font-size:10px;">You are receiving this because your organisation's group health policy is serviced by CapitUp India.</p>
  </td></tr>
</table>
</td></tr></table>"""

_GREETING_BLOCK = """<tr><td style="padding:0;">
    <img src="cid:launch_banner" alt="Wishing you a Happy &amp; Safe Diwali from Team CapitUp" width="640" style="width:100%;max-width:640px;height:auto;display:block;border:0;" />
  </td></tr>"""

_VIDEO_BLOCK = """<tr><td style="padding:6px 32px 8px 32px;" align="center">
    <a href="{{video_url}}" style="text-decoration:none;"><img src="cid:video_thumb" alt="Watch the 60-second dhAIrya tour" width="560" style="width:100%;max-width:560px;height:auto;border-radius:10px;display:block;border:0;" /></a>
    <p style="margin:8px 0 0 0;font-size:12px;color:#6B7785;">&#9654; Can't see the video? <a href="{{video_url}}" style="color:#128a78;font-weight:700;">Watch the 60-second tour here</a></p>
  </td></tr>"""

_BTN = "background:#128a78;color:#ffffff;text-decoration:none;font-size:15px;font-weight:800;padding:15px 34px;border-radius:8px;display:inline-block;border:2px solid #c9963f;"


def _row(icon, title, text):
    return (f'<tr><td valign="top" style="padding:7px 10px 7px 0;font-size:18px;width:30px;">{icon}</td>'
            f'<td style="padding:7px 0;font-size:14px;line-height:1.6;color:#333;"><strong style="color:#1c2430;">{title}</strong> {text}</td></tr>')


def hr_template():
    features = "".join([
        _row("&#128202;", "Executive Overview.", "Your policy, renewal date, live CD balance and covered lives on one screen. Send yourself a CD balance statement in one click."),
        _row("&#128101;", "Employee Directory.", "Your active roster with portal status. Export it to Excel and fix a missing employee e-mail with the pencil icon."),
        _row("&#128260;", "Endorsement Engine.", "Drop your joiners/leavers sheet in any layout. dhAIrya reads it, works out pro-rata premium and GST, and sends the batch to CapitUp for approval."),
        _row("&#128200;", "Claims Analytics.", "Upload your TPA claims MIS for loss ratio, top conditions, deductions and renewal pointers."),
        _row("&#127915;", "Request Queue.", "Raise a ticket to CapitUp, attach files and follow our replies in one place."),
        _row("&#10024;", "Ask Copilot.", "Ask about your policy in plain words, such as &ldquo;What is our room rent limit?&rdquo;"),
    ])
    notes = "".join(f'<li style="margin:0 0 6px 0;">{t}</li>' for t in [
        "<strong>Forgot your password?</strong> Self-service reset isn't built yet. Write to us and we'll issue a new login.",
        "<strong>Endorsements need CapitUp approval.</strong> You get an e-mail when a batch is approved (with your new CD balance) or sent back (with the reason).",
        "<strong>Idle sign-out.</strong> For safety, dhAIrya signs you out after 10 minutes without activity, with a &ldquo;Still there?&rdquo; warning first.",
        "<strong>AI answers can be wrong.</strong> The copilot reads your policy schedule, but the policy document is always final.",
        "<strong>Claims still go through your TPA</strong> as today. Employee claim filing and detail corrections are marked &ldquo;Coming soon&rdquo;.",
        "<strong>Top up ledger</strong> is handled by CapitUp for now. Ask us and we'll do it.",
        "<strong>Desktop first.</strong> It works in a mobile browser too; there's no app yet. A few page headings still use older names, like &ldquo;Endorsement Studio&rdquo;.",
    ])
    return _HEAD + """  {{greeting_block}}
  <tr><td style="padding:30px 32px 6px 32px;">
    <p style="margin:0 0 6px 0;font-size:12px;font-weight:800;letter-spacing:2px;color:#c9963f;">HAPPY DIWALI &middot; DHAIRYA IS LIVE FOR YOUR HR TEAM</p>
    <h1 style="margin:0;font-size:26px;line-height:1.25;color:#1c2430;font-family:Georgia,'Times New Roman',serif;">Every policy, roster and ledger on one desk.</h1>
  </td></tr>
  <tr><td style="padding:16px 32px 4px 32px;font-size:15px;line-height:1.7;color:#333;">
    <p style="margin:0 0 14px 0;">Dear <strong>{{name}}</strong>,</p>
    <p style="margin:0 0 14px 0;">Warm Diwali wishes to you, your family and everyone at <strong>{{company_name}}</strong>. May the festival bring light, good health and a safe celebration to all of you.</p>
    <p style="margin:0 0 14px 0;">We're happy to open <strong>dh<span style="color:#c9963f;">AI</span>rya</strong> to your HR team. <em>Dhairya</em> means steadiness, and that's what we want your group health benefits to feel like: your policy, your members, your CD balance and your requests to us, in one secure place.</p>
  </td></tr>
  <tr><td style="padding:6px 32px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#faf8f4;border-radius:12px;border:1px solid #EFE3C2;">
      <tr><td style="padding:18px 20px 4px 20px;font-size:13px;font-weight:800;letter-spacing:1px;color:#1c2430;">WHAT YOU CAN DO FROM DAY ONE</td></tr>
      <tr><td style="padding:4px 20px 14px 20px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0">""" + features + """</table></td></tr>
    </table>
  </td></tr>
  <tr><td style="padding:22px 32px 4px 32px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #CDEBE4;border-radius:12px;background:#F2FBF9;">
      <tr><td style="padding:16px 20px;font-size:14px;line-height:1.7;color:#333;">
        <strong style="color:#126f62;">&#128273; Signing in, in three steps</strong><br/>
        1. Open <a href="{{portal_url}}" style="color:#128a78;font-weight:700;">{{portal_url}}</a><br/>
        2. Enter your <strong>work e-mail</strong> and the <strong>temporary password</strong> we send you in a separate e-mail.<br/>
        3. Set your own permanent password when asked. You land on your Executive Overview, with a <em>Getting started</em> checklist to guide your first day.<br/>
        <span style="font-size:12px;color:#6B7785;">For your security, this e-mail never contains a password. Your login sees only {{company_name}}'s data.</span>
      </td></tr>
    </table>
  </td></tr>
  <tr><td align="center" style="padding:20px 32px 10px 32px;">
    <a href="{{portal_url}}" style=\"""" + _BTN + """\">Open dhAIrya &rarr;</a>
    <p style="margin:10px 0 0 0;font-size:12px;color:#6B7785;">Your step-by-step HR setup guide (PDF) is attached.</p>
  </td></tr>
  {{video_block}}
  <tr><td style="padding:16px 32px 4px 32px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border:1px dashed #1aab93;border-radius:12px;">
      <tr><td style="padding:16px 18px;font-size:13px;line-height:1.7;color:#333;">
        <strong style="color:#1c2430;">&#128106; Next: your employees.</strong> Each employee signs in at <a href="{{employee_url}}" style="color:#128a78;">{{employee_url}}</a> with their company, Employee ID and date of birth, and gets a 6-digit code on the e-mail in your roster. There they can see their cover and family members, download e-cards, ask the Policy Copilot, and request to add a dependent.<br/>
        <strong>One thing we need from you:</strong> an e-mail for every employee. In <em>Employee Directory</em>, the <em>Unboarded</em> filter shows who can't sign in yet. E-cards appear once we've uploaded your insurer's PDFs.
      </td></tr>
    </table>
  </td></tr>
  <tr><td style="padding:18px 32px 6px 32px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#1c2430;border-radius:12px;">
      <tr><td style="padding:22px 22px 8px 22px;">
        <p style="margin:0;font-size:12px;font-weight:800;letter-spacing:2px;color:#1aab93;">BETA &middot; BUILT WITH YOU</p>
        <p style="margin:8px 0 0 0;font-size:20px;line-height:1.35;color:#ffffff;font-family:Georgia,'Times New Roman',serif;">Our promise has always been <span style="color:#ecc976;">&ldquo;We Back You Up.&rdquo;</span> This time, we're asking you to back us up too.</p>
        <p style="margin:12px 0 0 0;font-size:14px;line-height:1.7;color:#C9D3DD;">dhAIrya is in <strong style="color:#ffffff;">beta</strong>. Some screens will change and a few things aren't perfect yet. Every note you send reaches the team building it. Let's back each other up.</p>
      </td></tr>
      <tr><td align="center" style="padding:10px 22px 24px 22px;">
        <a href="{{feedback_url}}" style="background:#1aab93;color:#1c2430;text-decoration:none;font-size:14px;font-weight:800;padding:13px 28px;border-radius:8px;display:inline-block;">&#128221; Share your feedback (2 minutes)</a>
        <p style="margin:10px 0 0 0;font-size:12px;color:#9FB0BF;">Something not working? Click <em>Report a problem</em> at the top of any page. It comes straight to us.</p>
      </td></tr>
    </table>
  </td></tr>
  <tr><td style="padding:18px 32px 4px 32px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-left:4px solid #c9963f;background:#FFFBF0;">
      <tr><td style="padding:14px 16px;font-size:13px;line-height:1.65;color:#444;">
        <strong style="color:#1c2430;">&#128197; How fresh is your data?</strong> dhAIrya is as current as the last active list, endorsement and claims file loaded for your company. Latest endorsement file we hold for {{company_name}}: <strong>{{data_cutoff}}</strong>. Anything sent after that appears once it is loaded and approved; <em>Recent batches</em> on the Endorsement Engine page shows exactly what's in.
      </td></tr>
    </table>
  </td></tr>
  <tr><td style="padding:16px 32px 4px 32px;">
    <p style="margin:0 0 8px 0;font-size:13px;font-weight:800;letter-spacing:1px;color:#1c2430;">GOOD TO KNOW IN THIS BETA</p>
    <ul style="margin:0;padding:0 0 0 18px;font-size:13px;line-height:1.6;color:#444;">""" + notes + """</ul>
  </td></tr>
  <tr><td style="padding:20px 32px 28px 32px;font-size:14px;line-height:1.7;color:#333;">
    Wishing you a bright, happy and safe Diwali,<br/>
    <strong style="color:#1c2430;">{{sender_name}}</strong>
  </td></tr>
""" + _FOOT


def user_template():
    return _HEAD + """  {{greeting_block}}
  <tr><td style="padding:30px 32px 6px 32px;">
    <p style="margin:0 0 6px 0;font-size:12px;font-weight:800;letter-spacing:2px;color:#c9963f;">HAPPY DIWALI FROM CAPITUP</p>
    <h1 style="margin:0;font-size:26px;line-height:1.25;color:#1c2430;font-family:Georgia,'Times New Roman',serif;">Your health cover, steady and clear.</h1>
  </td></tr>
  <tr><td style="padding:16px 32px 4px 32px;font-size:15px;line-height:1.7;color:#333;">
    <p style="margin:0 0 14px 0;">Dear <strong>{{name}}</strong>,</p>
    <p style="margin:0 0 14px 0;">Together with <strong>{{company_name}}</strong>, we're happy to welcome you to <strong>dh<span style="color:#c9963f;">AI</span>rya</strong>, your group health benefits portal.</p>
  </td></tr>
  <tr><td style="padding:6px 32px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#faf8f4;border-radius:12px;border:1px solid #EFE3C2;">
      <tr><td style="padding:18px 20px 16px 20px;font-size:14px;line-height:1.75;color:#333;">
        &#129706; <strong>See your cover and e-card</strong>, and download it as a PDF.<br/>
        &#128106; <strong>See who is covered</strong>: your family members on the policy.<br/>
        &#10024; <strong>Ask the Policy Copilot</strong> about room rent, maternity or waiting periods.<br/>
        &#10133; <strong>Request to add a dependent</strong> and track it in the Request tracker.
      </td></tr>
    </table>
  </td></tr>
  <tr><td style="padding:18px 32px 4px 32px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #CDEBE4;border-radius:12px;background:#F2FBF9;">
      <tr><td style="padding:16px 20px;font-size:14px;line-height:1.7;color:#333;">
        <strong style="color:#126f62;">&#128273; How to sign in</strong><br/>
        1. Open <a href="{{portal_url}}" style="color:#128a78;font-weight:700;">{{portal_url}}</a><br/>
        2. Pick <strong>{{company_name}}</strong>, enter your <strong>Employee ID</strong> and <strong>date of birth</strong>, then click <em>Send sign-in code</em>.<br/>
        3. Enter the 6-digit code sent to this e-mail address (valid for 10 minutes).
      </td></tr>
    </table>
  </td></tr>
  <tr><td align="center" style="padding:22px 32px 10px 32px;">
    <a href="{{portal_url}}" style=\"""" + _BTN + """\">Open my dashboard &rarr;</a>
  </td></tr>
  {{video_block}}
  <tr><td style="padding:18px 32px 6px 32px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#1c2430;border-radius:12px;">
      <tr><td style="padding:22px 22px 8px 22px;">
        <p style="margin:0;font-size:20px;line-height:1.35;color:#ffffff;font-family:Georgia,'Times New Roman',serif;">We back you up. <span style="color:#ecc976;">Back us up with your feedback.</span></p>
        <p style="margin:10px 0 0 0;font-size:14px;line-height:1.7;color:#C9D3DD;">dhAIrya is in <strong style="color:#ffffff;">beta</strong>. Tell us what works and what doesn't, and we'll keep making it better.</p>
      </td></tr>
      <tr><td align="center" style="padding:10px 22px 24px 22px;">
        <a href="{{feedback_url}}" style="background:#1aab93;color:#1c2430;text-decoration:none;font-size:14px;font-weight:800;padding:13px 28px;border-radius:8px;display:inline-block;">&#128221; Share your feedback</a>
      </td></tr>
    </table>
  </td></tr>
  <tr><td style="padding:18px 32px 26px 32px;font-size:12px;line-height:1.6;color:#6B7785;">
    Your e-card appears once your insurer's card has been uploaded. Claims still go through your TPA as usual. AI answers can be wrong; your policy document is final. Something not working? Use <em>Report a problem</em> at the top of any page, reply to this e-mail, or write to {{support_email}}.
  </td></tr>
""" + _FOOT


def data_date_for(cfg, company):
    """Match a company against the "Name = date" lines; first keyword hit wins."""
    def norm(x):
        return re.sub(r"[^a-z0-9]", "", str(x or "").lower())
    comp = norm(company)
    for line in str(cfg.get("company_data_dates", "")).splitlines():
        if "=" not in line:
            continue
        key, val = (x.strip() for x in line.split("=", 1))
        if norm(key) and val and norm(key) in comp:
            return val
    return cfg["data_cutoff"]


def render_email(tmpl, ctx, festive, has_video):
    """Fill a template. ctx values are escaped here; URLs are attribute-quoted."""
    body = tmpl.replace("{{greeting_block}}", _GREETING_BLOCK if festive else "")
    body = body.replace("{{video_block}}", _VIDEO_BLOCK if has_video else "")
    for k, v in ctx.items():
        safe = html.escape(str(v), quote=True)
        body = body.replace("{{" + k + "}}", safe)
    return body


def feedback_link(base, rid, name, company):
    sep = "&" if "?" in base else "?"
    return (f"{base}{sep}{FB_ENTRY_ID}={urllib.parse.quote(str(rid))}"
            f"&{FB_ENTRY_NAME}={urllib.parse.quote(str(name))}"
            f"&{FB_ENTRY_COMPANY}={urllib.parse.quote(str(company))}")


# ----------------------------------------------------------------------------
# Sending (one SMTP connection per batch)
# ----------------------------------------------------------------------------
class SmtpBatch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.server = None

    def _connect(self):
        port = int(self.cfg["port"])
        if port == 465:
            s = smtplib.SMTP_SSL(self.cfg["server"], port, timeout=30)
        else:
            s = smtplib.SMTP(self.cfg["server"], port, timeout=30)
            s.ehlo(); s.starttls(); s.ehlo()
        s.login(self.cfg["username"], self.cfg["password"])
        self.server = s

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *a):
        try:
            if self.server:
                self.server.quit()
        except Exception:
            pass

    def send(self, to_addr, msg):
        try:
            self.server.sendmail(self.cfg["username"], [to_addr], msg.as_string())
        except (smtplib.SMTPServerDisconnected, smtplib.SMTPSenderRefused, OSError):
            self._connect()  # one reconnect, then let errors surface
            self.server.sendmail(self.cfg["username"], [to_addr], msg.as_string())


def build_message(smtp_cfg, sender_name, to_addr, subject, body_html, images, attachments, support_email):
    msg = MIMEMultipart("mixed")
    msg["From"] = formataddr((sender_name, smtp_cfg["sender_email"]))
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Reply-To"] = support_email or smtp_cfg["sender_email"]
    msg["Message-ID"] = make_msgid(domain=smtp_cfg["sender_email"].split("@")[-1])
    if support_email:
        msg["List-Unsubscribe"] = f"<mailto:{support_email}?subject=Unsubscribe%20portal%20updates>"
    related = MIMEMultipart("related")
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(_html_to_text(body_html), "plain", "utf-8"))
    alt.attach(MIMEText(body_html, "html", "utf-8"))
    related.attach(alt)
    for cid, data in images.items():
        if data and f"cid:{cid}" in body_html:
            img = MIMEImage(data)
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
            related.attach(img)
    msg.attach(related)
    for fname, data in attachments:
        if data:
            att = MIMEApplication(data, _subtype="pdf")
            att.add_header("Content-Disposition", "attachment", filename=fname)
            msg.attach(att)
    return msg


# ----------------------------------------------------------------------------
# Audience storage
# ----------------------------------------------------------------------------
def upsert_hr_contact(db, email, name, company, policy_no=""):
    email = email.strip().lower()
    s = {"email": email, "updated_at": datetime.utcnow()}
    if name: s["name"] = name
    if company: s["company_name"] = company.strip().upper()
    if policy_no: s["policy_no"] = policy_no.strip().upper()
    db.hr_contacts.update_one({"email": email},
                              {"$set": s, "$setOnInsert": {"launch_announced": False, "created_at": datetime.utcnow()}},
                              upsert=True)


def upsert_employee(db, emp_id, name, email, policy_no, company):
    """Never overwrite a good value with a blank one and never change an existing role."""
    emp_id = emp_id.strip().upper()
    policy_no = (policy_no or "DEFAULT").strip().upper()
    s = {"emp_id": emp_id, "policy_no": policy_no, "updated_at": datetime.utcnow()}
    if name: s["name"] = name
    if email: s["email"] = email.strip().lower()
    if company: s["company_name"] = company.strip().upper()
    db.directory.update_one({"emp_id": emp_id, "policy_no": policy_no},
                            {"$set": s, "$setOnInsert": {"role": "EMPLOYEE"}}, upsert=True)


def _read_sheet(upload, clean_fn):
    raw = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    return clean_fn(raw)


def _guess(cols, words):
    for w in words:
        for c in cols:
            if w in str(c).upper():
                return cols.index(c)
    return 0


# ----------------------------------------------------------------------------
# Feedback widget (sidebar + employee view)
# ----------------------------------------------------------------------------
def render_feedback_widget(db, username, persona, company_hint="", key="fb"):
    with st.expander("\U0001f4ac Beta feedback: tell us what to improve", expanded=False):
        st.caption("This portal is in beta. Your feedback goes straight to the CapitUp team.")
        with st.form(f"{key}_form", clear_on_submit=True):
            rating = st.select_slider("How is your experience so far?", options=[1, 2, 3, 4, 5], value=4,
                                      format_func=lambda x: "⭐" * x)
            cat = st.selectbox("Topic", FEEDBACK_CATEGORIES, key=f"{key}_cat")
            comp = st.text_input("Company (optional)", value=company_hint, key=f"{key}_comp")
            msg = st.text_area("What should we fix, add or keep?", key=f"{key}_msg", height=100)
            if st.form_submit_button("Send feedback", type="primary", width="stretch"):
                if not msg.strip():
                    st.warning("Please write a short note before sending.")
                else:
                    try:
                        db.portal_feedback.insert_one({
                            "submitted_by": username or "ANONYMOUS", "persona": persona,
                            "company_name": comp.strip().upper(), "rating": int(rating), "category": cat,
                            "message": msg.strip()[:4000], "status": "NEW", "created_at": datetime.utcnow(),
                        })
                        st.success("Thank you! We back you up, and you just backed us up. \U0001f64f")
                    except Exception as e:
                        logging.error(f"feedback save failed: {e}")
                        st.error("Couldn't save feedback right now. Please email support@capitupindia.com.")


# ----------------------------------------------------------------------------
# Main tab
# ----------------------------------------------------------------------------
def render_launch_center(db, helpers):
    get_asset = helpers["get_asset"]
    save_asset = helpers["save_asset"]
    delete_asset = helpers["delete_asset"]
    log_email_dispatch = helpers["log_email_dispatch"]
    clean_df = helpers["clean_and_align_dataframe"]
    tenants = helpers["get_live_tenants_and_policies"]()

    ensure_indexes(db)
    cfg = load_config(db)

    st.markdown("### \U0001f680 Portal Launch & Feedback Center")
    st.caption("Launch the portal to tenant HR teams now and to employees once their emails are in. "
               "Every send is logged; failures can be retried; feedback lands in the Feedback Inbox.")

    # ---------------- Campaign settings ----------------
    with st.expander("⚙️ Campaign settings (saved for everyone)", expanded=False):
        c1, c2 = st.columns(2)
        new = dict(cfg)
        new["portal_name"] = c1.text_input("Portal name", cfg["portal_name"], key="lc_pname")
        new["portal_tagline"] = c2.text_input("Tagline (follows the name)", cfg["portal_tagline"], key="lc_ptag")
        new["hr_portal_url"] = c1.text_input("HR portal URL", cfg["hr_portal_url"], key="lc_hrurl")
        new["user_portal_url"] = c2.text_input("Employee portal URL", cfg["user_portal_url"], key="lc_uurl")
        new["feedback_form_url"] = st.text_input("Feedback Google Form (pre-fill URL)", cfg["feedback_form_url"], key="lc_fburl")
        new["data_cutoff"] = c1.text_input("Data date fallback", cfg["data_cutoff"], key="lc_cutoff",
                                           help="Used when a company has no line below. dhAIrya has no single 'data as of' date.")
        new["support_email"] = c2.text_input("Support / reply-to email", cfg["support_email"], key="lc_support")
        new["video_url"] = c1.text_input("Launch video link (YouTube / Drive)", cfg["video_url"], key="lc_video",
                                         help="Leave blank to hide the video block in the email.")
        new["sender_name"] = c2.text_input("Sender name", cfg["sender_name"], key="lc_sender")
        new["hr_subject"] = c1.text_input("HR email subject", cfg["hr_subject"], key="lc_hrsubj")
        new["user_subject"] = c2.text_input("Employee email subject", cfg["user_subject"], key="lc_usubj")
        new["company_data_dates"] = st.text_area(
            "Latest endorsement file per company (one 'Company = date' per line)", cfg["company_data_dates"],
            height=150, key="lc_dates", help="Matched by a word in the company name. Check each client's Recent batches before sending.")
        new["festive_greeting"] = st.toggle("\U0001fa94 Include Diwali greeting card", cfg["festive_greeting"], key="lc_festive")
        if st.button("\U0001f4be Save campaign settings", type="primary", key="lc_save"):
            save_config(db, new)
            st.success("Saved.")
            time.sleep(0.6); st.rerun()

    # ---------------- Shared assets ----------------
    with st.expander("\U0001f5bc️ Launch assets (guide PDF, greeting card, video thumbnail)", expanded=False):
        st.caption("Uploaded files replace the bundled CapitUp defaults. Delete an upload to go back to the default.")
        a1, a2, a3, a4 = st.columns(4)
        for col, key, label, types in (
            (a1, "hr_portal_guide", "HR setup guide (PDF)", ["pdf"]),
            (a2, "user_portal_guide", "Employee guide (PDF)", ["pdf"]),
            (a3, "hr_launch_banner", "Greeting banner (image)", ["png", "jpg", "jpeg"]),
            (a4, "video_thumb", "Video thumbnail (image)", ["png", "jpg", "jpeg"]),
        ):
            with col:
                if get_asset(key):
                    st.success(f"{label}: custom upload")
                    if st.button("Delete upload", key=f"lc_del_{key}"):
                        delete_asset(key); st.rerun()
                else:
                    default = {"hr_portal_guide": "hr_portal_guide", "hr_launch_banner": "diwali_card",
                               "video_thumb": "video_thumb"}.get(key)
                    st.info(f"{label}: " + ("CapitUp default" if default and _bundled_bytes(default) else "none"))
                    up = st.file_uploader("Upload", type=types, key=f"lc_up_{key}", label_visibility="collapsed")
                    if up:
                        save_asset(key, up.getvalue()); st.rerun()

    def asset(key, bundled_key=None):
        return get_asset(key) or (_bundled_bytes(bundled_key) if bundled_key else None)

    logo_b = asset("logo", "logo")
    banner_hr = asset("hr_launch_banner", "diwali_card")
    banner_user = get_asset("user_launch_banner") or banner_hr
    thumb_b = asset("video_thumb", "video_thumb")
    has_video = bool(cfg["video_url"].strip()) and bool(thumb_b)

    def preview(body, banner):
        out = body.replace("cid:logo_image", _data_uri(logo_b, "image/png") if logo_b else "")
        if banner: out = out.replace("cid:launch_banner", _data_uri(banner, "image/jpeg"))
        if thumb_b: out = out.replace("cid:video_thumb", _data_uri(thumb_b, "image/jpeg"))
        components.html(out, height=900, scrolling=True)

    smtp_ok = "smtp" in st.secrets

    sub_hr, sub_user, sub_fb, sub_audit = st.tabs([
        "\U0001f3e2 Tenant HR launch", "\U0001f464 Employee launch", "\U0001f4ac Feedback inbox", "\U0001f4ca Audit & retry"])

    # ========================= HR LAUNCH =========================
    with sub_hr:
        comp_opts = ["\U0001f310 All companies"] + sorted(tenants.keys())
        sel_comp = st.selectbox("Send to HR contacts of:", comp_opts, key="lc_hr_comp")
        scope_comp = None if sel_comp.startswith("\U0001f310") else sel_comp

        left, right = st.columns([1.25, 1])
        with right:
            st.subheader("\U0001f465 HR contacts")
            up = st.file_uploader("Upload HR contacts (CSV/Excel): name, email, company", type=["csv", "xlsx", "xls"], key="lc_hr_up")
            if up:
                try:
                    df = _read_sheet(up, clean_df)
                    cols = list(df.columns)
                    k1, k2 = st.columns(2)
                    n_col = k1.selectbox("Name column", cols, index=_guess(cols, ["NAME"]), key="lc_hr_n")
                    e_col = k2.selectbox("Email column", cols, index=_guess(cols, ["MAIL"]), key="lc_hr_e")
                    c_col = k1.selectbox("Company column", ["(use selected company)"] + cols,
                                         index=_guess(["(use selected company)"] + cols, ["COMPANY", "CLIENT", "CORP"]), key="lc_hr_c")
                    if st.button("➕ Add / update these contacts", type="primary", width="stretch", key="lc_hr_sync"):
                        added, skipped = 0, 0
                        for _, r in df.iterrows():
                            em = _clean(r[e_col]).lower()
                            comp = _clean(r[c_col]) if c_col != "(use selected company)" else (scope_comp or "")
                            if not _valid_email(em):
                                skipped += 1; continue
                            upsert_hr_contact(db, em, _clean(r[n_col]), comp)
                            added += 1
                        st.success(f"Saved {added} HR contacts. Skipped {skipped} rows without a valid email.")
                        time.sleep(1); st.rerun()
                except Exception as e:
                    st.error(f"Couldn't read that sheet: {e}")

            with st.form("lc_hr_manual", clear_on_submit=True):
                st.markdown("**Add one contact**")
                m1, m2 = st.columns(2)
                mn = m1.text_input("Name")
                me = m2.text_input("Email")
                mc = st.text_input("Company", value=scope_comp or "")
                if st.form_submit_button("Add contact", width="stretch"):
                    if _valid_email(me.strip()):
                        upsert_hr_contact(db, me, mn.strip(), mc); st.success("Added."); st.rerun()
                    else:
                        st.error("Enter a valid email.")

            legacy = list(db.directory.find({"role": "HR_ADMIN", "email": {"$regex": "@"}}))
            if legacy:
                if st.button(f"↪️ Import {len(legacy)} HR contacts saved by the old launch tab", width="stretch", key="lc_hr_legacy"):
                    for l in legacy:
                        upsert_hr_contact(db, l["email"], l.get("name", ""), l.get("company_name", ""))
                    st.success("Imported."); st.rerun()

            q = {"company_name": scope_comp} if scope_comp else {}
            contacts = list(db.hr_contacts.find(q).sort("company_name", 1))
            ready = [c for c in contacts if not c.get("launch_announced")]
            mc1, mc2 = st.columns(2)
            mc1.metric("\U0001f7e2 Ready to invite", len(ready))
            mc2.metric("✅ Already invited", len(contacts) - len(ready))
            if contacts:
                st.dataframe(pd.DataFrame([{
                    "Name": c.get("name", ""), "Email": c["email"], "Company": c.get("company_name", ""),
                    "Invited": "✅" if c.get("launch_announced") else "⏳",
                    "Sent (IST)": (c["launch_sent_at"] + timedelta(hours=5, minutes=30)).strftime("%d-%b %I:%M %p") if c.get("launch_sent_at") else "",
                } for c in contacts]), hide_index=True, width="stretch", height=220)

        with left:
            st.subheader("✉️ HR launch email")
            tmpl = st.text_area("Email HTML (placeholders in {{ }} are filled per recipient)", hr_template(),
                                height=160, key=f"lc_hr_tmpl_{TEMPLATE_VERSION}")
            sample = ready[0] if ready else (contacts[0] if contacts else {"name": "Priya Sharma", "company_name": scope_comp or "YOUR COMPANY", "email": "hr@example.com"})

            def ctx_for(c):
                comp = c.get("company_name") or scope_comp or "your organisation"
                return {
                    "name": c.get("name") or "HR Team", "company_name": comp,
                    "portal_name": cfg["portal_name"], "portal_tagline": cfg["portal_tagline"],
                    "portal_url": cfg["hr_portal_url"], "data_cutoff": data_date_for(cfg, comp),
                    "employee_url": cfg["user_portal_url"],
                    "support_email": cfg["support_email"], "sender_name": cfg["sender_name"],
                    "video_url": cfg["video_url"],
                    "feedback_url": feedback_link(cfg["feedback_form_url"], c.get("email", "HR"), c.get("name", "HR"), comp),
                }

            guide = asset("hr_portal_guide", "hr_portal_guide")
            st.caption(("\U0001f4ce Attaches: dhAIrya_HR_Setup_Guide.pdf" if guide else "⚠️ No setup guide PDF available") +
                       (" · \U0001fa94 Diwali card on" if cfg["festive_greeting"] else "") +
                       (" · ▶️ video block on" if has_video else " · video block off (add a link in settings)"))
            if st.checkbox("\U0001f441️ Preview (first recipient)", value=True, key="lc_hr_prev"):
                preview(render_email(tmpl, ctx_for(sample), cfg["festive_greeting"], has_video), banner_hr)

            images = {"logo_image": logo_b, "launch_banner": banner_hr, "video_thumb": thumb_b}
            atts = [("dhAIrya_HR_Setup_Guide.pdf", guide)]

            st.markdown("---")
            t1, t2 = st.columns([2, 1])
            test_to = t1.text_input("Send a test to", placeholder="you@capitupindia.com", key="lc_hr_test_to")
            if t2.button("\U0001f9ea Send test", width="stretch", disabled=not smtp_ok, key="lc_hr_test"):
                if _valid_email(test_to.strip()):
                    try:
                        with SmtpBatch(st.secrets["smtp"]) as sb:
                            body = render_email(tmpl, ctx_for(sample), cfg["festive_greeting"], has_video)
                            sb.send(test_to.strip(), build_message(st.secrets["smtp"], cfg["sender_name"], test_to.strip(),
                                                                   "[TEST] " + cfg["hr_subject"], body, images, atts, cfg["support_email"]))
                        st.success(f"Test sent to {test_to.strip()}.")
                    except Exception as e:
                        st.error(f"Test failed: {e}")
                else:
                    st.error("Enter a valid email.")

            b_lim = st.number_input("Batch size", 1, 200, min(25, max(1, len(ready))), key="lc_hr_blim")
            go = st.button(f"\U0001f680 Send {min(len(ready), b_lim)} HR invitations", type="primary", width="stretch",
                           disabled=(not ready or not smtp_ok), key="lc_hr_send")
            if not smtp_ok:
                st.warning("SMTP is not configured in secrets, so sending is disabled.")
            if go:
                _send_batch(db, ready[:b_lim], lambda c: c["email"], ctx_for, tmpl, cfg, cfg["hr_subject"], images, atts,
                            has_video, CAMPAIGN_HR, log_email_dispatch,
                            mark=lambda c: db.hr_contacts.update_one({"_id": c["_id"]}, {"$set": {"launch_announced": True, "launch_sent_at": datetime.utcnow()}}),
                            log_id=lambda c: c["email"])

    # ========================= EMPLOYEE LAUNCH =========================
    with sub_user:
        st.info("Use this once you have employees' email IDs. Employees get their own dashboard: family e-cards, coverage and policy chat.")
        all_pols = sorted(set(db.directory.distinct("policy_no", {"role": "EMPLOYEE"})) | {p for v in tenants.values() for p in v})
        pol_opts = ["\U0001f310 All policies"] + all_pols
        sel_pol = st.selectbox("Policy scope", pol_opts, key="lc_u_pol")
        scope_pol = None if sel_pol.startswith("\U0001f310") else sel_pol
        comp_default = st.selectbox("Company name for employees without one", ["(from directory)"] + sorted(tenants.keys()), key="lc_u_comp")

        left, right = st.columns([1.25, 1])
        with right:
            st.subheader("\U0001f465 Employee roster")
            upu = st.file_uploader("Upload employees (CSV/Excel): emp ID, name, email", type=["csv", "xlsx", "xls"], key="lc_u_up")
            if upu:
                try:
                    dfu = _read_sheet(upu, clean_df)
                    cols = list(dfu.columns)
                    k1, k2 = st.columns(2)
                    ecol = k1.selectbox("Emp ID column", cols, index=_guess(cols, ["EMP", "CODE", "ID"]), key="lc_u_e")
                    ncol = k2.selectbox("Name column", cols, index=_guess(cols, ["NAME"]), key="lc_u_n")
                    mcol = k1.selectbox("Email column", cols, index=_guess(cols, ["MAIL"]), key="lc_u_m")
                    rcol = k2.selectbox("Relationship column (optional)", ["(none: all are employees)"] + cols, key="lc_u_r")
                    if st.button("➕ Sync roster", type="primary", width="stretch", key="lc_u_sync"):
                        n_ok, n_skip = 0, 0
                        for _, r in dfu.iterrows():
                            if rcol != "(none: all are employees)" and _clean(r[rcol]).upper() not in ("SELF", "EMPLOYEE", "PRIMARY", "PROPOSER", ""):
                                continue
                            eid = _clean(r[ecol])
                            if not eid:
                                n_skip += 1; continue
                            em = _clean(r[mcol]).lower()
                            upsert_employee(db, eid, _clean(r[ncol]), em if _valid_email(em) else "", scope_pol or "DEFAULT",
                                            "" if comp_default.startswith("(") else comp_default)
                            n_ok += 1
                        st.success(f"Synced {n_ok} employees. Skipped {n_skip} rows without an Employee ID.")
                        time.sleep(1); st.rerun()
                except Exception as e:
                    st.error(f"Couldn't read that sheet: {e}")

            qu = {"role": "EMPLOYEE"}
            if scope_pol: qu["policy_no"] = scope_pol
            emps = list(db.directory.find(qu))
            ready_u = [u for u in emps if not u.get("launch_announced") and _valid_email(u.get("email", ""))]
            no_mail = [u for u in emps if not _valid_email(u.get("email", ""))]
            x1, x2, x3 = st.columns(3)
            x1.metric("\U0001f7e2 Ready", len(ready_u))
            x2.metric("✅ Invited", sum(1 for u in emps if u.get("launch_announced")))
            x3.metric("⚠️ No email", len(no_mail))

        with left:
            st.subheader("✉️ Employee launch email")
            tmpl_u = st.text_area("Email HTML", user_template(), height=160, key=f"lc_u_tmpl_{TEMPLATE_VERSION}")

            def ctx_u(u):
                comp = u.get("company_name") if u.get("company_name") not in (None, "", "UNKNOWN_CORP", "DEFAULT") else (
                    "" if comp_default.startswith("(") else comp_default)
                comp = comp or "your organisation"
                return {
                    "name": u.get("name") or "there", "company_name": comp,
                    "portal_name": cfg["portal_name"], "portal_tagline": cfg["portal_tagline"],
                    "portal_url": cfg["user_portal_url"], "data_cutoff": data_date_for(cfg, comp),
                    "employee_url": cfg["user_portal_url"],
                    "support_email": cfg["support_email"], "sender_name": cfg["sender_name"], "video_url": cfg["video_url"],
                    "feedback_url": feedback_link(cfg["feedback_form_url"], u.get("emp_id", ""), u.get("name", ""), comp),
                }

            sample_u = ready_u[0] if ready_u else {"name": "Rahul Verma", "emp_id": "EMP001", "company_name": ""}
            if st.checkbox("\U0001f441️ Preview", value=False, key="lc_u_prev"):
                preview(render_email(tmpl_u, ctx_u(sample_u), cfg["festive_greeting"], has_video), banner_user)
            images_u = {"logo_image": logo_b, "launch_banner": banner_user, "video_thumb": thumb_b}
            atts_u = [("CapitUp_Employee_Portal_Guide.pdf", get_asset("user_portal_guide"))]
            b_lim_u = st.number_input("Batch size", 1, 300, min(25, max(1, len(ready_u))), key="lc_u_blim")
            if st.button(f"\U0001f680 Send {min(len(ready_u), b_lim_u)} employee invitations", type="primary", width="stretch",
                         disabled=(not ready_u or not smtp_ok), key="lc_u_send"):
                _send_batch(db, ready_u[:b_lim_u], lambda u: u["email"], ctx_u, tmpl_u, cfg, cfg["user_subject"], images_u, atts_u,
                            has_video, CAMPAIGN_USER, log_email_dispatch,
                            mark=lambda u: db.directory.update_one({"_id": u["_id"]}, {"$set": {"launch_announced": True, "launch_sent_at": datetime.utcnow()}}),
                            log_id=lambda u: u["emp_id"])

    # ========================= FEEDBACK INBOX =========================
    with sub_fb:
        st.subheader("\U0001f4ac Beta feedback from the portal")
        st.caption("Collected from the in-portal feedback box. Google Form answers stay in the form's Responses tab / linked Sheet.")
        fbs = list(db.portal_feedback.find().sort("created_at", -1).limit(1000))
        if not fbs:
            st.info("No in-portal feedback yet. It appears here as soon as someone sends it.")
        else:
            dff = pd.DataFrame([{
                "id": str(f["_id"]), "When (IST)": (f["created_at"] + timedelta(hours=5, minutes=30)).strftime("%d-%b-%Y %I:%M %p"),
                "From": f.get("submitted_by", ""), "View": f.get("persona", ""), "Company": f.get("company_name", ""),
                "Rating": f.get("rating"), "Topic": f.get("category", ""), "Message": f.get("message", ""), "Status": f.get("status", "NEW"),
            } for f in fbs])
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Responses", len(dff))
            f2.metric("Avg rating", f"{dff['Rating'].dropna().mean():.1f} / 5" if dff["Rating"].notna().any() else "-")
            f3.metric("New", int((dff["Status"] == "NEW").sum()))
            f4.metric("Companies", dff["Company"].replace("", pd.NA).nunique())
            fc1, fc2 = st.columns(2)
            st_filter = fc1.multiselect("Status", FEEDBACK_STATUSES, default=["NEW", "REVIEWED", "PLANNED"], key="lc_fb_stat")
            topic_filter = fc2.multiselect("Topic", FEEDBACK_CATEGORIES, key="lc_fb_topic")
            view = dff[dff["Status"].isin(st_filter)] if st_filter else dff
            if topic_filter: view = view[view["Topic"].isin(topic_filter)]
            st.dataframe(view.drop(columns=["id"]), hide_index=True, width="stretch")
            st.download_button("\U0001f4e5 Download all feedback (.csv)", dff.drop(columns=["id"]).to_csv(index=False).encode(),
                               file_name=f"portal_feedback_{datetime.utcnow():%Y%m%d}.csv", mime="text/csv", key="lc_fb_dl")
            with st.form("lc_fb_status"):
                g1, g2 = st.columns([3, 1])
                pick = g1.selectbox("Update status of", view["id"].tolist(),
                                    format_func=lambda i: f"{dff.set_index('id').loc[i, 'When (IST)']} · {dff.set_index('id').loc[i, 'Message'][:60]}") if len(view) else None
                new_s = g2.selectbox("New status", FEEDBACK_STATUSES)
                if st.form_submit_button("Update") and pick:
                    from bson import ObjectId
                    db.portal_feedback.update_one({"_id": ObjectId(pick)}, {"$set": {"status": new_s}})
                    st.rerun()

    # ========================= AUDIT & RETRY =========================
    with sub_audit:
        st.subheader("\U0001f4ca Launch dispatch log (last 7 days)")
        logs = list(db.email_logs.find({"campaign_type": {"$in": [CAMPAIGN_HR, CAMPAIGN_USER]}}).sort("timestamp", -1).limit(500))
        if not logs:
            st.info("No launch emails sent in the last 7 days.")
        else:
            st.dataframe(pd.DataFrame([{
                "Time (IST)": (l["timestamp"] + timedelta(hours=5, minutes=30)).strftime("%d-%b %I:%M %p"),
                "Campaign": "HR" if l.get("campaign_type") == CAMPAIGN_HR else "Employee",
                "Recipient": l.get("recipient_email", ""), "Name": l.get("name", ""),
                "Status": ("✅ " if l["status"] == "DELIVERED" else "❌ ") + l["status"],
                "Error": l.get("error_reason") or "",
            } for l in logs]), hide_index=True, width="stretch")

        failed = [l for l in logs if l.get("status") == "FAILED"]
        r1, r2, r3 = st.columns(3)
        if r1.button(f"\U0001f504 Retry {len(failed)} failed", disabled=not failed, width="stretch", key="lc_retry"):
            hr_emails = list({l["recipient_email"] for l in failed if l.get("campaign_type") == CAMPAIGN_HR})
            u_emails = list({l["recipient_email"] for l in failed if l.get("campaign_type") == CAMPAIGN_USER})
            if hr_emails: db.hr_contacts.update_many({"email": {"$in": hr_emails}}, {"$set": {"launch_announced": False}})
            if u_emails: db.directory.update_many({"role": "EMPLOYEE", "email": {"$in": u_emails}}, {"$set": {"launch_announced": False}})
            db.email_logs.delete_many({"_id": {"$in": [l["_id"] for l in failed]}})
            st.success(f"Re-queued {len(hr_emails)} HR and {len(u_emails)} employee recipients."); time.sleep(1); st.rerun()

        with r2.popover("♻️ Re-send to everyone"):
            st.caption("Marks recipients as not invited so the next send goes to them again.")
            which = st.radio("Audience", ["HR contacts", "Employees"], key="lc_reset_who")
            comp_r = st.selectbox("Company (HR)", ["All"] + sorted(tenants.keys()), key="lc_reset_comp")
            if st.button("Confirm reset", type="primary", key="lc_reset_go"):
                if which == "HR contacts":
                    qq = {} if comp_r == "All" else {"company_name": comp_r}
                    n = db.hr_contacts.update_many(qq, {"$set": {"launch_announced": False}}).modified_count
                else:
                    n = db.directory.update_many({"role": "EMPLOYEE"}, {"$set": {"launch_announced": False}}).modified_count
                st.success(f"Reset {n} recipients."); time.sleep(1); st.rerun()

        if r3.button("\U0001f5d1️ Clear launch log", width="stretch", key="lc_clear_logs", disabled=not logs):
            db.email_logs.delete_many({"campaign_type": {"$in": [CAMPAIGN_HR, CAMPAIGN_USER]}})
            st.rerun()


def _send_batch(db, recipients, addr_fn, ctx_fn, tmpl, cfg, subject, images, atts, has_video, campaign, log_fn, mark, log_id):
    smtp_cfg = st.secrets["smtp"]
    prog = st.progress(0.0, text="Connecting to mail server…")
    sent, failed = 0, 0
    try:
        with SmtpBatch(smtp_cfg) as sb:
            for i, r in enumerate(recipients):
                to = addr_fn(r)
                ctx = ctx_fn(r)
                try:
                    body = render_email(tmpl, ctx, cfg["festive_greeting"], has_video)
                    sb.send(to, build_message(smtp_cfg, cfg["sender_name"], to, subject, body, images, atts, cfg["support_email"]))
                    mark(r)
                    log_fn(log_id(r), ctx["name"], to, r.get("policy_no", ctx["company_name"]), "DELIVERED", campaign_type=campaign)
                    sent += 1
                except Exception as e:
                    log_fn(log_id(r), ctx["name"], to, r.get("policy_no", ctx["company_name"]), "FAILED", str(e), campaign_type=campaign)
                    failed += 1
                prog.progress((i + 1) / len(recipients), text=f"Sent {sent} · failed {failed} · {i + 1}/{len(recipients)}")
                time.sleep(0.4)  # stay under provider rate limits
    except Exception as e:
        st.error(f"Mail server error: {e}. {sent} sent before the error; the rest stay in the queue.")
    if sent:
        st.success(f"\U0001f389 Sent {sent} invitations." + (f" {failed} failed: see Audit & retry." if failed else ""))
    elif failed:
        st.error(f"All {failed} sends failed. See Audit & retry for the reasons.")
    time.sleep(1.5)
    st.rerun()
