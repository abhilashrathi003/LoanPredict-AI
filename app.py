import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import html as html_lib
from src.pdf_generator import generate_prediction_pdf

from pathlib import Path

from src.database import (
    initialize_database,
    save_prediction,
    get_all_predictions,
    delete_prediction
)


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="LoanPredict AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("models/best_loan_model.joblib")
DATASET_PATH = Path("dataset/loan_dataset.csv")

RESULT_PATH = Path(
    "reports/model_evaluation/model_comparison.csv"
)

CONFUSION_PATH = Path(
    "reports/model_evaluation/confusion_matrix.png"
)

ROC_PATH = Path(
    "reports/model_evaluation/roc_curve.png"
)


# =====================================================
# DATABASE INITIALIZATION
# =====================================================

try:
    initialize_database()
except Exception as error:
    st.error(f"Database initialization failed: {error}")
    st.stop()


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp { background: #eaf0f9; color: #10234f; }
.block-container { max-width: 1600px; padding: 1.1rem 2.1rem 2.5rem; }
header[data-testid="stHeader"] { background: rgba(234,240,249,.96); }
h1,h2,h3,h4 { color: #10234f !important; }
p,label,.stCaption { color: #52658a; }
div[data-testid="stMetric"] {
    background: #f8faff !important; border: 1px solid #cfdaeb !important;
    border-radius: 15px !important; padding: 17px !important;
    box-shadow: 0 5px 16px rgba(26,65,135,.06);
}
div[data-testid="stMetric"] label { color: #617394 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #123b83 !important; font-weight: 800 !important;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background: #f8faff !important; border: 1px solid #d5e0f2 !important;
    border-radius: 9px !important;
}
div[data-baseweb="select"] *, div[data-baseweb="input"] * { color: #14264b !important; }
.stButton > button, .stFormSubmitButton > button {
    background: #1261e8; color: #ffffff !important; border: 0;
    border-radius: 10px; min-height: 43px; font-weight: 700;
}
.stButton > button p, .stButton > button span,
.stFormSubmitButton > button p, .stFormSubmitButton > button span,
.st-key-dashboard_view_all button, .st-key-dashboard_view_all button p, .st-key-dashboard_view_all button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover { background: #084dbf; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #dce6f6; border-radius: 16px;
    box-shadow: 0 4px 15px rgba(30,64,130,.045);
}
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e0e8f5; }
.result-approved { background:#ecfdf5; padding:22px; border-radius:14px; border-left:5px solid #16a34a; }
.result-approved h2 { color:#15803d !important; }
.result-approved p { color:#166534 !important; }
.result-rejected { background:#fff1f2; padding:22px; border-radius:14px; border-left:5px solid #e11d48; }
.result-rejected h2 { color:#be123c !important; }
.result-rejected p { color:#9f1239 !important; }
.top-brand {
    display:flex; align-items:center; gap:12px; color:white;
    font-size:21px; font-weight:850; white-space:nowrap;
}
.top-brand span { color:#60a5fa; }
.top-nav-wrap {
    background:linear-gradient(100deg,#06265e,#073b91);
    border-radius:14px; padding:10px 15px; margin:0 0 20px;
    box-shadow:0 6px 20px rgba(7,45,110,.18);
}
.st-key-main_nav_bar {
    background:linear-gradient(100deg,#06265e 0%,#073b91 60%,#0755bd 100%);
    border:1px solid rgba(147,197,253,.22);
    border-radius:14px;
    padding:12px 18px 8px;
    margin:0 0 24px 0;
    box-shadow:0 7px 20px rgba(7,45,110,.18);
}
.st-key-main_nav_bar [data-testid="stHorizontalBlock"] {
    align-items:center;
}
.st-key-main_nav_bar .top-brand,
.st-key-main_nav_bar .top-brand span {
    color:#ffffff !important;
}
.st-key-main_nav_bar div[role="radiogroup"] {
    display:flex;
    flex-wrap:nowrap;
    justify-content:flex-end;
    gap:5px;
}
.st-key-main_nav_bar div[role="radiogroup"] label {
    background:transparent !important;
    border:1px solid transparent !important;
    border-radius:9px !important;
    padding:8px 11px !important;
    white-space:nowrap;
}
.st-key-main_nav_bar div[role="radiogroup"] label p,
.st-key-main_nav_bar div[role="radiogroup"] label span {
    color:#eaf2ff !important;
    font-size:13px !important;
    font-weight:700 !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:has(input:checked) {
    background:#1671f5 !important;
    border-color:#69a7ff !important;
    box-shadow:0 3px 10px rgba(0,0,0,.15);
}
.st-key-main_nav_bar div[role="radiogroup"] label:has(input:checked) p,
.st-key-main_nav_bar div[role="radiogroup"] label:has(input:checked) span {
    color:#ffffff !important;
}
.st-key-main_nav_bar div[role="radiogroup"] input {
    accent-color:#ffffff;
}
div[role="radiogroup"] { gap:5px; }
div[role="radiogroup"] label {
    background:#e3ebf8; border:1px solid #d1ddef; border-radius:10px;
    padding:8px 12px; color:#17366f !important;
}
div[role="radiogroup"] label p { color:#17366f !important; font-weight:700; }
div[role="radiogroup"] label:has(input:checked) {
    background:#1261e8; border-color:#1261e8;
}
div[role="radiogroup"] label:has(input:checked) p { color:#ffffff !important; }
div[role="radiogroup"] input { accent-color:#ffffff; }
.dash-heading { color:#10234f; font-size:27px; font-weight:850; margin:2px 0 2px; }
.dash-sub { color:#53688e; font-size:14px; margin-bottom:12px; }
.dash-hero {
    min-height:190px; border-radius:20px; padding:28px 34px; margin:10px 0 16px;
    color:white; overflow:hidden; position:relative;
    background:
      radial-gradient(circle at 78% 45%,rgba(69,163,255,.55),transparent 25%),
      linear-gradient(110deg,#06317d 0%,#0755c9 58%,#0d70e8 100%);
    box-shadow:0 10px 25px rgba(12,67,150,.17);
}
.dash-hero:after {
    content:"🏦"; position:absolute; right:7%; top:22px;
    font-size:118px; opacity:.9; filter:drop-shadow(0 8px 10px rgba(0,0,0,.18));
}
.dash-hero h2 { color:#fff !important; font-size:30px; margin:9px 0 5px; }
.dash-hero p { color:#e6f0ff !important; max-width:610px; margin-bottom:12px; }
.dash-pill {
    display:inline-block; border:1px solid rgba(255,255,255,.3);
    background:rgba(255,255,255,.13); border-radius:30px;
    padding:5px 11px; font-size:11px; font-weight:800; letter-spacing:.05em;
}
.dash-kpi {
    background:#f8faff; border:1px solid #cfdaeb; border-radius:15px;
    padding:15px 17px; min-height:94px; box-shadow:0 4px 14px rgba(23,65,140,.06);
}
.dash-kpi-left { display:flex; align-items:center; gap:12px; min-width:0; }
.dash-kpi-ico {
    width:42px; height:42px; flex:0 0 42px; border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    font-size:17px; font-weight:850; line-height:1;
}
.dash-kpi-label { color:#52658a; font-size:12px; font-weight:750; line-height:1.25; }
.dash-kpi-value { color:#123b83; font-size:27px; font-weight:850; line-height:1.15; margin-top:2px; }
.dash-kpi-blue { border-color:#c9ddff; background:linear-gradient(120deg,#fff,#eff6ff); }
.dash-kpi-green { border-color:#b8efd5; background:linear-gradient(120deg,#fff,#ecfdf5); }
.dash-kpi-red { border-color:#fecdd3; background:linear-gradient(120deg,#fff,#fff1f2); }
.dash-kpi-purple { border-color:#ddd6fe; background:linear-gradient(120deg,#fff,#f5f3ff); }
.dash-section-title { color:#10234f; font-size:18px; font-weight:850; margin:16px 0 3px; }
.dash-section-caption { color:#617394; font-size:12px; margin-bottom:9px; }
.footer { text-align:center; color:#8190a6; padding:20px; font-size:13px; }

.loan-footer {
    margin-top:24px;
    padding:18px 20px 14px;
    text-align:center;
    color:#e8f1ff;
    border-radius:16px 16px 0 0;
    background:linear-gradient(110deg,#06265e 0%,#073b91 58%,#0755bd 100%);
    box-shadow:0 -5px 20px rgba(7,45,110,.08);
}
.loan-footer-brand {
    color:#ffffff;
    font-size:20px;
    font-weight:850;
    letter-spacing:.01em;
}
.loan-footer-brand span { color:#7db8ff; }
.loan-footer-title {
    color:#e6efff;
    font-size:13.5px;
    font-weight:650;
    margin-top:4px;
}
.loan-footer-copy {
    color:#9dbbe8;
    font-size:11px;
    margin-top:11px;
    padding-top:9px;
    border-top:1px solid rgba(219,234,254,.2);
}


/* Remove Streamlit's reserved top chrome and the pale strip above navigation */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    visibility: hidden !important;
}

/* Pull the app content up so the blue navbar is the first visible element */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

[data-testid="stMainBlockContainer"],
.main .block-container {
    padding-top: 0.65rem !important;
    margin-top: 0 !important;
}

/* Remove any translucent wrapper behind the custom navigation */
.st-key-main_nav_bar {
    margin-top: 0 !important;
    background: linear-gradient(100deg, #06265e 0%, #073b91 60%, #0755bd 100%) !important;
    border: 1px solid rgba(147,197,253,.22) !important;
    border-radius: 14px !important;
    box-shadow: 0 7px 20px rgba(7,45,110,.18) !important;
}

/* Keep navigation labels legible on the blue bar */
.st-key-main_nav_bar [data-testid="stMarkdownContainer"],
.st-key-main_nav_bar [data-testid="stMarkdownContainer"] p,
.st-key-main_nav_bar label,
.st-key-main_nav_bar label p {
    color: #ffffff !important;
}


/* ---------- Premium fintech refinement layer ---------- */
:root {
    --lp-navy: #10234f;
    --lp-blue: #1261e8;
    --lp-muted: #617394;
    --lp-border: #d8e3f3;
    --lp-surface: #fbfdff;
}
html, body, [class*="css"] {
    font-family: Inter, "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp { background: #eaf0f9 !important; }
.block-container {
    max-width: 1600px !important;
    padding: .7rem 2.1rem 1.8rem !important;
}
.st-key-main_nav_bar {
    padding: 10px 18px 7px !important;
    margin-bottom: 20px !important;
    border-radius: 14px !important;
    background: linear-gradient(105deg,#06265e 0%,#073b91 58%,#0755bd 100%) !important;
    border: 1px solid rgba(147,197,253,.25) !important;
    box-shadow: 0 8px 24px rgba(7,45,110,.16) !important;
}
.st-key-main_nav_bar div[role="radiogroup"] {
    display:flex !important;
    align-items:center !important;
    justify-content:flex-end !important;
    flex-wrap:wrap !important;
    gap:5px !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label {
    display:flex !important;
    align-items:center !important;
    background:transparent !important;
    border:1px solid transparent !important;
    border-radius:10px !important;
    padding:8px 11px !important;
    margin:0 !important;
    transition:background .18s ease,border-color .18s ease,transform .18s ease,box-shadow .18s ease !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:hover {
    background:rgba(255,255,255,.12) !important;
    border-color:rgba(191,219,254,.25) !important;
    transform:translateY(-1px);
}
.st-key-main_nav_bar div[role="radiogroup"] label p {
    color:#eaf2ff !important;
    font-size:13px !important;
    font-weight:650 !important;
    margin:0 !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(135deg,#1976ff,#0d5cdb) !important;
    border-color:rgba(191,219,254,.65) !important;
    box-shadow:0 4px 12px rgba(0,0,0,.16) !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:has(input:checked) p {
    color:#fff !important;
}
.st-key-main_nav_bar div[role="radiogroup"] input {
    display:none !important;
}
.top-brand { font-size:20px !important; letter-spacing:-.35px; }
.dash-heading { letter-spacing:-.65px; }
.dash-hero {
    min-height:164px !important;
    padding:24px 30px !important;
    margin:8px 0 15px !important;
    border:1px solid rgba(147,197,253,.28);
    box-shadow:0 9px 23px rgba(12,67,150,.14) !important;
    transition:box-shadow .2s ease,transform .2s ease;
}
.dash-hero:hover { box-shadow:0 13px 28px rgba(12,67,150,.18) !important; }
.dash-hero h2 { font-size:28px !important; letter-spacing:-.45px; }
.dash-hero:after { font-size:108px !important; top:18px !important; }
.dash-kpi {
    min-height:100px !important;
    height:100% !important;
    padding:16px 18px !important;
    border-radius:16px !important;
    box-shadow:0 5px 16px rgba(23,65,140,.055) !important;
    transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;
}
.dash-kpi:hover {
    transform:translateY(-3px);
    box-shadow:0 10px 22px rgba(23,65,140,.11) !important;
}
.dash-kpi-label {
    display:flex; align-items:center; gap:8px;
    font-size:12px !important; letter-spacing:.1px;
}
.dash-kpi-value { font-size:29px !important; line-height:1.2; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border:1px solid var(--lp-border) !important;
    border-radius:16px !important;
    box-shadow:0 5px 17px rgba(30,64,130,.045) !important;
    transition:box-shadow .18s ease,border-color .18s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color:#bfd2ee !important;
    box-shadow:0 8px 22px rgba(30,64,130,.075) !important;
}
.dash-section-title { margin-top:14px !important; }
.loan-footer {
    margin-top:22px !important;
    padding:16px 18px 13px !important;
    border-radius:16px 16px 0 0 !important;
}
.loan-footer-brand { font-size:20px !important; }
.loan-footer-title { margin-top:3px !important; }
.loan-footer-copy { margin-top:10px !important; padding-top:8px !important; }
.stButton > button, .stFormSubmitButton > button {
    transition:transform .16s ease,box-shadow .16s ease,background .16s ease !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform:translateY(-1px);
    box-shadow:0 5px 13px rgba(18,97,232,.2) !important;
}
@media (max-width: 900px) {
    .block-container { padding-left:1rem !important; padding-right:1rem !important; }
    .st-key-main_nav_bar div[role="radiogroup"] { justify-content:flex-start !important; }
    .dash-hero { padding:20px 22px !important; }
    .dash-hero:after { opacity:.22 !important; right:3% !important; }
}
@media (max-width: 600px) {
    .top-brand { font-size:17px !important; }
    .st-key-main_nav_bar { padding:9px 10px 7px !important; }
    .st-key-main_nav_bar div[role="radiogroup"] label { padding:7px 8px !important; }
    .st-key-main_nav_bar div[role="radiogroup"] label p { font-size:11px !important; }
    .dash-heading { font-size:23px !important; }
    .dash-hero h2 { font-size:22px !important; }
    .dash-hero p { font-size:13px !important; }
    .loan-footer { padding:15px 14px 12px !important; }
}


.recent-predictions-table th {
    color:#17366f !important;
    font-weight:750;
    border-bottom:1px solid #d5e0f0;
    white-space:nowrap;
}
.recent-predictions-table td {
    border-bottom:1px solid #e4ebf5;
    vertical-align:middle;
    padding:11px !important;
}
.recent-predictions-table tbody tr { transition:background .15s ease; }
.recent-predictions-table tbody tr:hover { background:#edf4ff !important; }
.recent-predictions-table tbody tr:last-child td { border-bottom:0; }


/* =========================================================
   LOANPREDICT AI — PREMIUM NAVIGATION BAR ONLY
   ========================================================= */
.st-key-main_nav_bar {
    min-height: 76px !important;
    display:flex;
    align-items:center;
    padding:10px 18px !important;
    margin:0 0 23px !important;
    border-radius:16px !important;
    background:linear-gradient(108deg,#061b49 0%,#082f78 48%,#075bd2 100%) !important;
    border:1px solid rgba(147,197,253,.24) !important;
    box-shadow:0 9px 25px rgba(7,45,110,.20), inset 0 1px 0 rgba(255,255,255,.08) !important;
}
.st-key-main_nav_bar [data-testid="stHorizontalBlock"] {
    align-items:center !important;
    gap:10px !important;
}
.st-key-main_nav_bar [data-testid="column"] {
    display:flex;
    align-items:center;
    min-width:0;
}
.st-key-main_nav_bar .top-brand {
    display:flex !important;
    align-items:center !important;
    gap:11px !important;
    min-height:48px;
    padding-right:18px;
    border-right:1px solid rgba(191,219,254,.28);
    color:#fff !important;
    white-space:nowrap;
}
.st-key-main_nav_bar .top-brand-mark {
    display:flex;
    align-items:center;
    justify-content:center;
    width:38px;
    height:38px;
    flex:0 0 38px;
    border-radius:11px;
    color:#dcecff !important;
    background:linear-gradient(145deg,rgba(255,255,255,.18),rgba(255,255,255,.06));
    border:1px solid rgba(191,219,254,.28);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.14);
}
.st-key-main_nav_bar .top-brand-mark svg {
    width:25px;
    height:25px;
}
.st-key-main_nav_bar .top-brand-name {
    color:#fff !important;
    font-size:20px;
    font-weight:800;
    letter-spacing:-.55px;
}
.st-key-main_nav_bar .top-brand-name b {
    color:#63b3ff !important;
    font-weight:850;
}
.st-key-main_nav_bar div[role="radiogroup"] {
    display:flex !important;
    align-items:center !important;
    justify-content:flex-end !important;
    flex-wrap:nowrap !important;
    gap:5px !important;
    width:100%;
}
.st-key-main_nav_bar div[role="radiogroup"] label {
    display:flex !important;
    align-items:center !important;
    position:relative;
    min-height:43px;
    padding:9px 12px 9px 36px !important;
    margin:0 !important;
    border:1px solid transparent !important;
    border-radius:11px !important;
    background:transparent !important;
    white-space:nowrap !important;
    cursor:pointer;
    opacity:.86;
    transition:background .2s ease,border-color .2s ease,box-shadow .2s ease,transform .2s ease,opacity .2s ease !important;
}

.st-key-main_nav_bar div[role="radiogroup"] input[type="radio"] {
    position:absolute !important;
    opacity:0 !important;
    width:1px !important;
    height:1px !important;
    margin:0 !important;
    pointer-events:none !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label > div:first-child {
    display:none !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label {
    padding:10px 13px 10px 40px !important;
    position:relative !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label::before {
    content:"" !important;
    display:block !important;
    position:absolute !important;
    left:13px !important;
    top:50% !important;
    width:19px !important;
    height:19px !important;
    transform:translateY(-50%) !important;
    background-repeat:no-repeat !important;
    background-position:center !important;
    background-size:19px 19px !important;
    opacity:.96 !important;
    pointer-events:none !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:nth-child(1)::before,
.st-key-main_nav_bar div[role="radiogroup"] > div:nth-child(1) label::before {
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f8fbff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 10 9-7 9 7'/%3E%3Cpath d='M5 9.5V21h14V9.5M9 21v-7h6v7'/%3E%3C/svg%3E") !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:nth-child(2)::before,
.st-key-main_nav_bar div[role="radiogroup"] > div:nth-child(2) label::before {
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f8fbff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpath d='M14 2v6h6M8 13h8M8 17h5'/%3E%3C/svg%3E") !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:nth-child(3)::before,
.st-key-main_nav_bar div[role="radiogroup"] > div:nth-child(3) label::before {
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f8fbff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 7v5l3 2'/%3E%3C/svg%3E") !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:nth-child(4)::before,
.st-key-main_nav_bar div[role="radiogroup"] > div:nth-child(4) label::before {
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f8fbff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 3v18h18'/%3E%3Cpath d='M8 16v-5m5 5V5m5 11V9'/%3E%3C/svg%3E") !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:nth-child(5)::before,
.st-key-main_nav_bar div[role="radiogroup"] > div:nth-child(5) label::before {
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f8fbff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='8' ry='3'/%3E%3Cpath d='M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3'/%3E%3C/svg%3E") !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:has(input:checked)::before {
    opacity:1 !important;
}

.st-key-main_nav_bar div[role="radiogroup"] label:hover {
    opacity:1;
    transform:translateY(-1px);
    background:rgba(219,234,254,.11) !important;
    border-color:rgba(191,219,254,.19) !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:has(input:checked) {
    opacity:1 !important;
    background:linear-gradient(135deg,#1978ff 0%,#075be0 100%) !important;
    border-color:rgba(191,219,254,.48) !important;
    box-shadow:0 4px 14px rgba(0,0,0,.18),0 0 0 1px rgba(96,165,250,.12) !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label p,
.st-key-main_nav_bar div[role="radiogroup"] label span {
    color:rgba(239,246,255,.91) !important;
    font-size:12.5px !important;
    line-height:1.2 !important;
    font-weight:650 !important;
    margin:0 !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:has(input:checked) p,
.st-key-main_nav_bar div[role="radiogroup"] label:has(input:checked) span {
    color:#fff !important;
    font-weight:750 !important;
}
.st-key-main_nav_bar div[role="radiogroup"] input {
    display:none !important;
}
.st-key-main_nav_bar .st-key-mobile_nav_menu {
    display:none !important;
}
.st-key-main_nav_bar button[kind="secondary"] {
    color:#fff !important;
    background:rgba(255,255,255,.11) !important;
    border:1px solid rgba(191,219,254,.3) !important;
    border-radius:11px !important;
    min-height:42px !important;
}
.st-key-main_nav_bar button[kind="secondary"]:hover {
    background:rgba(255,255,255,.2) !important;
    border-color:rgba(219,234,254,.55) !important;
}
@media (max-width: 1150px) {
    .st-key-main_nav_bar .top-brand-name { font-size:17px !important; }
    .st-key-main_nav_bar div[role="radiogroup"] label {
        padding-left:31px !important;
        padding-right:8px !important;
    }
    .st-key-main_nav_bar div[role="radiogroup"] label::before { left:9px; }
    .st-key-main_nav_bar div[role="radiogroup"] label p { font-size:11px !important; }
}
@media (max-width: 850px) {
    .st-key-main_nav_bar { min-height:66px !important; padding:9px 12px !important; }
    .st-key-main_nav_bar [data-testid="column"]:nth-child(2) { display:none !important; }
    .st-key-main_nav_bar .st-key-mobile_nav_menu { display:block !important; }
    .st-key-main_nav_bar .top-brand { border-right:0 !important; padding-right:0 !important; }
    .st-key-main_nav_bar [data-testid="column"]:last-child { margin-left:auto; }
}
@media (max-width: 480px) {
    .st-key-main_nav_bar { padding:8px 10px !important; border-radius:13px !important; }
    .st-key-main_nav_bar .top-brand-mark { width:34px; height:34px; flex-basis:34px; }
    .st-key-main_nav_bar .top-brand-name { font-size:16px !important; }
}


/* ---------- Navbar sizing, spacing and radio-indicator cleanup ---------- */
.st-key-main_nav_bar {
    min-height: 72px !important;
    padding: 8px 16px !important;
    margin-bottom: 20px !important;
    border-radius: 15px !important;
}
.st-key-main_nav_bar [data-testid="stHorizontalBlock"] {
    gap: 8px !important;
    align-items: center !important;
}
.st-key-main_nav_bar [data-testid="column"]:first-child {
    flex: 0 0 300px !important;
    width: 300px !important;
    max-width: 300px !important;
}
.st-key-main_nav_bar [data-testid="column"]:nth-child(2) {
    flex: 1 1 auto !important;
    min-width: 0 !important;
}
.st-key-main_nav_bar [data-testid="column"]:last-child {
    flex: 0 0 42px !important;
    width: 42px !important;
}
.st-key-main_nav_bar .top-brand {
    min-height: 46px !important;
    gap: 10px !important;
    padding-right: 16px !important;
}
.st-key-main_nav_bar .top-brand-mark {
    width: 36px !important;
    height: 36px !important;
    flex-basis: 36px !important;
}
.st-key-main_nav_bar .top-brand-mark svg {
    width: 23px !important;
    height: 23px !important;
}
.st-key-main_nav_bar .top-brand-name {
    font-size: 19px !important;
    letter-spacing: -.45px !important;
}
.st-key-main_nav_bar div[role="radiogroup"] {
    display: flex !important;
    width: 100% !important;
    justify-content: space-evenly !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    gap: 6px !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label {
    min-height: 44px !important;
    height: 44px !important;
    padding: 9px 12px 9px 38px !important;
    margin: 0 !important;
    flex: 0 1 auto !important;
    justify-content: center !important;
    border-radius: 10px !important;
}

.st-key-main_nav_bar [data-baseweb="radio"] > div:first-child,
.st-key-main_nav_bar label[data-baseweb="radio"] > div:first-child,
.st-key-main_nav_bar div[role="radiogroup"] label > div:first-child,
.st-key-main_nav_bar div[role="radiogroup"] label > span:first-child,
.st-key-main_nav_bar div[role="radiogroup"] label [data-testid="stRadioOption"] > div:first-child,
.st-key-main_nav_bar div[role="radiogroup"] label input[type="radio"] + div,
.st-key-main_nav_bar div[role="radiogroup"] label input[type="radio"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    min-width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label::before {
    left: 12px !important;
    width: 18px !important;
    height: 18px !important;
    background-size: 18px 18px !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label p,
.st-key-main_nav_bar div[role="radiogroup"] label span {
    font-size: 12.5px !important;
    line-height: 1.15 !important;
    white-space: nowrap !important;
}
.st-key-main_nav_bar div[role="radiogroup"] label:hover {
    transform: translateY(-1px) !important;
}
.st-key-main_nav_bar .st-key-mobile_nav_menu {
    display: none !important;
}

@media (max-width: 1250px) {
    .st-key-main_nav_bar [data-testid="column"]:first-child {
        flex-basis: 235px !important;
        width: 235px !important;
        max-width: 235px !important;
    }
    .st-key-main_nav_bar .top-brand-name { font-size: 17px !important; }
    .st-key-main_nav_bar div[role="radiogroup"] { gap: 3px !important; }
    .st-key-main_nav_bar div[role="radiogroup"] label {
        padding-left: 33px !important;
        padding-right: 8px !important;
    }
    .st-key-main_nav_bar div[role="radiogroup"] label::before {
        left: 9px !important;
    }
    .st-key-main_nav_bar div[role="radiogroup"] label p {
        font-size: 11px !important;
    }
}
@media (max-width: 850px) {
    .st-key-main_nav_bar {
        min-height: 64px !important;
        padding: 8px 12px !important;
    }
    .st-key-main_nav_bar [data-testid="column"]:first-child {
        flex: 1 1 auto !important;
        width: auto !important;
        max-width: none !important;
    }
    .st-key-main_nav_bar [data-testid="column"]:nth-child(2) {
        display: none !important;
    }
    .st-key-main_nav_bar [data-testid="column"]:last-child {
        flex: 0 0 42px !important;
        width: 42px !important;
        margin-left: auto !important;
    }
    .st-key-main_nav_bar .top-brand {
        border-right: 0 !important;
        padding-right: 0 !important;
    }
    .st-key-main_nav_bar .st-key-mobile_nav_menu {
        display: block !important;
    }
}
@media (max-width: 480px) {
    .st-key-main_nav_bar .top-brand-name { font-size: 16px !important; }
    .st-key-main_nav_bar .top-brand-mark {
        width: 34px !important;
        height: 34px !important;
        flex-basis: 34px !important;
    }
}


/* Navbar desktop buttons */
.st-key-main_nav_bar [class*="st-key-nav_"] button {
    min-height: 44px !important;
    height: 44px !important;
    width: 100% !important;
    padding: 0 11px 0 36px !important;
    border-radius: 11px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: #edf5ff !important;
    box-shadow: none !important;
    position: relative !important;
    white-space: nowrap !important;
    font-size: 12.5px !important;
    font-weight: 650 !important;
    transition: background .18s ease, border-color .18s ease, transform .18s ease !important;
}
.st-key-main_nav_bar [class*="st-key-nav_"] button:hover {
    background: rgba(219,234,254,.13) !important;
    border-color: rgba(191,219,254,.22) !important;
    color: #fff !important;
    transform: translateY(-1px);
}
.st-key-main_nav_bar [class*="st-key-nav_"] button[kind="primary"] {
    background: linear-gradient(135deg,#1978ff 0%,#075be0 100%) !important;
    border-color: rgba(191,219,254,.55) !important;
    color: #fff !important;
    box-shadow: 0 4px 13px rgba(0,0,0,.17) !important;
}
.st-key-main_nav_bar [class*="st-key-nav_"] button p {
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    margin: 0 !important;
    white-space: nowrap !important;
}
.st-key-main_nav_bar [class*="st-key-nav_"] button::before {
    content: "";
    position: absolute;
    left: 11px;
    top: 50%;
    width: 18px;
    height: 18px;
    transform: translateY(-50%);
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
    pointer-events: none;
}
.st-key-nav_loan_prediction button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 10.5 12 3l9 7.5'/%3E%3Cpath d='M5.5 9.5V21h13V9.5M9 21v-7h6v7'/%3E%3C/svg%3E");
}
.st-key-nav_dashboard button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpath d='M14 2v6h6M8 13h8M8 17h5'/%3E%3C/svg%3E");
}
.st-key-nav_prediction_history button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 7v5l3 2'/%3E%3C/svg%3E");
}
.st-key-nav_model_performance button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 3v18h18'/%3E%3Cpath d='M8 16v-5m5 5V5m5 11V9'/%3E%3C/svg%3E");
}
.st-key-nav_dataset_analytics button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='8' ry='3'/%3E%3Cpath d='M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3'/%3E%3C/svg%3E");
}
@media (max-width: 1250px) {
    .st-key-main_nav_bar [class*="st-key-nav_"] button {
        padding-left: 31px !important;
        padding-right: 7px !important;
        font-size: 11px !important;
    }
    .st-key-main_nav_bar [class*="st-key-nav_"] button::before {
        left: 8px;
        width: 17px;
        height: 17px;
    }
}
@media (max-width: 850px) {
    .st-key-main_nav_bar [data-testid="column"]:nth-child(2) { display:none !important; }
    .st-key-main_nav_bar [data-testid="column"]:last-child { margin-left:auto !important; }
}



/* Loan Prediction page — consolidated styles */
:root {
    --lp-navy: #102A5E;
    --lp-blue: #0866E8;
    --lp-border: #D5E3FA;
    --lp-bg: #EDF3FC;
}
.lp-pred-hero {
    position:relative; overflow:hidden; min-height:166px; padding:25px 34px;
    margin:4px 0 18px; border-radius:19px; color:#fff;
    background:
      radial-gradient(circle at 73% 25%,rgba(25,133,255,.42),transparent 25%),
      linear-gradient(110deg,#061b50 0%,#073b9b 52%,#075ee0 100%);
    border:1px solid rgba(105,165,255,.38);
    box-shadow:0 10px 25px rgba(9,54,132,.18);
}
.lp-pred-hero:before {
    content:""; position:absolute; right:18%; bottom:-110px; width:430px; height:230px;
    border-radius:50%; border:1px solid rgba(93,164,255,.19);
    box-shadow:0 0 0 28px rgba(93,164,255,.045),0 0 0 58px rgba(93,164,255,.035);
}
.lp-pred-hero-copy { position:relative; z-index:2; max-width:620px; }
.lp-pred-eyebrow {
    display:inline-flex; align-items:center; gap:8px; padding:5px 13px;
    border-radius:30px; color:#fff; background:rgba(32,112,255,.55);
    border:1px solid rgba(191,219,254,.28); font-size:11px; font-weight:800;
    letter-spacing:.045em;
}
.lp-pred-hero h1 { color:#fff !important; font-size:32px !important; line-height:1.15 !important; margin:8px 0 5px !important; }
.lp-pred-hero p { color:#e7f1ff !important; font-size:15px; line-height:1.5; margin:0 !important; max-width:580px; }
.lp-pred-hero-art { position:absolute; right:4%; top:8px; width:350px; height:150px; z-index:1; opacity:.98; }

.lp-pred-panel {
    background:#fff; border:1px solid var(--lp-border); border-radius:19px;
    padding:18px; box-shadow:0 5px 17px rgba(30,91,175,.075);
}
.lp-pred-panel-title { display:flex; align-items:center; gap:14px; margin:0 0 14px; padding:0 0 13px; border-bottom:1px solid #e2ebfa; }
.lp-pred-title-icon {
    width:48px; height:48px; flex:0 0 48px; border-radius:15px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(145deg,#eaf2ff,#dceaff); color:#0b48b7;
}
.lp-pred-title-icon svg { width:27px; height:27px; flex-shrink:0; }
.lp-pred-panel-title h2 { color:var(--lp-navy) !important; font-size:23px !important; line-height:1.15; margin:0 !important; font-weight:850; }
.lp-pred-panel-title p { color:#52688d !important; font-size:12px; margin:3px 0 0 !important; }
.lp-pred-section-head {
    display:flex; align-items:center; gap:10px; margin:14px 0 10px;
    padding:9px 14px; border-radius:10px 10px 0 0;
    background:linear-gradient(90deg,#dce9fc,#eaf2ff);
    color:var(--lp-navy); font-weight:800; font-size:14px; line-height:1.3;
    border-bottom:1px solid #cbdcf8;
}
.lp-pred-section-head svg { width:18px; height:18px; flex-shrink:0; color:var(--lp-blue); }

.lp-pred-result { border-radius:15px; padding:18px; margin-bottom:12px; border:1px solid #b9e9d0; background:linear-gradient(145deg,#f0fdf6,#fff); }
.lp-pred-result.rejected { border-color:#fecdd3; background:linear-gradient(145deg,#fff1f2,#fff); }
.lp-pred-result h3 { color:#087443 !important; margin:0 0 5px !important; font-size:21px !important; }
.lp-pred-result.rejected h3 { color:#be123c !important; }
.lp-pred-result p { color:#365b4a !important; margin:0 !important; font-size:13px; }
.lp-pred-result.rejected p { color:#8f3449 !important; }
.lp-pred-score { border:1px solid #d9e6f8; border-radius:12px; background:#fff; padding:13px; margin:8px 0; }
.lp-pred-score-label { color:#53698e; font-size:12px; font-weight:700; }
.lp-pred-score-value { color:#123b83; font-size:23px; font-weight:850; }

.st-key-loan_prediction_form_fields [data-testid="stWidgetLabel"],
.st-key-loan_prediction_form_fields [data-testid="stWidgetLabel"] *,
.st-key-loan_prediction_form_fields [data-testid="stSelectbox"] label,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] label,
.st-key-loan_prediction_form_fields [data-testid="stSelectbox"] label *,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] label * {
    color:var(--lp-navy) !important; -webkit-text-fill-color:var(--lp-navy) !important;
    font-size:13px !important; font-weight:750 !important; line-height:1.35 !important; opacity:1 !important;
}
.st-key-loan_prediction_form_fields [data-testid="stSelectbox"] [data-baseweb="select"],
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"] {
    background:#fff !important; border:1px solid var(--lp-border) !important;
    border-radius:10px !important; min-height:42px !important;
    box-shadow:0 1px 2px rgba(16,42,94,.035) !important;
    transition:border-color .16s ease,box-shadow .16s ease !important;
}
.st-key-loan_prediction_form_fields [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"] > div {
    background:#fff !important; border-color:var(--lp-border) !important; border-radius:10px !important;
}
.st-key-loan_prediction_form_fields [data-testid="stSelectbox"] [data-baseweb="select"]:hover,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"]:hover { border-color:#9ABEF2 !important; }
.st-key-loan_prediction_form_fields [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border-color:var(--lp-blue) !important; box-shadow:0 0 0 3px rgba(8,102,232,.14) !important;
}
.st-key-loan_prediction_form_fields [data-testid="stSelectbox"] *,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] input {
    color:var(--lp-navy) !important; -webkit-text-fill-color:var(--lp-navy) !important;
    caret-color:var(--lp-blue) !important; font-size:14px !important; font-weight:600 !important;
    background-color:transparent !important;
}
.st-key-loan_prediction_form_fields [data-testid="stSelectbox"] svg { color:var(--lp-navy) !important; fill:var(--lp-navy) !important; opacity:1 !important; }
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] button {
    background:#fff !important; color:var(--lp-blue) !important;
    border-left:1px solid var(--lp-border) !important; min-width:36px !important; min-height:40px !important;
}
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] button:hover { background:#EFF6FF !important; }
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] button svg { color:var(--lp-blue) !important; fill:var(--lp-blue) !important; }

div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="menu"],
div[data-baseweb="menu"] > div,
div[data-baseweb="menu"] ul,
ul[role="listbox"],
ul[role="listbox"] li,
li[role="option"] {
    background:#ffffff !important;
    background-color:#ffffff !important;
    color:#102a5e !important;
    -webkit-text-fill-color:#102a5e !important;
}
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
ul[role="listbox"] {
    border:1px solid #cfe0fa !important;
    border-radius:12px !important;
    box-shadow:0 10px 24px rgba(16,42,94,.14) !important;
    max-height:220px !important;
    overflow-y:auto !important;
}
div[data-baseweb="popover"] li *,
div[data-baseweb="menu"] li *,
ul[role="listbox"] li *,
li[role="option"] * {
    color:#102a5e !important;
    -webkit-text-fill-color:#102a5e !important;
    font-size:13px !important;
    font-weight:600 !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="menu"] li:hover,
ul[role="listbox"] li:hover,
li[role="option"]:hover {
    background:#edf4ff !important;
    background-color:#edf4ff !important;
}
li[role="option"][aria-selected="true"],
ul[role="listbox"] li[aria-selected="true"] {
    background:#dceaff !important;
    background-color:#dceaff !important;
    font-weight:750 !important;
}

.st-key-loan_prediction_form_fields [data-testid="stFormSubmitButton"] button {
    min-height:50px !important; border-radius:12px !important;
    background:linear-gradient(100deg,#0866E8 0%,#073ED0 100%) !important;
    color:#fff !important; font-size:15px !important; font-weight:800 !important;
    box-shadow:0 6px 15px rgba(8,82,218,.19) !important;
    transition:transform .18s ease,box-shadow .18s ease !important;
}
.st-key-loan_prediction_form_fields [data-testid="stFormSubmitButton"] button:hover {
    transform:translateY(-1px); box-shadow:0 9px 19px rgba(8,82,218,.25) !important;
}
.st-key-loan_prediction_form_fields [data-testid="stFormSubmitButton"] button * {
    color:#fff !important; -webkit-text-fill-color:#fff !important; fill:#fff !important;
}

.st-key-prediction_result_card {
    align-self:flex-start !important; width:100% !important; height:auto !important;
    min-height:0 !important; max-height:none !important; overflow:visible !important;
    padding:16px !important; background:linear-gradient(145deg,#fff,#f7faff) !important;
    border:1px solid var(--lp-border) !important; border-radius:19px !important;
    box-shadow:0 5px 17px rgba(30,91,175,.07) !important;
}
.st-key-prediction_result_card [data-testid="stVerticalBlock"] { gap:.65rem !important; }
.st-key-prediction_result_card .lp-pred-panel-title { margin:0 0 2px !important; padding-bottom:12px !important; }
.lp-pred-empty {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:9px; min-height:255px; height:auto; padding:18px 16px;
    text-align:center; background:linear-gradient(145deg,#fff,#f9fbff);
    border:1px solid #e1ebfb; border-radius:15px; overflow:visible;
}
.lp-pred-empty svg { display:block; width:150px; height:120px; max-width:75%; margin:0 auto 2px; flex:0 0 auto; }
.lp-pred-empty h3 { color:var(--lp-navy) !important; font-size:21px !important; line-height:1.25 !important; margin:0 !important; }
.lp-pred-empty p { color:#52688d !important; font-size:14px; line-height:1.5; max-width:350px; margin:0 auto !important; white-space:normal !important; }
.lp-pred-about {
    display:flex; align-items:flex-start; gap:14px;
    padding:18px 20px 26px; margin:14px 0 6px;
    border-radius:13px; background:#e6efff;
    border:1px solid #d8e6ff; color:#18376f;
    box-sizing:border-box;
}
.lp-pred-about svg { flex:0 0 23px; width:23px; height:23px; color:var(--lp-blue); }
.lp-pred-about > div { min-width:0; padding-top:1px; }
.lp-pred-about strong { display:block; color:var(--lp-navy); margin:0 0 7px; font-size:15px; line-height:1.35; }
.lp-pred-about p { margin:0; color:#38527e; font-size:13.5px; line-height:1.6; }

@media(max-width:1050px) {
    .lp-pred-hero-art { right:0; opacity:.42; }
    .lp-pred-hero-copy { max-width:75%; }
}
@media(max-width:900px) {
    .st-key-loan_prediction_form_fields [data-testid="stHorizontalBlock"] { gap:.75rem !important; }
    .lp-pred-empty { min-height:235px; }
}
@media(max-width:760px) {
    .lp-pred-hero { min-height:150px; padding:20px; }
    .lp-pred-hero h1 { font-size:26px !important; }
    .lp-pred-hero p { font-size:13px; max-width:100%; }
    .lp-pred-hero-art { width:230px; right:-25px; opacity:.2; }
    .lp-pred-panel { padding:13px; }
    .lp-pred-panel-title h2 { font-size:20px !important; }
    .st-key-prediction_result_card { padding:13px !important; }
    .lp-pred-empty { min-height:220px; padding:15px 12px; }
    .lp-pred-empty svg { width:135px; height:108px; }
}

.st-key-loan_prediction_form_fields [data-testid="stNumberInput"],
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] > div,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] > div > div,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"] {
    pointer-events:auto !important;
}
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] > div,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] > div > div {
    border:none !important;
    border-bottom:none !important;
    box-shadow:none !important;
    background:transparent !important;
}
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"] {
    background:#ffffff !important;
    border:1px solid #b8c7de !important;
    border-width:1px !important;
    border-style:solid !important;
    border-color:#b8c7de !important;
    border-radius:10px !important;
    min-height:42px !important;
    box-shadow:none !important;
    overflow:hidden !important;
    transition:border-color .16s ease,box-shadow .16s ease !important;
}
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"]::before,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"]::after,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"] > div::before,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"] > div::after {
    display:none !important;
    border:none !important;
    box-shadow:none !important;
}
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"]:hover {
    border-color:#8fb3e8 !important;
    box-shadow:none !important;
}
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border-color:var(--lp-blue) !important;
    box-shadow:0 0 0 3px rgba(8,102,232,.14) !important;
}
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"] > div {
    background:#ffffff !important;
    border:none !important;
    border-width:0 !important;
    border-bottom:none !important;
    box-shadow:none !important;
}
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] input,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] input[type="number"] {
    color:var(--lp-navy) !important;
    -webkit-text-fill-color:var(--lp-navy) !important;
    caret-color:var(--lp-blue) !important;
    font-weight:600 !important;
    background:#ffffff !important;
    border:none !important;
    border-width:0 !important;
    border-bottom:none !important;
    outline:none !important;
    box-shadow:none !important;
    pointer-events:auto !important;
}
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] button,
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"],
.st-key-loan_prediction_form_fields [data-testid="stNumberInput"] [data-baseweb="input"] button {
    display:none !important;
    visibility:hidden !important;
    width:0 !important;
    min-width:0 !important;
    padding:0 !important;
    border:0 !important;
    pointer-events:none !important;
}
.st-key-loan_prediction_form_fields [data-testid="stHorizontalBlock"]:has([class*="st-key-lp_dec_"]) {
    display:flex !important;
    flex-wrap:nowrap !important;
    gap:8px !important;
    align-items:center !important;
}
.st-key-loan_prediction_form_fields [data-testid="stHorizontalBlock"] [data-testid="column"]:has([data-testid="stNumberInput"]) {
    flex:1 1 auto !important;
    min-width:0 !important;
}
.st-key-loan_prediction_form_fields [data-testid="stHorizontalBlock"] [data-testid="column"]:has([class*="st-key-lp_dec_"]),
.st-key-loan_prediction_form_fields [data-testid="stHorizontalBlock"] [data-testid="column"]:has([class*="st-key-lp_inc_"]) {
    flex:0 0 40px !important;
    width:40px !important;
    min-width:40px !important;
    max-width:40px !important;
    padding:0 !important;
    margin:0 !important;
}
[class*="st-key-lp_dec_"] > div,
[class*="st-key-lp_inc_"] > div,
[class*="st-key-lp_dec_"] button,
[class*="st-key-lp_inc_"] button {
    width:40px !important;
    min-width:40px !important;
    max-width:40px !important;
    height:40px !important;
    min-height:40px !important;
    max-height:40px !important;
    padding:0 !important;
    margin:0 !important;
}
[class*="st-key-lp_dec_"] button,
[class*="st-key-lp_inc_"] button {
    display:inline-flex !important;
    align-items:center !important;
    justify-content:center !important;
    background:#edf4ff !important;
    color:#0866e8 !important;
    -webkit-text-fill-color:#0866e8 !important;
    border:1px solid #bfd5fa !important;
    border-radius:10px !important;
    font-size:17px !important;
    font-weight:800 !important;
    line-height:1 !important;
    cursor:pointer !important;
    pointer-events:auto !important;
    box-shadow:0 2px 6px rgba(8,102,232,.08) !important;
}
[class*="st-key-lp_dec_"] button p,
[class*="st-key-lp_inc_"] button p {
    display:inline-flex !important;
    align-items:center !important;
    justify-content:center !important;
    width:100% !important;
    height:100% !important;
    margin:0 !important;
    padding:0 !important;
    color:#0866e8 !important;
    -webkit-text-fill-color:#0866e8 !important;
    font-size:17px !important;
    font-weight:800 !important;
    line-height:1 !important;
}
[class*="st-key-lp_dec_"] button:hover,
[class*="st-key-lp_inc_"] button:hover {
    background:#1261e8 !important;
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    border-color:#1261e8 !important;
    cursor:pointer !important;
}
[class*="st-key-lp_dec_"] button:hover p,
[class*="st-key-lp_inc_"] button:hover p {
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
}
.st-key-lp_predict_btn button {
    display:inline-flex !important;
    align-items:center !important;
    justify-content:center !important;
    gap:10px !important;
    cursor:pointer !important;
    padding:0 20px !important;
}
.st-key-lp_predict_btn button p {
    display:inline-flex !important;
    align-items:center !important;
    gap:8px !important;
    letter-spacing:0.01em !important;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL AND DATASET
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found.")

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError("Dataset not found.")

    return pd.read_csv(DATASET_PATH)


try:
    model = load_model()
    df = load_dataset()

except Exception as error:
    st.error(f"Application loading error: {error}")
    st.stop()


# =====================================================
# TOP NAVIGATION
# =====================================================

pages = [
    "Loan Prediction",
    "Dashboard",
    "Prediction History",
    "Model Performance",
    "Dataset Analytics"
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "Loan Prediction"

def _set_active_page(page_name, sync_mobile_widget=True):
    st.session_state.active_page = page_name
    if sync_mobile_widget and "mobile_navigation" in st.session_state:
        st.session_state.mobile_navigation = page_name

with st.container(key="main_nav_bar"):
    nav_brand, nav_links, nav_mobile = st.columns(
        [1.12, 5.6, 0.28], vertical_alignment="center"
    )

    with nav_brand:
        st.markdown(
            """
            <div class="top-brand">
              <span class="top-brand-mark" aria-hidden="true">
                <svg viewBox="0 0 32 32" role="img">
                  <path d="M3 11.2 16 4l13 7.2H3Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                  <path d="M6 13.5h20M7.5 25h17M5 28h22M9 14v10m5-10v10m4-10v10m5-10v10" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
              </span>
              <span class="top-brand-name">LoanPredict <b>AI</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with nav_links:
        nav_columns = st.columns(5, gap="medium")
        nav_keys = [
            "loan_prediction",
            "dashboard",
            "prediction_history",
            "model_performance",
            "dataset_analytics"
        ]
        for nav_col, page_name, nav_key in zip(nav_columns, pages, nav_keys):
            with nav_col:
                if st.button(
                    page_name,
                    key=f"nav_{nav_key}",
                    type="primary" if st.session_state.active_page == page_name else "secondary",
                    use_container_width=True
                ):
                    _set_active_page(page_name)
                    st.rerun()

    with nav_mobile:
        with st.container(key="mobile_nav_menu"):
            with st.popover("☰", help="Open navigation", use_container_width=True):
                st.markdown("**Navigate to**")
                mobile_choice = st.selectbox(
                    "Mobile navigation",
                    pages,
                    index=pages.index(st.session_state.active_page),
                    label_visibility="collapsed",
                    key="mobile_navigation"
                )
                if mobile_choice != st.session_state.active_page:
                    _set_active_page(mobile_choice, sync_mobile_widget=False)
                    st.rerun()

page = st.session_state.active_page


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    def show_dashboard_chart(
        series, *, horizontal=False, color="#2585e8",
        colors_by_bar=None, height=2.45, x_label="Applicants"
    ):
        """Render every dashboard chart using one consistent light fintech style."""
        values = series
        fig, ax = plt.subplots(figsize=(5.0, 2.45), dpi=125)
        fig.patch.set_facecolor("#f8faff")
        ax.set_facecolor("#f8faff")

        labels = [str(x) for x in values.index]
        numbers = [float(x) for x in values.values]
        bar_colors = colors_by_bar or color
        maximum = max(numbers) if numbers else 0

        if horizontal:
            bars = ax.barh(
                labels, numbers, color=bar_colors, height=0.58,
                edgecolor="none", zorder=3
            )
            ax.invert_yaxis()
            ax.set_xlim(0, maximum * 1.20 if maximum > 0 else 1)
            ax.xaxis.grid(True, color="#dce6f4", linewidth=0.8, zorder=0)
            for bar, number in zip(bars, numbers):
                ax.text(
                    number + (maximum * .018 if maximum else .02),
                    bar.get_y() + bar.get_height()/2,
                    f"{int(number):,}", va="center", ha="left",
                    fontsize=8, color="#20365e", fontweight="semibold"
                )
            ax.set_xlabel(x_label, fontsize=8, color="#64748b", labelpad=5)
        else:
            bars = ax.bar(
                labels, numbers, color=bar_colors, width=.55,
                edgecolor="none", zorder=3
            )
            ax.set_ylim(0, maximum * 1.22 if maximum > 0 else 1)
            ax.yaxis.grid(True, color="#dce6f4", linewidth=.8, zorder=0)
            for bar, number in zip(bars, numbers):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (maximum * .025 if maximum else .02),
                    f"{int(number):,}", va="bottom", ha="center",
                    fontsize=8, color="#20365e", fontweight="semibold"
                )
            ax.set_ylabel("Applications", fontsize=8, color="#64748b", labelpad=5)

        ax.tick_params(axis="both", labelsize=8, colors="#344967", length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    total = len(df)
    approved = int((df["Loan_Status"] == "Y").sum())
    rejected = int((df["Loan_Status"] == "N").sum())
    approval_rate = approved / total * 100 if total else 0

    st.markdown("""
    <div class="lp-pred-hero">
      <div class="lp-pred-hero-copy">
        <div class="lp-pred-eyebrow">✦ &nbsp; MACHINE LEARNING • LOAN INTELLIGENCE</div>
        <h1>Loan Prediction Dashboard</h1>
        <p>Explore loan application patterns, review prediction activity, and monitor insights from the ML model.</p>
      </div>
      <svg class="lp-pred-hero-art" viewBox="0 0 380 165" fill="none" aria-hidden="true">
        <g opacity=".95">
          <path d="M195 42h90a8 8 0 0 1 8 8v100H187V50a8 8 0 0 1 8-8Z" fill="#f8fbff" stroke="#7eb8ff" stroke-width="3"/>
          <path d="M215 31h50v16h-50z" rx="5" fill="#9bd0ff" stroke="#e8f5ff" stroke-width="2"/>
          <circle cx="240" cy="31" r="7" fill="#d9efff" stroke="#fff" stroke-width="2"/>
          <text x="240" y="68" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">LOAN</text>
          <text x="240" y="83" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">DASHBOARD</text>
          <g stroke="#0877ed" stroke-width="3" stroke-linecap="round">
            <rect x="205" y="96" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 102 3 3 6-7M226 102h40"/>
            <rect x="205" y="119" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 125 3 3 6-7M226 125h40"/>
          </g>
          <path d="M300 80 342 58l35 22v6h-77z" fill="#d8eaff" stroke="#78baff" stroke-width="3"/>
          <path d="M306 87h68v61h-68z" fill="#e9f4ff" stroke="#78baff" stroke-width="3"/>
          <path d="M297 80h86M314 90v55m17-55v55m17-55v55m17-55v55M300 149h78" stroke="#c3ddff" stroke-width="5" stroke-linecap="round"/>
          <path d="M292 151h96" stroke="#fff" stroke-width="5" stroke-linecap="round"/>
          <path d="M324 69h34" stroke="#4d91e8" stroke-width="3" stroke-linecap="round"/>
          <text x="342" y="78" fill="#17428c" font-size="12" font-weight="800" text-anchor="middle">BANK</text>
          <g>
            <ellipse cx="147" cy="145" rx="29" ry="7" fill="#ffcf4b"/>
            <path d="M118 132h58v13h-58z" fill="#f5a900" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="132" rx="29" ry="7" fill="#ffd34f" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="116" rx="25" ry="7" fill="#ffcf4b"/>
            <path d="M122 103h50v13h-50z" fill="#eea500" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="103" rx="25" ry="7" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="88" rx="20" ry="6" fill="#ffd34f"/>
            <path d="M127 76h40v12h-40z" fill="#eaa000" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="76" rx="20" ry="6" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
          </g>
          <path d="M92 142c-14-21-13-37-2-49 2 19 9 29 21 37M99 128c-1-18 7-30 23-35-5 16-8 28-6 43" stroke="#29c5f5" stroke-width="5" stroke-linecap="round"/>
          <path d="M91 143c-13-3-22-11-26-24 15 3 24 10 26 24ZM111 144c4-14 13-22 28-25-4 14-13 22-28 25Z" fill="#17a9ec"/>
          <path d="M26 34h28m-14-14v28M72 57h17m-8-9v18" stroke="#7bdcff" stroke-width="2" stroke-linecap="round" opacity=".8"/>
          <circle cx="30" cy="87" r="3" fill="#7bdcff"/><circle cx="91" cy="22" r="2.5" fill="#7bdcff"/>
        </g>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="dash-kpi dash-kpi-blue" style="display:flex;align-items:center;justify-content:space-between;gap:10px;">'
            f'<div class="dash-kpi-left"><div class="dash-kpi-ico" style="background:#e6f0ff;color:#1261e8;">👥</div>'
            f'<div><div class="dash-kpi-label">Total Applications</div><div class="dash-kpi-value">{total:,}</div></div></div>'
            f'<svg width="74" height="36" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#1261e8" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            f'</div>',
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f'<div class="dash-kpi dash-kpi-green" style="display:flex;align-items:center;justify-content:space-between;gap:10px;">'
            f'<div class="dash-kpi-left"><div class="dash-kpi-ico" style="background:#ddf7e9;color:#07884b;">✓</div>'
            f'<div><div class="dash-kpi-label">Approved Loans</div><div class="dash-kpi-value" style="color:#07884b">{approved:,}</div></div></div>'
            f'<svg width="74" height="36" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#07884b" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            f'</div>',
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f'<div class="dash-kpi dash-kpi-red" style="display:flex;align-items:center;justify-content:space-between;gap:10px;">'
            f'<div class="dash-kpi-left"><div class="dash-kpi-ico" style="background:#ffe5e7;color:#c1273d;">✕</div>'
            f'<div><div class="dash-kpi-label">Rejected Loans</div><div class="dash-kpi-value" style="color:#c1273d">{rejected:,}</div></div></div>'
            f'<svg width="74" height="36" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#c1273d" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            f'</div>',
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f'<div class="dash-kpi dash-kpi-purple" style="display:flex;align-items:center;justify-content:space-between;gap:10px;">'
            f'<div class="dash-kpi-left"><div class="dash-kpi-ico" style="background:#ede9fe;color:#6236c9;">▥</div>'
            f'<div><div class="dash-kpi-label">Approval Rate</div><div class="dash-kpi-value" style="color:#6236c9">{approval_rate:.1f}%</div></div></div>'
            f'<svg width="74" height="36" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#6236c9" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="dash-section-title">Dataset Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-section-caption">Key patterns across the 614 historical loan applications used to train the model.</div>', unsafe_allow_html=True)

    chart1, chart2, chart3 = st.columns([1, 1, 1.05], gap="medium")

    with chart1:
        with st.container(border=True):
            st.markdown("### 📊 Historical Loan Status")
            st.caption("Approved vs rejected applications.")
            status_series = pd.Series(
                [approved, rejected], index=["Approved", "Rejected"]
            )
            show_dashboard_chart(status_series, colors_by_bar=["#18b887", "#ef5967"])
            st.markdown(
                f"**{approved:,} Approved** &nbsp;&nbsp; **{rejected:,} Rejected**",
                unsafe_allow_html=True
            )

    with chart2:
        with st.container(border=True):
            st.markdown("### 🎓 Applicant Education")
            st.caption("Distribution by education category.")
            education_counts = (
                df["Education"].fillna("Unknown").value_counts()
                .rename_axis("Education").to_frame("Applicants")
            )
            show_dashboard_chart(
                education_counts["Applicants"], color="#2585e8"
            )
            for label, count in education_counts["Applicants"].items():
                pct = count / total * 100 if total else 0
                st.caption(f"{label}: {count:,} ({pct:.1f}%)")

    with chart3:
        with st.container(border=True):
            st.markdown("### 📍 Property Area Distribution")
            st.caption("Applicants across property-area categories.")
            area_counts = (
                df["Property_Area"].fillna("Unknown").value_counts()
                .rename_axis("Property Area").to_frame("Applicants")
            )
            show_dashboard_chart(
                area_counts["Applicants"], horizontal=True,
                color="#2784ef"
            )

    lower_left, lower_right = st.columns([1, 1.25], gap="medium")

    with lower_left:
        with st.container(border=True):
            st.markdown("### ▤ Applicant Income Distribution")
            st.caption("Applicant counts across income bands.")
            income_values = pd.to_numeric(
                df["ApplicantIncome"], errors="coerce"
            ).dropna()
            income_bins = [
                0, 2500, 5000, 7500, 10000,
                15000, 20000, 30000, 50000, float("inf")
            ]
            income_labels = [
                "0–2.5k", "2.5k–5k", "5k–7.5k", "7.5k–10k",
                "10k–15k", "15k–20k", "20k–30k", "30k–50k", "50k+"
            ]
            income_groups = pd.cut(
                income_values, bins=income_bins, labels=income_labels,
                right=False, include_lowest=True
            )
            income_counts = (
                income_groups.value_counts(sort=False)
                .reindex(income_labels, fill_value=0)
                .rename_axis("Income Band").to_frame("Applicants")
            )
            show_dashboard_chart(
                income_counts["Applicants"], horizontal=True,
                color="#2585e8"
            )

    with lower_right:
        with st.container(border=True):
            recent_header, recent_link = st.columns([3, 1])
            with recent_header:
                st.markdown("### ◷ Recent Predictions")
                st.caption("Latest applications submitted using the prediction form.")
            with recent_link:
                st.button(
                    "View All →",
                    key="dashboard_view_all",
                    on_click=_set_active_page,
                    args=("Prediction History",)
                )

            records = get_all_predictions()
            if records:
                recent = pd.DataFrame(records).copy()
                if "created_at" in recent.columns:
                    recent = recent.sort_values(
                        "created_at", ascending=False, kind="stable"
                    )
                elif "id" in recent.columns:
                    recent = recent.sort_values("id", ascending=False, kind="stable")
                recent = recent.head(5)
                recent = recent.rename(
                    columns={
                        "id": "Application ID",
                        "prediction": "Prediction",
                        "approval_score": "Approval Score",
                        "created_at": "Submitted At",
                    }
                )
                recent["Approval Score"] = recent["Approval Score"].map(
                    lambda value: f"{float(value) * 100:.2f}%"
                )
                recent_view = recent[
                    ["Application ID", "Prediction", "Approval Score", "Submitted At"]
                ]
                table_rows = []
                for _, row in recent_view.iterrows():
                    status = str(row["Prediction"])
                    status_bg = "#dcfce7" if status == "Approved" else "#ffe4e6"
                    status_fg = "#15803d" if status == "Approved" else "#be123c"
                    table_rows.append(
                        "<tr>"
                        f"<td>{row['Application ID']}</td>"
                        f"<td><span style='display:inline-block;padding:4px 10px;"
                        f"border-radius:8px;background:{status_bg};color:{status_fg};"
                        f"font-weight:800'>{status}</span></td>"
                        f"<td>{row['Approval Score']}</td>"
                        f"<td>{row['Submitted At']}</td>"
                        "</tr>"
                    )
                html_table = (
                    '<div style="overflow-x:auto;border:1px solid #d5e0f0;border-radius:11px;margin-bottom:16px;">'
                    '<table class="recent-predictions-table" style="width:100%;border-collapse:collapse;background:#f8faff;color:#18345f;font-size:13px;">'
                    '<thead><tr style="background:#e5edf9;text-align:left;">'
                    '<th style="padding:11px;">Application ID</th>'
                    '<th style="padding:11px;">Prediction</th>'
                    '<th style="padding:11px;">Approval Score</th>'
                    '<th style="padding:11px;">Submitted At</th>'
                    '</tr></thead>'
                    f'<tbody>{"".join(table_rows)}</tbody>'
                    '</table></div>'
                )
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.info("No predictions have been submitted yet.")

    st.caption(
        "Dataset statistics represent historical training data. Recent Predictions "
        "shows records submitted through this application."
    )


# =====================================================
# LOAN PREDICTION
# =====================================================

elif page == "Loan Prediction":

    # Track fresh page load / browser refresh so every new session starts with empty fields,
    # while preserving any displayed result in st.session_state["last_prediction_result"]
    _lp_session_marker = st.session_state.get("_lp_rendered_on_page")
    if _lp_session_marker != "Loan Prediction":
        for _k in (
            "lp_gender", "lp_married", "lp_dependents", "lp_education",
            "lp_self_employed", "lp_property_area", "lp_applicant_income",
            "lp_loan_amount", "lp_coapplicant_income", "lp_loan_term",
            "lp_credit_history", "lp_validation_errors"
        ):
            st.session_state.pop(_k, None)
        st.session_state["_lp_rendered_on_page"] = "Loan Prediction"

    st.markdown("""
    <div class="lp-pred-hero">
      <div class="lp-pred-hero-copy">
        <div class="lp-pred-eyebrow">⚡ &nbsp; PREDICT LOAN APPROVAL</div>
        <h1>Loan Prediction</h1>
        <p>Enter applicant details to generate a machine learning prediction
        and assess the loan approval likelihood.</p>
      </div>
      <svg class="lp-pred-hero-art" viewBox="0 0 380 165" fill="none" aria-hidden="true">
        <g opacity=".95">
          <path d="M195 42h90a8 8 0 0 1 8 8v100H187V50a8 8 0 0 1 8-8Z" fill="#f8fbff" stroke="#7eb8ff" stroke-width="3"/>
          <path d="M215 31h50v16h-50z" rx="5" fill="#9bd0ff" stroke="#e8f5ff" stroke-width="2"/>
          <circle cx="240" cy="31" r="7" fill="#d9efff" stroke="#fff" stroke-width="2"/>
          <text x="240" y="68" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">LOAN</text>
          <text x="240" y="83" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">APPLICATION</text>
          <g stroke="#0877ed" stroke-width="3" stroke-linecap="round">
            <rect x="205" y="96" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 102 3 3 6-7M226 102h40"/>
            <rect x="205" y="119" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 125 3 3 6-7M226 125h40"/>
          </g>
          <path d="M300 80 342 58l35 22v6h-77z" fill="#d8eaff" stroke="#78baff" stroke-width="3"/>
          <path d="M306 87h68v61h-68z" fill="#e9f4ff" stroke="#78baff" stroke-width="3"/>
          <path d="M297 80h86M314 90v55m17-55v55m17-55v55m17-55v55M300 149h78" stroke="#c3ddff" stroke-width="5" stroke-linecap="round"/>
          <path d="M292 151h96" stroke="#fff" stroke-width="5" stroke-linecap="round"/>
          <path d="M324 69h34" stroke="#4d91e8" stroke-width="3" stroke-linecap="round"/>
          <text x="342" y="78" fill="#17428c" font-size="12" font-weight="800" text-anchor="middle">BANK</text>
          <g>
            <ellipse cx="147" cy="145" rx="29" ry="7" fill="#ffcf4b"/>
            <path d="M118 132h58v13h-58z" fill="#f5a900" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="132" rx="29" ry="7" fill="#ffd34f" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="116" rx="25" ry="7" fill="#ffcf4b"/>
            <path d="M122 103h50v13h-50z" fill="#eea500" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="103" rx="25" ry="7" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="88" rx="20" ry="6" fill="#ffd34f"/>
            <path d="M127 76h40v12h-40z" fill="#eaa000" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="76" rx="20" ry="6" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
          </g>
          <path d="M92 142c-14-21-13-37-2-49 2 19 9 29 21 37M99 128c-1-18 7-30 23-35-5 16-8 28-6 43" stroke="#29c5f5" stroke-width="5" stroke-linecap="round"/>
          <path d="M91 143c-13-3-22-11-26-24 15 3 24 10 26 24ZM111 144c4-14 13-22 28-25-4 14-13 22-28 25Z" fill="#17a9ec"/>
          <path d="M26 34h28m-14-14v28M72 57h17m-8-9v18" stroke="#7bdcff" stroke-width="2" stroke-linecap="round" opacity=".8"/>
          <circle cx="30" cy="87" r="3" fill="#7bdcff"/><circle cx="91" cy="22" r="2.5" fill="#7bdcff"/>
        </g>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    left_panel, right_panel = st.columns([1.58, 1.02], gap="medium")

    with left_panel:
        st.markdown("""
        <div class="lp-pred-panel">
          <div class="lp-pred-panel-title">
            <div class="lp-pred-title-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="8" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2H4Z"/>
              </svg>
            </div>
            <div><h2>Applicant Information</h2>
            <p>Provide the applicant's personal and financial details.</p></div>
          </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="lp-pred-section-head">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2"/></svg>
          Personal Information
        </div>
        """, unsafe_allow_html=True)

        with st.container(key="loan_prediction_form_fields"):
            st.markdown('<div class="lp-pred-form">', unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3, gap="medium")
            with p1:
                gender = st.selectbox(
                    "Gender",
                    ["Select Gender", "Male", "Female"],
                    index=0,
                    key="lp_gender"
                )
                married = st.selectbox(
                    "Marital Status",
                    ["Select Marital Status", "Yes", "No"],
                    index=0,
                    key="lp_married"
                )
            with p2:
                dependents = st.selectbox(
                    "Number of Dependents",
                    ["Select Dependents", "0", "1", "2", "3+"],
                    index=0,
                    key="lp_dependents"
                )
                education = st.selectbox(
                    "Education",
                    ["Select Education", "Graduate", "Not Graduate"],
                    index=0,
                    key="lp_education"
                )
            with p3:
                self_employed = st.selectbox(
                    "Self Employed",
                    ["Select Self Employed", "No", "Yes"],
                    index=0,
                    key="lp_self_employed"
                )
                property_area = st.selectbox(
                    "Property Area",
                    ["Select Property Area", "Urban", "Semiurban", "Rural"],
                    index=0,
                    key="lp_property_area"
                )

            st.markdown("""
            <div class="lp-pred-section-head">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4m0 0c-4 0-7 3-7 7v2a5 5 0 0 0 5 5h4a5 5 0 0 0 5-5v-2c0-4-3-7-7-7Z"/><path d="M9 14h6"/></svg>
              Financial Information
            </div>
            """, unsafe_allow_html=True)

            # Initialize numeric financial session state keys to None so inputs start blank
            # and step +/- buttons adjust values cleanly from any starting or blank state.
            for _num_k in ("lp_applicant_income", "lp_loan_amount", "lp_coapplicant_income"):
                if _num_k not in st.session_state:
                    st.session_state[_num_k] = None

            def _step_financial(key_name, delta, min_val, max_val):
                cur = st.session_state.get(key_name)
                if cur is None:
                    nxt = min_val if delta < 0 else (min_val if min_val > 0 else delta)
                else:
                    nxt = cur + delta
                nxt = max(min_val, min(max_val, nxt))
                st.session_state[key_name] = nxt

            loan_term_options = ["Select Loan Term", 12, 24, 36, 60, 120, 180, 240, 300, 360, 480]
            f1, f2 = st.columns(2, gap="large")
            with f1:
                app_in_col, app_dec_col, app_inc_col = st.columns([5.6, 0.85, 0.85], gap="xxsmall")
                with app_in_col:
                    applicant_income = st.number_input(
                        "Applicant Income (₹)",
                        min_value=0, max_value=1000000, step=500,
                        placeholder="Enter applicant income",
                        key="lp_applicant_income"
                    )
                with app_dec_col:
                    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
                    st.button(
                        "−", key="lp_dec_app_inc", use_container_width=True,
                        help="Decrease Applicant Income by ₹500",
                        on_click=_step_financial,
                        args=("lp_applicant_income", -500, 0, 1000000)
                    )
                with app_inc_col:
                    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
                    st.button(
                        "+", key="lp_inc_app_inc", use_container_width=True,
                        help="Increase Applicant Income by ₹500",
                        on_click=_step_financial,
                        args=("lp_applicant_income", 500, 0, 1000000)
                    )

                la_in_col, la_dec_col, la_inc_col = st.columns([5.6, 0.85, 0.85], gap="xxsmall")
                with la_in_col:
                    loan_amount = st.number_input(
                        "Loan Amount (Dataset Units)",
                        min_value=1, max_value=1000, step=5,
                        placeholder="Enter loan amount",
                        key="lp_loan_amount"
                    )
                with la_dec_col:
                    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
                    st.button(
                        "−", key="lp_dec_loan_amt", use_container_width=True,
                        help="Decrease Loan Amount by 5 units",
                        on_click=_step_financial,
                        args=("lp_loan_amount", -5, 1, 1000)
                    )
                with la_inc_col:
                    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
                    st.button(
                        "+", key="lp_inc_loan_amt", use_container_width=True,
                        help="Increase Loan Amount by 5 units",
                        on_click=_step_financial,
                        args=("lp_loan_amount", 5, 1, 1000)
                    )

            with f2:
                co_in_col, co_dec_col, co_inc_col = st.columns([5.6, 0.85, 0.85], gap="xxsmall")
                with co_in_col:
                    coapplicant_income = st.number_input(
                        "Co-applicant Income (₹)",
                        min_value=0, max_value=1000000, step=500,
                        placeholder="Enter co-applicant income",
                        key="lp_coapplicant_income"
                    )
                with co_dec_col:
                    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
                    st.button(
                        "−", key="lp_dec_co_inc", use_container_width=True,
                        help="Decrease Co-applicant Income by ₹500",
                        on_click=_step_financial,
                        args=("lp_coapplicant_income", -500, 0, 1000000)
                    )
                with co_inc_col:
                    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
                    st.button(
                        "+", key="lp_inc_co_inc", use_container_width=True,
                        help="Increase Co-applicant Income by ₹500",
                        on_click=_step_financial,
                        args=("lp_coapplicant_income", 500, 0, 1000000)
                    )

                loan_term = st.selectbox(
                    "Loan Term (Months)",
                    loan_term_options,
                    index=0,
                    format_func=lambda m: m if isinstance(m, str) else f"{m} Months",
                    key="lp_loan_term"
                )

            st.markdown("""
            <div class="lp-pred-section-head">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18M7 15h4"/></svg>
              Credit Information
            </div>
            """, unsafe_allow_html=True)

            credit_history = st.selectbox(
                "Credit History",
                ["Select Credit History", 1.0, 0.0],
                index=0,
                format_func=lambda x: (
                    x if isinstance(x, str)
                    else ("Good Credit History (1)" if x == 1.0 else "Poor Credit History (0)")
                ),
                key="lp_credit_history"
            )
            st.markdown("</div>", unsafe_allow_html=True)

            predict_clicked = st.button("⌕  Predict Loan Approval", use_container_width=True, key="lp_predict_btn")

            if predict_clicked:
                validation_errors = []
                if gender == "Select Gender":
                    validation_errors.append("Please select Gender.")
                if married == "Select Marital Status":
                    validation_errors.append("Please select Marital Status.")
                if dependents == "Select Dependents":
                    validation_errors.append("Please select Number of Dependents.")
                if education == "Select Education":
                    validation_errors.append("Please select Education.")
                if self_employed == "Select Self Employed":
                    validation_errors.append("Please select Self Employed.")
                if property_area == "Select Property Area":
                    validation_errors.append("Please select Property Area.")
                if applicant_income is None:
                    validation_errors.append("Please enter Applicant Income (₹).")
                if coapplicant_income is None:
                    validation_errors.append("Please enter Co-applicant Income (₹).")
                if loan_amount is None:
                    validation_errors.append("Please enter Loan Amount (Dataset Units).")
                if loan_term == "Select Loan Term":
                    validation_errors.append("Please select Loan Term (Months).")
                if credit_history == "Select Credit History":
                    validation_errors.append("Please select Credit History.")

                st.session_state["lp_validation_errors"] = validation_errors
            else:
                validation_errors = st.session_state.get("lp_validation_errors", [])

            if validation_errors:
                st.error(
                    "**Please complete all required fields before predicting:**\n- "
                    + "\n- ".join(validation_errors)
                )

        st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked and not validation_errors:
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_history),
            "Property_Area": property_area
        }])

        try:
            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            approval_score = float(probabilities[classes.index(1)])
            rejection_score = float(probabilities[classes.index(0)])
            prediction_label = "Approved" if prediction == 1 else "Rejected"

            record_id = save_prediction(
                input_data.iloc[0].to_dict(),
                prediction_label,
                approval_score,
                rejection_score
            )
            st.session_state["last_prediction_id"] = record_id
            st.session_state["last_prediction_result"] = {
                "prediction": prediction,
                "label": prediction_label,
                "approval_score": approval_score,
                "rejection_score": rejection_score,
                "record_id": record_id,
                "applicant": input_data.iloc[0].to_dict()
            }
        except Exception as error:
            st.error(f"Prediction error: {error}")

    with right_panel:
        with st.container(key="prediction_result_card", border=True):
            st.markdown("""
              <div class="lp-pred-panel-title">
                <div class="lp-pred-title-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 20h18M5 20V11h4v9M10 20V4h4v16M15 20v-7h4v7"/>
                  </svg>
                </div>
                <div><h2>Prediction Result</h2>
                <p>Loan approval outcome from the trained ML model.</p></div>
              </div>
            """, unsafe_allow_html=True)

            result = st.session_state.get("last_prediction_result")
            if result:
                is_approved = result["prediction"] == 1
                result_class = "" if is_approved else " rejected"
                result_title = "✓ Loan Likely Approved" if is_approved else "✕ Loan Likely Rejected"
                result_text = (
                    "The model predicts an approved loan application."
                    if is_approved else
                    "The model predicts a rejected loan application."
                )
                st.markdown(
                    f'<div class="lp-pred-result{result_class}"><h3>{result_title}</h3><p>{result_text}</p></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="lp-pred-score"><div class="lp-pred-score-label">Approval Model Score</div><div class="lp-pred-score-value">{result["approval_score"]*100:.2f}%</div></div>',
                    unsafe_allow_html=True
                )
                st.progress(max(0.0, min(1.0, result["approval_score"])))
                st.markdown(
                    f'<div class="lp-pred-score"><div class="lp-pred-score-label">Rejection Model Score</div><div class="lp-pred-score-value">{result["rejection_score"]*100:.2f}%</div></div>',
                    unsafe_allow_html=True
                )
                st.progress(max(0.0, min(1.0, result["rejection_score"])))
                st.success(f"Prediction saved successfully! Application ID: {result['record_id']}")
                st.caption("Scores are model estimates, not calibrated probabilities or guaranteed bank decisions.")
                with st.expander("View Applicant Information"):
                    st.json(result["applicant"])
            else:
                st.markdown("""
                <div class="lp-pred-empty">
                  <svg viewBox="0 0 220 175" fill="none" aria-hidden="true">
                    <ellipse cx="110" cy="88" rx="88" ry="67" fill="#edf4ff"/>
                    <ellipse cx="110" cy="88" rx="65" ry="52" fill="#e2edff"/>
                    <rect x="63" y="25" width="88" height="112" rx="12" fill="#fff" stroke="#4c78ff" stroke-width="4"/>
                    <path d="M82 53h48M82 69h38M82 85h29" stroke="#6c94ff" stroke-width="5" stroke-linecap="round"/>
                    <circle cx="142" cy="111" r="23" fill="#d7e7ff" stroke="#075de7" stroke-width="7"/>
                    <path d="m158 128 16 17" stroke="#075de7" stroke-width="10" stroke-linecap="round"/>
                    <path d="m178 23 0 13m-10-3-7 7m20-1 8 0" stroke="#3978ff" stroke-width="4" stroke-linecap="round"/>
                  </svg>
                  <h3>No Prediction Yet</h3>
                  <p>Fill in the applicant details and click “Predict Loan Approval” to see the result.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="lp-pred-about">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-9h.01"/></svg>
              <div><strong>About Prediction</strong>
              <p>The system uses a machine learning model trained on historical loan data to predict the approval likelihood.</p></div>
            </div>
            """, unsafe_allow_html=True)


# =====================================================
# PREDICTION HISTORY
# =====================================================

elif page == "Prediction History":

    st.markdown("""
    <style>
    /* Hero Banner */
    .history-hero {
        position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between;
        min-height:138px; padding:20px 32px; margin:2px 0 14px; border-radius:18px;
        color:#fff; background:linear-gradient(110deg,#051F55 0%,#084EC2 55%,#075EE0 100%);
        border:1px solid rgba(147,197,253,.28);
        box-shadow:0 8px 24px rgba(5,48,120,.18);
    }
    .history-hero:before {
        content:""; position:absolute; width:440px; height:240px; right:14%; bottom:-115px;
        border-radius:50%; border:1px solid rgba(125,184,255,.16);
        box-shadow:0 0 0 28px rgba(125,184,255,.04),0 0 0 58px rgba(125,184,255,.03);
    }
    .history-hero-copy { position:relative; z-index:2; max-width:580px; }
    .history-hero-tag {
        display:inline-flex; align-items:center; gap:6px; padding:4px 12px; margin-bottom:8px;
        border-radius:30px; background:rgba(255,255,255,.14); border:1px solid rgba(191,219,254,.25);
        color:#fff; font-size:11px; font-weight:800; letter-spacing:.4px;
    }
    .history-hero h1 { color:#fff !important; font-size:30px !important; font-weight:850 !important; margin:0 0 4px !important; letter-spacing:-.4px; }
    .history-hero p { color:#E6F0FF !important; font-size:14px; margin:0 !important; }
    .history-hero-art { position:relative; z-index:2; width:280px; height:120px; flex:0 0 280px; }

    /* KPI Summary Cards with Sparklines */
    .history-stat {
        display:flex; align-items:center; justify-content:space-between; gap:12px;
        min-height:92px; padding:14px 18px; border:1px solid #D4E3FC; border-radius:16px;
        background:linear-gradient(120deg,#FFFFFF 0%,#F4F8FF 100%);
        box-shadow:0 4px 14px rgba(11,36,87,.05); margin-bottom:12px;
    }
    .history-stat.approved { background:linear-gradient(120deg,#FFFFFF 0%,#EEFCF4 100%); border-color:#BCEAD3; }
    .history-stat.rejected { background:linear-gradient(120deg,#FFFFFF 0%,#FFF2F3 100%); border-color:#FAD0D4; }
    .history-stat-left { display:flex; align-items:center; gap:13px; min-width:0; }
    .history-stat-icon {
        display:flex; align-items:center; justify-content:center; width:46px; height:46px; flex:0 0 46px;
        border-radius:50%;
    }
    .history-stat-label { color:#102A5E; font-size:13px; font-weight:750; }
    .history-stat-value { color:#0B2457; font-size:30px; font-weight:850; line-height:1.1; margin:2px 0; }
    .history-stat-note { color:#627A9E; font-size:11.5px; }
    .history-sparkline { width:92px; height:44px; flex:0 0 92px; }

    /* Card Headers */
    .history-sec-head { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
    .history-sec-icon {
        width:36px; height:36px; flex:0 0 36px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        background:#EAF2FF; color:#0866E8;
    }
    .history-sec-icon.red { background:#FFE8EA; color:#E5484D; }
    .history-card-heading { color:#0B2457; font-size:17px; font-weight:850; margin:0; line-height:1.2; }
    .history-card-sub { color:#637C9F; font-size:12px; margin:2px 0 0; }

    /* Application Records HTML Table — compact columns so Submitted Date never clips */
    .history-table-wrap {
        width:100%; overflow-x:auto; border:1px solid #D3E2FA; border-radius:12px; background:#FFFFFF;
    }
    table.history-clean-table {
        width:100%; border-collapse:collapse; background:#FFFFFF; font-size:12px;
    }
    table.history-clean-table thead th {
        background:#EAF2FF; color:#0B2457; font-weight:800; text-align:left;
        padding:10px 8px; border-bottom:1px solid #CFE0FA; white-space:nowrap; font-size:11.5px;
    }
    table.history-clean-table tbody td {
        padding:10px 8px; color:#142E61; border-bottom:1px solid #E8F0FC;
        white-space:nowrap; font-weight:500; font-size:12px;
    }
    table.history-clean-table tbody tr:nth-child(even) td { background:#F8FBFF; }
    table.history-clean-table tbody tr:hover td { background:#EDF4FF; }
    table.history-clean-table tbody tr:last-child td { border-bottom:0; }
    .hp-badge {
        display:inline-flex; align-items:center; gap:5px; padding:4px 9px;
        border-radius:999px; font-size:11px; font-weight:800; line-height:1;
    }
    .hp-badge-approved { background:#DDF7E9; color:#117A4B; border:1px solid #B5E8CD; }
    .hp-badge-rejected { background:#FFE5E7; color:#CF2633; border:1px solid #F9C0C5; }
    .hp-badge-dot {
        display:inline-flex; align-items:center; justify-content:center;
        width:13px; height:13px; border-radius:50%; color:#fff; font-size:8.5px; font-weight:900;
    }

    /* Overview Tiles in Individual Application Details */
    .history-overview {
        display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:10px 0 12px;
    }
    .history-overview-tile {
        display:flex; align-items:center; gap:10px; min-width:0; padding:11px 12px;
        background:linear-gradient(125deg,#FFFFFF,#F5F9FF);
        border:1px solid #D7E5FB; border-radius:12px;
    }
    .history-overview-tile.rejected-tile {
        background:linear-gradient(125deg,#FFFFFF,#FFF3F4); border-color:#FAD1D5;
    }
    .history-overview-tile.approved-tile {
        background:linear-gradient(125deg,#FFFFFF,#EFFCF5); border-color:#BCEAD3;
    }
    .history-tile-ico {
        width:34px; height:34px; flex:0 0 34px; border-radius:9px;
        display:flex; align-items:center; justify-content:center;
        background:#EAF2FF; color:#0866E8;
    }
    .history-overview-label { color:#536E96; font-size:11px; font-weight:750; }
    .history-overview-value { color:#0B2457; font-size:17px; font-weight:850; margin-top:2px; overflow-wrap:anywhere; }

    /* Personal & Financial Info Split Card */
    .history-info-box {
        background:#F8FBFF; border:1px solid #D7E5FB; border-radius:13px; overflow:hidden;
        margin-bottom:14px;
    }
    .history-info-banner {
        background:linear-gradient(90deg,#DCE9FC,#EAF2FF); color:#102A5E;
        padding:9px 14px; font-size:12.5px; font-weight:800; border-bottom:1px solid #CFE0FA;
    }
    .history-info-grid {
        display:grid; grid-template-columns:1fr 1fr; gap:14px; padding:14px 16px 22px;
    }
    .history-info-col:first-child {
        border-right:1px solid #DCE7F8; padding-right:12px;
    }
    .history-info-col h4 {
        color:#0B2457 !important; font-size:13.5px !important; font-weight:850 !important;
        margin:0 0 8px !important; display:flex; align-items:center; gap:6px;
    }
    .history-info-line {
        display:flex; justify-content:space-between; gap:8px; padding:4px 0;
        color:#526B91; font-size:12px;
    }
    .history-info-line strong { color:#102A5E; text-align:right; font-weight:800; }

    /* Model Prediction Scores */
    .history-score-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }
    .history-score {
        background:linear-gradient(135deg,#FFFFFF,#F2FBF6); border:1px solid #BCEAD3;
        border-radius:13px; padding:13px 14px;
    }
    .history-score.rej {
        background:linear-gradient(135deg,#FFFFFF,#FFF3F4); border-color:#FAD0D4;
    }
    .history-score-top { display:flex; align-items:center; gap:10px; }
    .history-score-badge {
        width:30px; height:30px; flex:0 0 30px; border-radius:50%;
        display:flex; align-items:center; justify-content:center; color:#fff; font-weight:900; font-size:13px;
    }
    .history-score-label { color:#344D75; font-size:12px; font-weight:750; }
    .history-score-value { font-size:25px; font-weight:900; line-height:1.15; }
    .history-score-track { height:9px; background:#E2ECF8; border-radius:30px; overflow:hidden; margin-top:10px; }
    .history-score-fill { height:100%; border-radius:30px; }
    .history-score-disclaimer {
        margin-top:10px; padding:8px 12px; border-radius:10px;
        background:#F2F6FD; border:1px solid #DCE7F9;
        color:#4A648C; font-size:11.5px; text-align:center;
    }

    /* Force Crisp White Background on All Selectboxes and Text Inputs in Light Mode */
    .st-key-history_search [data-baseweb="input"],
    .st-key-history_search [data-baseweb="input"] > div,
    .st-key-history_search input {
        background:#FFFFFF !important;
        background-color:#FFFFFF !important;
        color:#102A5E !important;
        -webkit-text-fill-color:#102A5E !important;
        border-radius:10px !important;
        min-height:42px !important;
        box-shadow:none !important;
    }
    .st-key-history_search [data-baseweb="input"] { border:1px solid #C9DAF5 !important; }
    .st-key-history_search [data-baseweb="input"] > div { border:0 !important; }
    .st-key-history_search [data-baseweb="input"]:focus-within {
        border:1px solid #0866E8 !important;
        box-shadow:0 0 0 2px rgba(8,102,232,.14) !important;
    }

    /* Target EVERY nested element inside BaseWeb selectboxes so dark-theme #262730 never leaks */
    div[data-testid="stSelectbox"] [data-baseweb="select"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] [data-baseweb="select"] div,
    div[data-testid="stSelectbox"] [data-baseweb="select"] span,
    div[data-testid="stSelectbox"] [data-baseweb="select"] input {
        background:#FFFFFF !important;
        background-color:#FFFFFF !important;
        color:#102A5E !important;
        -webkit-text-fill-color:#102A5E !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border:1px solid #C9DAF5 !important;
        border-radius:10px !important;
        min-height:42px !important;
        box-shadow:none !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
        fill:#102A5E !important;
        color:#102A5E !important;
    }
    ul[role="listbox"], ul[role="listbox"] li {
        background:#FFFFFF !important;
        color:#102A5E !important;
    }
    ul[role="listbox"] li:hover {
        background:#EDF4FF !important;
    }

    div[data-testid="stTextInput"] input::placeholder { color:#7589A8 !important; opacity:1 !important; }
    div[data-testid="stTextInput"] label, div[data-testid="stSelectbox"] label,
    div[data-testid="stWidgetLabel"], div[data-testid="stWidgetLabel"] * {
        color:#173766 !important; font-weight:700 !important;
    }

    /* Ensure Action Buttons Never Truncate Text */
    div[data-testid="stDownloadButton"] > button,
    .st-key-history_request_delete button {
        width:100% !important;
        min-height:42px !important;
        padding:0 12px !important;
        border-radius:10px !important;
        font-size:12px !important;
        font-weight:800 !important;
        white-space:nowrap !important;
    }
    div[data-testid="stDownloadButton"] > button {
        background:linear-gradient(100deg,#0866E8,#0753C7) !important;
        color:#fff !important; border:0 !important;
    }
    div[data-testid="stDownloadButton"] > button *,
    div[data-testid="stButton"] > button * {
        color:#fff !important; -webkit-text-fill-color:#fff !important;
        white-space:nowrap !important; overflow:visible !important; text-overflow:clip !important;
    }
    div[data-testid="stDownloadButton"] > button:hover { background:#0649B5 !important; }
    .st-key-history_request_delete button {
        background:linear-gradient(100deg,#EF4444,#DC2626) !important;
        color:#fff !important; border:0 !important;
    }

    @media(max-width:950px) {
        .history-hero { padding:18px 20px; }
        .history-hero h1 { font-size:25px !important; }
        .history-hero-art { width:200px; flex-basis:200px; opacity:.45; }
        .history-info-grid { grid-template-columns:1fr; }
        .history-info-col:first-child { border-right:0; padding-right:0; border-bottom:1px solid #DCE7F8; padding-bottom:10px; }
    }
    @media(max-width:650px) {
        .history-overview, .history-score-grid { grid-template-columns:1fr; }
        .history-hero-art { display:none; }
    }
    </style>
    """, unsafe_allow_html=True)

    def safe_text(value):
        return html_lib.escape(str(value if value is not None else ""))

    # Hero Banner
    st.markdown("""
    <div class="lp-pred-hero">
      <div class="lp-pred-hero-copy">
        <div class="lp-pred-eyebrow">◷ &nbsp; PREDICTION HISTORY</div>
        <h1>Prediction History</h1>
        <p>View, search, filter and manage saved loan predictions.</p>
      </div>
      <svg class="lp-pred-hero-art" viewBox="0 0 380 165" fill="none" aria-hidden="true">
        <g opacity=".95">
          <path d="M195 42h90a8 8 0 0 1 8 8v100H187V50a8 8 0 0 1 8-8Z" fill="#f8fbff" stroke="#7eb8ff" stroke-width="3"/>
          <path d="M215 31h50v16h-50z" rx="5" fill="#9bd0ff" stroke="#e8f5ff" stroke-width="2"/>
          <circle cx="240" cy="31" r="7" fill="#d9efff" stroke="#fff" stroke-width="2"/>
          <text x="240" y="68" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">PREDICTION</text>
          <text x="240" y="83" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">HISTORY</text>
          <g stroke="#0877ed" stroke-width="3" stroke-linecap="round">
            <rect x="205" y="96" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 102 3 3 6-7M226 102h40"/>
            <rect x="205" y="119" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 125 3 3 6-7M226 125h40"/>
          </g>
          <path d="M300 80 342 58l35 22v6h-77z" fill="#d8eaff" stroke="#78baff" stroke-width="3"/>
          <path d="M306 87h68v61h-68z" fill="#e9f4ff" stroke="#78baff" stroke-width="3"/>
          <path d="M297 80h86M314 90v55m17-55v55m17-55v55m17-55v55M300 149h78" stroke="#c3ddff" stroke-width="5" stroke-linecap="round"/>
          <path d="M292 151h96" stroke="#fff" stroke-width="5" stroke-linecap="round"/>
          <path d="M324 69h34" stroke="#4d91e8" stroke-width="3" stroke-linecap="round"/>
          <text x="342" y="78" fill="#17428c" font-size="12" font-weight="800" text-anchor="middle">BANK</text>
          <g>
            <ellipse cx="147" cy="145" rx="29" ry="7" fill="#ffcf4b"/>
            <path d="M118 132h58v13h-58z" fill="#f5a900" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="132" rx="29" ry="7" fill="#ffd34f" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="116" rx="25" ry="7" fill="#ffcf4b"/>
            <path d="M122 103h50v13h-50z" fill="#eea500" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="103" rx="25" ry="7" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="88" rx="20" ry="6" fill="#ffd34f"/>
            <path d="M127 76h40v12h-40z" fill="#eaa000" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="76" rx="20" ry="6" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
          </g>
          <path d="M92 142c-14-21-13-37-2-49 2 19 9 29 21 37M99 128c-1-18 7-30 23-35-5 16-8 28-6 43" stroke="#29c5f5" stroke-width="5" stroke-linecap="round"/>
          <path d="M91 143c-13-3-22-11-26-24 15 3 24 10 26 24ZM111 144c4-14 13-22 28-25-4 14-13 22-28 25Z" fill="#17a9ec"/>
          <path d="M26 34h28m-14-14v28M72 57h17m-8-9v18" stroke="#7bdcff" stroke-width="2" stroke-linecap="round" opacity=".8"/>
          <circle cx="30" cy="87" r="3" fill="#7bdcff"/><circle cx="91" cy="22" r="2.5" fill="#7bdcff"/>
        </g>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    records = get_all_predictions()
    history_df = pd.DataFrame(records)

    if history_df.empty:
        st.info("No prediction history available. Submit a loan application first.")
    else:
        history_df["id"] = pd.to_numeric(history_df["id"], errors="coerce")
        history_df = history_df.sort_values(
            ["created_at", "id"], ascending=[False, False], kind="stable"
        ).reset_index(drop=True)

        total_records = len(history_df)
        approved_count = int(history_df["prediction"].eq("Approved").sum())
        rejected_count = int(history_df["prediction"].eq("Rejected").sum())

        # Three KPI Summary Cards with Clean Trend Sparklines
        stat_cols = st.columns(3, gap="medium")
        stat_specs = [
            (
                "Total Records", total_records, "Saved applications", "", "#E6F0FF", "#0866E8",
                '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0866E8" stroke-width="2"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>',
                '<svg class="history-sparkline" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#0866E8" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            ),
            (
                "Approved", approved_count, "Approved predictions", "approved", "#16A36A", "#FFFFFF",
                '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"><path d="m5 13 4 4L19 7"/></svg>',
                '<svg class="history-sparkline" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#16A36A" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            ),
            (
                "Rejected", rejected_count, "Rejected predictions", "rejected", "#E5484D", "#FFFFFF",
                '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.8" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
                '<svg class="history-sparkline" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#E5484D" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            ),
        ]
        for col, (label, value, note, card_cls, icon_bg, icon_fg, icon_svg, spark_svg) in zip(stat_cols, stat_specs):
            with col:
                st.markdown(
                    f'<div class="history-stat {card_cls}">'
                    f'<div class="history-stat-left">'
                    f'<div class="history-stat-icon" style="background:{icon_bg};color:{icon_fg};">{icon_svg}</div>'
                    f'<div><div class="history-stat-label">{label}</div>'
                    f'<div class="history-stat-value">{value:,}</div>'
                    f'<div class="history-stat-note">{note}</div></div>'
                    f'</div>{spark_svg}</div>',
                    unsafe_allow_html=True
                )

        # Horizontal Search & Filter Bar Card
        with st.container(border=True):
            s_title_col, s_input_col, s_label_col, s_select_col = st.columns(
                [1.15, 1.85, 0.78, 0.85], gap="small", vertical_alignment="center"
            )
            with s_title_col:
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:10px;">'
                    '<div class="history-sec-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg></div>'
                    '<div><div class="history-card-heading" style="font-size:15.5px;">Search Applications</div>'
                    '<div class="history-card-sub">Find by ID or applicant details.</div></div></div>',
                    unsafe_allow_html=True
                )
            with s_input_col:
                search = st.text_input(
                    "Search Applications",
                    placeholder="Enter application ID, gender, education, property area...",
                    key="history_search",
                    label_visibility="collapsed"
                )
            with s_label_col:
                st.markdown(
                    '<div style="display:flex;align-items:center;justify-content:flex-end;gap:7px;color:#0B2457;font-size:13px;font-weight:750;white-space:nowrap;">'
                    '<span class="history-sec-icon" style="width:30px;height:30px;flex-basis:30px;"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg></span>'
                    'Filter by Prediction</div>',
                    unsafe_allow_html=True
                )
            with s_select_col:
                status_filter = st.selectbox(
                    "Filter by Prediction",
                    ["All", "Approved", "Rejected"],
                    key="history_status_filter",
                    label_visibility="collapsed"
                )

        filtered_df = history_df.copy()
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["prediction"].eq(status_filter)]

        if search.strip():
            raw_query = search.strip()
            query = raw_query.casefold()
            normalized_id_token = query.lstrip("#").strip()

            exact_id_matches = pd.DataFrame()
            if normalized_id_token.isdigit():
                target_id = int(normalized_id_token)
                exact_id_matches = filtered_df[
                    pd.to_numeric(filtered_df["id"], errors="coerce") == target_id
                ]

            if not exact_id_matches.empty:
                filtered_df = exact_id_matches
            else:
                searchable_columns = [
                    "gender", "married", "dependents", "education", "self_employed",
                    "property_area", "applicant_income", "coapplicant_income",
                    "loan_amount", "prediction"
                ]
                search_values = pd.Series("", index=filtered_df.index, dtype="string")
                for column in searchable_columns:
                    if column in filtered_df.columns:
                        search_values = search_values.str.cat(
                            filtered_df[column].fillna("").astype(str).str.casefold(), sep=" "
                        )
                filtered_df = filtered_df[search_values.str.contains(query, regex=False, na=False)]

        filtered_df = filtered_df.sort_values(
            ["created_at", "id"], ascending=[False, False], kind="stable"
        ).reset_index(drop=True)

        detail_source = filtered_df if not filtered_df.empty else history_df
        detail_source = detail_source.sort_values(
            ["created_at", "id"], ascending=[False, False], kind="stable"
        )
        available_ids = detail_source["id"].astype(int).tolist()
        current_selected = st.session_state.get("history_detail_select")
        if current_selected not in available_ids:
            st.session_state["history_detail_select"] = available_ids[0]

        # Main Two-Column Layout (~57% Left / ~43% Right)
        left_column, right_column = st.columns([1.34, 1.0], gap="medium")

        with left_column:
            # 1. Application Records Table Card
            with st.container(border=True):
                st.markdown(
                    '<div class="history-sec-head">'
                    '<div class="history-sec-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M8 4v16"/></svg></div>'
                    '<div><div class="history-card-heading">Application Records</div>'
                    '<div class="history-card-sub">Newest applications appear first.</div></div></div>',
                    unsafe_allow_html=True
                )

                if filtered_df.empty:
                    st.info("No applications match your search and filter. Try clearing the search or selecting All.")
                else:
                    rows_html = []
                    for _, row in filtered_df.iterrows():
                        app_id = int(row["id"])
                        pred = str(row["prediction"])
                        is_app = pred == "Approved"
                        badge_cls = "hp-badge-approved" if is_app else "hp-badge-rejected"
                        dot_bg = "#16A36A" if is_app else "#E5484D"
                        dot_sym = "✓" if is_app else "✕"
                        inc_str = f"₹{float(row['applicant_income']):,.2f}"
                        loan_str = f"{float(row['loan_amount']):,.0f}"
                        app_sc = f"{float(row['approval_score']) * 100:.2f}%"
                        rej_sc = f"{float(row['rejection_score']) * 100:.2f}%"
                        dt_val = pd.to_datetime(row["created_at"], errors="coerce")
                        dt_str = dt_val.strftime("%Y-%m-%d %H:%M") if pd.notna(dt_val) else safe_text(row["created_at"])

                        rows_html.append(
                            f"<tr>"
                            f"<td><strong>#{app_id}</strong></td>"
                            f"<td><span class='hp-badge {badge_cls}'><span class='hp-badge-dot' style='background:{dot_bg};'>{dot_sym}</span>{safe_text(pred)}</span></td>"
                            f"<td>{safe_text(inc_str)}</td>"
                            f"<td>{safe_text(loan_str)}</td>"
                            f"<td>{safe_text(app_sc)}</td>"
                            f"<td>{safe_text(rej_sc)}</td>"
                            f"<td>{safe_text(dt_str)}</td>"
                            f"</tr>"
                        )

                    clean_table_html = (
                        '<div class="history-table-wrap">'
                        '<table class="history-clean-table">'
                        '<thead><tr>'
                        '<th>ID</th>'
                        '<th>Prediction Status</th>'
                        '<th>Applicant Income</th>'
                        '<th>Loan Amt</th>'
                        '<th>Approval</th>'
                        '<th>Rejection</th>'
                        '<th>Submitted Date</th>'
                        '</tr></thead>'
                        f'<tbody>{"".join(rows_html)}</tbody>'
                        '</table></div>'
                    )
                    st.markdown(clean_table_html, unsafe_allow_html=True)

                st.caption(f"Showing {len(filtered_df)} of {total_records} records.")

            # 2. Individual Application Details Card (Bottom-Left)
            with st.container(border=True):
                det_head_col, det_sel_col = st.columns([1.45, 1.0], gap="small", vertical_alignment="center")
                with det_head_col:
                    st.markdown(
                        '<div class="history-sec-head" style="margin-bottom:0;">'
                        '<div class="history-sec-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/></svg></div>'
                        '<div><div class="history-card-heading">Individual Application Details</div>'
                        '<div class="history-card-sub">Select an application to view its submitted information and model assessment.</div></div></div>',
                        unsafe_allow_html=True
                    )
                with det_sel_col:
                    selected_id = st.selectbox(
                        "Select Application ID to View",
                        options=available_ids,
                        format_func=lambda value: f"Application #{value}",
                        key="history_detail_select"
                    )

                selected_record = next(
                    record for record in records if int(record["id"]) == int(selected_id)
                )
                prediction = str(selected_record["prediction"])
                is_sel_approved = prediction == "Approved"
                prediction_color = "#16A36A" if is_sel_approved else "#E5484D"
                tile_cls = "approved-tile" if is_sel_approved else "rejected-tile"
                tile_badge_bg = "#16A36A" if is_sel_approved else "#E5484D"
                tile_badge_sym = "✓" if is_sel_approved else "✕"
                submitted_date = str(selected_record["created_at"]).split(" ")[0]

                st.markdown(
                    f'<div class="history-overview">'
                    f'<div class="history-overview-tile">'
                    f'<div class="history-tile-ico"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18M7 15h4"/></svg></div>'
                    f'<div><div class="history-overview-label">Application ID</div>'
                    f'<div class="history-overview-value">#{int(selected_record["id"])}</div></div>'
                    f'</div>'
                    f'<div class="history-overview-tile {tile_cls}">'
                    f'<div class="history-tile-ico" style="background:{tile_badge_bg};color:#fff;border-radius:50%;font-weight:900;">{tile_badge_sym}</div>'
                    f'<div><div class="history-overview-label">Prediction</div>'
                    f'<div class="history-overview-value" style="color:{prediction_color};">{safe_text(prediction)}</div></div>'
                    f'</div>'
                    f'<div class="history-overview-tile">'
                    f'<div class="history-tile-ico"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg></div>'
                    f'<div><div class="history-overview-label">Submitted On</div>'
                    f'<div class="history-overview-value" style="font-size:15px;">{safe_text(submitted_date)}</div></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                credit_label = (
                    "Good (1)"
                    if float(selected_record["credit_history"]) == 1.0
                    else "Poor (0)"
                )
                personal_items = [
                    ("Gender", selected_record["gender"]),
                    ("Marital Status", selected_record["married"]),
                    ("Dependents", selected_record["dependents"]),
                    ("Education", selected_record["education"]),
                    ("Self Employed", selected_record["self_employed"]),
                    ("Property Area", selected_record["property_area"]),
                ]
                financial_items = [
                    ("Applicant Income", f"₹{float(selected_record['applicant_income']):,.2f}"),
                    ("Co-applicant Income", f"₹{float(selected_record['coapplicant_income']):,.2f}"),
                    ("Loan Amount (Dataset Units)", f"{float(selected_record['loan_amount']):,.2f}"),
                    ("Loan Term", f"{selected_record['loan_term']} months"),
                    ("Credit History", credit_label),
                ]
                personal_html = "".join(
                    f'<div class="history-info-line"><span>{label}:</span><strong>{safe_text(value)}</strong></div>'
                    for label, value in personal_items
                )
                financial_html = "".join(
                    f'<div class="history-info-line"><span>{label}:</span><strong>{safe_text(value)}</strong></div>'
                    for label, value in financial_items
                )
                st.markdown(
                    f'<div class="history-info-box">'
                    f'<div class="history-info-banner">⌃ &nbsp; View Complete Applicant Details</div>'
                    f'<div class="history-info-grid">'
                    f'<div class="history-info-col"><h4>👤 Personal Information</h4>{personal_html}</div>'
                    f'<div class="history-info-col"><h4>💰 Financial Information</h4>{financial_html}</div>'
                    f'</div></div><div style="height:8px;"></div>',
                    unsafe_allow_html=True
                )

        with right_column:
            selected_id = int(st.session_state["history_detail_select"])
            selected_record = next(
                record for record in records if int(record["id"]) == selected_id
            )

            # 1. Model Prediction Scores Card (Top-Right)
            with st.container(border=True):
                st.markdown(
                    '<div class="history-sec-head">'
                    '<div class="history-sec-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M3 3v18h18"/><path d="M8 16v-5m5 5V5m5 11V9"/></svg></div>'
                    '<div><div class="history-card-heading">Model Prediction Scores</div>'
                    '<div class="history-card-sub">Model-estimated class scores for the selected application.</div></div></div>',
                    unsafe_allow_html=True
                )
                approval_score = max(0.0, min(1.0, float(selected_record["approval_score"])))
                rejection_score = max(0.0, min(1.0, float(selected_record["rejection_score"])))
                st.markdown(
                    f'<div class="history-score-grid">'
                    f'<div class="history-score">'
                    f'<div class="history-score-top">'
                    f'<div class="history-score-badge" style="background:#16A36A;">✓</div>'
                    f'<div><div class="history-score-label">Approval Score</div>'
                    f'<div class="history-score-value" style="color:#16A36A;">{approval_score*100:.2f}%</div></div>'
                    f'</div>'
                    f'<div class="history-score-track"><div class="history-score-fill" style="width:{approval_score*100:.2f}%;background:#16A36A;"></div></div>'
                    f'</div>'
                    f'<div class="history-score rej">'
                    f'<div class="history-score-top">'
                    f'<div class="history-score-badge" style="background:#E5484D;">✕</div>'
                    f'<div><div class="history-score-label">Rejection Score</div>'
                    f'<div class="history-score-value" style="color:#E5484D;">{rejection_score*100:.2f}%</div></div>'
                    f'</div>'
                    f'<div class="history-score-track"><div class="history-score-fill" style="width:{rejection_score*100:.2f}%;background:#E5484D;"></div></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # 2. Download Assessment Report Card (Middle-Right)
            with st.container(border=True):
                d_info, d_btn = st.columns([1.05, 1.0], gap="small", vertical_alignment="center")
                with d_info:
                    st.markdown(
                        '<div class="history-sec-head" style="margin-bottom:0;">'
                        '<div class="history-sec-icon red"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M9 15h6"/></svg></div>'
                        '<div><div class="history-card-heading" style="font-size:15px;">Download Assessment Report</div>'
                        '<div class="history-card-sub">Detailed PDF report for this application.</div></div></div>',
                        unsafe_allow_html=True
                    )
                with d_btn:
                    try:
                        pdf_bytes = generate_prediction_pdf(selected_record)
                        st.download_button(
                            "⬇ Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"Loan_Application_{selected_id}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"history_pdf_{selected_id}"
                        )
                    except Exception as error:
                        st.error(f"PDF generation failed: {error}")

            # 3. Export Records Card (Lower-Middle-Right)
            with st.container(border=True):
                e_info, e_btn = st.columns([1.05, 1.0], gap="small", vertical_alignment="center")
                with e_info:
                    st.markdown(
                        '<div class="history-sec-head" style="margin-bottom:0;">'
                        '<div class="history-sec-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg></div>'
                        '<div><div class="history-card-heading" style="font-size:15px;">Export Records</div>'
                        '<div class="history-card-sub">Export currently filtered records to CSV.</div></div></div>',
                        unsafe_allow_html=True
                    )
                with e_btn:
                    export_df = filtered_df.sort_values(
                        ["created_at", "id"], ascending=[False, False], kind="stable"
                    )
                    export_data = export_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download History CSV",
                        data=export_data,
                        file_name="loan_prediction_history.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="history_csv_download"
                    )

            # 4. Manage Records Card (Bottom-Right)
            with st.container(border=True):
                st.markdown(
                    '<div class="history-sec-head">'
                    '<div class="history-sec-icon red"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg></div>'
                    '<div><div class="history-card-heading" style="font-size:15px;">Manage Records</div>'
                    '<div class="history-card-sub">Permanently remove an application from prediction history.</div></div></div>',
                    unsafe_allow_html=True
                )
                record_options = {
                    f"Application #{int(row['id'])} — {row['prediction']} — {row['created_at']}": int(row["id"])
                    for _, row in history_df.iterrows()
                }
                m_sel_col, m_btn_col = st.columns([1.2, 1.0], gap="small", vertical_alignment="center")
                with m_sel_col:
                    delete_label = st.selectbox(
                        "Select Application to Delete",
                        options=list(record_options.keys()),
                        key="history_delete_select",
                        label_visibility="collapsed"
                    )
                delete_id = record_options[delete_label]
                with m_btn_col:
                    if st.button(
                        "🗑 Delete Application",
                        use_container_width=True,
                        key="history_request_delete"
                    ):
                        st.session_state["history_confirm_delete_id"] = delete_id

                if st.session_state.get("history_confirm_delete_id") == delete_id:
                    st.warning(
                        f"Confirm permanent deletion of Application #{delete_id}? This action cannot be undone."
                    )
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button(
                            "Confirm Delete", type="primary",
                            use_container_width=True, key=f"history_confirm_{delete_id}"
                        ):
                            if delete_prediction(delete_id):
                                st.session_state.pop("history_confirm_delete_id", None)
                                if st.session_state.get("history_detail_select") == delete_id:
                                    st.session_state.pop("history_detail_select", None)
                                st.success(f"Application #{delete_id} deleted successfully.")
                                st.rerun()
                            else:
                                st.error("Unable to delete the selected record.")
                    with cancel_col:
                        if st.button(
                            "Cancel", use_container_width=True,
                            key=f"history_cancel_{delete_id}"
                        ):
                            st.session_state.pop("history_confirm_delete_id", None)
                            st.rerun()


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.markdown("""
    <style>
    /* Scoped Model Performance Styling */
    .mp-hero {
        position:relative; overflow:hidden; min-height:142px; padding:22px 32px;
        margin:2px 0 14px; border-radius:18px; color:#fff;
        background:
          radial-gradient(circle at 75% 28%,rgba(25,133,255,.42),transparent 26%),
          linear-gradient(110deg,#061b50 0%,#073b9b 52%,#075ee0 100%);
        border:1px solid rgba(105,165,255,.36);
        box-shadow:0 9px 22px rgba(9,54,132,.16);
        display:flex; align-items:center; justify-content:space-between;
    }
    .mp-hero:before {
        content:""; position:absolute; right:18%; bottom:-110px; width:410px; height:210px;
        border-radius:50%; border:1px solid rgba(93,164,255,.18);
        box-shadow:0 0 0 26px rgba(93,164,255,.04),0 0 0 54px rgba(93,164,255,.03);
    }
    .mp-hero-copy { position:relative; z-index:2; max-width:620px; }
    .mp-hero-eyebrow {
        display:inline-flex; align-items:center; gap:7px; padding:4px 12px;
        border-radius:30px; color:#fff; background:rgba(32,112,255,.52);
        border:1px solid rgba(191,219,254,.28); font-size:11px; font-weight:800;
        letter-spacing:.045em;
    }
    .mp-hero h1 { color:#fff !important; font-size:29px !important; line-height:1.15 !important; margin:7px 0 4px !important; font-weight:850 !important; }
    .mp-hero p { color:#e7f1ff !important; font-size:14px; line-height:1.45; margin:0 !important; max-width:560px; }
    .mp-hero-art { position:relative; z-index:2; width:310px; height:132px; flex:0 0 310px; }

    /* KPI Cards */
    .mp-kpi {
        display:flex; align-items:center; justify-content:space-between; gap:12px;
        min-height:94px; height:100%; padding:15px 18px; border-radius:16px;
        border:1px solid #d4e3fc; background:linear-gradient(120deg,#ffffff 0%,#f3f8ff 100%);
        box-shadow:0 4px 14px rgba(16,42,94,.05);
        transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .mp-kpi:hover { transform:translateY(-2px); box-shadow:0 9px 20px rgba(16,42,94,.10); border-color:#b5cff8; }
    .mp-kpi.green { background:linear-gradient(120deg,#ffffff 0%,#eefcf4 100%); border-color:#bcead3; }
    .mp-kpi.purple { background:linear-gradient(120deg,#ffffff 0%,#f6f3ff 100%); border-color:#dcd3fe; }
    .mp-kpi-left { display:flex; align-items:center; gap:12px; min-width:0; }
    .mp-kpi-ico {
        width:42px; height:42px; flex:0 0 42px; border-radius:12px;
        display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:800;
    }
    .mp-kpi-label { color:#52658a; font-size:12px; font-weight:750; line-height:1.25; }
    .mp-kpi-value { color:#123b83; font-size:27px; font-weight:850; line-height:1.15; margin-top:2px; }
    .mp-kpi.green .mp-kpi-value { color:#07884b; }
    .mp-kpi.purple .mp-kpi-value { color:#6236c9; }
    .mp-kpi-spark { width:84px; height:38px; flex:0 0 84px; }

    /* Section Card Header */
    .mp-sec-head { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
    .mp-sec-ico {
        width:36px; height:36px; flex:0 0 36px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        background:#eaf2ff; color:#0866e8;
    }
    .mp-sec-title { color:#0b2457; font-size:17px; font-weight:850; margin:0; line-height:1.2; }
    .mp-sec-sub { color:#637c9f; font-size:12px; margin:2px 0 0; }

    /* Algorithm Comparison Table */
    .mp-table-wrap {
        width:100%; overflow-x:auto; border:1px solid #cfe0fa; border-radius:12px; background:#ffffff;
    }
    table.mp-comp-table {
        width:100%; border-collapse:collapse; background:#ffffff; font-size:12.5px;
    }
    table.mp-comp-table thead th {
        background:#eaf2ff; color:#0b2457; font-weight:800; text-align:left;
        padding:11px 11px; border-bottom:1px solid #cfe0fa; white-space:nowrap; font-size:12px;
    }
    table.mp-comp-table tbody td {
        padding:11px 11px; color:#142e61; border-bottom:1px solid #e8f0fc;
        white-space:nowrap; font-weight:500; height:42px;
    }
    table.mp-comp-table tbody tr:hover td { background:#edf4ff; }
    table.mp-comp-table tbody tr:last-child td { border-bottom:0; }
    .mp-badge-selected {
        display:inline-flex; align-items:center; gap:5px; padding:4px 10px;
        border-radius:999px; font-size:11px; font-weight:800;
        background:#ddf7e9; color:#117a4b; border:1px solid #b5e8cd;
    }
    .mp-badge-other {
        display:inline-flex; align-items:center; gap:5px; padding:4px 10px;
        border-radius:999px; font-size:11px; font-weight:700;
        background:#eef3fb; color:#4d668e; border:1px solid #d5e2f7;
    }
    .mp-selected-banner {
        display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;
        padding:11px 16px; margin-top:12px; margin-bottom:14px; border-radius:12px;
        background:linear-gradient(110deg,#ecfdf5 0%,#f3fbf8 100%);
        border:1px solid #b6e7ce; color:#0e6b40; font-size:13px; font-weight:700;
    }

    @media(max-width:950px) {
        .mp-hero { padding:18px 20px; }
        .mp-hero h1 { font-size:24px !important; }
        .mp-hero-art { width:210px; flex-basis:210px; opacity:.45; }
    }
    @media(max-width:650px) {
        .mp-hero-art { display:none; }
    }
    </style>
    """, unsafe_allow_html=True)

    # 1. Hero Banner
    st.markdown("""
    <div class="mp-hero">
      <div class="mp-hero-copy">
        <div class="mp-hero-eyebrow">📊 &nbsp; MACHINE LEARNING EVALUATION</div>
        <h1>Model Performance Analysis</h1>
        <p>Compare classification algorithms, inspect confusion matrix results, and review ROC-AUC discrimination performance.</p>
      </div>
      <svg class="mp-hero-art" viewBox="0 0 380 165" fill="none" aria-hidden="true">
        <g opacity=".95">
          <path d="M195 42h90a8 8 0 0 1 8 8v100H187V50a8 8 0 0 1 8-8Z" fill="#f8fbff" stroke="#7eb8ff" stroke-width="3"/>
          <path d="M215 31h50v16h-50z" rx="5" fill="#9bd0ff" stroke="#e8f5ff" stroke-width="2"/>
          <circle cx="240" cy="31" r="7" fill="#d9efff" stroke="#fff" stroke-width="2"/>
          <text x="240" y="68" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">MODEL</text>
          <text x="240" y="83" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">METRICS</text>
          <g stroke="#0877ed" stroke-width="3" stroke-linecap="round">
            <rect x="205" y="96" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 102 3 3 6-7M226 102h40"/>
            <rect x="205" y="119" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 125 3 3 6-7M226 125h40"/>
          </g>
          <path d="M300 80 342 58l35 22v6h-77z" fill="#d8eaff" stroke="#78baff" stroke-width="3"/>
          <path d="M306 87h68v61h-68z" fill="#e9f4ff" stroke="#78baff" stroke-width="3"/>
          <path d="M297 80h86M314 90v55m17-55v55m17-55v55m17-55v55M300 149h78" stroke="#c3ddff" stroke-width="5" stroke-linecap="round"/>
          <path d="M292 151h96" stroke="#fff" stroke-width="5" stroke-linecap="round"/>
          <path d="M324 69h34" stroke="#4d91e8" stroke-width="3" stroke-linecap="round"/>
          <text x="342" y="78" fill="#17428c" font-size="12" font-weight="800" text-anchor="middle">BANK</text>
          <g>
            <ellipse cx="147" cy="145" rx="29" ry="7" fill="#ffcf4b"/>
            <path d="M118 132h58v13h-58z" fill="#f5a900" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="132" rx="29" ry="7" fill="#ffd34f" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="116" rx="25" ry="7" fill="#ffcf4b"/>
            <path d="M122 103h50v13h-50z" fill="#eea500" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="103" rx="25" ry="7" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="88" rx="20" ry="6" fill="#ffd34f"/>
            <path d="M127 76h40v12h-40z" fill="#eaa000" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="76" rx="20" ry="6" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
          </g>
          <path d="M92 142c-14-21-13-37-2-49 2 19 9 29 21 37M99 128c-1-18 7-30 23-35-5 16-8 28-6 43" stroke="#29c5f5" stroke-width="5" stroke-linecap="round"/>
          <path d="M91 143c-13-3-22-11-26-24 15 3 24 10 26 24ZM111 144c4-14 13-22 28-25-4 14-13 22-28 25Z" fill="#17a9ec"/>
          <path d="M26 34h28m-14-14v28M72 57h17m-8-9v18" stroke="#7bdcff" stroke-width="2" stroke-linecap="round" opacity=".8"/>
          <circle cx="30" cy="87" r="3" fill="#7bdcff"/><circle cx="91" cy="22" r="2.5" fill="#7bdcff"/>
        </g>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    if RESULT_PATH.exists():
        results = pd.read_csv(RESULT_PATH)
        best = results.iloc[0]
        best_name = html_lib.escape(str(best["Model"]))

        # 2. Performance KPI Cards (driven dynamically from model_comparison.csv)
        k1, k2, k3 = st.columns(3, gap="medium")
        with k1:
            st.markdown(
                f'<div class="mp-kpi">'
                f'<div class="mp-kpi-left">'
                f'<div class="mp-kpi-ico" style="background:#e6f0ff;color:#1261e8;">🎯</div>'
                f'<div><div class="mp-kpi-label">Accuracy ({best_name})</div>'
                f'<div class="mp-kpi-value">{float(best["Accuracy"]):.2%}</div></div>'
                f'</div>'
                f'<svg class="mp-kpi-spark" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#1261e8" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                f'</div>',
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f'<div class="mp-kpi green">'
                f'<div class="mp-kpi-left">'
                f'<div class="mp-kpi-ico" style="background:#ddf7e9;color:#07884b;">⚖️</div>'
                f'<div><div class="mp-kpi-label">Macro F1 (Selection Metric)</div>'
                f'<div class="mp-kpi-value">{float(best["Macro F1"]):.2%}</div></div>'
                f'</div>'
                f'<svg class="mp-kpi-spark" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#07884b" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                f'</div>',
                unsafe_allow_html=True
            )
        with k3:
            st.markdown(
                f'<div class="mp-kpi purple">'
                f'<div class="mp-kpi-left">'
                f'<div class="mp-kpi-ico" style="background:#ede9fe;color:#6236c9;">📈</div>'
                f'<div><div class="mp-kpi-label">ROC-AUC (Tie-Breaker)</div>'
                f'<div class="mp-kpi-value">{float(best["ROC AUC"]):.2%}</div></div>'
                f'</div>'
                f'<svg class="mp-kpi-spark" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#6236c9" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # 3. First Row: Algorithm Comparison (~60%) & Model Accuracy Comparison (~40%)
        top_left, top_right = st.columns([1.5, 1.0], gap="medium")

        with top_left:
            with st.container(border=True):
                st.markdown(
                    '<div class="mp-sec-head">'
                    '<div class="mp-sec-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M8 4v16"/></svg></div>'
                    '<div><div class="mp-sec-title">Algorithm Comparison</div>'
                    '<div class="mp-sec-sub">Evaluated on a stratified 80/20 test split (random state 42).</div></div></div>',
                    unsafe_allow_html=True
                )

                comp_rows = []
                for idx, row in results.iterrows():
                    model_name = html_lib.escape(str(row["Model"]))
                    is_selected = idx == 0
                    status_pill = (
                        '<span class="mp-badge-selected">✓ Selected</span>'
                        if is_selected else
                        '<span class="mp-badge-other">Evaluated</span>'
                    )
                    row_bg = "background:#f2f8ff;" if is_selected else ""
                    comp_rows.append(
                        f"<tr style='{row_bg}'>"
                        f"<td><strong>{model_name}</strong></td>"
                        f"<td>{status_pill}</td>"
                        f"<td>{float(row['Accuracy']):.2%}</td>"
                        f"<td>{float(row['Precision']):.2%}</td>"
                        f"<td>{float(row['Recall']):.2%}</td>"
                        f"<td>{float(row['F1 Score']):.2%}</td>"
                        f"<td><strong>{float(row['Macro F1']):.2%}</strong></td>"
                        f"<td><strong>{float(row['ROC AUC']):.2%}</strong></td>"
                        f"</tr>"
                    )

                comp_table_html = (
                    '<div class="mp-table-wrap">'
                    '<table class="mp-comp-table">'
                    '<thead><tr>'
                    '<th>Model</th>'
                    '<th>Status</th>'
                    '<th>Accuracy</th>'
                    '<th>Precision</th>'
                    '<th>Recall</th>'
                    '<th>F1 Score</th>'
                    '<th>Macro F1</th>'
                    '<th>ROC-AUC</th>'
                    '</tr></thead>'
                    f'<tbody>{"".join(comp_rows)}</tbody>'
                    '</table></div>'
                )
                st.markdown(comp_table_html, unsafe_allow_html=True)

                st.markdown(
                    f'<div class="mp-selected-banner">'
                    f'<span>✓ &nbsp; Selected Production Model: <strong>{best_name}</strong></span>'
                    f'<span style="font-size:12px;color:#15803d;">Highest Macro F1 ({float(best["Macro F1"]):.2%}) &amp; ROC-AUC ({float(best["ROC AUC"]):.2%})</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with top_right:
            with st.container(border=True):
                st.markdown(
                    '<div class="mp-sec-head">'
                    '<div class="mp-sec-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M3 3v18h18"/><path d="M8 16v-5m5 5V5m5 11V9"/></svg></div>'
                    '<div><div class="mp-sec-title">Model Accuracy Comparison</div>'
                    '<div class="mp-sec-sub">Test accuracy across all four candidate models.</div></div></div>',
                    unsafe_allow_html=True
                )

                fig_acc, ax_acc = plt.subplots(figsize=(5.2, 2.75), dpi=135)
                fig_acc.patch.set_facecolor("#ffffff")
                ax_acc.set_facecolor("#f8faff")

                models_list = [str(m) for m in results["Model"]]
                acc_values = [float(a) * 100 for a in results["Accuracy"]]
                bar_colors = [
                    "#1261e8" if i == 0 else "#7aa8f8"
                    for i in range(len(models_list))
                ]

                bars = ax_acc.barh(
                    models_list, acc_values, color=bar_colors, height=0.52,
                    edgecolor="none", zorder=3
                )
                ax_acc.invert_yaxis()
                ax_acc.set_xlim(0, 102)
                ax_acc.xaxis.grid(True, color="#dce6f4", linewidth=0.8, zorder=0)
                for bar, val in zip(bars, acc_values):
                    ax_acc.text(
                        val + 1.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}%", va="center", ha="left",
                        fontsize=8.5, color="#10234f", fontweight="bold"
                    )
                ax_acc.set_xlabel("Accuracy (%)", fontsize=8.5, color="#52658a", labelpad=5)
                ax_acc.tick_params(axis="both", labelsize=8.5, colors="#20365e", length=0, pad=5)
                for spine in ax_acc.spines.values():
                    spine.set_visible(False)
                ax_acc.set_axisbelow(True)
                fig_acc.tight_layout(pad=0.9)
                st.pyplot(fig_acc, use_container_width=True)
                plt.close(fig_acc)

        # 4. Second Row: Compact Confusion Matrix (50%) & ROC Curve (50%)
        cm_col, roc_col = st.columns(2, gap="medium")

        # Compute actual test-set confusion matrix & ROC curve using the real saved model and dataset
        # (reusing exact stratified 80/20 split, random_state=42, and Loan_Status mapping)
        eval_ready = False
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

            X_eval = df.drop(columns=["Loan_ID", "Loan_Status"])
            y_eval = df["Loan_Status"].map({"Y": 1, "N": 0})
            _, X_test_eval, _, y_test_eval = train_test_split(
                X_eval, y_eval, test_size=0.2, stratify=y_eval, random_state=42
            )
            y_pred_eval = model.predict(X_test_eval)
            classes_eval = list(model.classes_)
            y_prob_eval = model.predict_proba(X_test_eval)[:, classes_eval.index(1)]
            cm_matrix = confusion_matrix(y_test_eval, y_pred_eval, labels=[0, 1])
            fpr_vals, tpr_vals, _ = roc_curve(y_test_eval, y_prob_eval)
            auc_val = roc_auc_score(y_test_eval, y_prob_eval)
            eval_ready = True
        except Exception:
            eval_ready = False

        with cm_col:
            with st.container(border=True):
                st.markdown(
                    '<div class="mp-sec-head">'
                    '<div class="mp-sec-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg></div>'
                    '<div><div class="mp-sec-title">Confusion Matrix</div>'
                    '<div class="mp-sec-sub">True vs. predicted classifications on the holdout test set.</div></div></div>',
                    unsafe_allow_html=True
                )
                if eval_ready:
                    fig_cm, ax_cm = plt.subplots(figsize=(5.0, 3.35), dpi=135)
                    fig_cm.patch.set_facecolor("#ffffff")
                    ax_cm.set_facecolor("#ffffff")

                    im = ax_cm.imshow(cm_matrix, cmap="Blues", aspect="auto")
                    cbar = fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8, colors="#344967")
                    cbar.outline.set_visible(False)

                    labels_cm = ["Rejected", "Approved"]
                    ax_cm.set_xticks([0, 1])
                    ax_cm.set_yticks([0, 1])
                    ax_cm.set_xticklabels(labels_cm, fontsize=9, fontweight="semibold", color="#10234f")
                    ax_cm.set_yticklabels(labels_cm, fontsize=9, fontweight="semibold", color="#10234f")
                    ax_cm.set_xlabel("Predicted Label", fontsize=9, fontweight="bold", color="#20365e", labelpad=6)
                    ax_cm.set_ylabel("Actual Label", fontsize=9, fontweight="bold", color="#20365e", labelpad=6)
                    ax_cm.set_title(f"Confusion Matrix — {best['Model']}", fontsize=10, fontweight="bold", color="#0b2457", pad=8)

                    thresh = cm_matrix.max() / 2.0
                    for i in range(2):
                        for j in range(2):
                            val = int(cm_matrix[i, j])
                            ax_cm.text(
                                j, i, f"{val}",
                                ha="center", va="center",
                                fontsize=13, fontweight="bold",
                                color="#ffffff" if val > thresh else "#10234f"
                            )

                    for spine in ax_cm.spines.values():
                        spine.set_edgecolor("#cfe0fa")
                        spine.set_linewidth(0.8)

                    fig_cm.tight_layout(pad=0.9)
                    st.pyplot(fig_cm, use_container_width=True)
                    plt.close(fig_cm)
                elif CONFUSION_PATH.exists():
                    st.image(str(CONFUSION_PATH), use_container_width=True)
                else:
                    st.info("Confusion matrix plot not found.")

        with roc_col:
            with st.container(border=True):
                st.markdown(
                    '<div class="mp-sec-head">'
                    '<div class="mp-sec-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 5-6"/></svg></div>'
                    '<div><div class="mp-sec-title">ROC Curve</div>'
                    '<div class="mp-sec-sub">Receiver Operating Characteristic curve and Area Under Curve (AUC).</div></div></div>',
                    unsafe_allow_html=True
                )
                if eval_ready:
                    fig_roc, ax_roc = plt.subplots(figsize=(5.0, 3.35), dpi=135)
                    fig_roc.patch.set_facecolor("#ffffff")
                    ax_roc.set_facecolor("#f8faff")

                    ax_roc.plot(
                        fpr_vals, tpr_vals,
                        color="#1261e8", linewidth=2.4,
                        label=f"{best['Model']} (AUC = {auc_val:.2f})",
                        zorder=4
                    )
                    ax_roc.fill_between(fpr_vals, tpr_vals, alpha=0.10, color="#1261e8", zorder=2)
                    ax_roc.plot(
                        [0, 1], [0, 1],
                        linestyle="--", linewidth=1.3, color="#94a3b8",
                        label="Random Classifier (AUC = 0.50)",
                        zorder=3
                    )

                    ax_roc.set_xlim(-0.02, 1.02)
                    ax_roc.set_ylim(-0.02, 1.04)
                    ax_roc.grid(True, color="#dce6f4", linewidth=0.8, zorder=0)
                    ax_roc.set_xlabel("False Positive Rate (Positive label: 1)", fontsize=9, fontweight="bold", color="#20365e", labelpad=6)
                    ax_roc.set_ylabel("True Positive Rate (Positive label: 1)", fontsize=9, fontweight="bold", color="#20365e", labelpad=6)
                    ax_roc.set_title(f"ROC Curve — {best['Model']}", fontsize=10, fontweight="bold", color="#0b2457", pad=8)
                    ax_roc.tick_params(axis="both", labelsize=8.5, colors="#344967")
                    ax_roc.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cfe0fa", fontsize=8.5)

                    for spine in ax_roc.spines.values():
                        spine.set_visible(False)

                    fig_roc.tight_layout(pad=0.9)
                    st.pyplot(fig_roc, use_container_width=True)
                    plt.close(fig_roc)
                elif ROC_PATH.exists():
                    st.image(str(ROC_PATH), use_container_width=True)
                else:
                    st.info("ROC curve plot not found.")

    else:
        st.warning("Model evaluation results not found.")


# =====================================================
# DATASET ANALYTICS
# =====================================================

elif page == "Dataset Analytics":

    st.markdown("""
    <style>
    /* Scoped Dataset Analytics Refinement */
    .da-kpi-grid {
        display:grid;
        grid-template-columns:repeat(4, minmax(0, 1fr));
        gap:14px;
        margin:2px 0 14px;
    }
    .da-kpi-card {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:10px;
        min-height:96px;
        padding:15px 18px;
        border-radius:16px;
        border:1px solid #d4e3fc;
        background:linear-gradient(120deg,#ffffff 0%,#f3f8ff 100%);
        box-shadow:0 4px 14px rgba(16,42,94,.05);
        transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .da-kpi-card:hover {
        transform:translateY(-2px);
        box-shadow:0 9px 20px rgba(16,42,94,.10);
        border-color:#b5cff8;
    }
    .da-kpi-card.green {
        background:linear-gradient(120deg,#ffffff 0%,#eefcf4 100%);
        border-color:#bcead3;
    }
    .da-kpi-card.red {
        background:linear-gradient(120deg,#ffffff 0%,#fff2f3 100%);
        border-color:#fad0d4;
    }
    .da-kpi-card.purple {
        background:linear-gradient(120deg,#ffffff 0%,#f6f3ff 100%);
        border-color:#dcd3fe;
    }
    .da-kpi-left { display:flex; align-items:center; gap:12px; min-width:0; }
    .da-kpi-ico {
        width:42px; height:42px; flex:0 0 42px; border-radius:12px;
        display:flex; align-items:center; justify-content:center;
        font-size:17px; font-weight:850;
    }
    .da-kpi-label { color:#52658a; font-size:12px; font-weight:750; line-height:1.25; }
    .da-kpi-val { color:#123b83; font-size:27px; font-weight:850; line-height:1.15; margin-top:2px; }
    .da-kpi-card.green .da-kpi-val { color:#07884b; }
    .da-kpi-card.red .da-kpi-val { color:#c1273d; }
    .da-kpi-card.purple .da-kpi-val { color:#6236c9; font-size:22px; }
    .da-kpi-spark { width:78px; height:36px; flex:0 0 78px; }

    .da-sec-head { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
    .da-sec-ico {
        width:36px; height:36px; flex:0 0 36px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        background:#eaf2ff; color:#0866e8;
    }
    .da-sec-title { color:#0b2457; font-size:17px; font-weight:850; margin:0; line-height:1.2; }
    .da-sec-sub { color:#637c9f; font-size:12px; margin:2px 0 0; }

    /* Fixed-height vertically scrollable preview table fitting all 13 columns without horizontal scroll */
    .da-preview-scroll {
        width:100%;
        max-width:100%;
        max-height:420px;
        overflow-y:auto;
        overflow-x:hidden;
        border:1px solid #cfe0fa;
        border-radius:12px;
        background:#ffffff;
        margin-bottom:8px;
    }
    table.da-preview-table {
        width:100%;
        max-width:100%;
        table-layout:fixed;
        border-collapse:separate;
        border-spacing:0;
        background:#ffffff;
        font-size:10.5px;
        line-height:1.25;
    }
    table.da-preview-table thead th {
        position:sticky;
        top:0;
        z-index:2;
        background:#eaf2ff;
        color:#0b2457;
        font-weight:800;
        text-align:left;
        padding:7px 5px;
        border-bottom:1px solid #c5d8f8;
        border-right:1px solid #dce8fa;
        white-space:normal;
        overflow-wrap:anywhere;
        word-break:break-word;
        hyphens:auto;
        font-size:10px;
        line-height:1.2;
        vertical-align:bottom;
    }
    table.da-preview-table thead th:last-child { border-right:0; }
    table.da-preview-table tbody td {
        padding:6px 5px;
        color:#142e61;
        border-bottom:1px solid #ebf1fb;
        border-right:1px solid #f0f5fc;
        white-space:normal;
        overflow-wrap:anywhere;
        word-break:break-word;
        font-weight:500;
        font-size:10.5px;
        vertical-align:middle;
    }
    table.da-preview-table tbody td:last-child { border-right:0; }
    table.da-preview-table tbody td.num-col { text-align:right; font-variant-numeric:tabular-nums; }
    table.da-preview-table tbody tr:nth-child(even) td { background:#f8fbff; }
    table.da-preview-table tbody tr:hover td { background:#edf4ff; }
    table.da-preview-table tbody tr:last-child td { border-bottom:0; }
    table.da-preview-table tbody td.da-na-cell {
        background:#fffbeb !important;
        color:#b45309 !important;
        font-weight:700;
        font-style:italic;
    }
    .da-preview-scroll .da-pill-y,
    .da-preview-scroll .da-pill-n {
        padding:2px 5px;
        font-size:9.5px;
        gap:2px;
        white-space:nowrap;
    }

    /* Statistical summary table with clean numeric alignment */
    .da-stats-scroll {
        width:100%;
        max-height:360px;
        overflow:auto;
        border:1px solid #cfe0fa;
        border-radius:12px;
        background:#ffffff;
        margin-bottom:8px;
    }
    table.da-stats-table {
        width:100%;
        border-collapse:separate;
        border-spacing:0;
        background:#ffffff;
        font-size:12px;
    }
    table.da-stats-table thead th {
        position:sticky;
        top:0;
        z-index:2;
        background:#eaf2ff;
        color:#0b2457;
        font-weight:800;
        padding:10px 11px;
        border-bottom:1px solid #c5d8f8;
        border-right:1px solid #dce8fa;
        white-space:nowrap;
        font-size:11.5px;
    }
    table.da-stats-table thead th:first-child { text-align:left; }
    table.da-stats-table thead th:not(:first-child) { text-align:right; }
    table.da-stats-table thead th:last-child { border-right:0; }
    table.da-stats-table tbody td {
        padding:9px 11px;
        color:#142e61;
        border-bottom:1px solid #ebf1fb;
        border-right:1px solid #f0f5fc;
        white-space:nowrap;
        font-variant-numeric:tabular-nums;
    }
    table.da-stats-table tbody td:first-child { text-align:left; font-weight:750; color:#0b2457; }
    table.da-stats-table tbody td:not(:first-child) { text-align:right; }
    table.da-stats-table tbody td:last-child { border-right:0; }
    table.da-stats-table tbody tr:nth-child(even) td { background:#f8fbff; }
    table.da-stats-table tbody tr:hover td { background:#edf4ff; }
    table.da-stats-table tbody tr:last-child td { border-bottom:0; }

    .da-pill-y {
        display:inline-flex; align-items:center; gap:5px; padding:3px 10px;
        border-radius:999px; font-size:11px; font-weight:800;
        background:#ddf7e9; color:#117a4b; border:1px solid #b5e8cd;
    }
    .da-pill-n {
        display:inline-flex; align-items:center; gap:5px; padding:3px 10px;
        border-radius:999px; font-size:11px; font-weight:800;
        background:#ffe5e7; color:#cf2633; border:1px solid #f9c0c5;
    }
    .da-status-ok {
        display:flex; align-items:center; gap:8px; padding:12px 16px;
        border-radius:12px; background:linear-gradient(110deg,#ecfdf5 0%,#f3fbf8 100%);
        border:1px solid #b6e7ce; color:#0e6b40; font-size:13px; font-weight:750;
        margin-bottom:8px;
    }

    @media(max-width:1050px) {
        .da-kpi-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    }
    @media(max-width:650px) {
        .da-kpi-grid { grid-template-columns:1fr; }
    }
    </style>
    """, unsafe_allow_html=True)

    # 1. Horizontal Blue-Gradient Hero Banner
    st.markdown("""
    <div class="lp-pred-hero">
      <div class="lp-pred-hero-copy">
        <div class="lp-pred-eyebrow">🗄️ &nbsp; EXPLORATORY DATA ANALYTICS</div>
        <h1>Dataset Analytics</h1>
        <p>Inspect the historical loan dataset structure, summary statistics, missing values, approval split, and income distribution.</p>
      </div>
      <svg class="lp-pred-hero-art" viewBox="0 0 380 165" fill="none" aria-hidden="true">
        <g opacity=".95">
          <path d="M195 42h90a8 8 0 0 1 8 8v100H187V50a8 8 0 0 1 8-8Z" fill="#f8fbff" stroke="#7eb8ff" stroke-width="3"/>
          <path d="M215 31h50v16h-50z" rx="5" fill="#9bd0ff" stroke="#e8f5ff" stroke-width="2"/>
          <circle cx="240" cy="31" r="7" fill="#d9efff" stroke="#fff" stroke-width="2"/>
          <text x="240" y="68" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">DATASET</text>
          <text x="240" y="83" fill="#123b91" font-size="13" font-weight="800" text-anchor="middle">INSIGHTS</text>
          <g stroke="#0877ed" stroke-width="3" stroke-linecap="round">
            <rect x="205" y="96" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 102 3 3 6-7M226 102h40"/>
            <rect x="205" y="119" width="12" height="12" rx="2" fill="#e5f4ff"/>
            <path d="m208 125 3 3 6-7M226 125h40"/>
          </g>
          <path d="M300 80 342 58l35 22v6h-77z" fill="#d8eaff" stroke="#78baff" stroke-width="3"/>
          <path d="M306 87h68v61h-68z" fill="#e9f4ff" stroke="#78baff" stroke-width="3"/>
          <path d="M297 80h86M314 90v55m17-55v55m17-55v55m17-55v55M300 149h78" stroke="#c3ddff" stroke-width="5" stroke-linecap="round"/>
          <path d="M292 151h96" stroke="#fff" stroke-width="5" stroke-linecap="round"/>
          <path d="M324 69h34" stroke="#4d91e8" stroke-width="3" stroke-linecap="round"/>
          <text x="342" y="78" fill="#17428c" font-size="12" font-weight="800" text-anchor="middle">BANK</text>
          <g>
            <ellipse cx="147" cy="145" rx="29" ry="7" fill="#ffcf4b"/>
            <path d="M118 132h58v13h-58z" fill="#f5a900" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="132" rx="29" ry="7" fill="#ffd34f" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="116" rx="25" ry="7" fill="#ffcf4b"/>
            <path d="M122 103h50v13h-50z" fill="#eea500" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="103" rx="25" ry="7" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
            <ellipse cx="147" cy="88" rx="20" ry="6" fill="#ffd34f"/>
            <path d="M127 76h40v12h-40z" fill="#eaa000" stroke="#ffe28a" stroke-width="2"/>
            <ellipse cx="147" cy="76" rx="20" ry="6" fill="#ffe078" stroke="#fff0a8" stroke-width="2"/>
          </g>
          <path d="M92 142c-14-21-13-37-2-49 2 19 9 29 21 37M99 128c-1-18 7-30 23-35-5 16-8 28-6 43" stroke="#29c5f5" stroke-width="5" stroke-linecap="round"/>
          <path d="M91 143c-13-3-22-11-26-24 15 3 24 10 26 24ZM111 144c4-14 13-22 28-25-4 14-13 22-28 25Z" fill="#17a9ec"/>
          <path d="M26 34h28m-14-14v28M72 57h17m-8-9v18" stroke="#7bdcff" stroke-width="2" stroke-linecap="round" opacity=".8"/>
          <circle cx="30" cy="87" r="3" fill="#7bdcff"/><circle cx="91" cy="22" r="2.5" fill="#7bdcff"/>
        </g>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    da_rows, da_cols = df.shape
    da_missing_total = int(df.isnull().sum().sum())
    da_approved_n = int((df["Loan_Status"] == "Y").sum())
    da_rejected_n = int((df["Loan_Status"] == "N").sum())

    # 2. Four Horizontally Aligned KPI Cards
    st.markdown(
        f'<div class="da-kpi-grid">'
        f'<div class="da-kpi-card">'
        f'<div class="da-kpi-left"><div class="da-kpi-ico" style="background:#e6f0ff;color:#1261e8;">📄</div>'
        f'<div><div class="da-kpi-label">Dataset Rows</div><div class="da-kpi-val">{da_rows:,}</div></div></div>'
        f'<svg class="da-kpi-spark" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#1261e8" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        f'</div>'
        f'<div class="da-kpi-card green">'
        f'<div class="da-kpi-left"><div class="da-kpi-ico" style="background:#ddf7e9;color:#07884b;">🧩</div>'
        f'<div><div class="da-kpi-label">Total Columns</div><div class="da-kpi-val">{da_cols:,}</div></div></div>'
        f'<svg class="da-kpi-spark" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#07884b" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        f'</div>'
        f'<div class="da-kpi-card red">'
        f'<div class="da-kpi-left"><div class="da-kpi-ico" style="background:#ffe5e7;color:#c1273d;">⚠️</div>'
        f'<div><div class="da-kpi-label">Raw Missing Cells</div><div class="da-kpi-val">{da_missing_total:,}</div></div></div>'
        f'<svg class="da-kpi-spark" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#c1273d" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        f'</div>'
        f'<div class="da-kpi-card purple">'
        f'<div class="da-kpi-left"><div class="da-kpi-ico" style="background:#ede9fe;color:#6236c9;">✓/✕</div>'
        f'<div><div class="da-kpi-label">Approved / Rejected</div><div class="da-kpi-val">{da_approved_n} / {da_rejected_n}</div></div></div>'
        f'<svg class="da-kpi-spark" viewBox="0 0 96 42" fill="none"><path d="M6 34 C18 32, 26 24, 38 25 C50 26, 58 17, 70 18 C79 19, 85 11, 91 8" stroke="#6236c9" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # 3. Dataset Preview Card (First 20 rows, all 13 columns, sticky header, scrollable)
    with st.container(border=True):
        st.markdown(
            '<div class="da-sec-head">'
            '<div class="da-sec-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M8 4v16"/></svg></div>'
            '<div><div class="da-sec-title">Dataset Preview</div>'
            '<div class="da-sec-sub">First 20 rows of loan_dataset.csv with sticky headers and missing-value highlights.</div></div></div>',
            unsafe_allow_html=True
        )

        # Proportional column widths (%) summing to 100% across all 13 dataset columns
        col_width_pct = {
            "Loan_ID": 7.5,
            "Gender": 6.5,
            "Married": 6.5,
            "Dependents": 7.5,
            "Education": 7.5,
            "Self_Employed": 8.0,
            "ApplicantIncome": 8.5,
            "CoapplicantIncome": 9.0,
            "LoanAmount": 6.5,
            "Loan_Amount_Term": 8.5,
            "Credit_History": 7.5,
            "Property_Area": 8.0,
            "Loan_Status": 8.5,
        }

        preview_df = df.head(20)
        colgroup_html = "".join(
            f'<col style="width:{col_width_pct.get(str(c), 100.0 / len(preview_df.columns)):.2f}%;">'
            for c in preview_df.columns
        )
        p_headers = "".join(
            f"<th>{html_lib.escape(str(c))}</th>" for c in preview_df.columns
        )
        p_rows = []
        for _, r in preview_df.iterrows():
            cells = []
            for c in preview_df.columns:
                val = r[c]
                if pd.isna(val):
                    cells.append("<td class='da-na-cell'>Missing</td>")
                elif c == "Loan_Status":
                    if str(val) == "Y":
                        cells.append("<td><span class='da-pill-y'>✓ Y</span></td>")
                    else:
                        cells.append("<td><span class='da-pill-n'>✕ N</span></td>")
                elif c in ("ApplicantIncome", "CoapplicantIncome"):
                    cells.append(f"<td class='num-col'>{float(val):,.0f}</td>")
                elif c in ("LoanAmount", "Loan_Amount_Term", "Credit_History"):
                    cells.append(f"<td class='num-col'>{float(val):g}</td>")
                else:
                    cells.append(f"<td>{html_lib.escape(str(val))}</td>")
            p_rows.append(f"<tr>{''.join(cells)}</tr>")

        st.markdown(
            f'<div class="da-preview-scroll"><table class="da-preview-table">'
            f'<colgroup>{colgroup_html}</colgroup>'
            f'<thead><tr>{p_headers}</tr></thead>'
            f'<tbody>{"".join(p_rows)}</tbody>'
            f'</table></div><div style="height:6px;"></div>',
            unsafe_allow_html=True
        )

    # 4. Statistical Summary Card (count, mean, std, min, 25%, 50%, 75%, max)
    with st.container(border=True):
        st.markdown(
            '<div class="da-sec-head">'
            '<div class="da-sec-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M3 3v18h18"/><path d="M8 16v-5m5 5V5m5 11V9"/></svg></div>'
            '<div><div class="da-sec-title">Statistical Summary</div>'
            '<div class="da-sec-sub">Descriptive statistics across numerical features (count, mean, std, min, 25%, 50%, 75%, max).</div></div></div>',
            unsafe_allow_html=True
        )

        desc_df = df.describe().T.reset_index().rename(columns={"index": "Feature"})
        stat_cols_order = ["Feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        present_cols = [c for c in stat_cols_order if c in desc_df.columns]
        s_headers = "".join(
            f"<th>{html_lib.escape(str(c))}</th>" for c in present_cols
        )
        s_rows = []
        for _, r in desc_df.iterrows():
            feat_name = html_lib.escape(str(r["Feature"]))
            cnt_val = f"{float(r['count']):,.0f}"
            mean_val = f"{float(r['mean']):,.2f}"
            std_val = f"{float(r['std']):,.2f}"
            min_val = f"{float(r['min']):,.2f}"
            q25_val = f"{float(r['25%']):,.2f}"
            q50_val = f"{float(r['50%']):,.2f}"
            q75_val = f"{float(r['75%']):,.2f}"
            max_val = f"{float(r['max']):,.2f}"
            s_rows.append(
                f"<tr>"
                f"<td>{feat_name}</td>"
                f"<td>{cnt_val}</td><td>{mean_val}</td><td>{std_val}</td>"
                f"<td>{min_val}</td><td>{q25_val}</td><td>{q50_val}</td>"
                f"<td>{q75_val}</td><td>{max_val}</td>"
                f"</tr>"
            )

        st.markdown(
            f'<div class="da-stats-scroll"><table class="da-stats-table">'
            f'<thead><tr>{s_headers}</tr></thead>'
            f'<tbody>{"".join(s_rows)}</tbody>'
            f'</table></div><div style="height:6px;"></div>',
            unsafe_allow_html=True
        )

    # 5. Missing Values & Loan Status Distribution Row (50% / 50%)
    miss_col, status_col = st.columns(2, gap="medium")

    with miss_col:
        with st.container(border=True):
            st.markdown(
                '<div class="da-sec-head">'
                '<div class="da-sec-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="9"/><path d="M12 8v4m0 4h.01"/></svg></div>'
                '<div><div class="da-sec-title">Missing Values</div>'
                '<div class="da-sec-sub">Missing entry counts per feature before preprocessing imputation.</div></div></div>',
                unsafe_allow_html=True
            )

            missing_series = df.isnull().sum()
            missing_series = missing_series[missing_series > 0].sort_values(ascending=False)

            if not missing_series.empty:
                fig_m, ax_m = plt.subplots(figsize=(5.4, 2.85), dpi=135)
                fig_m.patch.set_facecolor("#ffffff")
                ax_m.set_facecolor("#f8faff")

                m_labels = [str(x) for x in missing_series.index]
                m_vals = [int(x) for x in missing_series.values]
                m_colors = ["#1261e8" if i == 0 else "#4c8ef7" for i in range(len(m_vals))]
                bars_m = ax_m.barh(m_labels, m_vals, color=m_colors, height=0.54, zorder=3)
                ax_m.invert_yaxis()
                ax_m.set_xlim(0, max(m_vals) * 1.24)
                ax_m.xaxis.grid(True, color="#dce6f4", linewidth=0.8, zorder=0)
                for bar, v in zip(bars_m, m_vals):
                    pct_m = (v / da_rows * 100) if da_rows else 0
                    ax_m.text(
                        v + max(m_vals) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:,} ({pct_m:.1f}%)", va="center", ha="left",
                        fontsize=8.5, color="#10234f", fontweight="bold"
                    )
                ax_m.set_xlabel("Missing Records", fontsize=8.5, color="#52658a", labelpad=5)
                ax_m.set_ylabel("Feature", fontsize=8.5, color="#52658a", labelpad=5)
                ax_m.tick_params(axis="both", labelsize=8.5, colors="#20365e", length=0, pad=5)
                for spine in ax_m.spines.values():
                    spine.set_visible(False)
                ax_m.set_axisbelow(True)
                fig_m.tight_layout(pad=1.0)
                st.pyplot(fig_m, use_container_width=True)
                plt.close(fig_m)
            else:
                st.markdown(
                    '<div class="da-status-ok">✓ &nbsp; No missing values found in the dataset.</div>',
                    unsafe_allow_html=True
                )

    with status_col:
        with st.container(border=True):
            st.markdown(
                '<div class="da-sec-head">'
                '<div class="da-sec-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 20V10m6 10V4M6 20v-4"/></svg></div>'
                '<div><div class="da-sec-title">Loan Status Distribution</div>'
                '<div class="da-sec-sub">Target class balance between Approved (Y) and Rejected (N).</div></div></div>',
                unsafe_allow_html=True
            )

            s_counts = (
                df["Loan_Status"]
                .value_counts()
                .rename({"Y": "Approved", "N": "Rejected"})
                .reindex(["Approved", "Rejected"], fill_value=0)
            )

            fig_s, ax_s = plt.subplots(figsize=(5.4, 2.85), dpi=135)
            fig_s.patch.set_facecolor("#ffffff")
            ax_s.set_facecolor("#f8faff")

            s_labels = list(s_counts.index)
            s_vals = [int(v) for v in s_counts.values]
            s_colors = ["#16a36a", "#e5484d"]
            bars_s = ax_s.bar(s_labels, s_vals, color=s_colors, width=0.44, zorder=3)
            ax_s.set_ylim(0, max(s_vals) * 1.24 if max(s_vals) > 0 else 1)
            ax_s.yaxis.grid(True, color="#dce6f4", linewidth=0.8, zorder=0)
            for bar, v in zip(bars_s, s_vals):
                pct_s = (v / da_rows * 100) if da_rows else 0
                ax_s.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(s_vals) * 0.025,
                    f"{v:,} ({pct_s:.1f}%)", va="bottom", ha="center",
                    fontsize=8.5, color="#10234f", fontweight="bold"
                )
            ax_s.set_ylabel("Applications", fontsize=8.5, color="#52658a", labelpad=5)
            ax_s.set_xlabel("Loan Status", fontsize=8.5, color="#52658a", labelpad=5)
            ax_s.tick_params(axis="both", labelsize=8.5, colors="#20365e", length=0, pad=5)
            for spine in ax_s.spines.values():
                spine.set_visible(False)
            ax_s.set_axisbelow(True)
            fig_s.tight_layout(pad=1.0)
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)

    # 6. Applicant Income Distribution Card (clean X-axis tick spacing, labels, and titles)
    with st.container(border=True):
        st.markdown(
            '<div class="da-sec-head">'
            '<div class="da-sec-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 16v-4m4 4V8m4 8v-6m4 6v-3"/></svg></div>'
            '<div><div class="da-sec-title">Applicant Income Distribution</div>'
            '<div class="da-sec-sub">Applicant income distribution across 20 intervals in the dataset.</div></div></div>',
            unsafe_allow_html=True
        )

        inc_series = pd.to_numeric(df["ApplicantIncome"], errors="coerce").dropna()
        inc_binned = inc_series.value_counts(bins=20).sort_index()

        fig_inc, ax_inc = plt.subplots(figsize=(11.5, 3.15), dpi=135)
        fig_inc.patch.set_facecolor("#ffffff")
        ax_inc.set_facecolor("#f8faff")

        bin_labels = [
            f"{max(0, int(iv.left)):,}–{int(iv.right):,}"
            for iv in inc_binned.index
        ]
        bin_vals = [int(v) for v in inc_binned.values]
        x_pos = list(range(len(bin_labels)))

        bars_inc = ax_inc.bar(x_pos, bin_vals, color="#1261e8", width=0.60, zorder=3)
        ax_inc.set_ylim(0, max(bin_vals) * 1.22 if max(bin_vals) > 0 else 1)
        ax_inc.yaxis.grid(True, color="#dce6f4", linewidth=0.8, zorder=0)
        for bar, v in zip(bars_inc, bin_vals):
            if v > 0:
                ax_inc.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(bin_vals) * 0.022,
                    f"{v:,}", va="bottom", ha="center",
                    fontsize=7.5, color="#10234f", fontweight="bold"
                )

        # Show every 2nd tick label so the X-axis stays clean and never overlaps
        tick_indices = list(range(0, len(bin_labels), 2))
        ax_inc.set_xticks(tick_indices)
        ax_inc.set_xticklabels(
            [bin_labels[i] for i in tick_indices],
            rotation=25, ha="right",
            fontsize=8, color="#20365e", fontweight="semibold"
        )
        ax_inc.set_xlabel("Applicant Income Range", fontsize=8.5, color="#52658a", labelpad=6)
        ax_inc.set_ylabel("Applicants", fontsize=8.5, color="#52658a", labelpad=6)
        ax_inc.tick_params(axis="y", labelsize=8.5, colors="#20365e", length=0, pad=5)
        ax_inc.tick_params(axis="x", length=3, color="#b8cbee", pad=4)
        for spine in ax_inc.spines.values():
            spine.set_visible(False)
        ax_inc.set_axisbelow(True)
        fig_inc.tight_layout(pad=1.0)
        st.pyplot(fig_inc, use_container_width=True)
        plt.close(fig_inc)


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="loan-footer">
    <div class="loan-footer-brand">🏛️ LoanPredict <span>AI</span></div>
    <div class="loan-footer-title">Smart Loan Assessment &amp; Prediction</div>
    <div class="loan-footer-copy">© 2026 LoanPredict AI | Academic Project</div>
</div>
""", unsafe_allow_html=True)
