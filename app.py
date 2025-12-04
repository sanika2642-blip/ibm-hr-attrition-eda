# app.py — Clean HR Attrition Dashboard (visuals wrapped safely)
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import textwrap
import datetime

st.set_page_config(layout="wide", page_title="HR Attrition Dashboard", initial_sidebar_state="expanded")

# ---------------------------
# Minimal CSS (neon theme)
# ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {font-family: Inter, sans-serif; color: #cfeffd;}
    .stApp { background: radial-gradient(circle at 10% 10%, #001018 0%, #00050a 40%, #000000 100%); }
    .card { background: rgba(0,255,255,0.02); border:1px solid rgba(0,200,255,0.06);
            border-radius:10px; padding:12px; box-shadow: 0 6px 30px rgba(0,200,255,0.02); }
    .neon-title { color:#00eaff; font-weight:700; font-size:18px; margin-bottom:8px; }
    .small-muted { color:#96e9ff; opacity:0.75; font-size:13px; }
    .kpi { background:rgba(255,255,255,0.02); border-radius:10px; padding:10px; text-align:center; }
    .kpi .value { font-size:22px; font-weight:700; color:#00eaff; }
    .holo-line { height:2px; background: linear-gradient(90deg, rgba(0,255,255,0.12), rgba(0,255,255,0)); margin:12px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Data loading
# ---------------------------
DATA_PATH = "employee_attrition.csv"


@st.cache_data
def load_data(path=None, uploaded=None):
    try:
        if uploaded is not None:
            df_local = pd.read_csv(uploaded)
        elif path is not None:
            df_local = pd.read_csv(path)
        else:
            df_local = pd.DataFrame()
    except Exception:
        df_local = pd.DataFrame()
    df_local.columns = [c.strip() for c in df_local.columns]
    return df_local

uploaded = st.sidebar.file_uploader("Upload employee_attrition.csv (optional)", type=["csv"])
df = load_data(DATA_PATH if uploaded is None else None, uploaded)

if df.empty:
    st.sidebar.info("No dataset found. Upload employee_attrition.csv or place it at /mnt/data/employee_attrition.csv")
    st.title("HR Attrition Dashboard — Dataset missing")
    st.stop()

# ---------------------------
# Defensive preprocessing / KPIs
# ---------------------------
df.columns = [c.strip() for c in df.columns]

# Ensure Attrition_bool exists
if 'Attrition_bool' not in df.columns:
    if 'Attrition' in df.columns:
        df['Attrition_bool'] = df['Attrition'].apply(lambda x: 1 if str(x).strip().lower() in ['yes','y','true','1'] else 0)
    else:
        df['Attrition_bool'] = 0

# Age bucket if available
if 'Age' in df.columns:
    try:
        bins = [17,25,35,45,55,100]
        labels = ['18-25','26-35','36-45','46-55','55+']
        df['AgeBucket'] = pd.cut(df['Age'].fillna(-1), bins=bins, labels=labels, include_lowest=True)
    except Exception:
        df['AgeBucket'] = None

total_employees = len(df)
attrition_count = int(df['Attrition_bool'].sum()) if 'Attrition_bool' in df.columns else 0
attrition_rate = round(100 * attrition_count / total_employees, 1) if total_employees else 0
avg_age = round(df['Age'].mean(), 1) if 'Age' in df.columns and df['Age'].dropna().size>0 else "N/A"
avg_income = int(df['MonthlyIncome'].mean()) if 'MonthlyIncome' in df.columns and df['MonthlyIncome'].dropna().size>0 else "N/A"
avg_years = round(df['YearsAtCompany'].mean(), 1) if 'YearsAtCompany' in df.columns and df['YearsAtCompany'].dropna().size>0 else "N/A"

# Safe groups
grp_dept = pd.DataFrame()
grp_job = pd.DataFrame()
risk_by_dept = "N/A"
risk_by_job = "N/A"
if 'Department' in df.columns and 'Attrition_bool' in df.columns:
    try:
        grp_dept = df.groupby('Department').agg(total=('Attrition_bool','count'), left=('Attrition_bool','sum'))
        grp_dept['rate'] = 100 * grp_dept['left'] / grp_dept['total']
        if not grp_dept.empty:
            risk_by_dept = grp_dept['rate'].idxmax()
    except Exception:
        grp_dept = pd.DataFrame()

if 'JobRole' in df.columns and 'Attrition_bool' in df.columns:
    try:
        grp_job = df.groupby('JobRole').agg(total=('Attrition_bool','count'), left=('Attrition_bool','sum'))
        grp_job['rate'] = 100 * grp_job['left'] / grp_job['total']
        if not grp_job.empty:
            risk_by_job = grp_job['rate'].idxmax()
    except Exception:
        grp_job = pd.DataFrame()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Controls")
page = st.sidebar.radio("Go to", ["Overview","Visuals","Assistant","Predictor","Export"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset**  \nRows: {total_employees}  \nColumns: {df.shape[1]}")
st.sidebar.markdown(f"**Attrition:** {attrition_count} ({attrition_rate}%)")
st.sidebar.markdown("---")
st.sidebar.markdown("Made for OJT — Streamlit edition", unsafe_allow_html=True)

# helpers
def neon(txt):
    st.markdown(f'<div class="neon-title">🔹 {txt}</div>', unsafe_allow_html=True)

def tidy_layout(fig):
    fig.update_layout(margin=dict(l=6,r=6,t=6,b=6), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#bfefff'))
    return fig

# ---------------------------
# Overview
# ---------------------------
def page_overview():
    # ---------- MAIN HOLO HEADER ----------
    st.markdown(
        """
        <div style="margin-top:4px;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="
                width:34px;height:34px;border-radius:50%;
                background:radial-gradient(circle at 30% 30%, #00fff2, #007bff 60%, #000 100%);
                box-shadow:0 0 20px rgba(0,255,230,0.9);
                position:relative;overflow:hidden;">
              <div style="
                    position:absolute;inset:4px;
                    border-radius:50%;
                    border:1px solid rgba(0,0,0,0.9);
                    box-shadow:0 0 10px rgba(0,0,0,0.9) inset;"></div>
            </div>
            <div>
              <div style="font-size:14px;letter-spacing:0.15em;text-transform:uppercase;color:#7defff;">
                HR CORE NODE · ONLINE
              </div>
              <div style="font-size:20px;font-weight:700;color:#e6f9ff;">
                Attrition Intelligence · Live Telemetry
              </div>
            </div>
          </div>
          <div style="text-align:right;font-size:11px;color:#8dd5ff;">
            SESSION ID · OJT‑HR‑{sid}<br>
            LAST SYNC · {ts}
          </div>
        </div>
        """.format(
            sid=str(total_employees).zfill(4),
            ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        ),
        unsafe_allow_html=True,
    )

    # ---------- RING KPIs (TOP ROW) ----------
    top_html = f"""
    <div style="display:flex;gap:14px;margin-top:8px;margin-bottom:4px;flex-wrap:wrap;">
      <div style="flex:1;min-width:0;">
        <div class="card" style="position:relative;overflow:hidden;padding:14px 12px;">
          <div style="position:absolute;inset:-40px;
                      background:conic-gradient(from 220deg,
                        rgba(0,255,255,0.55) 0deg,
                        transparent 120deg,
                        transparent 240deg,
                        rgba(0,255,255,0.55) 360deg);
                      opacity:0.28;filter:blur(18px);"></div>
          <div style="position:relative;display:flex;align-items:center;gap:10px;">
            <div style="width:54px;height:54px;border-radius:50%;
                        background:radial-gradient(circle at 30% 25%, #001a26, #000 70%);
                        border:1px solid rgba(180,255,255,0.35);
                        box-shadow:0 0 20px rgba(0,255,255,0.55) inset,0 0 18px rgba(0,255,255,0.55);
                        display:flex;align-items:center;justify-content:center;">
              <span style="font-size:18px;font-weight:700;color:#00eaff;">
                {total_employees}
              </span>
            </div>
            <div style="flex:1;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.15em;color:#7adfff;">
                Total employees
              </div>
              <div style="font-size:12px;color:#c5eaff;">
                Active profiles in current dataset
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style="flex:1;min-width:0;">
        <div class="card" style="position:relative;overflow:hidden;padding:14px 12px;">
          <div style="position:absolute;inset:-40px;
                      background:conic-gradient(from 220deg,
                        rgba(255,122,217,0.55) 0deg,
                        transparent 120deg,
                        transparent 240deg,
                        rgba(255,122,217,0.55) 360deg);
                      opacity:0.28;filter:blur(18px);"></div>
          <div style="position:relative;display:flex;align-items:center;gap:10px;">
            <div style="width:54px;height:54px;border-radius:50%;
                        background:radial-gradient(circle at 30% 25%, #001a26, #000 70%);
                        border:1px solid rgba(180,255,255,0.35);
                        box-shadow:0 0 20px rgba(255,122,217,0.55) inset,0 0 18px rgba(255,122,217,0.55);
                        display:flex;align-items:center;justify-content:center;">
              <span style="font-size:18px;font-weight:700;color:#ff7ad9;">
                {attrition_rate}%
              </span>
            </div>
            <div style="flex:1;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.15em;color:#7adfff;">
                Attrition rate
              </div>
              <div style="font-size:12px;color:#c5eaff;">
                {attrition_count} employees flagged as left
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style="flex:1;min-width:0;">
        <div class="card" style="position:relative;overflow:hidden;padding:14px 12px;">
          <div style="position:absolute;inset:-40px;
                      background:conic-gradient(from 220deg,
                        rgba(255,204,106,0.55) 0deg,
                        transparent 120deg,
                        transparent 240deg,
                        rgba(255,204,106,0.55) 360deg);
                      opacity:0.28;filter:blur(18px);"></div>
          <div style="position:relative;display:flex;align-items:center;gap:10px;">
            <div style="width:54px;height:54px;border-radius:50%;
                        background:radial-gradient(circle at 30% 25%, #001a26, #000 70%);
                        border:1px solid rgba(180,255,255,0.35);
                        box-shadow:0 0 20px rgba(255,204,106,0.55) inset,0 0 18px rgba(255,204,106,0.55);
                        display:flex;align-items:center;justify-content:center;">
              <span style="font-size:18px;font-weight:700;color:#ffdd6a;">
                {avg_age}
              </span>
            </div>
            <div style="flex:1;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.15em;color:#7adfff;">
                Average age
              </div>
              <div style="font-size:12px;color:#c5eaff;">
                Experience distribution core
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style="flex:1;min-width:0;">
        <div class="card" style="position:relative;overflow:hidden;padding:14px 12px;">
          <div style="position:absolute;inset:-40px;
                      background:conic-gradient(from 220deg,
                        rgba(125,255,181,0.55) 0deg,
                        transparent 120deg,
                        transparent 240deg,
                        rgba(125,255,181,0.55) 360deg);
                      opacity:0.28;filter:blur(18px);"></div>
          <div style="position:relative;display:flex;align-items:center;gap:10px;">
            <div style="width:54px;height:54px;border-radius:50%;
                        background:radial-gradient(circle at 30% 25%, #001a26, #000 70%);
                        border:1px solid rgba(180,255,255,0.35);
                        box-shadow:0 0 20px rgba(125,255,181,0.55) inset,0 0 18px rgba(125,255,181,0.55);
                        display:flex;align-items:center;justify-content:center;">
              <span style="font-size:18px;font-weight:700;color:#7dffb5;">
                {avg_income}
              </span>
            </div>
            <div style="flex:1;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.15em;color:#7adfff;">
                Avg monthly income
              </div>
              <div style="font-size:12px;color:#c5eaff;">
                Compensation field (mean)
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
    st.markdown(top_html, unsafe_allow_html=True)

    st.markdown('<div class="holo-line"></div>', unsafe_allow_html=True)

    # ---------- CENTER GRID: STATUS + RISK PANEL ----------
    left, mid, right = st.columns([1.2, 1, 1.2])

    # LEFT: system status blocks
    with left:
        st.markdown(
            """
            <div class="card" style="position:relative;overflow:hidden;">
              <div style="position:absolute;right:-40px;top:-40px;width:120px;height:120px;
                          border-radius:50%;
                          border:1px solid rgba(0,255,255,0.35);
                          box-shadow:0 0 40px rgba(0,255,255,0.45);
                          opacity:0.25;"></div>
              <div class="neon-title">🛰 Network status</div>
              <div class="small-muted">Live stability for HR telemetry channels.</div>
              <div style="margin-top:10px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;">
                <div style="background:rgba(0,255,255,0.03);border-radius:8px;padding:6px 8px;border:1px solid rgba(0,255,255,0.12);">
                  <div style="font-size:11px;color:#7ce9ff;">DATA LINK</div>
                  <div style="font-size:13px;color:#cfeffd;">Stable · 100% rows loaded</div>
                </div>
                <div style="background:rgba(0,255,255,0.03);border-radius:8px;padding:6px 8px;border:1px solid rgba(0,255,255,0.12);">
                  <div style="font-size:11px;color:#7ce9ff;">NOISE FILTER</div>
                  <div style="font-size:13px;color:#cfeffd;">Outliers tolerated (defensive logic)</div>
                </div>
                <div style="background:rgba(0,255,255,0.03);border-radius:8px;padding:6px 8px;border:1px solid rgba(0,255,255,0.12);">
                  <div style="font-size:11px;color:#7ce9ff;">TIME FRAME</div>
                  <div style="font-size:13px;color:#cfeffd;">Static snapshot · CSV upload</div>
                </div>
                <div style="background:rgba(0,255,255,0.03);border-radius:8px;padding:6px 8px;border:1px solid rgba(0,255,255,0.12);">
                  <div style="font-size:11px;color:#7ce9ff;">ENGINE</div>
                  <div style="font-size:13px;color:#cfeffd;">Python · Streamlit · ML core ready</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # MID: circular risk radar (textual)
    with mid:
        dept_txt = risk_by_dept if isinstance(risk_by_dept, str) else "N/A"
        job_txt = risk_by_job if isinstance(risk_by_job, str) else "N/A"

        st.markdown(
            f"""
            <div class="card" style="text-align:center;position:relative;overflow:hidden;">
              <div style="position:absolute;inset:-70px;
                          background:radial-gradient(circle, rgba(0,255,255,0.12), transparent 60%);
                          opacity:0.8;"></div>
              <div style="position:relative;">
                <div class="neon-title">🧭 Risk focus core</div>
                <div class="small-muted">Holographic radar of current hot zones.</div>
                <div style="margin:12px auto;width:120px;height:120px;border-radius:50%;
                            border:1px dashed rgba(0,255,255,0.4);
                            display:flex;align-items:center;justify-content:center;
                            box-shadow:0 0 20px rgba(0,255,255,0.3);">
                  <div style="width:72px;height:72px;border-radius:50%;
                              background:radial-gradient(circle at 30% 20%, #00ffe5, #003c4f 70%);
                              box-shadow:0 0 25px rgba(0,255,213,0.9);
                              display:flex;flex-direction:column;align-items:center;justify-content:center;">
                    <div style="font-size:11px;color:#001218;background:rgba(255,255,255,0.85);
                                padding:2px 8px;border-radius:999px;margin-bottom:4px;">
                      RISK INDEX
                    </div>
                    <div style="font-size:20px;font-weight:800;color:#e6ffff;">
                      {attrition_rate}%
                    </div>
                  </div>
                </div>
                <div style="font-size:12px;color:#b7e8ff;margin-top:4px;">
                  Highest stress · <b>{dept_txt}</b><br>
                  Critical role cluster · <b>{job_txt}</b>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # RIGHT: compact “story” of the workforce
    with right:
        ot_pct = (
            f"{round(100 * df[df.get('OverTime', '') == 'Yes'].shape[0] / total_employees, 1)}%"
            if "OverTime" in df.columns
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="card" style="position:relative;overflow:hidden;">
              <div style="position:absolute;left:-60px;bottom:-60px;width:140px;height:140px;
                          border-radius:50%;
                          border:1px solid rgba(0,255,255,0.25);
                          box-shadow:0 0 30px rgba(0,255,255,0.4);
                          opacity:0.3;"></div>
              <div class="neon-title">📡 Workforce pulse</div>
              <div class="small-muted">Condensed summary of how this organisation looks right now.</div>
              <div style="margin-top:10px;font-size:12px;color:#cfeffd;line-height:1.4;">
                • <b>{total_employees}</b> active employee records in this dataset. <br>
                • <b>{attrition_count}</b> profiles marked as attrited, yielding a
                  <b>{attrition_rate}%</b> global attrition rate. <br>
                • Typical employee age hovers around <b>{avg_age}</b> years, staying for
                  roughly <b>{avg_years}</b> years on average. <br>
                • Approximate average monthly compensation is <b>{avg_income}</b>,
                  with overtime incidence near <b>{ot_pct}</b>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- BOTTOM: EVENT LOG / CONSOLE ----------
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card" style="max-height:170px;overflow:hidden;position:relative;">
          <div style="position:absolute;left:0;top:0;width:3px;height:100%;
                      background:linear-gradient(to bottom, #00eaff, transparent);"></div>
          <div style="font-size:12px;color:#7ce9ff;margin-bottom:6px;letter-spacing:0.16em;text-transform:uppercase;">
            LIVE EVENT LOG
          </div>
          <div style="font-family:monospace;font-size:12px;color:#b7ebff;max-height:130px;overflow-y:auto;">
            <div>> bootseq :: hr_core :: ok</div>
            <div>> ingest :: csv_rows = {rows} :: schema = {cols}</div>
            <div>> compute :: attrition_rate = {rate}% :: risk_vector[dept,role] loaded</div>
            <div>> engine :: predictor_online = true :: assistant_online = true</div>
            <div>> note :: overview page running in hologram mode · no charts rendered</div>
          </div>
        </div>
        """.format(
            rows=total_employees,
            cols=df.shape[1],
            rate=attrition_rate,
        ),
        unsafe_allow_html=True,
    )


# ===========================
# Visuals (wrapped safely)
# ===========================
def page_visuals():
    # extra CSS for cards / labels
    st.markdown("""
        <style>
        .viz-card {
            background: radial-gradient(circle at 0 0, rgba(0,255,255,0.14), rgba(0,10,20,0.9));
            border-radius: 14px;
            padding: 10px 12px 6px 12px;
            border: 1px solid rgba(0,255,255,0.20);
            box-shadow:
                0 0 24px rgba(0,255,255,0.16),
                0 0 35px rgba(0,120,255,0.18),
                inset 0 0 18px rgba(0,0,0,0.8);
        }
        .viz-title {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: #86efff;
            margin-bottom: 4px;
        }
        .viz-sub {
            font-size: 11px;
            color: #a9dfff;
            opacity: 0.9;
            margin-bottom: 6px;
        }
        .viz-tag {
            display:inline-block;
            font-size:10px;
            letter-spacing:0.18em;
            text-transform:uppercase;
            padding:2px 8px;
            border-radius:999px;
            border:1px solid rgba(0,255,255,0.4);
            color:#78f7ff;
            margin-bottom:4px;
        }
        </style>
    """, unsafe_allow_html=True)

    # header
    st.markdown(
        "<div style='text-align:center;color:#00eaff;font-size:22px;font-weight:800;letter-spacing:6px;margin-bottom:8px;'>VISUAL ANALYTICS HUB</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align:center;color:#8fdfff;font-size:12px;margin-bottom:16px;'>Hologram visualisations of your workforce patterns.</div>",
        unsafe_allow_html=True
    )

    # ================= ROW 1: metrics left, globe big right =================
    row1_left, row1_right = st.columns([1, 2])

    # SMALL METRIC PANEL (right now you like these numbers near globe)
    with row1_left:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown('<div class="viz-tag">GLOBAL SNAPSHOT</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-title">Key workforce indicators</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-sub">Quick stats aligned with the hologram Earth on the right.</div>', unsafe_allow_html=True)

        total_employees_local = len(df)
        attr_rate_local = round((df["Attrition"].astype(str).str.lower() == "yes").mean() * 100, 1) if "Attrition" in df.columns else "N/A"
        avg_age_local = round(df["Age"].mean(), 1) if "Age" in df.columns else "N/A"
        avg_inc_local = round(df["MonthlyIncome"].mean(), 1) if "MonthlyIncome" in df.columns else "N/A"

        st.markdown(
            f"""
            <div style="font-size:12px;color:#bffaff;line-height:1.5;">
              • <b>Total employees:</b> {total_employees_local}<br>
              • <b>Attrition rate:</b> {attr_rate_local}%<br>
              • <b>Average age:</b> {avg_age_local}<br>
              • <b>Average income:</b> {avg_inc_local}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # BIG CENTER GLOBE (kept from your previous version)
    with row1_right:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown('<div class="viz-tag">GLOBAL PRESENCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-title">Hologram Earth · employee field</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-sub">Glowing points represent workforce distribution (synthetic but visually rich).</div>', unsafe_allow_html=True)

        try:
            n = max(1600, len(df) * 4)
            np.random.seed(42)
            lat = np.random.normal(20, 28, n).clip(-60, 80)
            lon = np.random.uniform(-180, 180, n)

            lat_r = np.deg2rad(lat)
            lon_r = np.deg2rad(lon)
            x = np.cos(lat_r) * np.cos(lon_r)
            y = np.cos(lat_r) * np.sin(lon_r)
            z = np.sin(lat_r)

            if "Attrition" in df.columns:
                attr = (df["Attrition"].astype(str).str.lower() == "yes").astype(int).tolist()
                attr_long = (attr * ((n // len(attr)) + 1))[:n]
                colors = ["#00eaff" if a else "#007e99" for a in attr_long]
            else:
                colors = ["#00eaff" if i % 6 == 0 else "#007e99" for i in range(n)]

            dots = go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=2.4, color=colors, opacity=0.95),
                hoverinfo="none",
            )

            wire = []
            theta = np.linspace(-np.pi, np.pi, 150)
            phi = np.linspace(-np.pi / 2, np.pi / 2, 80)

            for L in np.linspace(-np.pi, np.pi, 12):
                xt = np.cos(phi) * np.cos(L)
                yt = np.cos(phi) * np.sin(L)
                zt = np.sin(phi)
                wire.append(
                    go.Scatter3d(
                        x=xt, y=yt, z=zt,
                        mode="lines",
                        line=dict(color="rgba(0,255,255,0.2)", width=1),
                        hoverinfo="none",
                    )
                )

            for lat0 in np.linspace(-np.pi / 3, np.pi / 3, 6):
                xt = np.cos(lat0) * np.cos(theta)
                yt = np.cos(lat0) * np.sin(theta)
                zt = np.full_like(theta, np.sin(lat0))
                wire.append(
                    go.Scatter3d(
                        x=xt, y=yt, z=zt,
                        mode="lines",
                        line=dict(color="rgba(0,255,255,0.12)", width=1),
                        hoverinfo="none",
                    )
                )

            arcs = []
            for a in [0.6, 1.9, 3.2]:
                t = np.linspace(0, np.pi, 160)
                r = 1.07 + 0.06 * np.sin(3 * t + a)
                arcs.append(
                    go.Scatter3d(
                        x=r * np.cos(t) * np.cos(a),
                        y=r * np.cos(t) * np.sin(a),
                        z=0.55 * np.sin(t),
                        mode="lines",
                        line=dict(color="rgba(0,255,255,0.3)", width=2),
                        hoverinfo="none",
                    )
                )

            u = np.linspace(0, 2 * np.pi, 45)
            v = np.linspace(0, np.pi, 25)
            su, sv = np.meshgrid(u, v)
            glow_x = 1.06 * np.cos(su) * np.sin(sv)
            glow_y = 1.06 * np.sin(su) * np.sin(sv)
            glow_z = 1.06 * np.cos(sv)

            glow = go.Surface(
                x=glow_x,
                y=glow_y,
                z=glow_z,
                colorscale=[[0, "rgba(0,255,255,0.05)"], [1, "rgba(0,255,255,0.05)"]],
                showscale=False,
                hoverinfo="none",
                opacity=0.20,
            )

            fig_globe = go.Figure(data=[dots] + wire + arcs + [glow])
            fig_globe.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=dict(eye=dict(x=1.7, y=1.6, z=0.8)),
                ),
            )
            st.plotly_chart(fig_globe, use_container_width=True)
        except Exception as e:
            st.error("Globe failed: " + str(e))

        st.markdown("</div>", unsafe_allow_html=True)

    # ================= ROW 2: two large panels =================
    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    row2_left, row2_right = st.columns([1.3, 1])

    # LEFT: JobRole vs MonthlyIncome vs Attrition
    with row2_left:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown('<div class="viz-tag">JOB ROLES · PAY · RISK</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-title">Role band vs compensation & attrition</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-sub">Each bar is a job role; height is salary, colour is attrition %.</div>', unsafe_allow_html=True)

        if "JobRole" in df.columns and "MonthlyIncome" in df.columns and "Attrition_bool" in df.columns:
            role_stats = (
                df.groupby("JobRole")
                .agg(
                    AvgIncome=("MonthlyIncome", "mean"),
                    AttritionRate=("Attrition_bool", "mean"),
                    Count=("Attrition_bool", "count"),
                )
                .reset_index()
            )
            role_stats["AttritionRatePct"] = role_stats["AttritionRate"] * 100

            fig_role = px.bar(
                role_stats.sort_values("AttritionRatePct", ascending=False),
                x="JobRole",
                y="AvgIncome",
                color="AttritionRatePct",
                color_continuous_scale=["#00eaff", "#ffb347", "#ff5c8a"],
            )
            fig_role.update_traces(marker=dict(line=dict(width=0)))
            fig_role.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=60),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cfeffd"),
                xaxis=dict(showgrid=False, tickangle=-30),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,255,255,0.12)"),
                coloraxis_colorbar=dict(title="Attrition %", ticksuffix="%", len=0.7),
            )
            st.plotly_chart(fig_role, use_container_width=True)
        else:
            st.info("Need JobRole, MonthlyIncome, Attrition_bool for this panel.")

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT: Correlation + Overtime stacked inside one card
    with row2_right:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown('<div class="viz-tag">RELATIONS · PRESSURE</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-title">Correlations & overtime impact</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-sub">Top numeric correlations plus overtime vs attrition breakdown.</div>', unsafe_allow_html=True)

        # correlation
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] >= 2:
            corr = numeric.corr()
            fig_corr = px.imshow(
                corr,
                color_continuous_scale=["#001219", "#0077b6", "#00eaff"],
                aspect="auto",
            )
            fig_corr.update_layout(
                height=200,
                margin=dict(l=50, r=10, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cfeffd", size=9),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation matrix.")

        # small separator
        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

        # overtime vs attrition
        if "OverTime" in df.columns and "Attrition" in df.columns:
            ot = df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
            fig_ot = px.bar(
                ot,
                x="OverTime",
                y="Count",
                color="Attrition",
                barmode="group",
                color_discrete_sequence=["#00eaff", "#ff6bcb"],
            )
            fig_ot.update_layout(
                height=160,
                margin=dict(l=10, r=10, t=4, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cfeffd", size=10),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,255,255,0.12)"),
            )
            st.plotly_chart(fig_ot, use_container_width=True)
        else:
            st.info("Need OverTime and Attrition columns for overtime panel.")

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
# Assistant
# ---------------------------
def page_assistant():
    st.markdown('<h2 style="color:#00eaff">Assistant</h2>', unsafe_allow_html=True)
    q = st.text_input("Ask (e.g. 'high risk jobs', 'overtime')", value="")
    if st.button("Send"):
        if 'high risk job' in q.lower():
            if not grp_job.empty:
                top = grp_job.sort_values('rate', ascending=False).head(6)
                st.write(top['rate'].round(1).astype(str) + '%')
            else:
                st.info("JobRole data missing")
        elif 'high risk dept' in q.lower():
            if not grp_dept.empty:
                top = grp_dept.sort_values('rate', ascending=False).head(6)
                st.write(top['rate'].round(1).astype(str) + '%')
            else:
                st.info("Department data missing")
        elif 'overtime' in q.lower():
            if 'OverTime' in df.columns:
                st.write(df.groupby(['OverTime','Attrition']).size())
            else:
                st.info("OverTime missing")
        else:
            st.info("Try: 'high risk job', 'high risk dept', 'overtime'")

# ---------------------------
# Predictor
# ---------------------------
def page_predictor():
    st.markdown('<h2 style="color:#00eaff">Predictor</h2>', unsafe_allow_html=True)
    features = [c for c in ['Age','MonthlyIncome','YearsAtCompany','DistanceFromHome','JobSatisfaction','OverTime'] if c in df.columns]
    if not features:
        st.info("Not enough features for predictor demo.")
        return
    X = df[features].copy()
    y = df['Attrition_bool'].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    model = Pipeline([('pre', pre), ('clf', LogisticRegression(solver='liblinear', max_iter=500))])
    if y.nunique() > 1 and len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error("Model training failed: " + str(e))
        return
    st.success("Model trained (demo)")
    # prediction form
    with st.form("pred"):
        inputs = {}
        for f in features:
            if f == 'OverTime':
                vals = sorted(df[f].dropna().unique().tolist()) if f in df.columns else ['Yes','No']
                inputs[f] = st.selectbox('OverTime', options=vals, index=0)
            else:
                default = float(df[f].median()) if f in df.columns and pd.api.types.is_numeric_dtype(df[f]) else 0.0
                inputs[f] = st.number_input(f, value=default)
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            proba = model.predict_proba(pd.DataFrame([inputs]))[0][1]
            st.success(f"Predicted attrition probability: {round(proba*100,1)}%")
        except Exception as e:
            st.error("Prediction error: " + str(e))

# ---------------------------
# Export
# ---------------------------
def page_export():
    st.markdown('<h2 style="color:#00eaff">Export</h2>', unsafe_allow_html=True)
    st.download_button("Download dataset CSV", df.to_csv(index=False).encode('utf-8'), "employee_attrition_export.csv")

# ---------------------------
# Router
# ---------------------------
if page == "Overview":
    page_overview()
elif page == "Visuals":
    page_visuals()
elif page == "Assistant":
    page_assistant()
elif page == "Predictor":
    page_predictor()
elif page == "Export":
    page_export()
