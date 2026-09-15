import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Job Salary Classification AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Compensation & Career Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #064E3B 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        margin-bottom: 10px;
    }

    .salary-card-high {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.06) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .salary-card-medium {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(29, 78, 216, 0.05) 100%);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .salary-card-low {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .salary-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 6px 0;
    }

    .comp-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_model():
    model_path = get_asset_path("job_salary_classifier.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">Talent Analytics & Compensation Intelligence AI</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">💼 Job Salary Classification</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Predict salary compensation brackets (Low, Medium, High) based on educational attainment, years of experience, seniority tier, industry domain, and skill repertoire.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Candidate Salary Band Evaluation", "📁 Batch Talent Screening (CSV)", "📊 Model Performance & Confusion Matrix"])

# --- TAB 1: Candidate Salary Band Evaluation ---
with tabs[0]:
    st.subheader("Candidate Career Credentials & Market Parameters")

    # Quick Presets
    p_cols = st.columns([1, 1, 1, 3])
    with p_cols[0]:
        load_high = st.button("💎 High Compensation Profile", width="stretch")
    with p_cols[1]:
        load_med = st.button("⚖️ Mid-Tier Profile", width="stretch")
    with p_cols[2]:
        load_entry = st.button("🌱 Entry-Level Profile", width="stretch")

    if load_high:
        st.session_state["edu"] = "Master"
        st.session_state["exp"] = 12
        st.session_state["level"] = "Lead"
        st.session_state["ind"] = "IT"
        st.session_state["loc"] = "Tier 1 City"
        st.session_state["skills"] = 9
    elif load_med:
        st.session_state["edu"] = "Bachelor"
        st.session_state["exp"] = 6
        st.session_state["level"] = "Mid"
        st.session_state["ind"] = "Healthcare"
        st.session_state["loc"] = "Tier 2 City"
        st.session_state["skills"] = 5
    elif load_entry:
        st.session_state["edu"] = "Diploma"
        st.session_state["exp"] = 1
        st.session_state["level"] = "Entry"
        st.session_state["ind"] = "Retail"
        st.session_state["loc"] = "Tier 3 City"
        st.session_state["skills"] = 2

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 🎓 Qualifications & Experience")
        education = st.selectbox(
            "Highest Education Level",
            ["PhD", "Master", "Bachelor", "Diploma", "High School"],
            index=["PhD", "Master", "Bachelor", "Diploma", "High School"].index(st.session_state.get("edu", "Master")),
            key="input_edu"
        )
        experience = st.slider(
            "Relevant Work Experience (Years)", 0, 20,
            value=int(st.session_state.get("exp", 8)), step=1,
            key="input_exp"
        )
        job_level = st.selectbox(
            "Role Seniority Level",
            ["Entry", "Junior", "Mid", "Senior", "Lead"],
            index=["Entry", "Junior", "Mid", "Senior", "Lead"].index(st.session_state.get("level", "Senior")),
            key="input_level"
        )

    with c_right:
        st.markdown("#### 🏢 Industry, Location & Skills")
        industry = st.selectbox(
            "Industry Sector",
            ["IT", "Finance", "Healthcare", "Manufacturing", "Retail", "Education"],
            index=["IT", "Finance", "Healthcare", "Manufacturing", "Retail", "Education"].index(st.session_state.get("ind", "IT")),
            key="input_ind"
        )
        location = st.selectbox(
            "Work Location Hub",
            ["Tier 1 City", "Tier 2 City", "Tier 3 City"],
            index=["Tier 1 City", "Tier 2 City", "Tier 3 City"].index(st.session_state.get("loc", "Tier 1 City")),
            key="input_loc", help="Tier 1 = Major metropolitan hubs with premium cost of living."
        )
        skill_count = st.slider(
            "Verified Specialized Skills Count", 1, 10,
            value=int(st.session_state.get("skills", 8)), step=1,
            key="input_skills", help="Number of validated technical/domain proficiencies."
        )

    st.markdown("---")
    eval_btn = st.button("🚀 Predict Salary Bracket", type="primary", width="stretch")

    candidate_df = pd.DataFrame([{
        "education": education,
        "experience_years": experience,
        "job_level": job_level,
        "industry": industry,
        "location_type": location,
        "skill_count": skill_count
    }])

    pred = model.predict(candidate_df)[0]
    probs = model.predict_proba(candidate_df)[0]
    classes = list(model.classes_)
    conf = max(probs) * 100

    st.markdown("### 📋 Compensation Bracket Outcome")
    r1, r2 = st.columns([1.3, 1.7], gap="medium")

    with r1:
        if pred == "High":
            card_class = "salary-card-high"
            color = "#10B981"
            badge = "Upper Percentile / Executive Bracket"
            icon = "💎"
        elif pred == "Medium":
            card_class = "salary-card-medium"
            color = "#3B82F6"
            badge = "Market Median / Competitive Bracket"
            icon = "⚖️"
        else:
            card_class = "salary-card-low"
            color = "#F59E0B"
            badge = "Entry / Foundational Base Bracket"
            icon = "🌱"

        st.markdown(f"""
        <div class="{card_class}">
            <span style="font-size: 2.8rem;">{icon}</span>
            <div style="color: {color}; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                {pred} Salary Tier
            </div>
            <div class="salary-title" style="color: {color};">
                {conf:.1f}% <span style="font-size: 1.1rem; color: #94A3B8;">Confidence</span>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">{badge}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Tier Probability Breakdown")
        for cls_name, p in zip(classes, probs):
            st.write(f"• **{cls_name} Tier:** {p * 100:.1f}%")
            st.progress(float(p))

    with r2:
        st.markdown("#### 🔍 Compensation Drivers & Insights")
        drivers = []
        if education in ["Master", "PhD"]:
            drivers.append(("Advanced Degree Premium", f"Holding a {education} substantially increases compensation leverage.", "info"))
        if experience >= 10:
            drivers.append(("Decade+ Domain Mastery", f"{experience} years of experience commands senior/lead market rates.", "info"))
        if job_level in ["Lead", "Senior"]:
            drivers.append(("Leadership & Ownership Tier", f"{job_level} roles include strategic scope and bonus multipliers.", "info"))
        if location == "Tier 1 City":
            drivers.append(("Metro Hub Cost-of-Living Adjustment", "Tier 1 metropolitan centers feature top-of-market compensation indexes.", "info"))
        if skill_count >= 7:
            drivers.append(("High Skill Breadth Multiplier", f"{skill_count} specialized skills place candidate in high demand.", "info"))

        if drivers:
            for title, desc, _ in drivers:
                st.success(f"**{title}**: {desc}")
        else:
            st.info("Candidate parameters correspond to entry/baseline compensation tables.")

        st.markdown(f"""
        <div class="comp-box">
            <strong style="color: #34D399;">Talent Acquisition & Compensation Advice:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                {'Extend top-band offer package with equity/incentive bonuses to close candidate.' if pred == 'High' else ('Offer standard competitive median package with milestone review in 6-12 months.' if pred == 'Medium' else 'Offer standard entry base package with structured upskilling and career laddering.')}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Features"):
        st.dataframe(candidate_df, width="stretch")

# --- TAB 2: Batch Talent Screening ---
with tabs[1]:
    st.subheader("Batch Talent Pool Compensation Screening")
    st.write("Upload candidate rosters or screen against the 1,500 record baseline job dataset.")

    csv_file = st.file_uploader("Upload Candidates CSV", type=["csv"], key="job_csv")
    df_jobs = None

    if csv_file is not None:
        df_jobs = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df_jobs)} candidate records from file.")
    else:
        sample_path = get_asset_path("data/job_salary_dataset.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load baseline job dataset (`data/job_salary_dataset.csv`)", value=True):
                df_jobs = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_jobs)} records from baseline job dataset.")

    if df_jobs is not None:
        req_cols = ["education", "experience_years", "job_level", "industry", "location_type", "skill_count"]
        missing = [c for c in req_cols if c not in df_jobs.columns]
        if missing:
            st.error(f"Missing required columns in dataset: {missing}")
        else:
            if st.button("⚡ Classify Entire Talent Pool", type="primary"):
                with st.spinner("Evaluating compensation bands..."):
                    preds = model.predict(df_jobs[req_cols])
                    probs = model.predict_proba(df_jobs[req_cols])

                    res_df = df_jobs.copy()
                    res_df["Predicted_Salary_Class"] = preds
                    res_df["Confidence_%"] = np.round(np.max(probs, axis=1) * 100, 1)

                    high_c = sum(preds == "High")
                    med_c = sum(preds == "Medium")
                    low_c = sum(preds == "Low")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Candidates", len(res_df))
                    m2.metric("High Bracket", high_c, delta=f"{(high_c/len(res_df))*100:.1f}%")
                    m3.metric("Medium Bracket", med_c)
                    m4.metric("Low Bracket", low_c)

                    f1, f2 = st.columns(2)
                    with f1:
                        f_salary = st.selectbox("Filter by Salary Class:", ["All", "High", "Medium", "Low"])
                    with f2:
                        f_ind = st.selectbox("Filter by Industry:", ["All"] + list(df_jobs["industry"].unique()))

                    view = res_df
                    if f_salary != "All":
                        view = view[view["Predicted_Salary_Class"] == f_salary]
                    if f_ind != "All":
                        view = view[view["industry"] == f_ind]

                    st.dataframe(view, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Classified Talent Pool as CSV",
                        data=csv_export,
                        file_name="job_salary_classifications.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Model Performance ---
with tabs[2]:
    st.subheader("Model Architecture & Classification Benchmark")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 Classification Architecture
        - **Algorithm**: `RandomForestClassifier(n_estimators=250, class_weight='balanced', random_state=42)`
        - **Pipeline Preprocessing**:
            - `OneHotEncoder` on categorical fields (`education`, `job_level`, `industry`, `location_type`)
            - Passthrough on numerical attributes (`experience_years`, `skill_count`)
        - **Accuracy Benchmark**:
            - **Overall Accuracy**: **86.00%** on Stratified Test Split
            - High recall on top-tier salaries (~0.91)
        """)

    with c2:
        img_path = get_asset_path("confusion_matrix.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Confusion Matrix on Test Split", width="stretch")
        else:
            st.info("Confusion matrix image not found.")

st.caption("Talent & HR Compensation Intelligence • Scikit-learn & Streamlit")
