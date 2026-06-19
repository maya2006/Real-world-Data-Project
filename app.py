# ============================================================
# Smart Diabetes Risk Prediction System
# app.py — Interactive Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ──────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'diabetes.csv')

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f1629 0%, #1a2340 50%, #0f1629 100%);
    color: #e8eaf6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141d35 0%, #1e2d50 100%);
    border-right: 1px solid rgba(76,155,232,0.2);
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(76,155,232,0.12) 0%, rgba(76,155,232,0.05) 100%);
    border: 1px solid rgba(76,155,232,0.3);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 10px;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .label { font-size: 12px; color: #8ba4c8; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.metric-card .value { font-size: 28px; font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #4C9BE8; }

/* ── Risk cards ── */
.risk-high {
    background: linear-gradient(135deg, rgba(232,84,76,0.18), rgba(232,84,76,0.05));
    border: 2px solid rgba(232,84,76,0.6);
    border-radius: 16px; padding: 24px; text-align: center;
}
.risk-medium {
    background: linear-gradient(135deg, rgba(232,168,76,0.18), rgba(232,168,76,0.05));
    border: 2px solid rgba(232,168,76,0.6);
    border-radius: 16px; padding: 24px; text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, rgba(76,232,122,0.18), rgba(76,232,122,0.05));
    border: 2px solid rgba(76,232,122,0.6);
    border-radius: 16px; padding: 24px; text-align: center;
}

/* ── Section headers ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 600;
    color: #4C9BE8; margin-bottom: 4px;
    border-left: 4px solid #4C9BE8;
    padding-left: 12px;
}
.section-sub {
    font-size: 13px; color: #6e88aa;
    padding-left: 16px; margin-bottom: 20px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4C9BE8, #2e7dd4);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; font-size: 15px;
    padding: 10px 28px; width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2e7dd4, #1a60b8);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(76,155,232,0.4);
}

/* ── Input sliders ── */
.stSlider [data-baseweb="slider"] { padding: 0 4px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(76,155,232,0.08);
    border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px;
    color: #8ba4c8; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(76,155,232,0.25) !important;
    color: #4C9BE8 !important;
}

/* ── Recommendation boxes ── */
.rec-box {
    background: rgba(76,155,232,0.08);
    border-left: 3px solid #4C9BE8;
    border-radius: 0 10px 10px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 14px;
}

/* ── Divider ── */
.custom-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(76,155,232,0.4), transparent);
    margin: 24px 0;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(76,155,232,0.15), rgba(46,125,212,0.08));
    border: 1px solid rgba(76,155,232,0.25);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex; align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Scaler ────────────────────────────────────
@st.cache_resource
def load_model():
    model_path  = os.path.join(MODEL_DIR, 'diabetes_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.pkl')

    if not os.path.exists(model_path):
        return None, None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    return model, scaler, metrics

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

model, scaler, model_metrics = load_model()
df = load_dataset()

# ─── Helper Functions ────────────────────────────────────────
def classify_risk(probability, glucose, bmi, age):
    """
    Multi-factor risk classification.
    Uses model probability + clinical thresholds.
    """
    score = probability * 100

    # Clinical boosters
    if glucose > 140: score += 15
    elif glucose > 110: score += 7

    if bmi > 30: score += 10
    elif bmi > 25: score += 5

    if age > 50: score += 8
    elif age > 40: score += 4

    score = min(score, 100)

    if score >= 60:
        return "High Risk", score, "🔴"
    elif score >= 35:
        return "Medium Risk", score, "🟡"
    else:
        return "Low Risk", score, "🟢"


def get_recommendations(risk_level, glucose, bmi, bp, insulin, age):
    """Return personalised lifestyle recommendations."""
    recs = []

    if risk_level == "High Risk":
        recs.append("🩺 **Consult a doctor immediately** — schedule a diabetes screening test (HbA1c).")
        recs.append("🥗 **Strict dietary changes** — eliminate sugary drinks, white rice, and processed foods.")
        recs.append("🏃 **Daily exercise** — aim for at least 45 minutes of moderate activity (brisk walking, cycling).")
        recs.append("📊 **Monitor blood sugar** — track glucose levels daily if possible.")
        recs.append("💊 **Follow medication** — take prescribed medications without skipping.")
        if bmi > 30:
            recs.append("⚖️ **Weight management** — target a 5–10% weight reduction; even small losses reduce diabetes risk significantly.")
        if bp > 90:
            recs.append("🫀 **Manage blood pressure** — reduce salt intake and follow the DASH diet.")

    elif risk_level == "Medium Risk":
        recs.append("🥦 **Adopt a balanced diet** — increase vegetables, whole grains, lean protein; reduce sugar.")
        recs.append("🚶 **Regular walking** — 30 minutes of brisk walking at least 5 days per week.")
        recs.append("🩺 **Annual check-up** — test fasting blood glucose and HbA1c once a year.")
        recs.append("🧘 **Stress management** — practice yoga or meditation to lower cortisol levels.")
        if insulin > 150:
            recs.append("⚗️ **Insulin resistance** — reduce refined carbs; consider consulting a dietitian.")
        if bmi > 25:
            recs.append("⚖️ **Healthy weight goal** — aim for BMI below 25 through diet and exercise.")

    else:  # Low Risk
        recs.append("✅ **Maintain your healthy lifestyle** — keep up the good habits!")
        recs.append("🥗 **Continue balanced nutrition** — fruits, vegetables, whole grains, lean proteins.")
        recs.append("🏋️ **Stay active** — 150 minutes of moderate exercise per week is the goal.")
        recs.append("😴 **Prioritise sleep** — aim for 7–9 hours per night; poor sleep raises blood sugar.")
        recs.append("🩺 **Routine screening** — get blood work done every 2–3 years as a preventive measure.")

    return recs


def dark_fig():
    """Return a figure with dark background matching app theme."""
    fig, ax = plt.subplots(facecolor='#1a2340')
    ax.set_facecolor('#1a2340')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e3f5e')
    ax.tick_params(colors='#8ba4c8')
    ax.xaxis.label.set_color('#8ba4c8')
    ax.yaxis.label.set_color('#8ba4c8')
    ax.title.set_color('#c8d8f0')
    return fig, ax


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🩺 Patient Input")
    st.markdown("Enter the patient's health parameters below.")
    st.markdown("---")

    pregnancies = st.slider("Pregnancies",     0,  17,  1)
    glucose     = st.slider("Glucose (mg/dL)", 50, 250, 110)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 30, 130, 72)
    skin_thickness = st.slider("Skin Thickness (mm)",    5,  99,  20)
    insulin     = st.slider("Insulin (μU/mL)", 0, 900, 80)
    bmi         = st.slider("BMI",             10.0, 70.0, 25.0, step=0.1)
    dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.35, step=0.01)
    age         = st.slider("Age (years)",     18, 90, 35)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#506278;line-height:1.6;'>
    <b>Dataset:</b> Pima Indians Diabetes<br>
    <b>Model:</b> Random Forest Classifier<br>
    <b>Domain:</b> Healthcare Analytics<br><br>
    <i>For educational purposes only.<br>
    Always consult a qualified doctor.</i>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════
# ── Hero Banner ──
st.markdown("""
<div class="hero-banner">
  <div>
    <h1 style="font-family:'Space Grotesk',sans-serif;font-size:30px;color:#4C9BE8;margin:0 0 6px 0;">
      🩺 Smart Diabetes Risk Prediction System
    </h1>
    <p style="color:#8ba4c8;margin:0;font-size:14px;">
      AI-powered healthcare analytics · Random Forest Classifier · 
      Pima Indians Diabetes Dataset · Final Year Project
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Check model availability ──
if model is None:
    st.error("⚠️ **Model not found!** Please run `python main.py` first to train and save the model.")
    st.info("Run this command in your terminal:\n```bash\npython main.py\n```")
    st.stop()

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction & Risk",
    "📊 Data Visualisation",
    "🤖 Model Performance",
    "📋 Dataset Explorer"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — Prediction & Risk
# ════════════════════════════════════════════════════════════
with tab1:
    if predict_btn or True:  # show input summary always
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        scaled_input = scaler.transform(input_data)

        if predict_btn:
            with st.spinner("Analysing patient data …"):
                prediction   = model.predict(scaled_input)[0]
                probability  = model.predict_proba(scaled_input)[0][1]
                risk_level, risk_score, risk_icon = classify_risk(
                    probability, glucose, bmi, age)

            # ── Prediction Result ──
            st.markdown('<div class="section-title">Prediction Result</div>'
                        '<div class="section-sub">Based on Random Forest Classifier</div>',
                        unsafe_allow_html=True)

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                diag_color = "#E8544C" if prediction == 1 else "#4CE87A"
                diag_label = "DIABETIC" if prediction == 1 else "NON-DIABETIC"
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Diagnosis</div>
                  <div class="value" style="color:{diag_color}">{diag_label}</div>
                </div>""", unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Probability</div>
                  <div class="value">{probability*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with col_r3:
                risk_colors = {"High Risk": "#E8544C",
                               "Medium Risk": "#E8A84C",
                               "Low Risk": "#4CE87A"}
                rc = risk_colors[risk_level]
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Risk Level</div>
                  <div class="value" style="color:{rc}">{risk_icon} {risk_level}</div>
                </div>""", unsafe_allow_html=True)

            # ── Risk Gauge ──
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Risk Score Gauge</div>', unsafe_allow_html=True)

            fig_g, ax_g = plt.subplots(figsize=(7, 1.4), facecolor='#1a2340')
            ax_g.set_facecolor('#1a2340')
            # Background track
            ax_g.barh(0, 100, height=0.55, color='#1e2d50', edgecolor='none', left=0)
            # Gradient segments
            for start, end, color in [(0, 35, '#4CE87A'), (35, 60, '#E8A84C'), (60, 100, '#E8544C')]:
                ax_g.barh(0, end-start, height=0.55, color=color,
                          alpha=0.35, edgecolor='none', left=start)
            # Actual score bar
            bar_color = risk_colors[risk_level]
            ax_g.barh(0, risk_score, height=0.55, color=bar_color,
                      alpha=0.85, edgecolor='none', left=0)
            # Score marker
            ax_g.axvline(risk_score, color='white', linewidth=2.5, alpha=0.9)
            ax_g.text(risk_score, 0.38, f'{risk_score:.0f}', color='white',
                      fontsize=11, fontweight='bold', ha='center')
            for val, label in [(17, 'Low'), (47, 'Medium'), (80, 'High')]:
                ax_g.text(val, -0.38, label, color='#6e88aa', fontsize=9, ha='center')
            ax_g.set_xlim(0, 100)
            ax_g.set_ylim(-0.6, 0.6)
            ax_g.axis('off')
            plt.tight_layout(pad=0.3)
            st.pyplot(fig_g)
            plt.close()

            # ── Risk Card ──
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            risk_class = risk_level.lower().replace(" ", "-")
            risk_descriptions = {
                "High Risk": "Immediate medical attention is strongly recommended.",
                "Medium Risk": "Preventive measures and lifestyle changes are advised.",
                "Low Risk": "Your health parameters look good — keep it up!"
            }
            st.markdown(f"""
            <div class="risk-{risk_class}">
              <h3 style="margin:0 0 6px 0;">{risk_icon} {risk_level}</h3>
              <p style="margin:0;font-size:14px;opacity:0.85;">{risk_descriptions[risk_level]}</p>
            </div>""", unsafe_allow_html=True)

            # ── Lifestyle Recommendations ──
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Lifestyle Recommendations</div>'
                        '<div class="section-sub">Personalised based on your health parameters</div>',
                        unsafe_allow_html=True)

            recs = get_recommendations(risk_level, glucose, bmi,
                                       blood_pressure, insulin, age)
            for rec in recs:
                st.markdown(f'<div class="rec-box">{rec}</div>', unsafe_allow_html=True)

            # ── Input Parameter Summary ──
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Input Parameter Summary</div>', unsafe_allow_html=True)

            params = {
                'Parameter'  : ['Pregnancies', 'Glucose', 'Blood Pressure',
                                 'Skin Thickness', 'Insulin', 'BMI',
                                 'Diabetes Pedigree', 'Age'],
                'Value'      : [pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, f'{bmi:.1f}',
                                f'{dpf:.2f}', age],
                'Normal Range': ['0 – 10', '70 – 99 mg/dL', '60 – 80 mm Hg',
                                  '10 – 50 mm', '16 – 166 μU/mL', '18.5 – 24.9',
                                  '0.0 – 1.0', '—'],
            }
            summary_df = pd.DataFrame(params)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        else:
            st.info("👈 **Enter patient data in the sidebar** and click **Predict Diabetes Risk** to see results.")
            st.markdown("""
            <div style='background:rgba(76,155,232,0.07);border-radius:14px;padding:24px;margin-top:16px;'>
              <h4 style='color:#4C9BE8;'>How It Works</h4>
              <ol style='color:#8ba4c8;line-height:2;'>
                <li>Enter patient health parameters using the sidebar sliders</li>
                <li>Click <b>Predict Diabetes Risk</b></li>
                <li>View the diagnosis, risk score, and risk level</li>
                <li>Read personalised lifestyle recommendations</li>
              </ol>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — Data Visualisation
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>'
                '<div class="section-sub">Visualisations generated from the Pima Indians Diabetes Dataset</div>',
                unsafe_allow_html=True)

    def show_image(filename, caption):
        path = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
        else:
            st.warning(f"Image not found: {filename} — run main.py first.")

    col1, col2 = st.columns(2)
    with col1:
        show_image('outcome_distribution.png',  'Outcome Distribution')
        show_image('glucose_analysis.png',       'Glucose Distribution by Outcome')
        show_image('age_glucose_scatter.png',    'Age vs Glucose Scatter')
    with col2:
        show_image('bmi_analysis.png',           'BMI Distribution by Outcome')
        show_image('feature_importance.png',     'Feature Importance — Random Forest')

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    show_image('feature_histograms.png', 'Feature Histograms')
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    show_image('correlation_heatmap.png', 'Feature Correlation Heatmap')

    # Live charts from loaded dataset
    if df is not None:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Live Dataset Insights</div>', unsafe_allow_html=True)

        # Preprocess zeros
        df_clean = df.copy()
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df_clean[col] = df_clean[col].replace(0, df_clean[col].median())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Age Distribution**")
            fig, ax = dark_fig()
            fig.set_size_inches(5, 3)
            ax.hist(df_clean[df_clean['Outcome']==0]['Age'], bins=20,
                    color='#4C9BE8', alpha=0.7, label='Non-Diabetic')
            ax.hist(df_clean[df_clean['Outcome']==1]['Age'], bins=20,
                    color='#E8544C', alpha=0.7, label='Diabetic')
            ax.legend(facecolor='#1e2d50', edgecolor='none',
                      labelcolor='#c8d8f0', fontsize=8)
            ax.set_xlabel('Age')
            ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with c2:
            st.markdown("**Glucose vs BMI (Interactive)**")
            fig, ax = dark_fig()
            fig.set_size_inches(5, 3)
            for outcome, color, label in [(0,'#4C9BE8','Non-Diabetic'),
                                          (1,'#E8544C','Diabetic')]:
                sub = df_clean[df_clean['Outcome']==outcome]
                ax.scatter(sub['BMI'], sub['Glucose'],
                           c=color, alpha=0.5, s=18, label=label)
            ax.legend(facecolor='#1e2d50', edgecolor='none',
                      labelcolor='#c8d8f0', fontsize=8)
            ax.set_xlabel('BMI')
            ax.set_ylabel('Glucose')
            plt.tight_layout()
            st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Model Performance Comparison</div>'
                '<div class="section-sub">Logistic Regression · Decision Tree · Random Forest</div>',
                unsafe_allow_html=True)

    if model_metrics:
        # Metric cards for Random Forest
        rf = model_metrics.get('Random Forest', {})
        cols = st.columns(4)
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [rf.get('accuracy',0), rf.get('precision',0),
                  rf.get('recall',0),   rf.get('f1',0)]
        for col, label, val in zip(cols, labels, values):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">{label}</div>
                  <div class="value">{val*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Comparison table
        rows = []
        for name, metrics in model_metrics.items():
            rows.append({
                'Model'    : name,
                'Accuracy' : f"{metrics['accuracy']*100:.2f}%",
                'Precision': f"{metrics['precision']*100:.2f}%",
                'Recall'   : f"{metrics['recall']*100:.2f}%",
                'F1-Score' : f"{metrics['f1']*100:.2f}%"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        path = os.path.join(IMAGE_DIR, 'model_comparison.png')
        if os.path.exists(path):
            st.image(path, caption='Model Comparison', use_column_width=True)
    with col_m2:
        path = os.path.join(IMAGE_DIR, 'confusion_matrix.png')
        if os.path.exists(path):
            st.image(path, caption='Confusion Matrix — Random Forest', use_column_width=True)

    # Why Random Forest?
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why Random Forest?</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, title, desc in [
        (c1, "🌳 Ensemble Method",
         "Combines 100 decision trees to reduce overfitting and improve generalisation."),
        (c2, "📈 High Accuracy",
         "Typically achieves the highest accuracy among baseline classifiers on this dataset."),
        (c3, "🏥 Healthcare Fit",
         "Handles non-linear feature interactions and imbalanced datasets well — ideal for clinical data."),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:rgba(76,155,232,0.07);border-radius:12px;
                        padding:18px;text-align:center;height:130px;'>
              <h4 style='color:#4C9BE8;margin:0 0 8px 0;'>{title}</h4>
              <p style='color:#8ba4c8;font-size:13px;margin:0;'>{desc}</p>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — Dataset Explorer
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Dataset Explorer</div>'
                '<div class="section-sub">Pima Indians Diabetes Dataset — UCI Machine Learning Repository</div>',
                unsafe_allow_html=True)

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
              <div class="label">Total Records</div>
              <div class="value">{len(df)}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
              <div class="label">Features</div>
              <div class="value">{df.shape[1]-1}</div></div>""", unsafe_allow_html=True)
        with c3:
            n_diab = df['Outcome'].sum()
            st.markdown(f"""<div class="metric-card">
              <div class="label">Diabetic</div>
              <div class="value" style="color:#E8544C">{n_diab}</div></div>""", unsafe_allow_html=True)
        with c4:
            n_non = len(df) - n_diab
            st.markdown(f"""<div class="metric-card">
              <div class="label">Non-Diabetic</div>
              <div class="value" style="color:#4CE87A">{n_non}</div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        search_col, filter_col = st.columns([3, 1])
        with filter_col:
            outcome_filter = st.selectbox("Filter by Outcome",
                                          ["All", "Diabetic (1)", "Non-Diabetic (0)"])
        filtered_df = df.copy()
        if outcome_filter == "Diabetic (1)":
            filtered_df = df[df['Outcome'] == 1]
        elif outcome_filter == "Non-Diabetic (0)":
            filtered_df = df[df['Outcome'] == 0]

        st.dataframe(filtered_df.reset_index(drop=True),
                     use_container_width=True, height=320)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Statistical Summary**")
        st.dataframe(df.describe().round(2), use_container_width=True)
    else:
        st.warning("Dataset not found. Place `diabetes.csv` inside the `dataset/` folder.")


# ─── Footer ──────────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#3d5270;font-size:12px;padding:10px 0 20px;'>
  Smart Diabetes Risk Prediction System &nbsp;·&nbsp;
  Final Year Data Science Project &nbsp;·&nbsp;
  Random Forest Classifier &nbsp;·&nbsp;
  Pima Indians Diabetes Dataset
</div>
""", unsafe_allow_html=True)
