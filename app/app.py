import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import MODEL_PATH

# Page config - compact layout
st.set_page_config(
    page_title="Mortality Risk (Within 24 Hours of Discharge)",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean dark theme styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&family=Ubuntu:wght@300;400;500;700&family=Inconsolata:wght@400;500;600;700;800;900&display=swap');

    /* Main dark background - no scrolling - professional dark theme */
    html, body {
        overflow: hidden !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        font-family: 'Rajdhani', sans-serif;
        color: #e6edf3 !important;
        overflow: hidden !important;
        height: 100vh !important;
        max-height: 100vh !important;
    }

    #root {
        overflow: hidden !important;
        height: 100vh !important;
    }

    /* Ensure Ubuntu font is loaded and available */
    @font-face {
        font-family: 'Ubuntu';
        font-style: normal;
        font-weight: 400;
        src: url('https://fonts.gstatic.com/s/ubuntu/v20/4iCs6KVjbNBYlgoKfw72nU6AFw.woff2') format('woff2');
    }

    /* Force all text to white */
    body, p, div, span, label, input, select, textarea, button {
        color: #ffffff !important;
    }

    /* Override Streamlit's default grey text */
    .stMarkdown, .stText, .stCaption, .stCode {
        color: #ffffff !important;
    }

    /* All Streamlit widget labels and text */
    .stSelectbox, .stSlider, .stRadio, .stNumberInput, .stTextInput {
        color: #ffffff !important;
    }

    .stSelectbox *, .stSlider *, .stRadio *, .stNumberInput *, .stTextInput * {
        color: #ffffff !important;
    }

    /* Ultra-compact spacing - no scrolling */
    .main {
        overflow: hidden !important;
        height: calc(100vh - 0.5rem) !important;
        max-height: calc(100vh - 0.5rem) !important;
    }

    .main > div {
        padding-top: 0 !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-bottom: 0.1rem !important;
        overflow: hidden !important;
        height: 100% !important;
        max-height: 100% !important;
    }

    /* Remove all margins to maximize space */
    * {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    h1, h3 {
        margin-top: 0 !important;
        margin-bottom: 0.1rem !important;
    }

    /* Prevent any element from causing overflow */
    [data-testid="stVerticalBlock"] {
        overflow: hidden !important;
        max-height: 100% !important;
    }

    /* Hide scrollbars completely */
    ::-webkit-scrollbar {
        display: none !important;
    }

    * {
        -ms-overflow-style: none !important;
        scrollbar-width: none !important;
    }

    /* Title - simple with glow */
    h1,
    [data-testid="stMarkdownContainer"] h1 {
        font-family: 'Inconsolata', monospace !important;
        font-size: 2.5rem !important;
        font-weight: 400 !important;
        margin: 0 auto 0.4rem auto !important;
        padding: 0.5rem 1.5rem !important;
        color: #ffffff !important;
        border: 1px solid rgba(88, 166, 255, 0.4) !important;
        border-radius: 12px !important;
        background-color: rgba(13, 17, 23, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.3),
                    0 0 40px rgba(88, 166, 255, 0.15) !important;
        display: inline-block !important;
        width: fit-content !important;
        text-align: center !important;
    }

    /* Center the title container */
    [data-testid="stMarkdownContainer"]:has(h1) {
        text-align: center !important;
    }

    h1 *,
    [data-testid="stMarkdownContainer"] h1 * {
        font-family: 'Inconsolata', monospace !important;
        color: #ffffff !important;
    }

    /* Section headers - professional styling */
    h3 {
        font-family: 'Inconsolata', monospace !important;
        font-size: 1.2rem !important;
        font-weight: 600;
        margin-top: 0.5rem !important;
        margin-bottom: 0.2rem !important;
        padding: 0.2rem 0.6rem !important;
        padding-top: 0 !important;
        color: #58a6ff !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        border-bottom: 1px solid rgba(88, 166, 255, 0.4) !important;
        display: inline-block !important;
    }

    /* Reduce spacing after section headers */
    h3 + div,
    h3 + [data-testid="stVerticalBlock"],
    h3 ~ [data-testid="stVerticalBlock"]:first-of-type {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Reduce spacing in columns after headers */
    h3 ~ div [data-testid="stVerticalBlock"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Ultra-compact spacing */
    .element-container {
        margin-bottom: 0.02rem !important;
        padding: 0 !important;
    }

    /* Selectbox styling - professional */
    .stSelectbox label {
        font-size: 0.65rem;
        color: #c9d1d9 !important;
        font-weight: 500;
    }

    .stSelectbox > div > div {
        background-color: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(88, 166, 255, 0.4) !important;
        border-radius: 6px !important;
        padding: 0.3rem 0.2rem !important;
        min-height: 2.2rem !important;
        height: auto !important;
        transition: all 0.2s ease !important;
        overflow: visible !important;
    }

    .stSelectbox [data-baseweb="select"] {
        min-height: 2.2rem !important;
        height: auto !important;
        padding: 0.3rem 0.5rem !important;
        line-height: 1.4 !important;
        overflow: visible !important;
    }

    .stSelectbox [data-baseweb="select"] > div {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
    }

    .stSelectbox [data-baseweb="select"] span {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        line-height: 1.5 !important;
    }

    .stSelectbox > div > div:hover {
        border-color: rgba(88, 166, 255, 0.4) !important;
        background-color: rgba(22, 27, 34, 0.95) !important;
    }

    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] * {
        color: #ffffff !important;
    }

    [data-baseweb="popover"] {
        background-color: rgba(26, 31, 58, 0.95) !important;
    }

    /* Force all dropdowns to open downward */
    [data-baseweb="popover"][data-placement^="top"] {
        transform: translateY(0) !important;
    }

    /* Dropdown options - dark grey text on white background */
    [data-baseweb="popover"] [data-baseweb="option"],
    [data-baseweb="popover"] [data-baseweb="option"] *,
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] li * {
        color: #333333 !important;
        background-color: #ffffff !important;
    }

    [data-baseweb="popover"] [data-baseweb="option"]:hover,
    [data-baseweb="popover"] [data-baseweb="option"]:hover * {
        background-color: #f0f0f0 !important;
        color: #333333 !important;
    }

    /* Slider label only */
    .stSlider label {
        font-size: 0.65rem;
        color: #c9d1d9 !important;
    }

    /* Slider range numbers styling only */
    .stSlider > div:last-child * {
        color: #ffffff !important;
        font-size: 0.65rem !important;
    }

    /* Radio button styling - professional */
    .stRadio label {
        font-size: 0.65rem;
        color: #c9d1d9 !important;
        font-weight: 500;
    }

    .stRadio input[type="radio"]:checked + label::before,
    .stRadio [data-baseweb="radio"] input:checked + label::before {
        background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 100%) !important;
        border-color: #58a6ff !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.4),
                    0 0 0 2px rgba(88, 166, 255, 0.2) !important;
        width: 18px !important;
        height: 18px !important;
    }

    .stRadio input[type="radio"]:checked,
    .stRadio [data-baseweb="radio"] input:checked {
        accent-color: #58a6ff !important;
    }

    /* Override Streamlit's red colors with blue */
    .stRadio input[type="radio"]:checked + label::after {
        background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 100%) !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.4) !important;
    }

    /* Unchecked radio buttons - subtle style */
    .stRadio input[type="radio"] + label::before,
    .stRadio [data-baseweb="radio"] input + label::before {
        border-color: rgba(88, 166, 255, 0.3) !important;
        border-radius: 50% !important;
    }

    /* Force blue on all radio elements */
    .stRadio * {
        accent-color: #58a6ff !important;
    }

    /* Change Streamlit's primary color variable - professional blue */
    :root {
        --primary-color: #58a6ff !important;
        --primary-fg-color: #58a6ff !important;
    }

    /* Override any red colors in widgets */
    [style*="rgb(255, 75, 75)"],
    [style*="rgb(255, 0, 0)"],
    [style*="rgb(239, 83, 80)"],
    [style*="#ff4b4b"],
    [style*="#ff0000"] {
        background-color: #58a6ff !important;
        background: #58a6ff !important;
    }

    /* Metric cards - professional styling */
    [data-testid="stMetricValue"] {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem !important;
        font-weight: 700;
        color: #58a6ff !important;
    }

    /* Mortality risk number - force red (override global white text) */
    div.mortality-risk-number,
    div.mortality-risk-number *,
    .mortality-risk-number,
    .mortality-risk-number *,
    [class*="mortality-risk-number"],
    [class*="mortality-risk-number"] * {
        color: #ff4444 !important;
        -webkit-text-fill-color: #ff4444 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.6rem !important;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }

    /* Metric container styling */
    [data-testid="stMetricContainer"] {
        background-color: rgba(22, 27, 34, 0.5) !important;
        border: 1px solid rgba(88, 166, 255, 0.4) !important;
        border-radius: 8px !important;
        padding: 0.8rem !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Progress bar - light blue fill, dark blue background */
    .stProgress > div > div {
        background-color: #1a3a5c !important;
        border-radius: 4px !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 100%) !important;
        background-color: #58a6ff !important;
        border-radius: 4px !important;
    }

    .stCaption {
        font-size: 0.65rem;
        color: #ffffff;
    }

    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Ultra-compact spacing */
    .stSelectbox, .stSlider, .stRadio {
        margin-bottom: 0.02rem !important;
        padding: 0 !important;
    }

    /* Remove extra spacing from Streamlit containers */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] {
        gap: 0.1rem !important;
    }


    /* CSS variables */
    :root {
        --primary-color: #00d4ff !important;
    }
    </style>
    <script>
    // Minimal - just remove highlights from range numbers
    setInterval(function() {
        document.querySelectorAll('.stSlider > div:last-child *').forEach(el => {
            el.style.background = 'transparent';
            el.style.backgroundColor = 'transparent';
        });

        // Simple: force all popovers with top placement to open downward
        document.querySelectorAll('[data-baseweb="popover"][data-placement^="top"]').forEach(popover => {
            const selectbox = popover.previousElementSibling?.closest('.stSelectbox') ||
                            document.querySelector('.stSelectbox:has([data-baseweb="select"]:focus)');
            if (selectbox) {
                const rect = selectbox.getBoundingClientRect();
                popover.style.top = (rect.bottom + window.scrollY) + 'px';
                popover.style.bottom = 'auto';
            }
        });
    }, 100);
    </script>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    return joblib.load(MODEL_PATH)

def create_feature_vector(user_inputs, model_features):
    """Convert user inputs to model feature vector"""
    # Initialize feature vector with zeros
    feature_dict = {feature: 0.0 for feature in model_features}

    # Set numerical features
    feature_dict['length_of_stay_hospital'] = user_inputs['length_of_stay_hospital']
    feature_dict['length_of_stay_unit'] = user_inputs['length_of_stay_unit']
    feature_dict['review_flag'] = user_inputs['review_flag']

    # Set datetime-derived features
    feature_dict['admission_hour'] = user_inputs['admission_hour']
    feature_dict['admission_day_of_week'] = user_inputs['admission_day_of_week']
    feature_dict['admission_month'] = user_inputs['admission_month']
    feature_dict['discharge_hour'] = user_inputs['discharge_hour']
    feature_dict['discharge_day_of_week'] = user_inputs['discharge_day_of_week']
    feature_dict['discharge_month'] = user_inputs['discharge_month']

    # Calculate stay hours from dates
    feature_dict['hospital_stay_hours'] = user_inputs['length_of_stay_hospital'] * 24
    feature_dict['unit_stay_hours'] = user_inputs['length_of_stay_unit'] * 24

    # Death-related features
    feature_dict['hours_to_death'] = user_inputs.get('hours_to_death', 0)
    feature_dict['has_death_date'] = user_inputs.get('has_death_date', 0)

    # Set one-hot encoded categorical features
    specialty_col = f"specialty_{user_inputs['specialty']}"
    if specialty_col in model_features:
        feature_dict[specialty_col] = 1.0

    ward_col = f"last_acute_ward_{user_inputs['last_acute_ward']}"
    if ward_col in model_features:
        feature_dict[ward_col] = 1.0

    diagnostic_col = f"diagnostic_ICD_{user_inputs['diagnostic_ICD']}"
    if diagnostic_col in model_features:
        feature_dict[diagnostic_col] = 1.0

    diagnostic_group_col = f"diagnostic_group_{user_inputs['diagnostic_group']}"
    if diagnostic_group_col in model_features:
        feature_dict[diagnostic_group_col] = 1.0

    # Convert to DataFrame with correct column order
    feature_df = pd.DataFrame([feature_dict])[model_features]
    return feature_df

def main():
    st.title("Mortality Risk (Within 24 Hours of Discharge)")

    # Load model first (needed for dropdown options)
    try:
        model_data = load_model()
        model = model_data['model']
        model_features = model_data['features']
        st.sidebar.markdown(f"<p style='color: #00d4ff; font-size: 0.7rem;'>MODEL: {model_data['model_name'].upper()}</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Extract available options from model features
    specialty_features = [f.replace('specialty_', '') for f in model_features if 'specialty' in f]
    ward_features = [f.replace('last_acute_ward_', '') for f in model_features if 'ward' in f.lower()]
    diagnostic_group_features = [f.replace('diagnostic_group_', '') for f in model_features if 'diagnostic_group' in f]
    diagnostic_icd_features = [f.replace('diagnostic_ICD_', '') for f in model_features if 'diagnostic_ICD' in f]

    # Ultra-compact layout - 6 columns for maximum density
    st.markdown("### TEMPORAL FACTORS")
    temp_col1, temp_col2, temp_col3, temp_col4, temp_col5, temp_col6 = st.columns(6)

    with temp_col1:
        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        admission_month = st.selectbox("Admission Month", options=list(month_names.keys()), format_func=lambda x: month_names[x], index=1, label_visibility="visible")

    with temp_col2:
        day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        admission_day = st.selectbox("Admission Day", options=list(day_names.keys()), format_func=lambda x: day_names[x], index=0, label_visibility="visible")

    with temp_col3:
        admission_hour = st.slider("Admission Hour", min_value=0, max_value=23, value=11, label_visibility="visible")

    with temp_col4:
        discharge_month = st.selectbox("Discharge Month", options=list(month_names.keys()), format_func=lambda x: month_names[x], index=1, label_visibility="visible")

    with temp_col5:
        discharge_day = st.selectbox("Discharge Day", options=list(day_names.keys()), format_func=lambda x: day_names[x], index=1, label_visibility="visible")

    with temp_col6:
        discharge_hour = st.slider("Discharge Hour", min_value=0, max_value=23, value=15, label_visibility="visible")

    st.markdown("### CLINICAL FACTORS")
    clin_col1, clin_col2, clin_col3, clin_col4, clin_col5, clin_col6 = st.columns(6)

    with clin_col1:
        specialty = st.selectbox("Specialty", options=specialty_features if specialty_features else ["West Roxbury Surgical"], label_visibility="visible")

    with clin_col2:
        last_acute_ward = st.selectbox("Unit", options=ward_features if ward_features else ["A-1 S WX"], label_visibility="visible")

    with clin_col3:
        diagnostic_group = st.selectbox("Diagnostic Group", options=diagnostic_group_features if diagnostic_group_features else ["Respiratory-M"], label_visibility="visible")

    with clin_col4:
        diagnostic_ICD = st.selectbox("ICD Code", options=diagnostic_icd_features if diagnostic_icd_features else ["R09.02"], label_visibility="visible")

    with clin_col5:
        length_of_stay_hospital = st.slider("Hospital Stay (days)", min_value=0.0, max_value=30.0, value=1.15, step=0.1, label_visibility="visible")

    with clin_col6:
        length_of_stay_unit = st.slider("Unit Stay (days)", min_value=0.0, max_value=30.0, value=1.15, step=0.1, label_visibility="visible")
        review_flag = st.radio("Review", options=[0, 1], format_func=lambda x: "Y" if x == 1 else "N", index=0, horizontal=True, label_visibility="visible")

    # Results section - ultra compact
    st.markdown("### PREDICTION")

    # Prepare inputs
    user_inputs = {
        'length_of_stay_hospital': length_of_stay_hospital,
        'length_of_stay_unit': length_of_stay_unit,
        'review_flag': review_flag,
        'admission_hour': admission_hour,
        'admission_day_of_week': admission_day,
        'admission_month': admission_month,
        'discharge_hour': discharge_hour,
        'discharge_day_of_week': discharge_day,
        'discharge_month': discharge_month,
        'specialty': specialty,
        'last_acute_ward': last_acute_ward,
        'diagnostic_ICD': diagnostic_ICD,
        'diagnostic_group': diagnostic_group,
        'hours_to_death': 0,
        'has_death_date': 0
    }

    # Create feature vector and predict
    try:
        feature_vector = create_feature_vector(user_inputs, model_features)
        probability = model.predict_proba(feature_vector)[0][1]
        prediction = model.predict(feature_vector)[0]

        # Display results compactly in one row
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)

        with result_col1:
            st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.65rem; color: #ffffff; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.2rem;">
                        MORTALITY RISK
                    </div>
                    <div class="mortality-risk-number" style="font-family: 'Rajdhani', sans-serif; font-size: 2.5rem; font-weight: 600;
                                 color: #ff4444 !important;
                                 text-shadow: 0 0 20px rgba(255, 68, 68, 0.5);">
                        {probability:.1%}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with result_col2:
            risk_level = "HIGH" if probability > 0.5 else "LOW"
            st.metric("RISK LEVEL", risk_level)

        with result_col3:
            prediction_text = "YES" if prediction == 1 else "NO"
            st.metric("PREDICTION", prediction_text)

        with result_col4:
            st.metric("PROBABILITY", f"{probability:.4f}")

        # Compact progress bar
        st.progress(probability)
        st.caption(f"{probability*100:.2f}%")

    except Exception as e:
        st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
