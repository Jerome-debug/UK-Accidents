import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE SETUP ---
st.set_page_config(
    page_title="UK Road Safety Analysis & Prediction",
    page_icon="🚗",
    layout="wide"
)

# --- CACHED MODEL LOADER ---
@st.cache_resource
def load_ml_components():
    """Loads the trained model and scaler safely if they exist."""
    model_path = "accident_severity_deployment_pipeline.pkl"
    scaler_path = "scaler.pkl"
    
    components = {"model": None, "scaler": None, "ready": False}
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            with open(model_path, "rb") as f:
                components["model"] = pickle.load(f)
            with open(scaler_path, "rb") as f:
                components["scaler"] = pickle.load(f)
            components["ready"] = True
        except Exception as e:
            st.error(f"Error loading model files: {e}")
            
    return components

ml_components = load_ml_components()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to:", ["Project Overview & EDA", "Accident Severity Predictor"])

# ==========================================
# MODE 1: PROJECT OVERVIEW & EDA
# ==========================================
if app_mode == "Project Overview & EDA":
    st.title("🇬🇧 UK Road Safety Accident Analysis Dashboard")
    st.markdown("""
    Based on the comprehensive dataset capturing over **1.8 million records** of structural road incidents compiled by the UK government.
    This interactive deployment app provides key visual indicators and predictive capability regarding accident severity metrics.
    """)
    
    # Core KPI Metrices
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Analyzed Records", value="1.5M+")
    with col2:
        st.metric(label="Primary Target Class", value="Accident Severity")
    with col3:
        st.metric(label="Temporal Scope", value="2000 - 2018")
    with col4:
        st.metric(label="Feature Count", value="33 Attributes")
        
    st.write("---")
    st.subheader("Key Analytical Trends (Kaggle Insights)")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("**📅 Distribution of Casualties by Day of Week**")
        # Creating a mock representation of the top weekly trend discovered in the Kaggle notebook
        days = ['Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Saturday', 'Monday', 'Sunday']
        casualties = [331934, 299044, 297756, 294476, 285261, 284043, 239532]
        chart_data = pd.DataFrame({"Casualties": casualties}, index=days)
        st.bar_chart(chart_data, color="#e94560")
        st.caption("Insight: Friday (Day 6) and middle weekdays see peak structural traffic and collision incident frequency.")

    with eda_col2:
        st.markdown("**🌦️ Environmental Hazard Mappings**")
        weather_mix = pd.DataFrame({
            'Condition': ['Fine without high winds', 'Raining without high winds', 'Other/Unknown', 'Snowing/Fog'],
            'Impact Share': [75, 15, 7, 3]
        })
        st.dataframe(weather_mix, use_container_width=True, hide_index=True)
        st.caption("Insight: Counter-intuitively, the vast majority of severe collisions happen during completely 'Fine' clear conditions, indicating driver speed/complacency issues rather than elemental visibility alone.")

# ==========================================
# MODE 2: ACCIDENT SEVERITY PREDICTOR
# ==========================================
elif app_mode == "Accident Severity Predictor":
    st.title("🔮 Real-Time Severity Prediction Pipeline")
    
    # Model Status Check Alert
    if not ml_components["ready"]:
        st.warning("⚠️ Predictive Inference Engine is in Simulation Mode. Place your structural `model.pkl` and `scaler.pkl` in the app directory to bind live production predictions.")
    else:
        st.success("✅ Operational: Production ML Models loaded successfully.")

    st.markdown("Provide structural incident environmental inputs below to project the localized collision profile:")
    
    # Setup interactive input grid form
    with st.form("prediction_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            num_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=20, value=2)
            light_conditions = st.selectbox("Lighting Conditions", [
                "Daylight: Street light present",
                "Darkness: Street lights present and lit",
                "Darkness: No street lighting",
                "Darkness: Street lights present but unlit"
            ])
            
        with col_in2:
            num_casualties = st.number_input("Expected Number of Casualties", min_value=1, max_value=50, value=1)
            weather_conditions = st.selectbox("Weather Conditions", [
                "Fine without high winds",
                "Raining without high winds",
                "Snowing without high winds",
                "Fine with high winds",
                "Raining with high winds",
                "Fog or mist"
            ])
            
        with col_in3:
            speed_limit = st.slider("Roadway Speed Limit (MPH)", min_value=10, max_value=70, value=30, step=10)
            road_surface = st.selectbox("Road Surface Conditions", [
                "Dry",
                "Wet/Damp",
                "Frost/Ice",
                "Snow",
                "Flood (over 3cm deep)"
            ])
            
        submit_prediction = st.form_submit_button("Run Diagnostic Inference")
        
    if submit_prediction:
        # 1. Feature Map/Encoding matching the Kaggle preprocessing steps
        # mapping string values cleanly to numeric values matching your model encoders
        light_encoded = {"Daylight: Street light present": 1, "Darkness: Street lights present and lit": 2, "Darkness: No street lighting": 3, "Darkness: Street lights present but unlit": 4}.get(light_conditions, 1)
        weather_encoded = {"Fine without high winds": 1, "Raining without high winds": 2, "Snowing without high winds": 3, "Fine with high winds": 4, "Raining with high winds": 5, "Fog or mist": 6}.get(weather_conditions, 1)
        road_encoded = {"Dry": 1, "Wet/Damp": 2, "Frost/Ice": 3, "Snow": 4, "Flood (over 3cm deep)": 5}.get(road_surface, 1)
        
        raw_features = np.array([[num_vehicles, num_casualties, speed_limit, light_encoded, weather_encoded, road_encoded]])
        
        st.write("---")
        st.subheader("Diagnostic Results")
        
        # 2. Live Inference Engine Logic
        if ml_components["ready"]:
            try:
                # Standardize inputs using saved scaler configuration
                scaled_features = ml_components["scaler"].transform(raw_features)
                prediction_class = ml_components["model"].predict(scaled_features)[0]
                prediction_proba = ml_components["model"].predict_proba(scaled_features)[0]
                
                # Dynamic mapping according to UK classification outputs (1: Fatal/Severe, 2: Serious, 3: Slight)
                severity_mapping = {1: "Fatal/Critical", 2: "Serious", 3: "Slight / Minor Injury"}
                result_label = severity_mapping.get(prediction_class, f"Class {prediction_class}")
                
                st.metric(label="Predicted Accident Severity Profile", value=result_label)
                st.write(f"Confidence score of predictive categorization: `{max(prediction_proba)*100:.2f}%`")
                
            except Exception as inference_error:
                st.error(f"Inference pipeline execution error. Verify model dimension inputs: {inference_error}")
        else:
            # Simulation Fallback Matrix
            st.info("💡 Simulation Run: (Live model files not attached). Processing dummy matrix verification...")
            calculated_score = (num_vehicles * 0.2) + (num_casualties * 0.3) + (speed_limit * 0.05) + (road_encoded * 0.1)
            
            if calculated_score > 3.5:
                st.error("⚠️ Predicted Profile Target: Category 1 (Highly Severe / Fatal Likelihood)")
            elif calculated_score > 2.0:
                st.warning("⚠️ Predicted Profile Target: Category 2 (Serious Injury Risk Cluster)")
            else:
                st.success("✅ Predicted Profile Target: Category 3 (Slight/Minor Local Damage)")




