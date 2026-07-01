Project Name: UK Road Safety Analysis & Prediction
Repository: https://github.com/Jerome-debug/UK-Accidents
Author: Jerome-debug (Data Analyst)

--- Core Components ---
1. Dataset:
   - Source: UK Government (1.8 million+ records).
   - Temporal Scope: 2000 - 2018.
   - Size: 1.5M+ analyzed records.
   - Features: 33 attributes (Weather, Lighting, Road Class, Number of Vehicles, etc.).
   - Target: Accident Severity (Predicting the impact level of a collision).


2. Streamlit Application Features:
   - Navigation: Sidebar with two modes.
   - Mode 1: Project Overview & EDA
     - KPI Metrics (Total records, Target class, Temporal scope, Feature count).
     - Visualizations: Casualties by Day of Week (Bar chart), Environmental Hazard Mappings (Table).
     - Insights: Peak incidents on Fridays; most severe accidents happen in "Fine" weather (driver behavior factor).
   - Mode 2: Accident Severity Predictor
     - Real-time inference pipeline.
     - User inputs: Number of vehicles, lighting conditions, casualties, weather conditions, etc.
     - Model status check (production vs. simulation mode).


3. Technical Workflow (from Notebook):
   - Data Cleaning: Dropping identifiers and granular geographical data.
   - Feature Engineering: One-Hot Encoding for categorical variables (Day of Week, Road Class, etc.).
   - Preprocessing: Standard Scaling for numerical stability.
   - Modeling: Stratified sampling to handle class imbalance.
   - Deployment: Pipeline exported as .pkl for low-latency web inference.
