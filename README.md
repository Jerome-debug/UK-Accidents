Project Name: UK Road Safety Analysis & Prediction
Repository: https://github.com/Jerome-debug/UK-Accidents
Author: Jerome-debug (Data Analyst)



--- Technical Setup Instructions ---
- Environment: Python 3.x
- IDE: VS Code
- Setup Steps:
  1. Create a virtual environment: `python -m venv .venv`
  2. Activate virtual environment: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
  3. Install dependencies: `pip install -r requirements.txt` (or install manually: streamlit, pandas, numpy, scikit-learn, xgboost, imbalanced-learn)
  4. Connect Jupyter Kernel in VS Code: Select the created `.venv` as the interpreter/kernel for the .ipynb file and run your notebook.
  5. Run Streamlit: `streamlit run app.py
--- Core Components ---
1. Dataset:
   - Source: UK Government (1.8 million+ records).
   - Temporal Scope: 2000 - 2018.
   - Size: 1.5M+ analyzed records.
   - Features: 33 attributes (Weather, Lighting, Road Class, Number of Vehicles, etc.).
   - Target: Accident Severity (Predicting the impact level of a collision).



3. Technical Workflow (from Notebook):
   - Data Cleaning: Dropping identifiers and granular geographical data.
   - Feature Engineering: One-Hot Encoding for categorical variables (Day of Week, Road Class, etc.).
   - Preprocessing: Standard Scaling for numerical stability.
   - Modeling: Stratified sampling to handle class imbalance.
   - Deployment: Pipeline exported as .pkl for low-latency web inference.
  
4. --- Technical ML Details ---
1. Preprocessing:
   - Dropping Irrelevant Columns: Accident_Index, Location coordinates, granular LSOA, etc.
   - Handling Missing Values: Row-wise deletion after feature selection.
   - Categorical Encoding: One-Hot Encoding for Day_of_Week, Police_Force, Road Classes, Urban/Rural Area.
   - Feature Scaling: StandardScaler (with_mean=False for sparse compatibility).
   - Class Imbalance Handling: SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous) to balance Accident_Severity classes.

2. Models & Evaluation:
   - Models Used: XGBClassifier (XGBoost) - chosen for its high performance on structured data and ability to handle complex non-linear relationships.
   - Target Variable: Accident_Severity (3 classes, converted to 0-indexed for XGBoost).
   - Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
   - Validation: Stratified Train-Test Split (80/20) to ensure class proportions are maintained.
  

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


3. Findings:
   - Temporal Trends: Fridays and middle weekdays see peak incident frequency.
   - Environmental Factors: Counter-intuitively, most severe accidents occur in "Fine" weather conditions, suggesting human factors like speed or complacency are primary drivers.
   - Feature Importance: Factors like Lighting Conditions, Number of Vehicles, and Road Class are significant predictors of severity.

4. Recommendations:
   - Targeted Enforcement: Increase police presence on Fridays and during peak weekday hours.
   - Awareness Campaigns: Focus on driver complacency during clear weather conditions.
   - Infrastructure: Improve lighting in areas identified as high-risk for severe nighttime accidents.
   - Real-time Monitoring: Integrate the predictive model with emergency dispatch systems for proactive resource positioning.

