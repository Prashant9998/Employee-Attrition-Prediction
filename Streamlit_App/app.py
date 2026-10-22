import sys
import os

def is_streamlit_running():
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False

if not is_streamlit_running():
    # Export WSGI application for platforms like Vercel expecting a top-level `app` variable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from flask_app import app as _flask_app
        app = _flask_app
        application = _flask_app
    except Exception as e:
        from flask import Flask
        app = Flask(__name__)
        @app.route('/')
        def fallback():
            return "Employee Attrition Prediction Service is active."
        application = app
else:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import plotly.express as px
    import plotly.graph_objects as go
    import requests
    from streamlit_lottie import st_lottie

    # Set page configuration
    st.set_page_config(page_title="Employee Attrition Prediction", page_icon="🏢", layout="wide")

    # Load models and encoders
    @st.cache_resource
    def load_assets():
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'Model'),
            os.path.join(os.path.dirname(__file__), 'Model'),
            os.path.join(os.getcwd(), 'Model'),
            os.path.join(os.getcwd(), '..', 'Model')
        ]
        model_dir = candidates[0]
        for c in candidates:
            if os.path.exists(os.path.join(c, 'best_model.pkl')):
                model_dir = c
                break
        
        try:
            model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
            scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            encoders = joblib.load(os.path.join(model_dir, 'encoder.pkl'))
            return model, scaler, encoders
        except Exception as e:
            return None, None, None

    model, scaler, encoders = load_assets()

    # Function to get risk level
    def get_risk_level(probability):
        if probability < 0.3:
            return "Low Risk", "green"
        elif probability < 0.7:
            return "Medium Risk", "orange"
        else:
            return "High Risk", "red"

    def load_lottieurl(url: str):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Page", ["Prediction Dashboard", "Model Information", "About Project"])

    if app_mode == "Prediction Dashboard":
        col_t1, col_t2 = st.columns([1, 6])
        with col_t1:
            lottie_hr = load_lottieurl("https://lottie.host/80eeb733-1ec8-45be-bb33-68d7cd9a4059/zC2y2VlYyU.json")
            if lottie_hr:
                st_lottie(lottie_hr, height=100, key="hr_anim")
        with col_t2:
            st.title("🏢 Employee Attrition Prediction")
            
        st.markdown("Enter the employee's details below to predict the likelihood of attrition.")
        
        if model is None:
            st.error("Model assets not found. Please ensure you have run the Machine Learning notebook first to generate the models.")
        else:
            # User Input Form
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    age = st.number_input("Age", min_value=18, max_value=65, value=30)
                    business_travel = st.selectbox("Business Travel", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
                    daily_rate = st.number_input("Daily Rate", min_value=100, max_value=2000, value=800)
                    department = st.selectbox("Department", ['Research & Development', 'Sales', 'Human Resources'])
                    distance_from_home = st.number_input("Distance From Home (miles)", min_value=1, max_value=50, value=10)
                    education = st.selectbox("Education Level", [1, 2, 3, 4, 5], format_func=lambda x: {1:'Below College', 2:'College', 3:'Bachelor', 4:'Master', 5:'Doctor'}[x])
                    education_field = st.selectbox("Education Field", ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
                    environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
                    gender = st.selectbox("Gender", ['Male', 'Female'])
                    hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=60)
                    
                with col2:
                    job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
                    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
                    job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
                    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
                    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
                    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000)
                    monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=30000, value=15000)
                    num_companies_worked = st.number_input("Num Companies Worked", min_value=0, max_value=10, value=1)
                    over_time = st.selectbox("Over Time", ['Yes', 'No'])
                    percent_salary_hike = st.number_input("Percent Salary Hike (%)", min_value=10, max_value=25, value=15)
                    
                with col3:
                    performance_rating = st.selectbox("Performance Rating", [3, 4], format_func=lambda x: {3:'Excellent', 4:'Outstanding'}[x])
                    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
                    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
                    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
                    training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
                    work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
                    years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=3)
                    years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=2)
                    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
                    years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)

                submit_button = st.form_submit_button(label='Predict Attrition')

            if submit_button:
                # 1. Prepare Input DataFrame
                input_data = pd.DataFrame({
                    'Age': [age],
                    'BusinessTravel': [business_travel],
                    'DailyRate': [daily_rate],
                    'Department': [department],
                    'DistanceFromHome': [distance_from_home],
                    'Education': [education],
                    'EducationField': [education_field],
                    'EnvironmentSatisfaction': [environment_satisfaction],
                    'Gender': [gender],
                    'HourlyRate': [hourly_rate],
                    'JobInvolvement': [job_involvement],
                    'JobLevel': [job_level],
                    'JobRole': [job_role],
                    'JobSatisfaction': [job_satisfaction],
                    'MaritalStatus': [marital_status],
                    'MonthlyIncome': [monthly_income],
                    'MonthlyRate': [monthly_rate],
                    'NumCompaniesWorked': [num_companies_worked],
                    'OverTime': [over_time],
                    'PercentSalaryHike': [percent_salary_hike],
                    'PerformanceRating': [performance_rating],
                    'RelationshipSatisfaction': [relationship_satisfaction],
                    'StockOptionLevel': [stock_option_level],
                    'TotalWorkingYears': [total_working_years],
                    'TrainingTimesLastYear': [training_times_last_year],
                    'WorkLifeBalance': [work_life_balance],
                    'YearsAtCompany': [years_at_company],
                    'YearsInCurrentRole': [years_in_current_role],
                    'YearsSinceLastPromotion': [years_since_last_promotion],
                    'YearsWithCurrManager': [years_with_curr_manager]
                })

                # 2. Encode Categorical Data
                for col, le in encoders.items():
                    if col in input_data.columns:
                        input_data[col] = input_data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        input_data[col] = le.transform(input_data[col])
                
                # 3. Scale Features
                try:
                    input_scaled = scaler.transform(input_data)
                except Exception as e:
                    st.error(f"Error scaling features: {e}. Ensure all features are provided exactly as during training.")
                    input_scaled = scaler.transform(input_data)

                # 4. Predict
                pred = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
                
                # 5. Display Results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                risk_label, risk_color = get_risk_level(probability)
                
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("Prediction", "Likely to Leave" if pred == 1 else "Likely to Stay")
                with res_col2:
                    st.metric("Attrition Probability", f"{probability * 100:.2f}%")
                with res_col3:
                    st.markdown(f"**Risk Level:** <span style='color:{risk_color}; font-size:24px'><b>{risk_label}</b></span>", unsafe_allow_html=True)
                
                # Gauge Chart for Probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Attrition Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "salmon"}],
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Key Factors affecting prediction (using Feature Importances if available)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Key Factors Influencing This Prediction")
                    importances = model.feature_importances_
                    
                    feat_imp = pd.DataFrame({
                        'Feature': input_data.columns,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).head(5)
                    
                    fig2 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Top 5 Influential Features")
                    fig2.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig2, use_container_width=True)

    elif app_mode == "Model Information":
        st.title("🧠 Model Information & 3D Analytics")
        
        st.markdown("### 3D Dataset Exploration")
        st.markdown("Explore the clusters of employee attrition based on Age, Monthly Income, and Working Years in 3D space. **(Drag to rotate)**")
        
        try:
            candidates = [
                os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'WA_Fn-UseC_-HR-Employee-Attrition.csv'),
                os.path.join(os.getcwd(), 'Dataset', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
            ]
            dataset_path = candidates[0]
            for c in candidates:
                if os.path.exists(c):
                    dataset_path = c
                    break
            df = pd.read_csv(dataset_path)
            fig3d = px.scatter_3d(df, x='Age', y='MonthlyIncome', z='TotalWorkingYears',
                                  color='Attrition',
                                  color_discrete_map={'Yes': 'red', 'No': 'green'},
                                  opacity=0.7)
            fig3d.update_layout(margin=dict(l=0, r=0, b=0, t=10))
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.warning("3D Visualization currently unavailable. Ensure dataset exists in the Dataset folder.")

        st.markdown("""
        ### Machine Learning Pipeline
        This application uses a sophisticated Machine Learning pipeline trained on the IBM HR Analytics dataset.
        
        **Preprocessing Steps:**
        - Data Cleaning (Handling duplicates and irrelevant columns)
        - Categorical Encoding (Label Encoding)
        - Feature Scaling (StandardScaler)
        - Dataset Balancing (SMOTE - Synthetic Minority Over-sampling Technique)
        
        **Algorithm:**
        - The model running behind this application is dynamically selected as the **Best Model** based on the highest F1-Score during the training phase.
        - Potential candidates included Random Forest, XGBoost, Logistic Regression, SVM, etc.
        
        **Evaluation Metrics considered:**
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - ROC-AUC
        """)

    elif app_mode == "About Project":
        st.title("ℹ️ About the Project")
        st.markdown("""
        ### Employee Attrition Prediction System
        
        **Objective:** 
        To predict whether an employee is likely to leave the organization based on various demographic, job-related, and satisfaction metrics.
        
        **Why is this important?**
        Employee attrition costs companies significantly in terms of recruitment, training, and lost productivity. By identifying at-risk employees early, HR departments can take proactive retention measures, improve workplace satisfaction, and reduce turnover costs.
        
        **Features:**
        - End-to-End Machine Learning Lifecycle
        - Interactive Dashboard for real-time predictions
        - Detailed confidence and probability scoring
        - Identification of key attrition factors
        
        **Developed for:** B.Tech/MCA Final Year Project
        """)
