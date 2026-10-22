import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string

def get_model_dir():
    candidates = [
        os.path.join(os.path.dirname(__file__), 'Model'),
        os.path.join(os.path.dirname(__file__), '..', 'Model'),
        os.path.join(os.getcwd(), 'Model'),
        os.path.join(os.getcwd(), '..', 'Model'),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, 'best_model.pkl')):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])

def load_ml_assets():
    model_dir = get_model_dir()
    try:
        model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        encoders = joblib.load(os.path.join(model_dir, 'encoder.pkl'))
        return model, scaler, encoders
    except Exception as e:
        print(f"Warning: could not load model assets from {model_dir}: {e}")
        return None, None, None

model, scaler, encoders = load_ml_assets()

def calculate_prediction(data):
    if model is None or scaler is None or encoders is None:
        raise ValueError("Model assets not loaded")

    field_specs = [
        ('Age', 30, int),
        ('BusinessTravel', 'Travel_Rarely', str),
        ('DailyRate', 800, int),
        ('Department', 'Research & Development', str),
        ('DistanceFromHome', 10, int),
        ('Education', 3, int),
        ('EducationField', 'Life Sciences', str),
        ('EnvironmentSatisfaction', 3, int),
        ('Gender', 'Male', str),
        ('HourlyRate', 60, int),
        ('JobInvolvement', 3, int),
        ('JobLevel', 2, int),
        ('JobRole', 'Research Scientist', str),
        ('JobSatisfaction', 3, int),
        ('MaritalStatus', 'Single', str),
        ('MonthlyIncome', 5000, int),
        ('MonthlyRate', 15000, int),
        ('NumCompaniesWorked', 1, int),
        ('OverTime', 'No', str),
        ('PercentSalaryHike', 15, int),
        ('PerformanceRating', 3, int),
        ('RelationshipSatisfaction', 3, int),
        ('StockOptionLevel', 0, int),
        ('TotalWorkingYears', 5, int),
        ('TrainingTimesLastYear', 2, int),
        ('WorkLifeBalance', 3, int),
        ('YearsAtCompany', 3, int),
        ('YearsInCurrentRole', 2, int),
        ('YearsSinceLastPromotion', 1, int),
        ('YearsWithCurrManager', 2, int)
    ]

    clean_data = {}
    for col, default_val, cast_type in field_specs:
        raw_val = data.get(col, default_val)
        try:
            clean_data[col] = [cast_type(raw_val)]
        except (ValueError, TypeError):
            clean_data[col] = [default_val]

    input_df = pd.DataFrame(clean_data)

    # Encode categoricals
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = int(model.predict(input_scaled)[0])
    prob = float(model.predict_proba(input_scaled)[0][1]) if hasattr(model, 'predict_proba') else float(pred)

    if prob < 0.3:
        risk_level = "Low Risk"
        risk_color = "#10B981"
        badge_class = "bg-green-100 text-green-800 border-green-300"
    elif prob < 0.7:
        risk_level = "Medium Risk"
        risk_color = "#F59E0B"
        badge_class = "bg-amber-100 text-amber-800 border-amber-300"
    else:
        risk_level = "High Risk"
        risk_color = "#EF4444"
        badge_class = "bg-red-100 text-red-800 border-red-300"

    prediction_text = "Likely to Leave" if pred == 1 else "Likely to Stay"

    # Top influential factors
    top_factors = []
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        factor_pairs = sorted(zip(clean_data.keys(), importances), key=lambda x: x[1], reverse=True)[:5]
        top_factors = [{'feature': f, 'importance': round(float(imp) * 100, 2)} for f, imp in factor_pairs]
    else:
        # For SVM without feature_importances, provide domain-weighted contributing risk factors
        key_signals = [
            {'feature': 'OverTime', 'value': data.get('OverTime', 'No')},
            {'feature': 'MonthlyIncome', 'value': f"${data.get('MonthlyIncome', 5000):,}"},
            {'feature': 'JobSatisfaction', 'value': f"{data.get('JobSatisfaction', 3)}/4"},
            {'feature': 'WorkLifeBalance', 'value': f"{data.get('WorkLifeBalance', 3)}/4"},
            {'feature': 'YearsAtCompany', 'value': f"{data.get('YearsAtCompany', 3)} yrs"}
        ]
        top_factors = key_signals

    return {
        'prediction': pred,
        'prediction_text': prediction_text,
        'probability': round(prob, 4),
        'probability_percent': round(prob * 100, 2),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'badge_class': badge_class,
        'top_factors': top_factors
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Employee Attrition Prediction System</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @keyframes pulse-subtle { 0%, 100% { opacity: 1; } 50% { opacity: 0.85; } }
    .pulse-meter { animation: pulse-subtle 2s infinite ease-in-out; }
  </style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen">
  <!-- Navigation Header -->
  <header class="bg-indigo-900 text-white shadow-md border-b border-indigo-800">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5 flex flex-col md:flex-row justify-between items-center gap-4">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-lg bg-indigo-500/30 flex items-center justify-center text-2xl">🏢</div>
        <div>
          <h1 class="text-xl font-bold tracking-tight">Employee Attrition Prediction</h1>
          <p class="text-xs text-indigo-200">AI-Powered HR Retention Analytics & Machine Learning Pipeline</p>
        </div>
      </div>
      <div class="flex items-center space-x-3 text-xs">
        <span class="inline-flex items-center px-2.5 py-1 rounded-full bg-emerald-900/60 text-emerald-300 border border-emerald-500/40">
          <span class="w-2 h-2 mr-1.5 rounded-full bg-emerald-400 animate-ping"></span> Model Online
        </span>
        <a href="#api-section" class="px-3 py-1.5 rounded-md bg-indigo-800 hover:bg-indigo-700 text-indigo-100 transition">REST API Docs</a>
      </div>
    </div>
  </header>

  <!-- Main Content Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- Left Column: Prediction Form (7 cols) -->
      <div class="lg:col-span-7 space-y-6">
        <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 sm:p-8">
          <div class="flex items-center justify-between pb-4 border-b border-slate-100 mb-6">
            <div>
              <h2 class="text-lg font-bold text-slate-800 flex items-center gap-2">
                <i class="fa-solid fa-user-pen text-indigo-600"></i> Employee Information Form
              </h2>
              <p class="text-sm text-slate-500">Provide employee attributes to evaluate retention risk probability.</p>
            </div>
            <button type="button" id="fillSampleBtn" class="text-xs font-medium text-indigo-600 hover:text-indigo-800 hover:underline">
              Load Sample Data
            </button>
          </div>

          <form id="predictionForm" class="space-y-6">
            <!-- Section 1: Demographics -->
            <div>
              <h3 class="text-xs font-bold uppercase tracking-wider text-indigo-600 mb-3 flex items-center gap-2">
                <i class="fa-solid fa-address-card"></i> 1. Demographics & Background
              </h3>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Age</label>
                  <input type="number" name="Age" id="Age" value="30" min="18" max="65" required class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Gender</label>
                  <select name="Gender" id="Gender" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="Male" selected>Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Marital Status</label>
                  <select name="MaritalStatus" id="MaritalStatus" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="Single" selected>Single</option>
                    <option value="Married">Married</option>
                    <option value="Divorced">Divorced</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Distance From Home (miles)</label>
                  <input type="number" name="DistanceFromHome" id="DistanceFromHome" value="10" min="1" max="50" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Education Level</label>
                  <select name="Education" id="Education" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="1">1 - Below College</option>
                    <option value="2">2 - College</option>
                    <option value="3" selected>3 - Bachelor</option>
                    <option value="4">4 - Master</option>
                    <option value="5">5 - Doctor</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Education Field</label>
                  <select name="EducationField" id="EducationField" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="Life Sciences" selected>Life Sciences</option>
                    <option value="Medical">Medical</option>
                    <option value="Marketing">Marketing</option>
                    <option value="Technical Degree">Technical Degree</option>
                    <option value="Human Resources">Human Resources</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- Section 2: Job Role & Compensation -->
            <div class="pt-4 border-t border-slate-100">
              <h3 class="text-xs font-bold uppercase tracking-wider text-indigo-600 mb-3 flex items-center gap-2">
                <i class="fa-solid fa-briefcase"></i> 2. Role & Compensation
              </h3>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Department</label>
                  <select name="Department" id="Department" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="Research & Development" selected>Research & Development</option>
                    <option value="Sales">Sales</option>
                    <option value="Human Resources">Human Resources</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Job Role</label>
                  <select name="JobRole" id="JobRole" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="Sales Executive">Sales Executive</option>
                    <option value="Research Scientist" selected>Research Scientist</option>
                    <option value="Laboratory Technician">Laboratory Technician</option>
                    <option value="Manufacturing Director">Manufacturing Director</option>
                    <option value="Healthcare Representative">Healthcare Representative</option>
                    <option value="Manager">Manager</option>
                    <option value="Sales Representative">Sales Representative</option>
                    <option value="Research Director">Research Director</option>
                    <option value="Human Resources">Human Resources</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Job Level (1-5)</label>
                  <select name="JobLevel" id="JobLevel" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="1">1 - Entry</option>
                    <option value="2" selected>2 - Intermediate</option>
                    <option value="3">3 - Mid</option>
                    <option value="4">4 - Senior</option>
                    <option value="5">5 - Executive</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Monthly Income ($)</label>
                  <input type="number" name="MonthlyIncome" id="MonthlyIncome" value="5000" min="1000" max="25000" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Over Time</label>
                  <select name="OverTime" id="OverTime" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="No" selected>No</option>
                    <option value="Yes">Yes</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Business Travel</label>
                  <select name="BusinessTravel" id="BusinessTravel" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="Travel_Rarely" selected>Travel Rarely</option>
                    <option value="Travel_Frequently">Travel Frequently</option>
                    <option value="Non-Travel">Non-Travel</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Daily Rate ($)</label>
                  <input type="number" name="DailyRate" id="DailyRate" value="800" min="100" max="2000" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Hourly Rate ($)</label>
                  <input type="number" name="HourlyRate" id="HourlyRate" value="60" min="30" max="100" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Monthly Rate ($)</label>
                  <input type="number" name="MonthlyRate" id="MonthlyRate" value="15000" min="2000" max="30000" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Stock Option Level (0-3)</label>
                  <select name="StockOptionLevel" id="StockOptionLevel" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="0" selected>0 - None</option>
                    <option value="1">1 - Basic</option>
                    <option value="2">2 - High</option>
                    <option value="3">3 - Executive</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Percent Salary Hike (%)</label>
                  <input type="number" name="PercentSalaryHike" id="PercentSalaryHike" value="15" min="10" max="25" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Performance Rating</label>
                  <select name="PerformanceRating" id="PerformanceRating" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="3" selected>3 - Excellent</option>
                    <option value="4">4 - Outstanding</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- Section 3: Satisfaction & Tenures -->
            <div class="pt-4 border-t border-slate-100">
              <h3 class="text-xs font-bold uppercase tracking-wider text-indigo-600 mb-3 flex items-center gap-2">
                <i class="fa-solid fa-chart-line"></i> 3. Satisfaction & Experience
              </h3>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Job Satisfaction (1-4)</label>
                  <select name="JobSatisfaction" id="JobSatisfaction" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="1">1 - Low</option>
                    <option value="2">2 - Medium</option>
                    <option value="3" selected>3 - High</option>
                    <option value="4">4 - Very High</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Environment Satisfaction (1-4)</label>
                  <select name="EnvironmentSatisfaction" id="EnvironmentSatisfaction" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="1">1 - Low</option>
                    <option value="2">2 - Medium</option>
                    <option value="3" selected>3 - High</option>
                    <option value="4">4 - Very High</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Relationship Satisfaction (1-4)</label>
                  <select name="RelationshipSatisfaction" id="RelationshipSatisfaction" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="1">1 - Low</option>
                    <option value="2">2 - Medium</option>
                    <option value="3" selected>3 - High</option>
                    <option value="4">4 - Very High</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Job Involvement (1-4)</label>
                  <select name="JobInvolvement" id="JobInvolvement" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="1">1 - Low</option>
                    <option value="2">2 - Medium</option>
                    <option value="3" selected>3 - High</option>
                    <option value="4">4 - Very High</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Work Life Balance (1-4)</label>
                  <select name="WorkLifeBalance" id="WorkLifeBalance" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="1">1 - Bad</option>
                    <option value="2">2 - Good</option>
                    <option value="3" selected>3 - Better</option>
                    <option value="4">4 - Best</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Total Working Years</label>
                  <input type="number" name="TotalWorkingYears" id="TotalWorkingYears" value="5" min="0" max="40" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Years At Company</label>
                  <input type="number" name="YearsAtCompany" id="YearsAtCompany" value="3" min="0" max="40" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Years In Current Role</label>
                  <input type="number" name="YearsInCurrentRole" id="YearsInCurrentRole" value="2" min="0" max="20" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Years Since Last Promotion</label>
                  <input type="number" name="YearsSinceLastPromotion" id="YearsSinceLastPromotion" value="1" min="0" max="15" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Years With Current Manager</label>
                  <input type="number" name="YearsWithCurrManager" id="YearsWithCurrManager" value="2" min="0" max="20" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Num Companies Worked</label>
                  <input type="number" name="NumCompaniesWorked" id="NumCompaniesWorked" value="1" min="0" max="10" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-700 mb-1">Training Times Last Year</label>
                  <input type="number" name="TrainingTimesLastYear" id="TrainingTimesLastYear" value="2" min="0" max="6" class="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
              </div>
            </div>

            <!-- Submit Button -->
            <button type="submit" id="submitBtn" class="w-full py-3.5 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-lg shadow-indigo-200 transition duration-150 flex items-center justify-center gap-2">
              <i class="fa-solid fa-wand-magic-sparkles"></i> Predict Attrition Risk
            </button>
          </form>
        </div>
      </div>

      <!-- Right Column: Prediction Results Dashboard (5 cols) -->
      <div class="lg:col-span-5 space-y-6">
        
        <!-- Live Prediction Card -->
        <div id="resultCard" class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 sm:p-8 space-y-6">
          <div class="flex items-center justify-between pb-4 border-b border-slate-100">
            <h3 class="text-base font-bold text-slate-800 flex items-center gap-2">
              <i class="fa-solid fa-chart-pie text-indigo-600"></i> Prediction Outcome
            </h3>
            <span id="riskBadge" class="text-xs font-bold px-3 py-1 rounded-full border bg-green-100 text-green-800 border-green-300">
              Low Risk
            </span>
          </div>

          <!-- Main Status Hero -->
          <div class="text-center py-4 bg-slate-50 rounded-xl border border-slate-100">
            <p class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Estimated Status</p>
            <div id="predOutcome" class="text-2xl font-extrabold text-emerald-600">
              Likely to Stay
            </div>
            <p id="predSubtext" class="text-xs text-slate-500 mt-1">Employee demonstrates high retention probability.</p>
          </div>

          <!-- Probability Meter -->
          <div>
            <div class="flex justify-between items-baseline mb-2">
              <span class="text-xs font-semibold text-slate-600">Attrition Probability</span>
              <span id="probText" class="text-xl font-bold text-slate-800">5.55%</span>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-3.5 overflow-hidden p-0.5 border border-slate-200">
              <div id="probBar" class="h-full rounded-full transition-all duration-700 bg-emerald-500" style="width: 5.55%"></div>
            </div>
            <div class="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>0% (Safe)</span>
              <span>30%</span>
              <span>70%</span>
              <span>100% (High Risk)</span>
            </div>
          </div>

          <!-- Key Influential Indicators -->
          <div class="pt-4 border-t border-slate-100">
            <h4 class="text-xs font-bold uppercase tracking-wider text-slate-600 mb-3">Key Influential Factors</h4>
            <div id="factorsList" class="space-y-2 text-xs">
              <div class="flex justify-between items-center p-2 rounded-lg bg-slate-50 border border-slate-100">
                <span class="font-medium text-slate-600">Over Time</span>
                <span class="font-bold text-slate-800">No</span>
              </div>
              <div class="flex justify-between items-center p-2 rounded-lg bg-slate-50 border border-slate-100">
                <span class="font-medium text-slate-600">Monthly Income</span>
                <span class="font-bold text-slate-800">$5,000</span>
              </div>
              <div class="flex justify-between items-center p-2 rounded-lg bg-slate-50 border border-slate-100">
                <span class="font-medium text-slate-600">Job Satisfaction</span>
                <span class="font-bold text-slate-800">3/4</span>
              </div>
            </div>
          </div>
        </div>

        <!-- System Pipeline & Model Info -->
        <div class="bg-indigo-950 text-white rounded-2xl shadow-sm p-6 space-y-4">
          <h3 class="text-sm font-bold flex items-center gap-2 text-indigo-300">
            <i class="fa-solid fa-microchip"></i> Model Architecture
          </h3>
          <p class="text-xs text-indigo-200 leading-relaxed">
            Trained on the IBM HR Analytics dataset using 8 machine learning algorithms. The production model is an optimized Support Vector Classifier with SMOTE dataset balancing and StandardScaler normalization.
          </p>
          <div class="grid grid-cols-2 gap-2 pt-2 border-t border-indigo-800 text-[11px]">
            <div>
              <span class="text-indigo-400 block">Classifier:</span>
              <span class="font-semibold text-white">SVM / SVC</span>
            </div>
            <div>
              <span class="text-indigo-400 block">Preprocessing:</span>
              <span class="font-semibold text-white">SMOTE + Scaler</span>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- API Reference Section -->
    <section id="api-section" class="mt-12 bg-white rounded-2xl shadow-sm border border-slate-200 p-6 sm:p-8">
      <h2 class="text-lg font-bold text-slate-800 mb-2 flex items-center gap-2">
        <i class="fa-solid fa-code text-indigo-600"></i> REST API Documentation
      </h2>
      <p class="text-sm text-slate-500 mb-4">
        You can make programmatic predictions by sending a POST request to <code class="bg-slate-100 text-indigo-600 px-2 py-0.5 rounded text-xs">/api/predict</code>.
      </p>

      <div class="bg-slate-900 text-slate-200 rounded-xl p-4 text-xs font-mono overflow-x-auto">
        <p class="text-slate-400 mb-2"># Sample cURL request:</p>
        <p>curl -X POST /api/predict \\</p>
        <p class="pl-4">-H "Content-Type: application/json" \\</p>
        <p class="pl-4">-d '{"Age": 30, "MonthlyIncome": 5000, "OverTime": "No", "JobRole": "Research Scientist"}'</p>
      </div>
    </section>
  </main>

  <footer class="text-center py-6 text-xs text-slate-400 border-t border-slate-200 mt-12">
    Employee Attrition Prediction System &bull; Machine Learning Project &bull; Compatible with Vercel & Streamlit
  </footer>

  <script>
    const form = document.getElementById('predictionForm');
    const fillSampleBtn = document.getElementById('fillSampleBtn');
    const predOutcome = document.getElementById('predOutcome');
    const predSubtext = document.getElementById('predSubtext');
    const probText = document.getElementById('probText');
    const probBar = document.getElementById('probBar');
    const riskBadge = document.getElementById('riskBadge');
    const factorsList = document.getElementById('factorsList');

    fillSampleBtn.addEventListener('click', () => {
      document.getElementById('Age').value = 42;
      document.getElementById('OverTime').value = 'Yes';
      document.getElementById('MonthlyIncome').value = 2400;
      document.getElementById('JobSatisfaction').value = 1;
      document.getElementById('EnvironmentSatisfaction').value = 1;
      document.getElementById('WorkLifeBalance').value = 1;
      document.getElementById('DistanceFromHome').value = 28;
      document.getElementById('MaritalStatus').value = 'Single';
      form.requestSubmit();
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(form);
      const data = Object.fromEntries(formData.entries());

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
        const result = await res.json();
        if (result.success) {
          const p = result.data;
          predOutcome.textContent = p.prediction_text;
          predOutcome.className = p.prediction === 1 ? 'text-2xl font-extrabold text-red-600' : 'text-2xl font-extrabold text-emerald-600';
          predSubtext.textContent = p.prediction === 1 ? 'High risk indicators detected. Proactive retention recommended.' : 'Employee demonstrates steady retention indicators.';
          
          probText.textContent = p.probability_percent + '%';
          probBar.style.width = Math.min(Math.max(p.probability_percent, 2), 100) + '%';
          probBar.style.backgroundColor = p.risk_color;

          riskBadge.textContent = p.risk_level;
          riskBadge.className = 'text-xs font-bold px-3 py-1 rounded-full border ' + p.badge_class;

          if (p.top_factors && p.top_factors.length > 0) {
            factorsList.innerHTML = p.top_factors.map(f => `
              <div class="flex justify-between items-center p-2 rounded-lg bg-slate-50 border border-slate-100">
                <span class="font-medium text-slate-600">${f.feature}</span>
                <span class="font-bold text-slate-800">${f.value ?? (f.importance + '%')}</span>
              </div>
            `).join('');
          }
        } else {
          alert('Prediction error: ' + (result.error || 'Unknown error'));
        }
      } catch (err) {
        console.error(err);
        alert('Failed to connect to prediction server.');
      }
    });
  </script>
</body>
</html>
"""

def create_app():
    flask_app = Flask(__name__)

    @flask_app.route('/', methods=['GET'])
    def home():
        return render_template_string(HTML_TEMPLATE)

    @flask_app.route('/api/predict', methods=['POST'])
    def predict_endpoint():
        try:
            req_data = request.get_json(silent=True) or request.form.to_dict() or {}
            result = calculate_prediction(req_data)
            return jsonify({'success': True, 'data': result})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

    @flask_app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'encoders_loaded': encoders is not None
        })

    return flask_app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
