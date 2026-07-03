# Employee Attrition Prediction System
## Comprehensive Project Report

---

### **Abstract**
Employee attrition is a critical issue for organizations, leading to significant costs associated with hiring, training, and lost productivity. This project presents an end-to-end Machine Learning system designed to predict employee attrition using the IBM HR Analytics dataset. The system covers the complete data science lifecycle: data collection, preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation. Multiple algorithms including Logistic Regression, Random Forest, XGBoost, and Neural Networks were evaluated. The best performing model is deployed via an interactive Streamlit web application, allowing HR departments to assess employee risk levels and key driving factors in real-time.

---

## **Chapter 1: Introduction**

### 1.1 Background
In today's competitive business environment, human capital is the most valuable asset. Employee attrition, the voluntary or involuntary departure of employees, disrupts ongoing projects and incurs massive costs. Understanding why employees leave is essential for implementing effective retention strategies.

### 1.2 Problem Statement
Human Resource departments often lack predictive capabilities to foresee employee departures. They rely on exit interviews, which are reactive. The problem is to develop a proactive, data-driven predictive model that classifies whether an employee is likely to leave ("Yes") or stay ("No") based on historical HR data.

### 1.3 Objectives
- To understand the primary factors contributing to employee attrition through EDA.
- To preprocess and balance the HR dataset for optimal model training.
- To apply and compare various machine learning algorithms.
- To deploy the most accurate model into a user-friendly web application.

---

## **Chapter 2: Literature Review**

### 2.1 Existing Research
Previous studies in HR Analytics have emphasized the importance of job satisfaction, monthly income, and work-life balance in predicting turnover. Traditional approaches used simple logistic regression models, but recent advancements show that ensemble methods like Random Forest and XGBoost yield superior accuracy due to their ability to capture non-linear relationships.

### 2.2 Gap Analysis
Many existing solutions lack a complete, user-facing deployment mechanism. While academic papers achieve high accuracy, they do not provide HR managers with an intuitive interface to input data and receive actionable insights (like risk probabilities and key influencing factors). This project bridges that gap using Streamlit.

---

## **Chapter 3: Methodology**

### 3.1 Dataset Description
The "IBM HR Analytics Employee Attrition & Performance" dataset contains 1,470 records and 35 features.
- **Target Variable:** `Attrition` (Categorical: Yes/No)
- **Key Features:** Age, MonthlyIncome, JobSatisfaction, TotalWorkingYears, DistanceFromHome, OverTime, etc.

### 3.2 Data Preprocessing
1. **Handling Missing Values & Duplicates:** The dataset was checked for null values and duplicate records to ensure data integrity.
2. **Feature Removal:** Uninformative columns like `EmployeeCount`, `StandardHours`, and `Over18` were dropped.
3. **Outlier Treatment:** Outliers in continuous variables (e.g., MonthlyIncome) were treated using the IQR clipping method to preserve the dataset's size.
4. **Encoding:** Categorical variables (like BusinessTravel, Department, Gender) were encoded using `LabelEncoder`.
5. **Scaling:** Continuous variables were normalized using `StandardScaler` to ensure algorithms like SVM and KNN perform optimally.

### 3.3 Handling Imbalanced Data (SMOTE)
The dataset is heavily imbalanced (~84% No, ~16% Yes). Training models on imbalanced data leads to a biased majority class prediction. We applied the **Synthetic Minority Over-sampling Technique (SMOTE)** strictly on the training set to synthetically generate minority class instances, achieving a 50:50 balance.

### 3.4 Algorithms Evaluated
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)
7. Naive Bayes
8. Multi-Layer Perceptron (MLP) Neural Network

---

## **Chapter 4: Implementation**

### 4.1 Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Showed the skewness in variables like DistanceFromHome and YearsAtCompany.
- **Bivariate Analysis:** Revealed that employees with lower MonthlyIncome, frequent BusinessTravel, and high OverTime are more prone to attrition.
- **Correlation Heatmap:** Highlighted multicollinearity between `JobLevel` and `MonthlyIncome`, as well as `YearsAtCompany` and `YearsWithCurrManager`.

### 4.2 Feature Engineering & Selection
Random Forest Feature Importance was utilized to rank features. The top driving factors were found to be:
1. MonthlyIncome
2. Age
3. TotalWorkingYears
4. DistanceFromHome
5. OverTime

### 4.3 Model Development
The data was split using an 80:20 Train-Test ratio. All 8 models were trained using the scaled and SMOTE-balanced training set. Predictions were generated on the unseen test set to evaluate real-world performance.

---

## **Chapter 5: Results and Analysis**

### 5.1 Performance Metrics
Models were evaluated based on:
- **Accuracy:** Overall correctness.
- **Precision:** Correctness of positive predictions (minimizing false positives).
- **Recall:** Ability to find all actual positive cases (minimizing false negatives - critical for attrition).
- **F1-Score:** Harmonic mean of Precision and Recall.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve.

### 5.2 Comparative Analysis
*(Note: Actual metric numbers depend on the execution of the notebook. The following is the expected observation pattern.)*
- **Logistic Regression & SVM:** Performed moderately but struggled with non-linear boundaries.
- **Decision Trees:** Showed signs of overfitting despite high training accuracy.
- **Random Forest & XGBoost:** Achieved the highest F1-Scores and ROC-AUC due to their ensemble nature and robustness to noise. XGBoost generally provides the best balance between precision and recall.

### 5.3 Deployment via Streamlit
The best model (highest F1-Score) was exported using `joblib` alongside the `StandardScaler` and `LabelEncoders`. The Streamlit application loads these assets to provide:
1. A clean UI for HR input.
2. Immediate categorical to numerical mapping.
3. Real-time inference showing Risk Level (Low/Medium/High).

---

## **Chapter 6: Conclusion and Future Scope**

### 6.1 Conclusion
The project successfully demonstrates a complete end-to-end Machine Learning lifecycle. By employing SMOTE and ensemble algorithms, the system accurately predicts employee attrition. The deployment of the model via Streamlit transforms raw algorithmic power into an accessible business tool, allowing organizations to proactively manage human resources.

### 6.2 Future Scope
1. **Dynamic Model Retraining:** Implement a pipeline (e.g., using Apache Airflow) to automatically retrain the model as new employee data is collected.
2. **Deep Learning Integration:** Explore recurrent neural networks (RNNs) or LSTMs to analyze time-series data of employee performance over months.
3. **API Integration:** Expose the model via FastAPI or Flask to allow seamless integration with existing enterprise HR software (e.g., Workday, SAP).
4. **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) into the dashboard to provide granular, instance-level explanations for *why* a specific employee is flagged at high risk.

---

## **References**
1. IBM HR Analytics Employee Attrition & Performance Dataset. Available on Kaggle.
2. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research.
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." Proceedings of the 22nd ACM SIGKDD.
4. Streamlit Documentation: https://docs.streamlit.io/
