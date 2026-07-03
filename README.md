# Employee Attrition Prediction System

## Project Description
Employee attrition is a major concern for organizations as it affects productivity, morale, and incurs significant recruitment and training costs. This project is a complete End-to-End Machine Learning system designed to predict whether an employee will leave the organization (Attrition) based on various factors like job role, satisfaction level, salary, and more. 

The project encompasses the entire Machine Learning Lifecycle from Data Collection and Preprocessing, Exploratory Data Analysis (EDA), Feature Engineering, Model Building, Hyperparameter Tuning, to final Deployment using Streamlit.

## Features
- **Comprehensive EDA:** Univariate, Bivariate, and Multivariate analysis using high-quality Seaborn and Matplotlib visualizations.
- **Advanced Preprocessing:** Missing value handling, outlier detection, scaling, categorical encoding, and dataset balancing using SMOTE.
- **Feature Selection:** Automated selection of top features using Random Forest Feature Importance, Mutual Information, and SelectKBest.
- **Extensive Modeling:** Trains and compares 8 different Machine Learning algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - MLP Neural Network
- **Deployment:** A modern, interactive Streamlit web application providing real-time predictions, risk levels, and probability scores.
- **Detailed Documentation:** Complete academic-style project report (20-25 pages) and PowerPoint presentation content.

## Dataset
- **Name:** IBM HR Analytics Employee Attrition & Performance
- **Source:** Kaggle / Public Repositories
- **Target Variable:** `Attrition` (Yes/No)
- **Format:** CSV

## Technologies Used
- **Programming Language:** Python 3.11+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
- **Data Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit
- **Environment & Notebook:** Jupyter Notebook, nbformat

## Folder Structure
```text
Employee_Attrition_Prediction/
├── Dataset/
│   ├── download_dataset.py
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── Notebook/
│   ├── build_notebook.py
│   └── Employee_Attrition.ipynb
├── Model/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
├── Streamlit_App/
│   ├── app.py
│   └── requirements.txt
├── Documentation/
│   └── Project_Report.md
├── PPT/
│   └── Employee_Attrition_Presentation.md
├── Images/
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository or download the project files.
2. Navigate to the project directory:
   ```bash
   cd Employee_Attrition_Prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Download Dataset:**
   Navigate to the `Dataset/` folder and run the download script:
   ```bash
   python Dataset/download_dataset.py
   ```
2. **Build and Run Notebook:**
   Navigate to the `Notebook/` folder and run the builder script to generate the Jupyter Notebook:
   ```bash
   python Notebook/build_notebook.py
   ```
   Open `Employee_Attrition.ipynb` in Jupyter Notebook or VS Code to explore the ML lifecycle and train the models. Running the notebook will save the best model and scalers to the `Model/` directory.
3. **Run Streamlit App:**
   Once models are trained and saved, navigate to the `Streamlit_App/` folder and run:
   ```bash
   streamlit run app.py
   ```

## Sample Output
The Streamlit app will present a sidebar for navigation and a main form. Upon submitting an employee's details, the application outputs:
- **Prediction:** "Employee is Likely to Leave" or "Employee is Likely to Stay"
- **Confidence Score & Probability**
- **Risk Level:** Low, Medium, or High
- **Key Factors Visualization:** A plot indicating which features most influenced the decision.

## Future Scope
- Integration with external HR systems via REST APIs.
- Incorporating deep learning architectures (e.g., LSTMs) for time-series attrition tracking if historical timeline data becomes available.
- Implementing an automated retraining pipeline when new employee data is ingested.
- Generating automated custom reports for HR managers based on attrition trends across specific departments.
