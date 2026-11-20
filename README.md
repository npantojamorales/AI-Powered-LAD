# 🎓 An Integrated AI-Powered Learning Analytics Dashboard for Student Success and Early Intervention

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-powered-lad-bnjyyq6law7eyvcbtp8hdb.streamlit.app/)

This repository contains the machine learning pipeline and interactive Streamlit dashboard designed to proactively identify at-risk students (withdrawal or failure) using the Open University Learning Analytics Dataset (OULAD). By integrating academic performance scores with behavioral engagement metrics (VLE clicks), the system provides educators with actionable early intervention recommendations and Explainable AI (SHAP) insights.

## 🚀 Live Access (No Installation Required)
Educators can access the live production dashboard directly in any web browser via Streamlit Cloud:  
👉 **[Launch Educator Learning Analytics Dashboard](https://ai-powered-lad-bnjyyq6law7eyvcbtp8hdb.streamlit.app/)**

## ✨ Features
- **Multi-Source Data Integration:** Merges academic performance and behavioral VLE clickstream data from the OULAD dataset.
- **Predictive AI Engine:** A Random Forest classifier (81.21% accuracy, 0.8998 AUC-ROC) trained to classify students as `On Track` or `At-Risk`.
- **Explainable AI (SHAP):** Uses SHAP's `TreeExplainer` to visualize global feature importance so non-technical educators understand which factors drive risk classifications.
- **Educator Dashboard:** An interactive three-tab Streamlit interface (**Individual Student Profile**, **Class Overview**, and **AI Transparency**) with automated, rule-based intervention recommendations.

## 🛠️ Prerequisites
- Python 3.13+ (a virtual environment is recommended)
- Install all required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure
- `streamlit_app.py`: Core web application running the educator dashboard.
- `training_model.py`: Preprocesses raw OULAD data, handles imputation and encoding, trains the Random Forest classifier, generates the SHAP plot, and exports model artifacts.
- `eda_script.py`: Computes missing value statistics across OULAD tables and generates the correlation matrix heatmap.
- `compare_models.py`: Executes post-hoc algorithm benchmarking (Random Forest vs. XGBoost and SVM).
- `dashboard_student_data.csv`: Preprocessed lightweight student dataset consumed directly by the dashboard.
- `risk_prediction_model.joblib`: Trained Scikit-learn `RandomForestClassifier` artifact.
- `model_features.joblib`: Encoded feature names and ordering for inference alignment.
- `shap_summary_plot.png`: Precomputed SHAP feature importance plot.
- `requirements.txt`: Python package dependencies for local and cloud deployment.

## 🏃‍♂️ How to Run Locally

### Quick Start (Out of the Box)
Because of GitHub's 100 MB file size limit, the raw OULAD files are excluded from this repository. The dashboard runs immediately using the preprocessed dataset and model artifacts already included.

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the dashboard:**
   ```bash
   python -m streamlit run streamlit_app.py
   ```
   The local server will start and open the dashboard at `http://localhost:8501`.

### Optional: Retraining the Model Locally
If you want to re-run EDA, model training, or algorithm benchmarking from scratch:
1. Download the [raw OULAD files](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad/data) (`studentInfo.csv`, `studentAssessment.csv`, and `studentVle.csv`) and place them in the project root directory.
2. **(Optional) Run Exploratory Data Analysis:**
   ```bash
   python eda_script.py
   ```
3. **Train the model and export artifacts** (`.joblib`, `.csv`, and `shap_summary_plot.png`):
   ```bash
   python training_model.py
   ```
4. **(Optional) Run post-hoc algorithm benchmarking:**
   ```bash
   python compare_models.py
   ```
