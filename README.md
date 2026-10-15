# 🎓 An Integrated AI-Powered Learning Analytics Dashboard for Student Success and Early Intervention

This repository contains the machine learning pipeline and interactive Streamlit dashboard designed to predict student risk (withdrawal or failure) using the Open University Learning Analytics Dataset (OULAD). By integrating academic performance scores with behavioral engagement metrics (VLE clicks), the system provides educators with early intervention recommendations and Explainable AI (SHAP) insights.

## 🚀 Features
- **Data Integration:** Merges academic performance and behavioral engagement data from the OULAD dataset.
- **Predictive AI Engine:** A Random Forest classifier trained to identify at-risk students based on historical engagement, grades, and demographic profiles.
- **Explainable AI (SHAP):** Visualizes feature importance so educators understand *why* a student was flagged (e.g., highlighting drops in virtual learning environment activity).
- **Educator Dashboard:** An interactive Streamlit frontend that allows users to search for specific students, view live KPIs, and receive automated, data-driven intervention recommendations.

## 🛠️ Prerequisites
Ensure you have Python installed. You can install all necessary dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn shap streamlit matplotlib seaborn joblib
```

## 📁 Project Structure
* `eda_script.py`: Generates missing values tables and correlation heatmaps for exploratory data analysis.
* `train_model.py`: Preprocesses the OULAD data, trains the Random Forest classifier, generates the SHAP summary plot, and exports the backend model/data.
* `app.py`: The main Streamlit dashboard application.
* **Data Requirements**: The system requires the raw OULAD files (`studentInfo.csv`, `studentAssessment.csv`, and `studentVle.csv`) to be placed in the project root folder prior to execution.

## 🏃‍♂️ How to Run the System

**Step 1: Exploratory Data Analysis (Optional)**
Generate data quality metrics and the correlation heatmap (`oulad_correlation_heatmap.png`).
```bash
python eda_script.py
```

**Step 2: Train the Model and Export Backend**
Run the training script to process the data, evaluate the baseline Random Forest model, and export the required .joblib and .csv files for the frontend dashboard. This will also generate the shap_summary_plot.png.
```bash
python training_model.py
```

**Step 3: Launch the Dashboard**
Start the interactive local web application. The dashboard will automatically read the exported machine learning model and compiled student dataset.
```bash
streamlit run app.py
```

# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

Prerequisite: install `uv` if you don't already have it.

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync the dependencies

   ```
   $ uv sync
   ```

2. Run the app

   ```
   $ uv run streamlit run streamlit_app.py
   ```
