import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Learning Analytics Dashboard", layout="wide")

# 1. Load the Backend Model and Data
@st.cache_data # Caches the data so it doesn't reload on every click
def load_backend():
    model = joblib.load('risk_prediction_model.joblib')
    features = joblib.load('model_features.joblib')
    data = pd.read_csv('dashboard_student_data.csv')
    return model, features, data

try:
    rf_model, model_features, student_data = load_backend()
except FileNotFoundError:
    st.error("Backend files not found. Please run train_model.py first.")
    st.stop()

# Dashboard Header
st.title("🎓 Integrated AI-Powered Learning Analytics Dashboard")
st.markdown("### Student Success & Early Intervention System")
st.markdown("---")

# 2. Sidebar - Real Student Selection
st.sidebar.header("Student Selection")
# Get a list of unique student IDs from the real data
student_list = student_data['id_student'].unique()
selected_student_id = st.sidebar.selectbox("Search Student ID", student_list)

# 3. Fetch Real Data for Selected Student
# Get the row matching the selected ID
student_record = student_data[student_data['id_student'] == selected_student_id].iloc[0]

# Prepare the data for the model (matching the exact columns used in training)
student_features = pd.DataFrame([student_record])
student_features = pd.get_dummies(student_features)
# Ensure all required columns are present, fill missing with 0
student_features = student_features.reindex(columns=model_features, fill_value=0)

# 4. Make Live AI Prediction
prediction = rf_model.predict(student_features)[0]
risk_status = "At-Risk" if prediction == 1 else "Not At-Risk"
risk_color = "inverse" if prediction == 1 else "normal"

# 5. Display Real KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Current Average Score", f"{student_record['avg_score']:.1f}%")
col2.metric("VLE Engagement (Clicks)", f"{int(student_record['total_vle_clicks'])}")
col3.metric("AI Risk Prediction", risk_status, delta_color=risk_color)

st.markdown("---")

# Main Content Split
left_column, right_column = st.columns([2, 1])

with left_column:
    st.subheader("💡 AI Intervention Recommendation")
    if prediction == 1:
        st.error("**Recommended Action:** Schedule proactive 1-on-1 Advising. The model flags this student as At-Risk based on their current engagement and academic profile.")
    else:
        st.success("**Recommended Action:** No immediate intervention required. Continue standard monitoring.")

with right_column:
    st.subheader("🔍 Explainable AI (SHAP)")
    try:
        st.image("shap_summary_plot.png", caption="Global Feature Impact on Risk Prediction", use_container_width=True)
    except FileNotFoundError:
        st.warning("SHAP plot not found.")
