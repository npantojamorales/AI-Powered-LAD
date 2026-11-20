import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Educator Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# 1. Load the Backend Model and Data
@st.cache_data
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

# Clean up module names for the teacher view
subject_mapping = {
    "AAA": "Social Sciences (AAA)", "BBB": "STEM Fundamentals (BBB)", 
    "CCC": "Computer Science (CCC)", "DDD": "Business (DDD)",
    "EEE": "Humanities (EEE)", "FFF": "IT & Computing (FFF)", 
    "GGG": "Design (GGG)"
}
if 'code_module' in student_data.columns:
    student_data['Subject'] = student_data['code_module'].map(subject_mapping).fillna(student_data['code_module'])
else:
    student_data['Subject'] = "General Course"

# Dashboard Header
st.title("🎓 Educator Learning Analytics Dashboard")
st.markdown("Proactive Student Success & Early Intervention System")
st.markdown("---")

# 2. Sidebar - Teacher Controls
st.sidebar.header("📋 Classroom Controls")

# Subject/Class Filter
available_subjects = student_data['Subject'].unique()
selected_subject = st.sidebar.selectbox("Select Class / Subject", available_subjects)

# Filter data by selected subject
class_data = student_data[student_data['Subject'] == selected_subject]

# Student Filter
student_list = class_data['id_student'].unique()
selected_student_id = st.sidebar.selectbox("Select Student ID", student_list)

st.sidebar.markdown("---")
st.sidebar.info("Data synced with Open University LMS.")

# 3. Main Layout - 3 Dedicated Tabs
tab1, tab2, tab3 = st.tabs(["🧑‍🎓 Individual Student Profile", "📊 Class Overview", "🧠 AI Transparency"])

# --- TAB 1: Student Profile & Action ---
with tab1:
    # Fetch Real Data for Selected Student
    student_record = class_data[class_data['id_student'] == selected_student_id].iloc[0]

    # Prepare the data for the model
    student_features = pd.DataFrame([student_record])
    student_features = pd.get_dummies(student_features)
    student_features = student_features.reindex(columns=model_features, fill_value=0)

    # Make Live AI Prediction
    prediction = rf_model.predict(student_features)[0]
    risk_status = "⚠️ At-Risk" if prediction == 1 else "✅ On Track"
    
    # KPI Metrics
    raw_score = student_record['avg_score']
    clicks = int(student_record['total_vle_clicks']) if pd.notna(student_record['total_vle_clicks']) else 0
    display_score = "N/A (No Submissions)" if pd.isna(raw_score) else f"{raw_score:.1f}%"

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Average Score", display_score)
    col2.metric("VLE Engagement (Clicks)", f"{clicks}")
    col3.metric("AI Risk Status", risk_status)

    st.markdown("---")

    # Actionable Intervention (Full Width for Readability)
    st.subheader("Actionable AI Intervention")
    
    if prediction == 1:
        if pd.isna(raw_score) or clicks < 50:
            st.error(f"**Diagnostic:** This student has virtually abandoned the digital classroom ({clicks} total clicks) and has no graded submissions on record.")
            st.warning("**Recommended Action:** Send an immediate wellness check-in email and verify their enrollment status with academic counseling.")
        elif raw_score < 70:
            st.error(f"**Diagnostic:** The student is actively struggling with the course material, currently holding a {display_score} average despite engaging with the platform.")
            st.warning("**Recommended Action:** Schedule a 1-on-1 academic tutoring session to review recent failed assessments.")
        else:
            st.error(f"**Diagnostic:** The student's engagement patterns show early warning signs of dropping out, despite passing grades.")
            st.warning("**Recommended Action:** Reach out with a proactive message encouraging them to maintain their participation.")
    else:
        st.success(f"**Diagnostic:** Student is maintaining a healthy {display_score} average with solid online participation.")
        st.info("**Recommended Action:** No immediate intervention required. Continue standard monitoring.")

# --- TAB 2: Class Roster ---
with tab2:
    st.subheader(f"Class Roster: {selected_subject}")
    st.write("Overview of all students currently enrolled in this subject, including their AI-predicted risk status and recommended interventions.")
    
    # --- NEW LOGIC: Generate live predictions for the entire class ---
    # 1. Prepare the data for the whole class exactly like Tab 1
    class_features = pd.get_dummies(class_data)
    class_features = class_features.reindex(columns=model_features, fill_value=0)
    
    # 2. Make Live AI Predictions for everyone
    class_predictions = rf_model.predict(class_features)
    
    # 3. Create display dataframe and assign the live AI predictions
    class_data_display = class_data.copy()
    class_data_display['Live_Prediction'] = class_predictions
    
    # Update the summary metrics to use the AI's live predictions
    total_students = len(class_data_display)
    at_risk_count = class_data_display['Live_Prediction'].sum()
    on_track_count = total_students - at_risk_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Enrolled", total_students)
    col2.metric("🟢 On Track", on_track_count)
    col3.metric("🔴 At-Risk", at_risk_count)

    st.divider()
    
    # 4. Map the Status using the Live_Prediction (NOT the old CSV column)
    class_data_display['Predicted Status'] = class_data_display['Live_Prediction'].apply(lambda x: "🔴 At-Risk" if x == 1 else "🟢 On Track")
    
    # Define the rule-based logic for the entire class (updated to use Live_Prediction)
    def determine_action(row):
        if row['Live_Prediction'] == 0:
            return "✅ Monitor"
        else:
            clicks = row['total_vle_clicks']
            score = row['avg_score']
            if pd.isna(score) or clicks < 50:
                return "🚨 Wellness Check"
            elif score < 70:
                return "📚 1-on-1 Tutoring"
            else:
                return "✉️ Proactive Message"
                
    # Apply the logic to create the new column
    class_data_display['Recommended Action'] = class_data_display.apply(determine_action, axis=1)
    
    # 5. Select and rename the columns to make it look clean
    display_df = class_data_display[['id_student', 'Predicted Status', 'Recommended Action', 'gender', 'highest_education', 'total_vle_clicks', 'avg_score']]
    display_df.columns = ['Student ID', 'Predicted Status', 'Recommended Action', 'Gender', 'Education Level', 'Total Clicks', 'Average Score']
    
    # 6. Render the updated table
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- TAB 3: SHAP Chart & Logic ---
with tab3:
    st.subheader("How the AI Makes Decisions")
    st.markdown("""
This platform uses a machine learning model to evaluate student risk. The chart below illustrates the **global decision logic** of the AI.

**How to read this chart:**
- **Features at the top** (like VLE Clicks and Average Score) have the heaviest impact on the student's final risk status.
- **Blue dots pushed to the right** indicate that low academic scores and low digital engagement actively drive a student toward the 'At-Risk' category.
- **Red dots pushed to the left** indicate that high engagement and strong scores protect a student and keep them 'On Track'.
""")
    
    try:
        st.image("shap_summary_plot.png", width=500)
    except FileNotFoundError:
        st.warning("Global AI logic plot not found.")
