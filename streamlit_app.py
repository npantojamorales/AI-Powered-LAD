import streamlit as st
import pandas as pd
import numpy as np

# Configure the page layout
st.set_page_config(page_title="Learning Analytics Dashboard", layout="wide")

# Dashboard Header
st.title("🎓 Integrated AI-Powered Learning Analytics Dashboard")
st.markdown("### Student Success & Early Intervention System")
st.markdown("---")

# Sidebar for Navigation / Filtering
st.sidebar.header("Student Selection")
student_id = st.sidebar.selectbox("Search Student ID", ["STU-99201", "STU-99202", "STU-99203"])
course_module = st.sidebar.selectbox("Course Module", ["Data Structures", "Software Measurement"])

# Top Level KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Current Average Score", "68%", "-5% from last assessment")
col2.metric("VLE Engagement (Clicks)", "845", "+20 this week")
col3.metric("AI Risk Prediction", "At-Risk", "High Probability", delta_color="inverse")

st.markdown("---")

# Main Content Split
left_column, right_column = st.columns([2, 1])

with left_column:
    st.subheader("📊 Academic & Behavioral Trends")
    # Generate some mock data to make the UI look active for your screenshot
    chart_data = pd.DataFrame(
        np.random.randn(20, 2) * [5, 15] + [70, 150],
        columns=['Assessment Scores', 'VLE Activity']
    )
    st.line_chart(chart_data)

    st.subheader("💡 Intervention Recommendation System")
    st.info(
        "**Recommended Action:** Schedule proactive 1-on-1 Advising.\n\n"
        "**System Reasoning:** The Random Forest model indicates that a recent 15% drop in VLE engagement, "
        "combined with a borderline score on Assessment 2, strongly predicts risk of withdrawal. Early outreach is advised."
    )

with right_column:
    st.subheader("🧠 Well-Being Monitoring (PoC)")
    st.write("Self-Reported Student Metrics")
    st.slider("Reported Stress Level", 1, 10, 8, disabled=True)
    st.slider("Reported Sleep Quality", 1, 10, 3, disabled=True)
    st.warning("⚠️ Alert: Elevated stress levels and poor sleep detected over the last 7 days.")

    st.subheader("🔍 Explainable AI (SHAP)")
    st.markdown(
        """
        <div style='border: 2px dashed gray; padding: 30px; text-align: center; border-radius: 10px;'>
            <em>SHAP Summary Plot will render here in final build.</em><br>
            (Displays feature impact on the risk prediction)
        </div>
        """, 
        unsafe_allow_html=True
    )
