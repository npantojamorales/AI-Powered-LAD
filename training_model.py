import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt

print("Loading and preparing datasets...")
student_info = pd.read_csv('studentInfo.csv')
student_assessment = pd.read_csv('studentAssessment.csv')
student_vle = pd.read_csv('studentVle.csv')

# 1. Feature Engineering (Replicating Days 1-2 structure)
avg_scores = student_assessment.groupby('id_student')['score'].mean().reset_index()
avg_scores.rename(columns={'score': 'avg_score'}, inplace=True)

total_clicks = student_vle.groupby('id_student')['sum_click'].sum().reset_index()
total_clicks.rename(columns={'sum_click': 'total_vle_clicks'}, inplace=True)

# Merge datasets
df = student_info.merge(avg_scores, on='id_student', how='left')
df = df.merge(total_clicks, on='id_student', how='left')

# 2. Define the Target Variable (Risk Prediction)
# Group 'Fail' and 'Withdrawn' as At-Risk (1), 'Pass' and 'Distinction' as Not At-Risk (0)
df['is_at_risk'] = df['final_result'].isin(['Fail', 'Withdrawn']).astype(int)

# 3. Data Preprocessing (Handling Missing Values and Encoding)
print("Preprocessing features...")
# Drop identifiers and the original target variable
X = df.drop(columns=['id_student', 'final_result', 'is_at_risk'])
y = df['is_at_risk']

# One-hot encode categorical variables (e.g., gender, disability, highest_education)
X = pd.get_dummies(X, drop_first=True)

# Impute missing numeric values with the median
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# 5. Train the Random Forest Classifier
print("Training the Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# 6. Evaluate the Baseline Model
predictions = rf_model.predict(X_test)
print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 7. Generate Explainable AI (SHAP) Summary Plot
print("Generating SHAP summary plot...")
explainer = shap.TreeExplainer(rf_model)
X_sample = X_test.sample(n=500, random_state=42)

# Calculate SHAP values
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 6))

# Handle SHAP version array differences for binary classification
if isinstance(shap_values, list):
    shap_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_to_plot = shap_values[:, :, 1]
else:
    shap_to_plot = shap_values

shap.summary_plot(shap_to_plot, X_sample, show=False)

# Save the plot for the midterm report
output_filename = 'shap_summary_plot.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=300)
print(f"\nSHAP plot saved successfully as '{output_filename}'.")

# 8. Export the Model and Data for Streamlit
print("\nSaving model and data for the dashboard...")
# Save the trained Random Forest model
joblib.dump(rf_model, 'risk_prediction_model.joblib')

# Save the final imputed feature names so Streamlit knows exactly what to feed the model
joblib.dump(X_imputed.columns, 'model_features.joblib')

# Save the merged dataset so the dashboard can look up real student records
df.to_csv('dashboard_student_data.csv', index=False)

print("Export complete! Files saved for Streamlit integration.")