import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load processed data
df = pd.read_csv("dashboard_student_data.csv") 
X = df.drop(columns=['id_student', 'is_at_risk', 'final_result', 'Subject'], errors='ignore')
y = df['is_at_risk']

# Convert text categories into numbers
X = pd.get_dummies(X, drop_first=True)

# Fill missing numerical blanks with the median
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 1: Algorithm Comparison
models = {
    "Random Forest": RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
    "SVM": SVC(probability=True, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # Get probabilities for the positive class (1) to calculate AUC-ROC
    y_prob = model.predict_proba(X_test_scaled)[:, 1] 
    
    results.append({
        "Model": name,
        "Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
        "Precision": f"{precision_score(y_test, y_pred):.4f}",
        "Recall": f"{recall_score(y_test, y_pred):.4f}",
        "F1-Score": f"{f1_score(y_test, y_pred):.4f}",
        "AUC-ROC": f"{roc_auc_score(y_test, y_prob):.4f}"
    })

print("\n--- Algorithm Comparison Table ---")
print(pd.DataFrame(results).to_markdown(index=False))

# Step 2: Academic-Only Baseline
X_academic = X_imputed.drop(columns=['total_vle_clicks'], errors='ignore')
X_train_ac, X_test_ac, y_train_ac, y_test_ac = train_test_split(X_academic, y, test_size=0.2, random_state=42)

rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_baseline.fit(X_train_ac, y_train_ac)
y_pred_ac = rf_baseline.predict(X_test_ac)

print("\n--- Academic-Only Baseline ---")
print(f"Accuracy without VLE Clicks: {accuracy_score(y_test_ac, y_pred_ac):.4f}")
print(f"F1-Score without VLE Clicks: {f1_score(y_test_ac, y_pred_ac):.4f}\n")
