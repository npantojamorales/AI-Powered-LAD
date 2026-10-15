import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the OULAD datasets
print("Loading OULAD datasets...")
try:
    student_info = pd.read_csv('studentInfo.csv')
    student_assessment = pd.read_csv('studentAssessment.csv')
    student_vle = pd.read_csv('studentVle.csv')
    print("Data loaded successfully!\n")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the CSV files are in the same directory as this script.")
    exit()

# 2. Generate Missing Values Tables
print("--- Missing Values Analysis ---")
def get_missing_table(df, name):
    missing = df.isnull().sum()
    percent = (df.isnull().sum() / len(df)) * 100
    table = pd.DataFrame({'Missing Values': missing, 'Percentage (%)': percent})
    # Filter only columns that actually have missing data
    table = table[table['Missing Values'] > 0].round(2)
    
    if not table.empty:
        print(f"\nMissing values in {name}:")
        print(table)
    else:
        print(f"\nNo missing values found in {name}.")
    return table

info_missing = get_missing_table(student_info, 'studentInfo.csv')
assess_missing = get_missing_table(student_assessment, 'studentAssessment.csv')
vle_missing = get_missing_table(student_vle, 'studentVle.csv')

# 3. Feature Engineering for Correlation
print("\n--- Generating Correlation Heatmap ---")
# Calculate average assessment score per student (Academic data)
avg_scores = student_assessment.groupby('id_student')['score'].mean().reset_index()
avg_scores.rename(columns={'score': 'avg_score'}, inplace=True)

# Calculate total VLE clicks per student (Behavioral data)
# Note: studentVle is large, this groupby may take a few seconds
total_clicks = student_vle.groupby('id_student')['sum_click'].sum().reset_index()
total_clicks.rename(columns={'sum_click': 'total_vle_clicks'}, inplace=True)

# Merge datasets together based on student ID
eda_df = student_info.merge(avg_scores, on='id_student', how='left')
eda_df = eda_df.merge(total_clicks, on='id_student', how='left')

# 4. Correlation Heatmap
# Select only the numerical columns for correlation
numeric_df = eda_df.select_dtypes(include=['float64', 'int64'])

# Drop 'id_student' as it is just an identifier and skews the matrix
if 'id_student' in numeric_df.columns:
    numeric_df = numeric_df.drop('id_student', axis=1)

# Generate the plot
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix: Academic & Behavioral Features', pad=20)
plt.tight_layout()

# Save the plot directly for your midterm report
output_filename = 'oulad_correlation_heatmap.png'
plt.savefig(output_filename, dpi=300)
print(f"\nHeatmap saved successfully as '{output_filename}'.")

# Display the plot
plt.show()