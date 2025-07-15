import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Step 1: Create Binary Target Label from CAGR
# Let’s add a new Performance_Label:
# 1 if fund's CAGR is in the top 25%
# 0 otherwise

df = pd.read_csv(r"C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/mutual-fund-recommender/app/data/processed/preprocessed_mutual_funds.csv")
# Calculate the 75th percentile threshold for CAGR
cagr_threshold = df['CAGR'].quantile(0.75)

# Assign binary performance label
df['Performance_Label'] = df['CAGR'].apply(lambda x: 1 if x >= cagr_threshold else 0 if pd.notnull(x) else None)

# Preview label distribution
label_counts = df['Performance_Label'].value_counts(dropna=False)

label_counts

# Step 2: Feature Selection + Preprocessing
# Numerical: NAV, Rolling_Std_NAV

# Categorical: Fund_House, Scheme_Type, Scheme_Category
# Ignore: Date, Scheme_Name, Scheme_Code (ID fields)
# We’ll do data imputation using simple strategy:
# Fill NaN in numerical columns with the mean
# For categorical columns, we’ve already label-encoded, so we’re good there

# Prepare data again
model_df = df.dropna(subset=['Performance_Label'])
features = ['NAV', 'Rolling_Std_NAV', 'Fund_House', 'Scheme_Type', 'Scheme_Category']
X = model_df[features]
y = model_df['Performance_Label'].astype(int)

# Encode categorical columns
X_encoded = X.copy()
label_encoders = {}
for col in ['Fund_House', 'Scheme_Type', 'Scheme_Category']:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

# Impute missing values in numerical columns
num_cols = ['NAV', 'Rolling_Std_NAV']
imputer = SimpleImputer(strategy='mean')
X_encoded[num_cols] = imputer.fit_transform(X_encoded[num_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(report)
print("Confusion Matrix:\n", conf_matrix)
print("ROC AUC Score:", roc_auc)

# Get feature importances from the trained Random Forest model
importances = clf.feature_importances_
feature_names = X_encoded.columns

# Create a sorted dataframe for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette='crest', dodge=False, legend=False)
plt.title('Feature Importance – What Drives Mutual Fund Success?')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
