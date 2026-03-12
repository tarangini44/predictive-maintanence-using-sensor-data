# ==============================
# AI4I Predictive Maintenance Project
# ==============================

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
df = pd.read_csv("predictive_maintenance.csv")

print("Dataset shape:", df.shape)
print(df.head())

# 3. Drop unnecessary columns (AI4I has ID columns)
if "UDI" in df.columns and "Product ID" in df.columns:
    df = df.drop(["UDI", "Product ID"], axis=1)

# 4. Convert categorical column "Type" into numbers
if "Type" in df.columns:
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)

# 5. Define Features and Target
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# 6. Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Create Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 8. Train Model
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)

# 10. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 11. Feature Importance
importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.bar(feature_names, importance)
plt.xticks(rotation=45, ha="right")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
