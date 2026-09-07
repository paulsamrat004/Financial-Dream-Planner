import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
print("[*] Loading salary dataset...")
df = pd.read_csv("salary_data.csv")
X = df[["Age", "City", "Education", "Job_Role"]]
y = df["Monthly_Salary"]
categorical_features = ["City", "Education", "Job_Role"]
numerical_features = ["Age"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_preds)
lr_mae = mean_absolute_error(y_test, lr_preds)
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_r2 = r2_score(y_test, dt_preds)
dt_mae = mean_absolute_error(y_test, dt_preds)
print("\n--- Model Evaluation Results ---")
print(f"Linear Regression  -> R2: {lr_r2:.4f} | MAE: Rs. {lr_mae:.2f}")
print(f"Decision Tree      -> R2: {dt_r2:.4f} | MAE: Rs. {dt_mae:.2f}")
if lr_r2 >= dt_r2:
    best_model = lr_model
    selected_name = "Linear Regression"
else:
    best_model = dt_model
    selected_name = "Decision Tree Regressor"

print(f"\n[+] Selected Model for Deployment: {selected_name}")
joblib.dump(best_model, "best_salary_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
metrics = {
    "lr_r2": round(float(lr_r2), 4),
    "lr_mae": round(float(lr_mae), 2),
    "dt_r2": round(float(dt_r2), 4),
    "dt_mae": round(float(dt_mae), 2),
    "selected": selected_name,
}
joblib.dump(metrics, "model_metrics.pkl")
print("[+] Model artifacts saved successfully as '.pkl' files!")