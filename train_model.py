import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

print("Loading salary dataset...")
df = pd.read_csv("salary_data.csv")

# Defining feature matrix (X) and target vector (y)
X = df[["Age", "City", "Education", "Job_Role"]]
y = df["Monthly_Salary"]

categorical_features = ["City", "Education", "Job_Role"]
numerical_features = ["Age"]

# Build preprocessor with explicit column transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

# Fit and transform dataset features
X_processed = preprocessor.fit_transform(X)

# Split dataset into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_preds)
lr_mae = mean_absolute_error(y_test, lr_preds)

# Train Decision Tree Regressor model
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_r2 = r2_score(y_test, dt_preds)
dt_mae = mean_absolute_error(y_test, dt_preds)

# Output evaluation metrics to terminal
print("\n--- Model Evaluation Results ---")
print(f"Linear Regression  -> R2: {lr_r2:.4f} | MAE: Rs. {lr_mae:.2f}")
print(f"Decision Tree      -> R2: {dt_r2:.4f} | MAE: Rs. {dt_mae:.2f}")

# Model Selection Logic based on R2 Score
if lr_r2 >= dt_r2:
    best_model = lr_model
    selected_name = "Linear Regression"
else:
    best_model = dt_model
    selected_name = "Decision Tree Regressor"

print(f"\nSelected Model for Deployment: {selected_name}")

# Export trained artifacts strictly matching Streamlit app expectations
joblib.dump(best_model, "best_salary_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

# Save dictionary with scalar metrics for Tab 4 visualization
metrics = {
    "lr_r2": float(round(lr_r2, 4)),
    "lr_mae": float(round(lr_mae, 2)),
    "dt_r2": float(round(dt_r2, 4)),
    "dt_mae": float(round(dt_mae, 2)),
    "selected": selected_name,
}

joblib.dump(metrics, "model_metrics.pkl")
print("Model artifacts saved successfully as '.pkl' files!")