from pathlib import Path

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split
)
from sklearn.model_selection import (
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from preprocessing import clean_features

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Car details v3.csv"
df = pd.read_csv(DATA_PATH)
df.drop_duplicates(inplace=True)

X = df.drop(columns=["selling_price"])
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

numeric_columns = [
    "year",
    "km_driven",
    "engine",
    "max_power",
    "seats",
    "mileage_value"
]

categorical_columns = [
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "brand",
    "mileage_unit"
]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_columns),
    ("cat", cat_pipeline, categorical_columns)
])

full_pipeline = Pipeline([
    (
        "feature_cleaning",
        FunctionTransformer(
            clean_features,
            validate=False
        )
    ),
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    full_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

cv_rmse = -cv_results["test_score"]

print(
    f"CV RMSE: {cv_rmse.mean():.2f} "
    f"+/- {cv_rmse.std():.2f}"
)

full_pipeline.fit(X_train, y_train)
y_pred_full = full_pipeline.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred_full)
test_mse = mean_squared_error(y_test, y_pred_full)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, y_pred_full)

print(f"Final test MAE: {test_mae:.2f}")
print(f"Final test RMSE: {test_rmse:.2f}")
print(f"Final test R²: {test_r2:.4f}")

error_analysis = X_test[
    ["name", "year", "km_driven"]
].copy()

error_analysis["actual_price"] = y_test
error_analysis["predicted_price"] = y_pred_full

error_analysis["residual"] = (
    error_analysis["actual_price"]
    - error_analysis["predicted_price"]
)

error_analysis["absolute_error"] = (
    error_analysis["residual"].abs()
)

largest_errors = (
    error_analysis
    .sort_values("absolute_error", ascending=False)
    .head(10)
)

print("\nTen largest prediction errors:")
print(largest_errors.to_string())

MODEL_PATH = BASE_DIR / "models" / "car_price_model.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Final production training using all available data
full_pipeline.fit(X, y)

joblib.dump(
    full_pipeline,
    MODEL_PATH,
    compress=3
)

print(f"Model saved to: {MODEL_PATH}")