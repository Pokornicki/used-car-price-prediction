from pathlib import Path

import joblib
import pandas as pd

# Required when loading FunctionTransformer
from preprocessing import clean_features


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model.joblib"

model = joblib.load(MODEL_PATH)

new_car = pd.DataFrame([{
    "name": "Hyundai i20 Sportz",
    "year": 2020,
    "km_driven": 50000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": "18.6 kmpl",
    "engine": "1197 CC",
    "max_power": "81.8 bhp",
    "torque": "114Nm@ 4000rpm",
    "seats": 5.0
}])

for year in range(2015, 2021):
    test_car = new_car.copy()
    test_car.loc[0, "year"] = year

    predicted_price = model.predict(test_car)[0]

    print(
        f"Year {year}: "
        f"{predicted_price:,.2f} INR"
    )