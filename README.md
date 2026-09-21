# Used Car Price Prediction

An end-to-end machine learning regression project that predicts the selling price of a used car from its age, mileage, engine specifications, fuel type, transmission, ownership history, and other available characteristics.

## Project Goal

The goal is to estimate a used car's selling price in Indian rupees (INR). This is a regression problem because the target, `selling_price`, is a continuous numerical value.

RMSE is the primary evaluation metric because large prediction errors are particularly costly. MAE and R² are also reported to provide a more complete evaluation.

## Dataset

The project uses `Car details v3.csv`.

- Original observations: 8,128
- Observations after removing duplicates: 6,926
- Target: `selling_price`
- Train/test split: 80%/20%
- Random state: 42

The dataset is not included in this repository. Place it at:

```text
data/Car details v3.csv
```

## Data Preparation

The preprocessing workflow:

- removes duplicate observations;
- extracts `brand` from the car name;
- converts `engine` and `max_power` from text to numeric values;
- separates mileage into `mileage_value` and `mileage_unit`;
- removes the original `name`, `mileage`, and `torque` columns;
- fills missing numerical values with the median;
- fills missing categorical values with the most frequent value;
- standardizes numerical features;
- one-hot encodes categorical features and safely handles unseen categories.

The feature-cleaning function, preprocessing steps, and model are combined into one scikit-learn pipeline. Consequently, the saved model accepts raw car data in the same format as the original dataset.

### Numerical features

- `year`
- `km_driven`
- `engine`
- `max_power`
- `seats`
- `mileage_value`

### Categorical features

- `fuel`
- `seller_type`
- `transmission`
- `owner`
- `brand`
- `mileage_unit`

## Exploratory Data Analysis

The selling-price distribution is strongly right-skewed. Its mean is higher than its median because a small number of very expensive cars extend the right tail.

Important observations included:

- newer cars generally have higher selling prices;
- automatic cars have a higher median price than manual cars;
- dealer listings have a higher median price than individual listings;
- first-owner cars are generally more expensive than cars with multiple previous owners;
- the most expensive cars produce the largest prediction errors.

## Models Compared

Five-fold shuffled cross-validation was performed only on the training data.

| Model | CV MAE | CV RMSE | CV R² |
|---|---:|---:|---:|
| Dummy Regressor | 297,969.19 | 528,033.83 | -0.0036 |
| Linear Regression | 139,852.91 | 274,460.59 | 0.7286 |
| Ridge Regression | 141,100.29 | 273,932.94 | 0.7290 |
| Decision Tree | 105,398.85 | 231,996.30 | 0.8066 |
| Random Forest | **81,594.11** | **177,344.71** | **0.8828** |

Random Forest produced the lowest cross-validation RMSE and was selected as the final model.

## Final Model

The final estimator is:

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

A randomized hyperparameter search was also performed, but it did not improve the primary cross-validation metric. The simpler Random Forest configuration was therefore retained.

An additional `model_family` feature, built from the first two words of the car name, was tested as well. It changed CV RMSE from `177,344.71` to `178,222.45`, so it was removed.

## Results

### Cross-validation

```text
CV RMSE: 177,344.71 +/- 26,679.28 INR
```

### Final test set

```text
MAE:   73,035.75 INR
RMSE: 126,891.25 INR
R²:         0.9266
```

The model explains approximately 92.66% of the variation in prices in the test set. Its typical absolute error is approximately 73,036 INR. RMSE is higher than MAE because a small number of expensive cars have large prediction errors.

After evaluation, the final pipeline is refitted using all available observations and saved to `models/car_price_model.joblib`.

## Project Structure

```text
used-car-price-prediction/
├── data/
│   └── Car details v3.csv
├── models/
│   └── car_price_model.joblib
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Place the CSV file in the `data` directory and run the training script:

```bash
python src/train.py
```

This script evaluates the model, retrains it on all available data, and saves the complete pipeline in the `models` directory.

Run an example prediction with:

```bash
python src/predict.py
```

To predict another car, edit the raw car information stored in `new_car` inside `src/predict.py`.

## Limitations

- The target distribution is strongly right-skewed, and expensive cars are harder to predict accurately.
- Random Forest predictions are not guaranteed to increase smoothly as the production year increases.
- Only the manufacturer is retained from the car name, so the model does not know the exact model or trim.
- The first-two-word `model_family` experiment did not improve cross-validation.
- Predictions are most reliable for cars similar to those represented in the training dataset.
- The results describe this particular dataset and may not represent current market prices.

## Future Improvements

- engineer a more reliable model and trim feature;
- collect more observations for rare and expensive cars;
- evaluate a logarithmic target transformation;
- compare gradient-boosting models;
- investigate monotonic constraints for features such as production year;
- add input validation and a web interface or API.

## Key Learning Outcomes

This project demonstrates:

- regression problem formulation;
- exploratory data analysis without using the test set;
- feature engineering from mixed text and numeric columns;
- leakage-safe preprocessing with scikit-learn pipelines;
- baseline creation and model comparison;
- cross-validation and hyperparameter search;
- residual and large-error analysis;
- saving and loading a complete production pipeline.
