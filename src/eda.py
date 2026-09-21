from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Car details v3.csv"


# ----------------------------
# 1. Load and split data
# ----------------------------

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

train_df = X_train.copy()
train_df["selling_price"] = y_train


# ----------------------------
# 2. Target statistics
# ----------------------------

price = train_df["selling_price"]

print(f"Mean: {price.mean():.2f}")
print(f"Median: {price.median():.2f}")
print(f"Skewness: {price.skew():.2f}")

print("\nPrice percentiles:")
print(price.quantile([0.90, 0.95, 0.99]))


# ----------------------------
# 3. Categorical analysis
# ----------------------------

categorical_columns = [
    "fuel",
    "transmission",
    "seller_type",
    "owner"
]

for column in categorical_columns:
    category_results = (
        train_df.groupby(column)["selling_price"]
        .agg(["count", "median"])
        .sort_values("median", ascending=False)
    )

    print(f"\nPrice statistics by {column}:")
    print(category_results)


# ----------------------------
# 4. Charts
# ----------------------------

figure, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(18, 5)
)

price.plot.hist(
    bins=50,
    edgecolor="black",
    ax=axes[0]
)
axes[0].set_title("Selling price distribution")
axes[0].set_xlabel("Selling price")

train_df.plot.scatter(
    x="year",
    y="selling_price",
    alpha=0.3,
    ax=axes[1]
)
axes[1].set_title("Year vs selling price")

train_df.plot.scatter(
    x="km_driven",
    y="selling_price",
    alpha=0.3,
    ax=axes[2]
)
axes[2].set_title("Kilometres driven vs selling price")

plt.tight_layout()
plt.show()