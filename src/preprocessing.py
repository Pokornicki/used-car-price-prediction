import pandas as pd


def clean_features(data):
    cleaned = data.copy()

    cleaned["brand"] = cleaned["name"].str.split().str[0]

    cleaned["mileage_unit"] = cleaned["mileage"].str.extract(
        r"(kmpl|km/kg)",
        expand=False
    )

    cleaned["mileage_value"] = pd.to_numeric(
        cleaned["mileage"].str.extract(
            r"(\d+(?:\.\d+)?)",
            expand=False
        ),
        errors="coerce"
    )

    cleaned["engine"] = pd.to_numeric(
        cleaned["engine"].str.extract(
            r"(\d+(?:\.\d+)?)",
            expand=False
        ),
        errors="coerce"
    )

    cleaned["max_power"] = pd.to_numeric(
        cleaned["max_power"].str.extract(
            r"(\d+(?:\.\d+)?)",
            expand=False
        ),
        errors="coerce"
    )

    cleaned = cleaned.drop(
        columns=["name", "mileage", "torque"]
    )

    return cleaned