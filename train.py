import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ─────────────────────────────
# LOAD DATA
# ─────────────────────────────
data = pd.read_csv("raw_data_zameen.csv")

print("Original shape:", data.shape)


# ─────────────────────────────
# PRICE CLEANING
# ─────────────────────────────
def convert_price(price):
    try:
        price = str(price).lower().replace("pkr", "").replace("\n", " ")

        num = re.findall(r"[\d\.]+", price)
        if not num:
            return np.nan

        value = float(num[0])

        if "crore" in price:
            return value * 10000000
        elif "lakh" in price:
            return value * 100000
        else:
            return np.nan
    except:
        return np.nan


# ─────────────────────────────
# AREA CLEANING
# ─────────────────────────────
def convert_area(area):
    try:
        area = str(area).lower()

        num = re.findall(r"[\d\.]+", area)
        if not num:
            return np.nan

        value = float(num[0])

        if "kanal" in area:
            return value * 20
        elif "marla" in area:
            return value
        else:
            return np.nan
    except:
        return np.nan


# ─────────────────────────────
# FEATURE SELECTION (IMPROVED)
# ─────────────────────────────
data = data[[
    "area",
    "bedroom",
    "bath",
    "location_city",
    "location_province",
    "purpose",
    "price"
]]


# ─────────────────────────────
# CLEANING
# ─────────────────────────────
data["price"] = data["price"].apply(convert_price)
data["area"] = data["area"].apply(convert_area)

data["bedroom"] = pd.to_numeric(data["bedroom"], errors="coerce")
data["bath"] = pd.to_numeric(data["bath"], errors="coerce")

# drop extreme outliers (VERY IMPORTANT)
data = data[data["price"] < data["price"].quantile(0.99)]

data = data.dropna()


print("After cleaning:", data.shape)


# ─────────────────────────────
# ENCODING (FASTER METHOD)
# ─────────────────────────────
data = pd.get_dummies(
    data,
    columns=["location_city", "location_province", "purpose"],
    drop_first=True
)


# ─────────────────────────────
# FEATURES / TARGET
# ─────────────────────────────
X = data.drop("price", axis=1)
y = data["price"]


# ─────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ─────────────────────────────
# MODEL (UPGRADED)
# ─────────────────────────────
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=None
)

model.fit(X_train, y_train)


# ─────────────────────────────
# EVALUATION
# ─────────────────────────────
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\nMAE:", mae)
print("R2 Score:", r2)


# ─────────────────────────────
# SAVE MODEL
# ─────────────────────────────
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("model_features.pkl", "wb"))

print("\nModel saved successfully!")