import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report
# LOAD DATA
# Debugging: List contents of the path to find the correct filename
print(f"Contents of {path}: {os.listdir(path)}")

df = pd.read_csv(os.path.join(path, "Global finance data.csv"))
print(df.columns)
# DEFINE TARGET & FEATURES
TARGET = "Daily_Change_Percent"

DROP_COLS = [
    "Country",
    "Stock_Index",
    "Date", # Add 'Date' to DROP_COLS as it's non-numeric and causing the imputer error
    "Currency_Code", # Add 'Currency_Code' to DROP_COLS as it's non-numeric
    "Credit_Rating", # Add 'Credit_Rating' to DROP_COLS as it's non-numeric
    "Banking_Sector_Health" # Add 'Banking_Sector_Health' to DROP_COLS as it's non-numeric
]

X = df.drop(columns=DROP_COLS + [TARGET])
y = df[TARGET]

# HANDLE MISSING VALUES
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# MODEL 1: LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\nLINEAR REGRESSION RESULTS")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2:", r2_score(y_test, y_pred_lr))

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nLINEAR REGRESSION COEFFICIENTS")
print(coef_df)

# MODEL 2: RANDOM FOREST REGRESSOR
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRANDOM FOREST REGRESSION RESULTS")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2:", r2_score(y_test, y_pred_rf))

# FEATURE IMPORTANCE
importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFEATURE IMPORTANCE (RANDOM FOREST)")
print(importances)

plt.figure(figsize=(10, 6))
importances.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# OPTIONAL: MARKET REGIME CLASSIFICATION

def market_regime(change):
    if change > 0.5:
        return "Bull"
    elif change < -0.5:
        return "Bear"
    else:
        return "Neutral"

df["market_regime"] = df[TARGET].apply(market_regime)

le = LabelEncoder()
y_cls = le.fit_transform(df["market_regime"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cls,
    test_size=0.25,
    random_state=42
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

clf.fit(X_train, y_train)
y_pred_cls = clf.predict(X_test)

print("\nMARKET REGIME CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred_cls, target_names=le.classes_))
