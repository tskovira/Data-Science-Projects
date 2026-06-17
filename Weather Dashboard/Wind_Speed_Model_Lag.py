import pandas as pd
import numpy as np
from meteostat import Stations, Hourly
import datetime

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

start = datetime.datetime(2024, 1, 1)
end   = datetime.datetime(2024, 12, 31)

stations = (
    Stations()
    .nearby(40.4406, -79.9959)  # Pittsburgh
    .fetch(10)
)

df = pd.DataFrame()
selected_station = None

for station_id in stations.index:
    attempt = Hourly(station_id, start, end).fetch()

    if len(attempt) > 0:
        df = attempt
        selected_station = station_id
        break
    else:
        print("No data available.")

#Lag Features
df["hour"] = df.index.hour
df["month"] = df.index.month
df["wspd_lag1"] = df["wspd"].shift(1)
df["wspd_lag3"] = df["wspd"].shift(3)
df["wspd_lag6"] = df["wspd"].shift(6)

df = df.dropna(subset=["wspd", "wspd_lag1", "wspd_lag3", "wspd_lag6"])

feature_cols = [
    "temp", "dwpt", "pres", "rhum", "wdir",
    "hour", "month",
    "wspd_lag1", "wspd_lag3", "wspd_lag6"
]

X = df[feature_cols]
y = df["wspd"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler())
        ]), feature_cols)
    ]
)

linear_model = Pipeline([
    ("prep", preprocessor),
    ("lr", LinearRegression())
])

boost_model = Pipeline([
    ("prep", preprocessor),
    ("hgb", HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.05
    ))
])

n_splits = min(5, max(2, len(df) // 500))# 5 splits

tscv = TimeSeriesSplit(n_splits=n_splits)

def evaluate_model(model, name):
    mae_scores = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error"))
    r2_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")

    print(f"\n{name}")
    print(f"MAE:  {mae_scores.mean():.3f}")
    print(f"RMSE: {rmse_scores.mean():.3f}")
    print(f"R²:   {r2_scores.mean():.3f}")

evaluate_model(linear_model, "Linear Regression")
evaluate_model(boost_model, "HistGradientBoostingRegressor")