import pandas as pd
import numpy as np
from meteostat import Point, Stations, Hourly
import datetime

from great_tables import GT, md, style, loc

import altair as alt
from vega_datasets import data as vega_datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import numpy as np

def get_data():
    stations = Stations()
    locs=[ (40.4406, -79.9959),
    (33.9425, -118.4081),
    (25.7959, -80.2870),
    (39.8617, -104.6731),
    (41.9742, -87.9073),
    (47.4502, -122.3088)]

    def _get_nearby_stations(latitude, longitude, n=6):  
        stations = Stations()
        
        stations = stations.nearby(latitude, longitude)
        stations = stations.fetch(n)    

        #rank stations based on distance to main station 
        stations["rank"] = stations["distance"].rank(method="first").astype(int)

        #print(stations.head())


        return stations.reset_index()
    def _get_data(stations, start = datetime.datetime(2024, 1, 1), end = datetime.datetime(2024, 12, 31)):
    
        station_ids = stations['id'].tolist()
        data = Hourly(station_ids, start, end).fetch().reset_index()

        # attach main station ID to all rows BEFORE pivot
        main_station = stations.iloc[0]["id"]
        data["main_station"] = main_station

        # Merge with station coords/rank
        data = data.merge(
            stations[["id", "longitude", "latitude", "rank"]],
            left_on="station",
            right_on="id",
            how="left"
        )

        # compute u/v
        data["u"] = data["wspd"] * np.sin(np.deg2rad(data["wdir"] + 180))
        data["v"] = data["wspd"] * np.cos(np.deg2rad(data["wdir"] + 180))
        data = data[["time","rank","main_station","u","v","longitude","latitude"]]

        # average duplicate rows
        data = data.groupby(["time","rank","main_station"], as_index=False).mean()

        # pivot, but KEEP main_station!
        data = data.pivot(index=["time","main_station"], columns="rank", values=["u","v","longitude","latitude"])

        # flatten
        data.columns = [f"{var}_{rank}" for var, rank in data.columns]
        data = data.dropna().reset_index()

        # rename for consistency
        data = data.rename(columns={"main_station": "station"})

        return data

    
    all_stations=pd.DataFrame()
    data=pd.DataFrame()
    for lat, lon in locs:
        stations_loc = _get_nearby_stations(lat, lon, n=6)
        data_loc = _get_data(stations_loc)
        all_stations = pd.concat([all_stations, stations_loc]).drop_duplicates()
        data = pd.concat([data, data_loc]).drop_duplicates()
        
    stations = all_stations.reset_index()
    #print(data.columns)

    return data, stations


def train_model(data):

    y = data[["u_1", "v_1"]]
    station = data[["station"]]

    # Drop non-numeric features
    X = data.drop(columns=["u_1", "v_1", "station", "time"])

    # Remove rows with NaNs
    mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    station = station[mask]

    # Split
    X_train, X_test, y_train, y_test, station_train, station_test = train_test_split(
        X, y, station, test_size=0.2, shuffle=True, random_state=42
    )

    # Fit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    print("R²:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    return station_test, y_test.to_numpy(), y_pred


def visualize_predictions(station_test, y_test, y_pred, stations):
    alt.data_transformers.disable_max_rows()

    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Build dataframe
    wind_df = pd.DataFrame({
        'u_obs': y_test[:, 0],
        'v_obs': y_test[:, 1],
        'u_pred': y_pred[:, 0],
        'v_pred': y_pred[:, 1],
        'station': station_test.values.ravel()
    })

    wind_df = wind_df.merge(
        stations[['id', 'longitude', 'latitude']],
        left_on='station',
        right_on='id',
        how='left'
    )
    
    scale = .5
    wind_df["end_lon"] = wind_df["longitude"] + wind_df["u_obs"] * scale
    wind_df["end_lat"] = wind_df["latitude"] + wind_df["v_obs"] * scale
    wind_df["end_lon_pred"] = wind_df["longitude"] + wind_df["u_pred"] * scale
    wind_df["end_lat_pred"] = wind_df["latitude"] + wind_df["v_pred"] * scale
    wind_df.sort_values(by="station", inplace=True)

    wind_df = wind_df.reset_index(drop=True).reset_index().rename(columns={"index": "arrow_id"})
    

    # Slider parameter
    selector = alt.param(
        name="selected_arrow",
        bind=alt.binding_range(min=0, max=len(wind_df)-1, step=1, name="Arrow:"),
        value=0
    )

    # US map
    us_states = alt.topo_feature(vega_datasets.us_10m.url, "states")

    base = (
        alt.Chart(us_states)
        .mark_geoshape(fill="#f0f0f0", stroke="black")
        .project("albersUsa")
        .interactive()
    )

    # Stations
    points = (
        alt.Chart(stations)
        .mark_circle(size=200, opacity=0.8)
        .encode(
            longitude="longitude:Q",
            latitude="latitude:Q",
            color="rank:N",
            tooltip=["name:N", "id:N", "longitude:Q", "latitude:Q",
                     "elevation:Q", "rank:N", "distance:Q"]
        )
    )

    # Observed arrow
    arrows_obs = (
        alt.Chart(wind_df)
        .mark_line()
        .encode(
            longitude='longitude:Q',
            latitude='latitude:Q',
            longitude2='end_lon:Q',
            latitude2='end_lat:Q',
            color=alt.value("blue"),
            strokeWidth=alt.value(4)
        )
        .transform_filter("datum.arrow_id == selected_arrow")
        
    )

    # Predicted arrow
    arrows_pred = (
        alt.Chart(wind_df)
        .mark_line()
        .encode(
            longitude='longitude:Q',
            latitude='latitude:Q',
            longitude2='end_lon_pred:Q',
            latitude2='end_lat_pred:Q',
            color=alt.value("red"),
            strokeWidth=alt.value(4)
        )
        .transform_filter("datum.arrow_id == selected_arrow")
    )

    # Layer all, attach params ONCE
    map_chart = (
        (base + points + arrows_obs + arrows_pred)
        .add_params(selector)
        .properties(
            width=600,
            height=400,
            title="Observed vs Predicted Wind Vector"
        )
    )

    return map_chart


def make_wind_model_charts():
    import pandas as pd
    import numpy as np
    import altair as alt
    from datetime import datetime, timedelta
    from meteostat import Point, Hourly
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.ensemble import HistGradientBoostingRegressor
    np.random.seed(42)

    study_locations = {
        'Chicago': (41.8781, -87.6298),
        'Denver': (39.7392, -104.9903),
        'Miami': (25.7617, -80.1918),
        'Phoenix': (33.4484, -112.0740),
        'Seattle': (47.6062, -122.3321)
    }

    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)

    def collect_weather_data(locations, start_date, end_date):
        all_data = []
        exclude_vars = ['tsun', 'snow', 'wpgt', 'prcp']
        
        for city_name, (lat, lon) in locations.items():
            try:
                point = Point(lat, lon)
                data = Hourly(point, start_date, end_date).fetch()
                
                try:
                    from meteostat import Stations
                    stations = Stations()
                    stations = stations.nearby(lat, lon)
                    station_data = stations.fetch(1)
                    if not station_data.empty:
                        elevation = station_data.iloc[0]['elevation']
                    else:
                        elevation = None
                except:
                    elevation = None
                
                for col in exclude_vars:
                    if col in data.columns:
                        data = data.drop(columns=[col])
                
                if 'coco' in data.columns:
                    def group_condition_codes(code):
                        if pd.isna(code):
                            return 'Unknown'
                        code = int(code)
                        if code in [1, 2]:
                            return 'Clear'
                        elif code in [3, 4, 5, 6]:
                            return 'Cloudy/Foggy'
                        elif code in [7, 8, 9, 17]:
                            return 'Rain'
                        elif code in [10, 11, 12, 13, 14, 15, 16, 19, 21]:
                            return 'Cold Precip'
                        elif code in [18, 20, 22, 23, 24, 25, 26, 27]:
                            return 'Extreme'
                        else:
                            return 'Other'
                    
                    data['condition_group'] = data['coco'].apply(group_condition_codes)
                
                data['city'] = city_name
                data['latitude'] = lat
                data['longitude'] = lon
                data['elevation'] = elevation
                data['datetime'] = data.index
                data = data.reset_index()
                
                all_data.append(data)
                    
            except Exception as e:
                pass
                
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            empty_cols = []
            for col in combined_data.columns:
                if combined_data[col].isna().all():
                    empty_cols.append(col)
            
            if empty_cols:
                combined_data = combined_data.drop(columns=empty_cols)
            
            return combined_data
        else:
            return pd.DataFrame()

    weather_data = collect_weather_data(study_locations, start_date, end_date)

    def prepare_wind_direction_features(df):
        df_clean = df.copy()
        
        df_clean['time'] = pd.to_datetime(df_clean['time'])
        df_clean = df_clean.sort_values(['city', 'time']).reset_index(drop=True)
        
        df_clean['hour'] = df_clean['time'].dt.hour
        df_clean['month'] = df_clean['time'].dt.month
        
        # Create circular encodings for hour and month
        df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['hour'] / 24)
        df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['hour'] / 24)
        
        df_clean['month_sin'] = np.sin(2 * np.pi * (df_clean['month'] - 1) / 12)
        df_clean['month_cos'] = np.cos(2 * np.pi * (df_clean['month'] - 1) / 12)
        
        numerical_features = ['temp', 'dwpt', 'rhum', 'pres', 'wspd', 
                            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                            'latitude', 'longitude', 'elevation']
        categorical_features = ['condition_group', 'city']
        
        available_numerical = [col for col in numerical_features if col in df_clean.columns]
        available_categorical = [col for col in categorical_features if col in df_clean.columns]
        
        for col in available_numerical:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        for col in available_categorical:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
        
        return df_clean, available_numerical, available_categorical

    weather_clean, numerical_cols, categorical_cols = prepare_wind_direction_features(weather_data)

    def prepare_wind_direction_model(df):
        """
        Prepare data for circular wind direction prediction
        """
        
        model_data = df[df['wdir'].notna()].copy()
        
        numerical_features = ['temp', 'dwpt', 'rhum', 'pres', 'wspd', 
                            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                            'latitude', 'longitude', 'elevation']
        numerical_available = [col for col in numerical_features if col in model_data.columns]
        
        categorical_features = ['condition_group', 'city']
        categorical_available = [col for col in categorical_features if col in model_data.columns]
        
        # Prepare features
        X_numerical = model_data[numerical_available].copy()
        X_categorical = model_data[categorical_available].copy()
        
        # Fill missing values
        X_numerical.fillna(X_numerical.median(), inplace=True)
        X_categorical.fillna(X_categorical.mode().iloc[0], inplace=True)
        
        # Convert wind direction to circular components
        wind_radians = np.radians(model_data['wdir'])
        y_sin = np.sin(wind_radians)
        y_cos = np.cos(wind_radians)
        
        return X_numerical, X_categorical, y_sin, y_cos, numerical_available, categorical_available

    X_numerical, X_categorical, y_sin, y_cos, numerical_cols, categorical_cols = prepare_wind_direction_model(weather_clean)

    if len(categorical_cols) > 0:
        onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_categorical_encoded = onehot_encoder.fit_transform(X_categorical)

        feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
        X_categorical_encoded = pd.DataFrame(X_categorical_encoded, 
                                        columns=feature_names, 
                                        index=X_categorical.index)
        
        X = pd.concat([X_numerical, X_categorical_encoded], axis=1)
    else:
        X = X_numerical.copy()

    # Split data for both sine and cosine components
    X_train, X_test, y_sin_train, y_sin_test = train_test_split(X, y_sin, test_size=0.2, random_state=42)
    _, _, y_cos_train, y_cos_test = train_test_split(X, y_cos, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear Regression models for sine and cosine
    linear_sin = LinearRegression()
    linear_cos = LinearRegression()
    linear_sin.fit(X_train_scaled, y_sin_train)
    linear_cos.fit(X_train_scaled, y_cos_train)

    y_sin_pred_linear = linear_sin.predict(X_test_scaled)
    y_cos_pred_linear = linear_cos.predict(X_test_scaled)

    # Train HistGradientBoosting models for sine and cosine
    hist_sin = HistGradientBoostingRegressor(random_state=42)
    hist_cos = HistGradientBoostingRegressor(random_state=42)
    hist_sin.fit(X_train, y_sin_train)
    hist_cos.fit(X_train, y_cos_train)

    y_sin_pred_hist = hist_sin.predict(X_test)
    y_cos_pred_hist = hist_cos.predict(X_test)

    # Convert back to degrees
    def circular_to_degrees(sin_pred, cos_pred):
        return np.degrees(np.arctan2(sin_pred, cos_pred)) % 360

    y_pred_linear = circular_to_degrees(y_sin_pred_linear, y_cos_pred_linear)
    y_pred_hist = circular_to_degrees(y_sin_pred_hist, y_cos_pred_hist)

    # Convert test data back to degrees for comparison
    y_test_degrees = np.degrees(np.arctan2(y_sin_test, y_cos_test)) % 360

    # Calculate circular R² scores
    def circular_r2(actual_degrees, predicted_degrees):
        # Convert to circular errors
        errors = np.abs(actual_degrees - predicted_degrees)
        circular_errors = np.minimum(errors, 360 - errors)
        
        # Calculate total sum of squares (circular)
        mean_direction = np.degrees(np.arctan2(np.mean(np.sin(np.radians(actual_degrees))), 
                                            np.mean(np.cos(np.radians(actual_degrees))))) % 360
        mean_errors = np.abs(actual_degrees - mean_direction)
        mean_circular_errors = np.minimum(mean_errors, 360 - mean_errors)
        
        ss_tot = np.sum(mean_circular_errors**2)
        ss_res = np.sum(circular_errors**2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    linear_r2 = circular_r2(y_test_degrees, y_pred_linear)
    hist_r2 = circular_r2(y_test_degrees, y_pred_hist)

    # Select best model
    if hist_r2 > linear_r2:
        best_model_name = "HistGradientBoosting"
        best_predictions = y_pred_hist
        best_r2 = hist_r2
    else:
        best_model_name = "Linear Regression"
        best_predictions = y_pred_linear
        best_r2 = linear_r2

    # Sample data for visualization (Altair has 5000 row limit)
    sample_size = min(2000, len(y_test_degrees))
    sample_indices = np.random.choice(len(y_test_degrees), sample_size, replace=False)

    diagonal_line = pd.DataFrame({'x': [0, 360], 'y': [0, 360]})

    # Linear Regression plot
    linear_data = pd.DataFrame({
        'Actual': y_test_degrees.iloc[sample_indices],
        'Predicted': y_pred_linear[sample_indices]
    })

    linear_scatter = alt.Chart(linear_data).mark_circle(size=30, opacity=0.6).encode(
        x=alt.X('Actual:Q', title='Actual Wind Direction (degrees)', scale=alt.Scale(domain=[0, 360], nice=False)),
        y=alt.Y('Predicted:Q', title='Predicted Wind Direction (degrees)', scale=alt.Scale(domain=[0, 360], nice=False)),
        color=alt.value('blue')
    )

    linear_line = alt.Chart(diagonal_line).mark_line(color='red', strokeWidth=3).encode(
        x='x:Q',
        y='y:Q'
    )

    linear_chart = (linear_scatter + linear_line).properties(
        width=400,
        height=400,
        title=f'Linear Regression - R² = {linear_r2:.3f}'
    )

    # HistGradientBoosting plot
    hist_data = pd.DataFrame({
        'Actual': y_test_degrees.iloc[sample_indices],
        'Predicted': y_pred_hist[sample_indices]
    })

    hist_scatter = alt.Chart(hist_data).mark_circle(size=30, opacity=0.6).encode(
        x=alt.X('Actual:Q', title='Actual Wind Direction (degrees)', scale=alt.Scale(domain=[0, 360], nice=False)),
        y=alt.Y('Predicted:Q', title='Predicted Wind Direction (degrees)', scale=alt.Scale(domain=[0, 360], nice=False)),
        color=alt.value('green')
    )

    hist_line = alt.Chart(diagonal_line).mark_line(color='red', strokeWidth=3).encode(
        x='x:Q',
        y='y:Q'
    )

    hist_chart = (hist_scatter + hist_line).properties(
        width=400,
        height=400,
        title=f'HistGradientBoosting - R² = {hist_r2:.3f}'
    )

    return alt.hconcat(linear_chart, hist_chart)


def weather_classification_model(visual_type="confusion_matrix", random_state=42, memory_efficient=False):
    """
    Weather Classification Model with XGBoost
    
    Parameters:
    visual_type (str): Type of visual to return
                      - "confusion_matrix": Confusion matrix heatmap
                      - "feature_importance": Feature importance bar chart  
                      - "geographic_performance": Map of city performance
                      - "station_map": Map of all weather stations
    random_state (int): Random state for reproducibility
    
    Returns:
    altair.Chart: The requested visualization
    """
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    
    # Data collection
    stations = Stations()
    stations = stations.nearby(40.4406, -79.9959)
    stations = stations.fetch()
    
    all_stations = stations[stations["country"]=="US"].sample(50, random_state=random_state)
    train_stations = all_stations.sample(40, random_state=random_state)
    test_stations = all_stations.drop(train_stations.index)
    
    data = Hourly(all_stations, datetime.datetime(2024, 1, 1), datetime.datetime(2024, 12, 31))
    data = data.fetch()
    data = data.drop(columns=["tsun", "snow", "prcp","wpgt"])
    
    # Data preprocessing
    data = data.copy()
    data.index = data.index.set_levels(pd.to_datetime(data.index.levels[1]), level=1)
    data = data.sort_index(level=["station", "time"])
    
    # Feature engineering
    data = data.sort_index()
    data['pres_change'] = data.groupby(level='station')['pres'].diff()
    data['rhum_change'] = data.groupby(level='station')['rhum'].diff()
    data['wspd_change'] = data.groupby(level='station')['wspd'].diff()
    data['pres_rhum_ratio'] = data['pres'] / (data['rhum'] + 0.01)
    data['wspd_rhum_product'] = data['wspd'] * data['rhum']
    data['pres_wspd_ratio'] = data['pres'] / (data['wspd'] + 0.01)
    data['rhum_squared'] = data['rhum'] ** 2
    data['wspd_squared'] = data['wspd'] ** 2
    data['pres_deviation'] = np.abs(data['pres'] - data['pres'].mean())
    data['rhum_extreme'] = np.where(data['rhum'].fillna(0) > 90, 1, 0)
    data['low_pressure'] = np.where(data['pres'].fillna(data['pres'].mean()) < data['pres'].quantile(0.1), 1, 0)
    data['high_wind'] = np.where(data['wspd'].fillna(0) > data['wspd'].quantile(0.9), 1, 0)
    data['pressure_trend'] = data.groupby(level='station')['pres'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    data['humidity_volatility'] = data.groupby(level='station')['rhum'].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    data['wind_consistency'] = data.groupby(level='station')['wspd'].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    data['temp_change'] = data.groupby(level='station')['temp'].diff()
    data['dwpt_change'] = data.groupby(level='station')['dwpt'].diff()
    data['temp_dwpt_spread'] = data['temp'] - data['dwpt']
    data['heat_index'] = data['temp'] + 0.5 * (data['rhum'] / 100) * (data['temp'] - 14.5)
    data['temp_extreme_hot'] = np.where(data['temp'].fillna(data['temp'].mean()) > data['temp'].quantile(0.9), 1, 0)
    data['temp_extreme_cold'] = np.where(data['temp'].fillna(data['temp'].mean()) < data['temp'].quantile(0.1), 1, 0)
    data['temp_pres_ratio'] = data['temp'] / (data['pres'] / 1000 + 0.01)
    data['temp_wspd_product'] = data['temp'] * data['wspd']
    data['dwpt_rhum_ratio'] = data['dwpt'] / (data['rhum'] + 0.01)
    data['temp_volatility'] = data.groupby(level='station')['temp'].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    data['dwpt_trend'] = data.groupby(level='station')['dwpt'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    data['temp_trend'] = data.groupby(level='station')['temp'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    data['comfort_index'] = 0
    data['frost_risk'] = 0
    data['condensation_risk'] = 0
    
    temp_valid = data['temp'].notna()
    dwpt_valid = data['dwpt'].notna() 
    rhum_valid = data['rhum'].notna()
    
    comfort_mask = temp_valid & rhum_valid & (data['temp'] >= 18) & (data['temp'] <= 24) & (data['rhum'] >= 30) & (data['rhum'] <= 70)
    data.loc[comfort_mask, 'comfort_index'] = 1
    
    frost_mask = temp_valid & dwpt_valid & (data['temp'] <= 2) & (data['dwpt'] <= 0)
    data.loc[frost_mask, 'frost_risk'] = 1
    
    condensation_mask = temp_valid & dwpt_valid & (np.abs(data['temp'] - data['dwpt']) <= 2)
    data.loc[condensation_mask, 'condensation_risk'] = 1
    
    # Lag features
    cols = list(data.columns)
    cols.remove("coco")
    
    for lag in [1, 2, 3]:
        lagged = data.groupby(level="station")[cols].shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in cols]
        data = data.join(lagged)
    
    data = data.dropna()
    
    # Merge coordinates
    data_reset = data.reset_index()
    data_with_coords = data_reset.merge(
        all_stations[['latitude', 'longitude', 'elevation']],
        left_on='station',
        right_index=True,
        how='left'
    )
    data = data_with_coords.set_index(['station', 'time'])
    
    # Enhanced features
    enhanced_data = data.copy()
    time_data = enhanced_data.index.get_level_values('time')
    
    enhanced_data['hour'] = time_data.hour
    enhanced_data['hour_sin'] = np.sin(2 * np.pi * enhanced_data['hour'] / 24)
    enhanced_data['hour_cos'] = np.cos(2 * np.pi * enhanced_data['hour'] / 24)
    enhanced_data['day_of_year'] = time_data.dayofyear
    enhanced_data['season_sin'] = np.sin(2 * np.pi * enhanced_data['day_of_year'] / 365.25)
    enhanced_data['season_cos'] = np.cos(2 * np.pi * enhanced_data['day_of_year'] / 365.25)
    enhanced_data['month'] = time_data.month
    enhanced_data['is_winter'] = ((enhanced_data['month'] == 12) | (enhanced_data['month'] <= 2)).astype(int)
    enhanced_data['is_spring'] = ((enhanced_data['month'] >= 3) & (enhanced_data['month'] <= 5)).astype(int) 
    enhanced_data['is_summer'] = ((enhanced_data['month'] >= 6) & (enhanced_data['month'] <= 8)).astype(int)
    enhanced_data['is_fall'] = ((enhanced_data['month'] >= 9) & (enhanced_data['month'] <= 11)).astype(int)
    
    def saturation_vapor_pressure(temp_celsius):
        return 0.6112 * np.exp((17.67 * temp_celsius) / (temp_celsius + 243.5))
    
    enhanced_data['sat_vapor_press'] = saturation_vapor_pressure(enhanced_data['temp'])
    enhanced_data['actual_vapor_press'] = enhanced_data['sat_vapor_press'] * enhanced_data['rhum'] / 100
    enhanced_data['vapor_pressure_deficit'] = enhanced_data['sat_vapor_press'] - enhanced_data['actual_vapor_press']
    enhanced_data['wet_bulb_temp'] = enhanced_data['temp'] * np.arctan(0.151977 * (enhanced_data['rhum'] + 8.313659) ** 0.5) + np.arctan(enhanced_data['temp'] + enhanced_data['rhum']) - np.arctan(enhanced_data['rhum'] - 1.676331) + 0.00391838 * (enhanced_data['rhum'] ** 1.5) * np.arctan(0.023101 * enhanced_data['rhum']) - 4.686035
    
    def wind_chill(temp, wind_speed):
        mask = temp <= 10
        wc = np.full_like(temp, temp)
        wc[mask] = 13.12 + 0.6215 * temp[mask] - 11.37 * (wind_speed[mask] ** 0.16) + 0.3965 * temp[mask] * (wind_speed[mask] ** 0.16)
        return wc
    
    enhanced_data['wind_chill'] = wind_chill(enhanced_data['temp'], enhanced_data['wspd'])
    
    def heat_index(temp, humidity):
        mask = temp >= 26
        hi = np.full_like(temp, temp)
        t = temp[mask]
        h = humidity[mask]
        hi[mask] = -42.379 + 2.04901523*t + 10.14333127*h - 0.22475541*t*h - 0.00683783*t*t - 0.05481717*h*h + 0.00122874*t*t*h + 0.00085282*t*h*h - 0.00000199*t*t*h*h
        return hi
    
    enhanced_data['heat_index_advanced'] = heat_index(enhanced_data['temp'], enhanced_data['rhum'])
    enhanced_data['mixing_ratio'] = 0.622 * enhanced_data['actual_vapor_press'] / (enhanced_data['pres'] - enhanced_data['actual_vapor_press'])
    enhanced_data['potential_temp'] = enhanced_data['temp'] * (1000 / enhanced_data['pres']) ** 0.286
    
    def solar_elevation(lat, lon, datetime_utc):
        lat = np.array(lat)
        lon = np.array(lon) 
        day_of_year = datetime_utc.dayofyear
        hour_angle = 15 * (datetime_utc.hour - 12) + lon
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        lat_rad = np.radians(lat)
        decl_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)
        elevation = np.arcsin(np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad))
        return np.degrees(elevation)
    
    solar_elevations = []
    for idx in enhanced_data.index:
        station_id, timestamp = idx
        lat = enhanced_data.loc[idx, 'latitude']
        lon = enhanced_data.loc[idx, 'longitude'] 
        elevation = solar_elevation(lat, lon, timestamp)
        solar_elevations.append(elevation)
    
    enhanced_data['solar_elevation'] = solar_elevations
    enhanced_data['is_daylight'] = (enhanced_data['solar_elevation'] > 0).astype(int)
    enhanced_data['solar_intensity'] = np.maximum(0, np.sin(np.radians(enhanced_data['solar_elevation'])))
    
    def beaufort_scale(wind_speed):
        conditions = [
            wind_speed < 1, wind_speed < 5.5, wind_speed < 11.9, wind_speed < 19.3,
            wind_speed < 28.7, wind_speed < 38.9, wind_speed < 49.9, wind_speed < 61.8,
            wind_speed < 74.6, wind_speed < 88.1, wind_speed < 102.4, wind_speed < 117.4,
        ]
        choices = list(range(12))
        return np.select(conditions, choices, default=12)
    
    enhanced_data['beaufort_scale'] = beaufort_scale(enhanced_data['wspd'])
    enhanced_data['wind_power'] = enhanced_data['wspd'] ** 3
    
    # Remove storm conditions and group weather codes
    enhanced_data = enhanced_data[~enhanced_data['coco'].isin([23, 24, 25, 26, 27])]
    
    def group_condition_codes(code):
        if code in [1, 2]:
            return "Clear"
        elif code in [3, 4, 5, 6]:
            return "Cloudy"
        elif code in [7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20]:
            return "Rain"
        elif code in [14, 15, 16, 21, 22]:
            return "Snow"
    
    enhanced_data["coco"] = enhanced_data['coco'].apply(group_condition_codes)
    
    # Model data preparation
    model_data = enhanced_data.copy()
    train_data = model_data[model_data.index.get_level_values('station').isin(train_stations.index)]
    test_data = model_data[model_data.index.get_level_values('station').isin(test_stations.index)]
    
    X_train = train_data.drop('coco', axis=1)
    y_train = train_data['coco']
    X_test = test_data.drop('coco', axis=1)
    y_test = test_data['coco']
    
    # Model training
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    direct_weights = {
        'Clear': 1.5,
        'Cloudy': 3.0, 
        'Rain': 20.0,
        'Snow': 325.0
    }
    
    class_weight_dict = {}
    for i, class_name in enumerate(label_encoder.classes_):
        class_weight_dict[i] = direct_weights[class_name]
    
    model = XGBClassifier(
        n_estimators=300, 
        learning_rate=0.1, 
        max_depth=None, 
        random_state=random_state, 
        n_jobs=1, 
        eval_metric='mlogloss'
    )
    
    sample_weights = np.array([class_weight_dict[y] for y in y_train_encoded])
    model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate requested visualization
    if visual_type == "station_map":
        us_states = alt.topo_feature(data.us_10m.url, 'states')
        base = (
            alt.Chart(us_states)
            .mark_geoshape(fill="#f0f0f0", stroke="black")
            .project("albersUsa")
        )
        
        points = (
            alt.Chart(all_stations.reset_index())
            .mark_circle(size=200, opacity=0.8)
            .encode(
                longitude="longitude:Q",
                latitude="latitude:Q",
                color="name:N",
                tooltip=["name:N","id:N" , "longitude:Q", "latitude:Q"]
            )
        )
        return (base + points).properties(
            width=600,
            height=400,
            title="50 Weather Stations (40 Train + 10 Test)"
        )
    
    elif visual_type == "feature_importance":
        xgb_importance = pd.Series(model.feature_importances_, index=X_train.columns)
        xgb_importance = xgb_importance.sort_values(ascending=False)
        
        importance_df = pd.DataFrame({
            'feature': xgb_importance.head(15).index,
            'importance': xgb_importance.head(15).values
        })
        
        return alt.Chart(importance_df).mark_bar().encode(
            x=alt.X('importance:Q', title='Feature Importance'),
            y=alt.Y('feature:N', sort='-x', title='Feature'),
            color=alt.Color('importance:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['feature:N', alt.Tooltip('importance:Q', format='.4f')]
        ).properties(
            width=600,
            height=400,
            title='Top 15 Features - XGBoost Importance'
        )
    
    elif visual_type == "confusion_matrix":
        cm = confusion_matrix(y_test, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        cm_data = []
        for i, true_label in enumerate(label_encoder.classes_):
            for j, pred_label in enumerate(label_encoder.classes_):
                cm_data.append({
                    'True Label': true_label,
                    'Predicted Label': pred_label,
                    'Percentage': cm_percent[i, j],
                    'Text': f'{cm_percent[i, j]:.1f}%'
                })
        
        cm_df = pd.DataFrame(cm_data)
        
        heatmap = alt.Chart(cm_df).mark_rect().encode(
            x=alt.X('Predicted Label:N', title='Predicted Label'),
            y=alt.Y('True Label:N', title='True Label'),
            color=alt.Color('Percentage:Q', 
                           scale=alt.Scale(scheme='blues', domain=[0, 100]),
                           title='Percentage (%)'),
            tooltip=['True Label:N', 'Predicted Label:N', alt.Tooltip('Percentage:Q', format='.1f')]
        ).properties(
            width=400,
            height=400,
            title=f'XGBoost Confusion Matrix (Row Percentages) - Accuracy: {accuracy:.2%}'
        )
        
        text = alt.Chart(cm_df).mark_text(
            align='center',
            baseline='middle',
            fontSize=12,
            fontWeight='bold'
        ).encode(
            x='Predicted Label:N',
            y='True Label:N',
            text='Text:N',
            color=alt.condition(alt.datum.Percentage > 50, alt.value('white'), alt.value('black'))
        )
        
        return heatmap + text
    
    elif visual_type == "geographic_performance":
        test_data_with_stations = test_data.reset_index()
        test_predictions = model.predict(X_test)
        test_predictions = label_encoder.inverse_transform(test_predictions)
        
        city_performance = []
        test_station_names = test_stations.index.tolist()
        
        for station_id in test_station_names:
            station_mask = test_data_with_stations['station'] == station_id
            station_indices = test_data_with_stations[station_mask].index
            valid_indices = [i for i in station_indices if i < len(y_test)]
            
            if len(valid_indices) > 10:
                station_y_true = y_test.iloc[valid_indices]
                station_y_pred = test_predictions[valid_indices]
                city_accuracy = accuracy_score(station_y_true, station_y_pred)
                
                city_performance.append({
                    'station_id': station_id,
                    'name': test_stations.loc[station_id, 'name'],
                    'longitude': test_stations.loc[station_id, 'longitude'],
                    'latitude': test_stations.loc[station_id, 'latitude'],
                    'accuracy': city_accuracy,
                    'samples': len(valid_indices)
                })
        
        city_perf_df = pd.DataFrame(city_performance)
        
        us_states = alt.topo_feature(vega_datasets.us_10m.url, 'states')
        base_map = (
            alt.Chart(us_states)
            .mark_geoshape(fill="#f0f0f0", stroke="black")
            .project("albersUsa")
        )
        
        train_points = (
            alt.Chart(train_stations.reset_index())
            .mark_circle(size=100, opacity=0.6)
            .encode(
                longitude="longitude:Q",
                latitude="latitude:Q",
                color=alt.value("gray"),
                tooltip=["name:N", "id:N"]
            )
        )
        
        test_points = (
            alt.Chart(city_perf_df)
            .mark_circle(size=300, opacity=0.9, stroke="black", strokeWidth=2)
            .encode(
                longitude="longitude:Q",
                latitude="latitude:Q",
                color=alt.Color("accuracy:Q", 
                               scale=alt.Scale(scheme="viridis", domain=[0.4, 1.0]),
                               title="XGBoost Accuracy"),
                tooltip=[
                    alt.Tooltip("name:N", title="City"),
                    alt.Tooltip("accuracy:Q", title="XGBoost Accuracy", format=".3f"),
                    alt.Tooltip("samples:Q", title="Test Samples"),
                    alt.Tooltip("longitude:Q", title="Longitude", format=".2f"),
                    alt.Tooltip("latitude:Q", title="Latitude", format=".2f")
                ]
            )
        )
        
        return (base_map + train_points + test_points).properties(
            width=800,
            height=500,
            title="XGBoost Model Performance: 40 Training Cities (Gray) + 10 Test Cities (Colored by Accuracy)"
        ).resolve_scale(
            color='independent'
        )
    
    else:
        raise ValueError(f"Unknown visual_type: {visual_type}. Choose from: 'confusion_matrix', 'feature_importance', 'geographic_performance', 'station_map'")