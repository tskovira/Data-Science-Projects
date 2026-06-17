import pandas as pd
import numpy as np
from meteostat import Point, Hourly
import datetime
from great_tables import GT, md, style, loc

import altair as alt
from vega_datasets import data

cities_by_region = {
    'Northeast': {
        'New York, NY': (40.7128, -74.0060),
        'Boston, MA': (42.3601, -71.0589),
        'Philadelphia, PA': (39.9526, -75.1652),
        'Pittsburgh, PA': (40.4406, -79.9959),
        'Buffalo, NY': (42.8864, -78.8784)
    },
    'Southeast': {
        'Miami, FL': (25.7617, -80.1918),
        'Atlanta, GA': (33.7490, -84.3880),
        'Charlotte, NC': (35.2271, -80.8431),
        'Jacksonville, FL': (30.3322, -81.6557),
        'Tampa, FL': (27.9506, -82.4572)
    },
    'Midwest': {
        'Chicago, IL': (41.8781, -87.6298),
        'Detroit, MI': (42.3314, -83.0458),
        'Minneapolis, MN': (44.9778, -93.2650),
        'Cleveland, OH': (41.4993, -81.6944),
        'Milwaukee, WI': (43.0389, -87.9065)
    },
    'West': {
        'Los Angeles, CA': (34.0522, -118.2437),
        'San Francisco, CA': (37.7749, -122.4194),
        'Seattle, WA': (47.6062, -122.3321),
        'Portland, OR': (45.5152, -122.6784),
        'Denver, CO': (39.7392, -104.9903)
    }
}

def wind_stats(city_name, lat, lon, start_year=2020, end_year=2024):
    try:
        point = Point(lat, lon)
        start = datetime.datetime(start_year, 1, 1)
        end = datetime.datetime(end_year, 12, 31)
        
        data = Hourly(point, start, end).fetch()
            
        # Calculate averaged stats
        wind_speed_avg = data['wspd'].mean() if 'wspd' in data.columns else np.nan
        # Convert to radians then back to degrees to avoid circular mean issues
        if 'wdir' in data.columns:
            wind_directions = data['wdir'].dropna()
            if len(wind_directions) > 0:
                directions_rad = np.radians(wind_directions)
                
                sin_mean = np.mean(np.sin(directions_rad))
                cos_mean = np.mean(np.cos(directions_rad))
                
                wind_dir_avg = np.degrees(np.arctan2(sin_mean, cos_mean))
                
                # Ensure result is between 0 and 360
                if wind_dir_avg < 0:
                    wind_dir_avg += 360
        else:
            wind_dir_avg = np.nan
        
        return {
            'city_name': city_name,
            'latitude': lat,
            'longitude': lon,
            'avg_wind_speed_kmh': wind_speed_avg,
            'avg_wind_direction_deg': wind_dir_avg,
        }
        
    except Exception as e:
        return {
            'city_name': city_name,
            'latitude': lat,
            'longitude': lon,
            'avg_wind_speed_kmh': np.nan,
            'avg_wind_direction_deg': np.nan,
        }
    
def create_wind_table():
    def wind_stats_local(city_name, lat, lon, start_year=2020, end_year=2024):
        try:
            point = Point(lat, lon)
            start = datetime.datetime(start_year, 1, 1)
            end = datetime.datetime(end_year, 12, 31)
            data = Hourly(point, start, end).fetch()

            wind_speed_avg = data['wspd'].mean() if 'wspd' in data.columns else np.nan

            if 'wdir' in data.columns:
                wind_directions = data['wdir'].dropna()
                if len(wind_directions) > 0:
                    directions_rad = np.radians(wind_directions)
                    sin_mean = np.mean(np.sin(directions_rad))
                    cos_mean = np.mean(np.cos(directions_rad))
                    wind_dir_avg = np.degrees(np.arctan2(sin_mean, cos_mean))
                    if wind_dir_avg < 0:
                        wind_dir_avg += 360
                else:
                    wind_dir_avg = np.nan
            else:
                wind_dir_avg = np.nan

            return {
                'city_name': city_name,
                'latitude': lat,
                'longitude': lon,
                'avg_wind_speed_kmh': wind_speed_avg,
                'avg_wind_direction_deg': wind_dir_avg,
            }
        except Exception:
            return {
                'city_name': city_name,
                'latitude': lat,
                'longitude': lon,
                'avg_wind_speed_kmh': np.nan,
                'avg_wind_direction_deg': np.nan,
            }

    results = []
    for region, cities in cities_by_region.items():
        for city_name, (lat, lon) in cities.items():
            r = wind_stats_local(city_name, lat, lon)
            r['region'] = region
            results.append(r)

    wind_df_local = pd.DataFrame(results)

    def get_direction_color(direction_deg):
        if pd.isna(direction_deg):
            return "#ffffff"
        direction = float(direction_deg) % 360
        cardinals = {0: (0.2, 0.4, 0.9), 90: (0.9, 0.3, 0.3), 180: (0.9, 0.9, 0.2), 270: (0.3, 0.8, 0.3)}
        cardinal_angles = [0, 90, 180, 270]
        distances = []
        for cardinal in cardinal_angles:
            if cardinal == 0:
                dist = min(direction, 360 - direction)
            else:
                dist = abs(direction - cardinal)
            distances.append(dist)
        min_distance = min(distances)
        closest_cardinal = cardinal_angles[distances.index(min_distance)]
        if min_distance == 45:
            return "#ffffff"
        max_distance = 45
        intensity = 1.0 - (min_distance / max_distance)
        min_intensity = 0.1
        intensity = min_intensity + intensity * (1.0 - min_intensity)
        cardinal_color = cardinals[closest_cardinal]
        white = (1.0, 1.0, 1.0)
        final_color = tuple(intensity * cardinal_color[i] + (1 - intensity) * white[i] for i in range(3))
        r, g, b = [int(c * 255) for c in final_color]
        return f"#{r:02x}{g:02x}{b:02x}"

    # Prepare table data
    table_data_sorted = wind_df_local[[
        'city_name', 'region', 'avg_wind_speed_kmh', 'avg_wind_direction_deg'
    ]].copy()
    table_data_sorted['avg_wind_speed_kmh'] = table_data_sorted['avg_wind_speed_kmh'].round(1)
    table_data_sorted['avg_wind_direction_deg'] = table_data_sorted['avg_wind_direction_deg'].round(0)
    table_data_sorted = table_data_sorted.sort_values(['region', 'avg_wind_speed_kmh'], ascending=[True, False]).reset_index(drop=True)
    table_data_sorted['dir_color'] = table_data_sorted['avg_wind_direction_deg'].apply(get_direction_color)

    min_speed = table_data_sorted['avg_wind_speed_kmh'].min()
    max_speed = table_data_sorted['avg_wind_speed_kmh'].max()

    # Build the table
    table = (
        GT(table_data_sorted.drop(columns=['dir_color']))
        .tab_header(
            title=md("**Regional Wind Analysis by Wind Speed and Direction**"),
            subtitle=md("20 Major Cities on 5-Year Hourly Averages (2020-2024)")
        )
        .cols_label(
            city_name="City",
            avg_wind_speed_kmh="Speed (km/h)",
            avg_wind_direction_deg="Direction (°)",
        )
        .tab_spanner(
            label="Wind Statistics",
            columns=["avg_wind_speed_kmh", "avg_wind_direction_deg"]
        )
        .tab_stub(
            rowname_col="city_name",
            groupname_col="region",
        )
        .data_color(
            columns=['avg_wind_speed_kmh'],
            palette=["#f8f9fa", "#e9ecef", "#dee2e6", "#ced4da", "#adb5bd", "#6c757d"],
            domain=[min_speed, max_speed],
        )
    )

    for idx, row in table_data_sorted.iterrows():
        table = table.tab_style(
            style=style.fill(color=get_direction_color(row['avg_wind_direction_deg'])),
            locations=loc.body(columns=['avg_wind_direction_deg'], rows=[idx])
        )

    table = (
        table
        .cols_align(
            align='center',
            columns=['avg_wind_speed_kmh', 'avg_wind_direction_deg']
        )
        .tab_source_note(
            source_note=md("**Data:** Meteostat | **Speed:** Stronger→Darker | **Direction:** 🔵**North** 🔴**East** 🟡**South** 🟢**West**  Closer to single direction→Darker color")
        )
    )

    globals()['wind_df'] = wind_df_local

    return table



def get_tornado_data():
    tornados_df = pd.read_csv("Tornado_Tracks_1950_2017_1_7964592706304725094.csv")
    tornados_df = tornados_df[tornados_df["Year"]==2024]


    cat_4 =  tornados_df[tornados_df["Magnitude"]==4]
    cat_3 = tornados_df[tornados_df["Magnitude"]==3].sample(n=5,random_state=42)
    cat_2 = tornados_df[tornados_df["Magnitude"]==2].sample(n=5,random_state=42)
    cat_1 = tornados_df[tornados_df["Magnitude"]==1].sample(n=5,random_state=42)
    cat_0 = tornados_df[tornados_df["Magnitude"]==0].sample(n=5,random_state=42)

    case_study= pd.concat([cat_4, cat_3, cat_2, cat_1, cat_0])
    #print(case_study.columns)
    case_study[["State Abbreviation", "Tornado Number", "Year","Month","Day", "Time","Magnitude", "Starting Latitude",	"Starting Longitude",	"Ending Latitude",	"Ending Longitude"]]




    def get_tornado_weekly_wind(tornado_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch hourly wind data (speed & direction) for a week surrounding each tornado event.
        
        Args:
            tornado_df (pd.DataFrame): DataFrame with columns:
                ["State Abbreviation", "Tornado Number", "Year", "Month", "Day", "Time",
                "Magnitude", "Starting Latitude", "Starting Longitude", 
                "Ending Latitude", "Ending Longitude"]
                
        Returns:
            pd.DataFrame: Combined DataFrame with columns:
                ['tornado_number', 'state', 'magnitude', 'lat', 'lon', 
                'time', 'wspd', 'wdir']
        """

        all_data = []

        for idx, row in tornado_df.iterrows():
            try:
                # Parse tornado info
                tornado_num = row["Tornado Number"]
                state = row["State Abbreviation"]
                magnitude = row["Magnitude"]
                lat = row["Starting Latitude"]
                lon = row["Starting Longitude"]
                
                # Parse date and time (safe fallback)
                try:
                    tornado_time = datetime.datetime(
                        int(row["Year"]),
                        int(row["Month"]),
                        int(row["Day"])
                    )
                except Exception:
                    #print(f"⚠️ Skipping tornado {tornado_num}: Invalid date")
                    continue

                # Define ±3 days around event
                start = tornado_time - datetime.timedelta(days=3)
                end = tornado_time + datetime.timedelta(days=3)

                #print(f"Fetching wind for Tornado {tornado_num} ({state}) from {start.date()} to {end.date()}")

                # Create a Meteostat point and fetch hourly data
                point = Point(lat, lon)
                data = Hourly(point, start, end).fetch()

                if not data.empty:
                    data = data.reset_index()
                    data["tornado_number"] = tornado_num
                    data["state"] = state
                    data["magnitude"] = magnitude
                    data["lat"] = lat
                    data["lon"] = lon
                    data["time_delta"] = (data["time"] - tornado_time).dt.total_seconds() / (3600) # gets time delta in hours
                    # Keep only relevant columns
                    data = data[[
                        "tornado_number", "state", "magnitude", "lat", "lon",
                        "time","time_delta", "wspd", "wdir"
                    ]]
                    all_data.append(data)
                #else:
                    
                    #print(f"⚠️ No wind data found for Tornado {tornado_num} ({state})")

            except Exception as e:
                #print(f"❌ Error fetching Tornado {tornado_num}: {e}")
                continue

        if not all_data:
            #print("No data retrieved for any tornado events.")
            return pd.DataFrame(columns=[
                "tornado_number", "state", "magnitude", "lat", "lon", "time", "wspd", "wdir"
            ])

        result = pd.concat(all_data, ignore_index=True)
        result["day_of_year"] = result["time"].dt.day_of_year
        
        return result
    tornado = get_tornado_weekly_wind(case_study)
    return tornado


def create_wind_table():
    """
    Build a GreatTable summarizing average wind speed (mph) and wind direction
    for every city in cities_by_region.
    """

    results = []
    start = datetime.datetime(2024, 1, 1)
    end = datetime.datetime(2024, 12, 31)

    for region, cities in cities_by_region.items():
        for city_name, (lat, lon) in cities.items():
            try:
                point = Point(lat, lon)
                data = Hourly(point, start, end).fetch()

                # --- Wind Speed (m/s → mph) ---
                if "wspd" in data.columns:
                    wind_speed_avg = data["wspd"].mean() * 2.236936   # mph
                else:
                    wind_speed_avg = np.nan

                # --- Wind Direction (circular mean) ---
                if "wdir" in data.columns and len(data["wdir"].dropna()) > 0:
                    dirs = np.radians(data["wdir"].dropna())
                    wind_dir_avg = np.degrees(
                        np.arctan2(np.mean(np.sin(dirs)), np.mean(np.cos(dirs)))
                    )
                    if wind_dir_avg < 0:
                        wind_dir_avg += 360
                else:
                    wind_dir_avg = np.nan

                results.append({
                    "region": region,
                    "city_name": city_name,
                    "latitude": lat,
                    "longitude": lon,
                    "avg_wind_speed_mph": wind_speed_avg,
                    "avg_wind_direction_deg": wind_dir_avg
                })

            except Exception:
                results.append({
                    "region": region,
                    "city_name": city_name,
                    "latitude": lat,
                    "longitude": lon,
                    "avg_wind_speed_mph": np.nan,
                    "avg_wind_direction_deg": np.nan
                })

    # --- Build DataFrame ---
    df = pd.DataFrame(results)
    df["avg_wind_speed_mph"] = df["avg_wind_speed_mph"].round(1)
    df["avg_wind_direction_deg"] = df["avg_wind_direction_deg"].round(0)

    df_sorted = df.sort_values(
        ["region", "avg_wind_speed_mph"],
        ascending=[True, False]
    ).reset_index(drop=True)

    min_speed = df_sorted["avg_wind_speed_mph"].min()
    max_speed = df_sorted["avg_wind_speed_mph"].max()

    # --- Direction → color mapping ---
    def get_dir_color(deg):
        if pd.isna(deg):
            return "#ffffff"
        deg = float(deg) % 360
        if deg >= 315 or deg < 45:
            return "#007bff"   # North
        if 45 <= deg < 135:
            return "#dc3545"   # East
        if 135 <= deg < 225:
            return "#ffc107"   # South
        return "#28a745"       # West

    # --- Build GreatTable ---
    table = (
        GT(df_sorted)
        .tab_header(
            title=md("**Regional Wind Analysis by Speed and Direction**"),
            subtitle=md("Hourly Averages 2024 | Data: Meteostat")
        )
        .cols_label(
            city_name="City",
            avg_wind_speed_mph="Speed (mph)",
            avg_wind_direction_deg="Direction (°)"
        )
        .tab_spanner(
            label="Wind Statistics",
            columns=["avg_wind_speed_mph", "avg_wind_direction_deg"]
        )
        .tab_stub(
            rowname_col="city_name",
            groupname_col="region"
        )
        .data_color(
            columns=["avg_wind_speed_mph"],
            palette=["#f8f9fa", "#adb5bd", "#6c757d"],
            domain=[min_speed, max_speed]
        )
        .tab_options(
            table_font_size=11,
            data_row_padding=1,
            heading_title_font_size=16,
            heading_subtitle_font_size=12,
            row_group_font_size=10,
            source_notes_font_size=9        
        )
        .cols_align(
            align="center",
            columns=["avg_wind_speed_mph", "avg_wind_direction_deg"]
        )
        .tab_source_note(
            source_note=md("**Legend:** 🔵North 🔴East 🟡South 🟢West | Darker = Stronger")
        )
    )

    # --- Apply direction color styling ---
    for i, row in df_sorted.iterrows():
        table = table.tab_style(
            style=style.fill(color=get_dir_color(row["avg_wind_direction_deg"])),
            locations=loc.body(columns=["avg_wind_direction_deg"], rows=[i])
        )

    globals()["wind_df"] = df_sorted
    return table

def get_regional_daily_wind():
    """
    Fetch daily average wind speed (mph) for all cities in cities_by_region,
    then compute regional averages for each day of 2024.

    Returns:
        pandas.DataFrame with columns: ['date', 'Region', 'wspd_mph']
    """

    from meteostat import Daily
    records = []

    start = datetime.datetime(2024, 1, 1)
    end = datetime.datetime(2024, 12, 31)

    for region, cities in cities_by_region.items():
        for city, (lat, lon) in cities.items():

            point = Point(lat, lon)
            df1 = Daily(point, start, end).fetch()

            # Ensure wspd_mph always exists, even if station lacks wspd
            if not df1.empty:
                if "wspd" in df1.columns:
                    # km/h → mph
                    df1["wspd_mph"] = df1["wspd"] * 0.621371
                else:
                    # Station lacks wind data
                    df1["wspd_mph"] = np.nan

                df1["date"] = df1.index.date

                daily_avg = df1[["date", "wspd_mph"]].reset_index(drop=True)
                daily_avg["Region"] = region
                daily_avg["City"] = city
                records.append(daily_avg)

            else:
                # Station returned no data at all
                blank = pd.DataFrame({
                    "date": pd.date_range(start, end).date,
                    "wspd_mph": np.nan,
                    "Region": region,
                    "City": city
                })
                records.append(blank)

    # If nothing returned (unlikely)
    if not records:
        return pd.DataFrame(columns=["date", "Region", "wspd_mph"])

    wind_all = pd.concat(records, ignore_index=True)

    # Regional daily averages
    regional_daily = (
        wind_all.groupby(["date", "Region"])["wspd_mph"]
        .mean()
        .reset_index()
    )

    regional_daily["date"] = pd.to_datetime(regional_daily["date"])

    return regional_daily


def chart_regional_daily_wind(regional_daily):
    """
    Create the interactive dropdown-based line chart for
    average daily wind speed by region.

    Args:
        regional_daily (DataFrame): Output of get_regional_daily_wind()

    Returns:
        alt.Chart: Fully interactive line chart
    """

    # Dropdown selection
    region_dropdown = alt.binding_select(
        options=sorted(regional_daily["Region"].unique().tolist()),
        name="Select Region:  "
    )
    region_selection = alt.selection_point(
        fields=["Region"], 
        bind=region_dropdown
    )

    # Dynamic styling
    color_condition = (
        alt.when(region_selection)
        .then(alt.Color("Region:N"))
        .otherwise(alt.value("lightgray"))
    )

    opacity_condition = (
        alt.when(region_selection)
        .then(alt.value(1))
        .otherwise(alt.value(0.4))
    )

    # Chart
    chart = (
        alt.Chart(regional_daily)
        .mark_line(point=False)
        .encode(
            x=alt.X("date:T", title="Month"),
            y=alt.Y("wspd_mph:Q", title="Avg Daily Wind Speed (mph)"),
            color=color_condition,
            opacity=opacity_condition,
            tooltip=["Region", "wspd_mph"]
        )
        .add_params(region_selection)
        .properties(
            title="Average Daily Wind Speed by Region",
            width=800,
            height=200
        )
        .interactive()
    )

    return chart



def get_geo_feature_temperature():
    """
    Fetch daily average temperature for selected geographic-feature cities in 2024.

    Returns:
        DataFrame with columns: ['date', 'Feature', 'temp_f']
    """

    geo_cities = {
        "Great Lakes (Chicago, IL)": (41.8781, -87.6298),
        "Coastal (Miami, FL)": (25.7617, -80.1918),
        "Mountains (Denver, CO)": (39.7392, -104.9903),
        "Desert (Phoenix, AZ)": (33.4484, -112.0740),
        "Volcanic Islands (Honolulu, HI)": (21.3069, -157.8583),
        "Appalachian (Pittsburgh, PA)": (40.4406, -79.9959)
    }

    geo_records = []
    start = datetime.datetime(2024, 1, 1)
    end = datetime.datetime(2024, 12, 31)

    for feature, (lat, lon) in geo_cities.items():
        point = Point(lat, lon)
        df = Hourly(point, start, end).fetch()

        if not df.empty:

            # Temperature safety check
            if "temp" in df.columns:
                df["temp_f"] = df["temp"] * 9/5 + 32   # °C → °F
            else:
                df["temp_f"] = np.nan

            df["date"] = df.index.date

            daily_avg = (
                df.groupby("date")["temp_f"]
                .mean()
                .reset_index()
            )
            daily_avg["Feature"] = feature
            geo_records.append(daily_avg)

        else:
            # No data returned → create blank rows
            blank = pd.DataFrame({
                "date": pd.date_range(start, end).date,
                "temp_f": np.nan,
                "Feature": feature,
            })
            geo_records.append(blank)

    if not geo_records:
        return pd.DataFrame(columns=["date", "Feature", "temp_f"])

    geo_all = pd.concat(geo_records, ignore_index=True)

    geo_summary = (
        geo_all.groupby(["date", "Feature"])["temp_f"]
        .mean()
        .reset_index()
    )

    geo_summary["date"] = pd.to_datetime(geo_summary["date"])

    return geo_summary


def chart_geo_feature_temperature(geo_summary):
    """
    Build the temperature-only chart by geographic feature.

    Args:
        geo_summary (DataFrame): Output from get_geo_feature_temperature()

    Returns:
        alt.Chart: Temperature line chart
    """

    temp_chart = (
        alt.Chart(geo_summary.dropna(subset=["temp_f"]))
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Month"),
            y=alt.Y("temp_f:Q", title="Avg Temperature (°F)"),
            color=alt.Color("Feature:N", legend=alt.Legend(title="Geographic Feature")),
            tooltip=["Feature", "temp_f"]
        )
        .properties(
            title="Average Daily Temperature by Geographic Feature",
            width=800,
            height=200
        )
        .interactive()
    )

    return temp_chart


















def generate_tornado_map(tornado):
    alt.data_transformers.enable("vegafusion")
    # ['tornado_number', 'state', 'magnitude', 'lat', 'lon', 'time', 'wspd', 'wdir']

    # --------------------------------
    # Tornado Locations (unique per tornado)
    # --------------------------------
    df = (
        tornado[['tornado_number', 'state', 'magnitude', 'lat', 'lon']]
        .drop_duplicates(subset=['tornado_number'])
    )

    # --------------------------------
    # HOURLY WIND DATA
    # --------------------------------
    # No grouping, keep each hourly record
    hourly_wind = tornado[['tornado_number', 'time','time_delta', 'wspd',"magnitude",'lat','lon']].copy()
    hourly_wind = hourly_wind.rename(columns={'wspd': 'wind_speed'})
    hourly_wind['wind_speed'] = hourly_wind['wind_speed'] * 0.621371 # convert to mph    

    # --------------------------------
    # INTERACTIVE SELECTION
    # --------------------------------
    geographic_brush  = alt.selection_point(fields=['tornado_number'])
    
    # U.S. map base layer
    us_states = alt.topo_feature(data.us_10m.url, feature='states')

    base_map = alt.Chart(us_states).mark_geoshape(
        fill="#f0f0f0", stroke='black'
    ).project("albersUsa").properties(width=600, height=300)

    # Tornado starting points
    points = alt.Chart(df).mark_circle(size=200, opacity=0.8).encode(
        longitude='lon:Q',
        latitude='lat:Q',
        color=alt.Color('magnitude:N'),
        tooltip=['tornado_number:N', 'state:N', 'magnitude:Q'],
        opacity=alt.condition(geographic_brush, 
                          alt.value(0.8), alt.value(0.4))
    ).add_params(
        geographic_brush
    )

    map_layer = base_map + points

    # --------------------------------
    # HOURLY LINE CHART (Filtered by Selection)
    # --------------------------------
    
    wind_chart = alt.Chart(hourly_wind).mark_line(point=True).encode(
        x=alt.X('time_delta',
            title='Time from tornado start',
            axis=alt.Axis(labelExpr="floor(datum.value/24) + 'd ' + floor(datum.value%24) + 'h'"),
            scale=alt.Scale(domain=[-3*24, 3*24])
        ),
        y=alt.Y('wind_speed:Q', title='Wind Speed (mph)'),
        color=alt.Color('magnitude:N'),
        detail='tornado_number:N',
        tooltip=['tornado_number:N', 'time:T', 'wind_speed:Q'],
        opacity=alt.value(0.6), 
    ).transform_filter(
        geographic_brush

        ).properties(
        width=650,
        height=300,
        title='Hourly Wind Speed (Filtered by Tornado Selection)'
    )

    # --------------------------------
    # FINAL DASHBOARD
    # --------------------------------
    final_chart = alt.hconcat(map_layer, wind_chart).resolve_scale()

    return final_chart

def get_hourly_wind(cities: dict, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """
    Fetch hourly wind data (speed & direction) for multiple cities.
    
    Args:
        cities (dict): Mapping of city names to (lat, lon) tuples.
        start (datetime): Start date for data.
        end (datetime): End date for data.
    
    Returns:
        pd.DataFrame: Combined DataFrame with columns:
                      ['city', 'time', 'wspd', 'wdir']
    """
    all_data = []

    for city, (lat, lon) in cities.items():
        try:
            #print(f"Fetching: {city} ({lat}, {lon})")
            point = Point(lat, lon)
            data = Hourly(point, start, end).fetch()

            if not data.empty:
                data = data.reset_index()
                data['city'] = city
                # keep only relevant columns
                data = data[['city', 'time', 'wspd', 'wdir']]
                
                all_data.append(data)
            #else:
                #print(f"⚠️ No data for {city}")

        except Exception as e:
            #print(f"❌ Error fetching {city}: {e}")
            continue

    if not all_data:
        #print("No data retrieved for any city.")
        return pd.DataFrame(columns=['city', 'time', 'wspd', 'wdir'])

    return pd.concat(all_data, ignore_index=True)

def generate_cities_map(cities,wind_df):
    alt.data_transformers.enable("vegafusion")
    df = pd.DataFrame([
    {'city': name, 'lat': coords[0], 'lon': coords[1]}
    for name, coords in cities.items()
    ])

    time_range = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 12, 31)]


    daily_avg = (
        wind_df.groupby(['city', 'day_of_year'], as_index=False)['wspd']
        .mean()
        .rename(columns={'wspd': 'wind_speed'})
    )
    daily_avg['wind_speed'] = daily_avg['wind_speed'] * 0.621371 # convert to mph
    #print(daily_avg)
    #print(data)


    brush = alt.selection_point(fields=['city'])

    # World map
    world = alt.topo_feature(data.us_10m.url, feature='states')
    base_map = alt.Chart(world).mark_geoshape(
        fill="#f0f0f0", stroke='black'
    ).project("albersUsa").properties(width=600, height=300)


    # City points with selection
    points = alt.Chart(df).mark_circle(size=200, opacity=0.8).encode(
        longitude='lon:Q',
        latitude='lat:Q',
        color='city:N',
        tooltip=['city:N'],
        opacity=alt.condition(brush, 
                          alt.value(0.8), alt.value(0.4))
    ).add_params(
        brush
    )

    map_layer = base_map + points

    # Wind line chart filtered by selected cities
    wind_chart = alt.Chart(daily_avg).mark_line().encode(
        x=alt.X('day_of_year:Q',scale=alt.Scale(domain=[0,366])),
        y=alt.Y('wind_speed:Q', title='Wind Speed (mph)'),
        color='city:N',
        tooltip=['city:N', 'day_of_year:Q', 'wind_speed:Q'],
        opacity= alt.value(0.6)
    ).transform_filter(
        brush
    ).properties(
        width=650,
        height=300,
        title='Daily Average Wind Speed by City (Filtered by Map Selection)'
    )

    final_chart = alt.hconcat(map_layer, wind_chart)
    return final_chart


def create_complete_dashboard_std():
    """
    Creates a complete wind analysis dashboard with geographic mapping, 
    wind vector visualization, and standard deviation analysis.
    
    Returns:
        altair.VConcatChart: Complete interactive dashboard
    """
    import pandas as pd
    import numpy as np
    import altair as alt
    from meteostat import Point, Hourly
    import datetime

    # Use inline data instead of JSON files for Quarto compatibility
    alt.data_transformers.enable('default')
    alt.data_transformers.disable_max_rows()

    cities_by_region = {
        'Northeast': {
            'New York, NY': (40.7128, -74.0060), 'Boston, MA': (42.3601, -71.0589),
            'Portland, ME': (43.6591, -70.2568), 'Providence, RI': (41.8240, -71.4128),
            'Newark, NJ': (40.7357, -74.1724), 'Buffalo, NY': (42.8864, -78.8784),
            'Rochester, NY': (43.1566, -77.6088), 'Erie, PA': (42.1292, -80.0851),
            'Burlington, VT': (44.4759, -73.2121), 'Manchester, NH': (42.9956, -71.4548),
            'Montpelier, VT': (44.2601, -72.5806), 'Pittsburgh, PA': (40.4406, -79.9959),
            'Hartford, CT': (41.7658, -72.6734), 'Waterbury, CT': (41.5582, -73.0515),
            'Philadelphia, PA': (39.9526, -75.1652), 'Albany, NY': (42.6526, -73.7562),
            'Syracuse, NY': (43.0481, -76.1474), 'Worcester, MA': (42.2626, -71.8023),
            'Bridgeport, CT': (41.1865, -73.1952), 'Lowell, MA': (42.6334, -71.3162)
        },
        'Southeast': {
            'Miami, FL': (25.7617, -80.1918), 'Jacksonville, FL': (30.3322, -81.6557),
            'Tampa, FL': (27.9506, -82.4572), 'Charleston, SC': (32.7765, -79.9311),
                'Savannah, GA': (32.0835, -81.0998), 'Nashville, TN': (36.1627, -86.7816),
            'Knoxville, TN': (35.9606, -83.9207), 'Huntsville, AL': (34.7304, -86.5861),
            'Atlanta, GA': (33.7490, -84.3880), 'Charlotte, NC': (35.2271, -80.8431),
            'Orlando, FL': (28.5383, -81.3792), 'Birmingham, AL': (33.5186, -86.8104),
            'Richmond, VA': (37.5407, -77.4360), 'Raleigh, NC': (35.7796, -78.6382),
            'Greensboro, NC': (36.0726, -79.7920), 'Columbia, SC': (34.0007, -81.0348),
            'Tallahassee, FL': (30.4518, -84.2807), 'Lexington, KY': (38.0406, -84.5037),
            'Mobile, AL': (30.6954, -88.0399), 'Virginia Beach, VA': (36.8529, -75.9780)
        },
        'Midwest': {
            'Chicago, IL': (41.8781, -87.6298), 'Detroit, MI': (42.3314, -83.0458),
            'Cleveland, OH': (41.4993, -81.6944), 'Milwaukee, WI': (43.0389, -87.9065),
            'Grand Rapids, MI': (42.9634, -85.6681), 'Toledo, OH': (41.6528, -83.5379),
            'Kansas City, MO': (39.0997, -94.5786), 'Omaha, NE': (41.2565, -95.9345),
            'Des Moines, IA': (41.5868, -93.6250), 'Wichita, KS': (37.6872, -97.3301),
            'Oklahoma City, OK': (35.4676, -97.5164), 'Tulsa, OK': (36.1540, -95.9928),
            'St. Louis, MO': (38.6270, -90.1994), 'Louisville, KY': (38.2527, -85.7585),
            'Memphis, TN': (35.1495, -90.0490), 'Minneapolis, MN': (44.9778, -93.2650),
            'Indianapolis, IN': (39.7684, -86.1581), 'Columbus, OH': (39.9612, -82.9988),
            'Madison, WI': (43.0731, -89.4012), 'Springfield, IL': (39.7817, -89.6501)
        },
        'West': {
            'Los Angeles, CA': (34.0522, -118.2437), 'San Francisco, CA': (37.7749, -122.4194),
            'Seattle, WA': (47.6062, -122.3321), 'Portland, OR': (45.5152, -122.6784),
            'San Diego, CA': (32.7157, -117.1611), 'Denver, CO': (39.7392, -104.9903),
            'Boise, ID': (43.6150, -116.2023), 'Colorado Springs, CO': (38.8339, -104.8214),
            'Reno, NV': (39.5296, -119.8138), 'Spokane, WA': (47.6587, -117.4260),
            'Flagstaff, AZ': (35.1983, -111.6513), 'Phoenix, AZ': (33.4484, -112.0740),
            'Las Vegas, NV': (36.1699, -115.1398), 'Tucson, AZ': (32.2226, -110.9747),
            'Albuquerque, NM': (35.0844, -106.6504), 'El Paso, TX': (31.7619, -106.4850),
            'Bakersfield, CA': (35.3733, -119.0187), 'Fresno, CA': (36.7378, -119.7871),
            'Stockton, CA': (37.9577, -121.2908), 'Modesto, CA': (37.6391, -120.9969)
        }
        }

    def wind_stats(city_name, lat, lon, start_year=2024, end_year=2024):
        try:
            point = Point(lat, lon)
            start = datetime.datetime(start_year, 1, 1)
            end = datetime.datetime(end_year, 12, 31)
            data = Hourly(point, start, end).fetch()
            
            wind_speed_avg = data['wspd'].mean() if 'wspd' in data.columns else np.nan
            wind_speed_95th = data['wspd'].quantile(0.95) if 'wspd' in data.columns else np.nan
            wind_speed_max = data['wspd'].max() if 'wspd' in data.columns else np.nan
            wind_speed_std = data['wspd'].std() if 'wspd' in data.columns else np.nan
            
            if 'wdir' in data.columns:
                wind_directions = data['wdir'].dropna()
                if len(wind_directions) > 0:
                    directions_rad = np.radians(wind_directions)
                    sin_mean = np.mean(np.sin(directions_rad))
                    cos_mean = np.mean(np.cos(directions_rad))
                    wind_dir_avg = np.degrees(np.arctan2(sin_mean, cos_mean))
                    if wind_dir_avg < 0:
                        wind_dir_avg += 360
                else:
                    wind_dir_avg = np.nan
            else:
                wind_dir_avg = np.nan
            
            return {
                'city_name': city_name, 'latitude': lat, 'longitude': lon,
                'avg_wind_speed_kmh': wind_speed_avg, 'max_wind_speed_kmh': wind_speed_max,
                'wind_speed_95th_kmh': wind_speed_95th, 'wind_speed_std_kmh': wind_speed_std,
                'avg_wind_direction_deg': wind_dir_avg,
            }
        except Exception as e:
            return {
                'city_name': city_name, 'latitude': lat, 'longitude': lon,
                'avg_wind_speed_kmh': np.nan, 'max_wind_speed_kmh': np.nan,
                'wind_speed_95th_kmh': np.nan, 'wind_speed_std_kmh': np.nan,
                'avg_wind_direction_deg': np.nan,
            }
        
    results = []
    for region, cities in cities_by_region.items():
        for city_name, (lat, lon) in cities.items():
            result = wind_stats(city_name, lat, lon)
            result['region'] = region
            results.append(result)

        wind_df = pd.DataFrame(results)

    geographic_features = {
        'Coastal': [
            'New York, NY', 'Boston, MA', 'Portland, ME', 'Providence, RI', 'Newark, NJ',
            'Miami, FL', 'Jacksonville, FL', 'Tampa, FL', 'Charleston, SC', 'Savannah, GA',
            'Los Angeles, CA', 'San Francisco, CA', 'Seattle, WA', 'Portland, OR', 'San Diego, CA'
        ],
        'Great Lakes': [
            'Buffalo, NY', 'Rochester, NY', 'Erie, PA',
            'Chicago, IL', 'Detroit, MI', 'Cleveland, OH', 'Milwaukee, WI', 'Grand Rapids, MI', 'Toledo, OH'
        ],
        'Mountain': [
            'Burlington, VT', 'Manchester, NH', 'Montpelier, VT',
            'Denver, CO', 'Boise, ID', 'Colorado Springs, CO', 'Reno, NV', 'Spokane, WA', 'Flagstaff, AZ'
        ],
        'Desert': [
            'Phoenix, AZ', 'Las Vegas, NV', 'Tucson, AZ', 'Albuquerque, NM', 'El Paso, TX', 'Bakersfield, CA'
        ],
        'Valley': [
            'Pittsburgh, PA', 'Hartford, CT', 'Waterbury, CT',
            'Nashville, TN', 'Knoxville, TN', 'Huntsville, AL', 
            'St. Louis, MO', 'Louisville, KY', 'Memphis, TN',
            'Fresno, CA', 'Stockton, CA', 'Modesto, CA'
        ],
        'Plains': [
            'Kansas City, MO', 'Omaha, NE', 'Des Moines, IA', 'Wichita, KS', 'Oklahoma City, OK', 'Tulsa, OK'
        ],
        'Standard': [
            'Philadelphia, PA', 'Albany, NY', 'Syracuse, NY', 'Worcester, MA', 'Bridgeport, CT', 'Lowell, MA',
            'Atlanta, GA', 'Charlotte, NC', 'Orlando, FL', 'Birmingham, AL', 'Richmond, VA', 'Raleigh, NC', 
            'Greensboro, NC', 'Columbia, SC', 'Tallahassee, FL', 'Lexington, KY', 'Mobile, AL', 'Virginia Beach, VA',
            'Minneapolis, MN', 'Indianapolis, IN', 'Columbus, OH', 'Madison, WI', 'Springfield, IL'
        ]
    }

    wind_df_enhanced = wind_df.copy()
    wind_df_enhanced['geographic_feature'] = 'Standard'

    for feature, cities in geographic_features.items():
        if feature != 'Standard':
            mask = wind_df_enhanced['city_name'].isin(cities)
            wind_df_enhanced.loc[mask, 'geographic_feature'] = feature

    regions = ['Northeast', 'Southeast', 'Midwest', 'West']
    geo_features = ['Coastal', 'Great Lakes', 'Mountain', 'Desert', 'Valley', 'Plains', 'Standard']
    region_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def create_clean_wind_vector_chart(df, width=600, height=600):
        df_vectors = df.copy()
        df_vectors['direction_rad'] = np.radians((df_vectors['avg_wind_direction_deg'] - 90) % 360)
        
        max_wind_speed = df['avg_wind_speed_kmh'].max()
        vector_scale = 1
        
        df_vectors['dx'] = -df_vectors['avg_wind_speed_kmh'] * np.cos(df_vectors['direction_rad']) * vector_scale
        df_vectors['dy'] = df_vectors['avg_wind_speed_kmh'] * np.sin(df_vectors['direction_rad']) * vector_scale
        df_vectors['center_x'] = 0
        df_vectors['center_y'] = 0
        df_vectors['vector_end_x'] = df_vectors['dx']
        df_vectors['vector_end_y'] = df_vectors['dy']
        
        circle_radius = max_wind_speed
        domain_size = circle_radius * 2.4
        chart_pixel_size = width
        pixels_per_coord_unit = chart_pixel_size / domain_size
        circle_radius_pixels = circle_radius * pixels_per_coord_unit
        circle_area = np.pi * (circle_radius_pixels ** 2)
        circle_size = circle_area * (4 / np.pi)
        
        domain_min = -circle_radius * 1.2
        domain_max = circle_radius * 1.2
        
        outer_circle = alt.Chart(pd.DataFrame({'x': [0], 'y': [0]})).mark_circle(
            size=circle_size, stroke='black', strokeWidth=3, fill=None, opacity=1.0
        ).encode(
            x=alt.X('x:Q').scale(domain=[domain_min, domain_max]).title('').axis(labels=False),
            y=alt.Y('y:Q').scale(domain=[domain_min, domain_max]).title('').axis(labels=False)
        )
        
        center_point = alt.Chart(pd.DataFrame({'x': [0], 'y': [0]})).mark_circle(
            size=200, stroke='black', strokeWidth=3, fill='white'
        ).encode(
            x=alt.X('x:Q').scale(domain=[domain_min, domain_max]),
            y=alt.Y('y:Q').scale(domain=[domain_min, domain_max])
        )
        
        vectors = alt.Chart(df_vectors).mark_rule(strokeWidth=3, opacity=0.8).encode(
            x=alt.X('center_x:Q'), y=alt.Y('center_y:Q'), x2='vector_end_x:Q', y2='vector_end_y:Q',
            color=alt.Color('region:N').scale(domain=regions, range=region_colors).title('Region'),
            tooltip=[
                'city_name:N', 'region:N', 'geographic_feature:N',
                alt.Tooltip('avg_wind_speed_kmh:Q', title='Wind Speed (km/h)', format='.1f'),
                alt.Tooltip('avg_wind_direction_deg:Q', title='Wind Direction (°)', format='.0f')
            ]
        )
        
        arrow_scale = vector_scale * 0.8
        df_vectors['arrow_dx'] = -arrow_scale * np.cos(df_vectors['direction_rad'] + np.pi * 5/6)
        df_vectors['arrow_dy'] = arrow_scale * np.sin(df_vectors['direction_rad'] + np.pi * 5/6)
        df_vectors['arrow_dx2'] = -arrow_scale * np.cos(df_vectors['direction_rad'] - np.pi * 5/6)
        df_vectors['arrow_dy2'] = arrow_scale * np.sin(df_vectors['direction_rad'] - np.pi * 5/6)
        df_vectors['arrow1_end_x'] = df_vectors['vector_end_x'] + df_vectors['arrow_dx']
        df_vectors['arrow1_end_y'] = df_vectors['vector_end_y'] + df_vectors['arrow_dy']
        df_vectors['arrow2_end_x'] = df_vectors['vector_end_x'] + df_vectors['arrow_dx2']
        df_vectors['arrow2_end_y'] = df_vectors['vector_end_y'] + df_vectors['arrow_dy2']
        
        arrow1 = alt.Chart(df_vectors).mark_rule(strokeWidth=2, opacity=0.8).encode(
            x=alt.X('vector_end_x:Q'), y=alt.Y('vector_end_y:Q'), x2='arrow1_end_x:Q', y2='arrow1_end_y:Q',
            color=alt.Color('region:N').scale(domain=regions, range=region_colors)
        )
        
        arrow2 = alt.Chart(df_vectors).mark_rule(strokeWidth=2, opacity=0.8).encode(
            x=alt.X('vector_end_x:Q'), y=alt.Y('vector_end_y:Q'), x2='arrow2_end_x:Q', y2='arrow2_end_y:Q',
            color=alt.Color('region:N').scale(domain=regions, range=region_colors)
        )
        
        vector_endpoints = alt.Chart(df_vectors).mark_circle(size=60).encode(
            x=alt.X('vector_end_x:Q'), y=alt.Y('vector_end_y:Q'),
            color=alt.Color('region:N').scale(domain=regions, range=region_colors),
            stroke=alt.value('white'), strokeWidth=alt.value(2),
            tooltip=[
                'city_name:N', 'region:N', 'geographic_feature:N',
                alt.Tooltip('avg_wind_speed_kmh:Q', title='Wind Speed (km/h)', format='.1f'),
                alt.Tooltip('avg_wind_direction_deg:Q', title='Wind Direction (°)', format='.0f')
            ]
        )
        
        return (outer_circle + center_point + vectors + arrow1 + arrow2 + vector_endpoints).resolve_scale(
            color='shared'
        ).properties(width=width, height=height, title='Inversed Direction Vectors: Direction Wind is Traveling to (Magnitude = Speed)')

    # Create the wind vector chart
    wind_vector_clean = create_clean_wind_vector_chart(wind_df_enhanced, width=600, height=600)
    
    # Create individual filterable wind vector components
    def create_filterable_wind_vector_chart(df, width=600, height=600):
        df_vectors = df.copy()
        df_vectors['direction_rad'] = np.radians((df_vectors['avg_wind_direction_deg'] - 90) % 360)
        
        max_wind_speed = df['avg_wind_speed_kmh'].max()
        vector_scale = 1
        
        df_vectors['dx'] = -df_vectors['avg_wind_speed_kmh'] * np.cos(df_vectors['direction_rad']) * vector_scale
        df_vectors['dy'] = df_vectors['avg_wind_speed_kmh'] * np.sin(df_vectors['direction_rad']) * vector_scale
        df_vectors['center_x'] = 0
        df_vectors['center_y'] = 0
        df_vectors['vector_end_x'] = df_vectors['dx']
        df_vectors['vector_end_y'] = df_vectors['dy']
        
        circle_radius = max_wind_speed
        domain_size = circle_radius * 2.4
        domain_min = -circle_radius * 1.2
        domain_max = circle_radius * 1.2
        
        # Static elements (not filtered)
        chart_pixel_size = width
        pixels_per_coord_unit = chart_pixel_size / domain_size
        circle_radius_pixels = circle_radius * pixels_per_coord_unit
        circle_area = np.pi * (circle_radius_pixels ** 2)
        circle_size = circle_area * (4 / np.pi)
        
        outer_circle = alt.Chart(pd.DataFrame({'x': [0], 'y': [0]})).mark_circle(
            size=circle_size, stroke='black', strokeWidth=3, fill=None, opacity=1.0
        ).encode(
            x=alt.X('x:Q').scale(domain=[domain_min, domain_max]).title('').axis(labels=False),
            y=alt.Y('y:Q').scale(domain=[domain_min, domain_max]).title('').axis(labels=False)
        )
        
        center_point = alt.Chart(pd.DataFrame({'x': [0], 'y': [0]})).mark_circle(
            size=200, stroke='black', strokeWidth=3, fill='white'
        ).encode(
            x=alt.X('x:Q').scale(domain=[domain_min, domain_max]),
            y=alt.Y('y:Q').scale(domain=[domain_min, domain_max])
        )
        
        # Filterable data elements - create base chart that can be filtered
        base_chart = alt.Chart(df_vectors)
        
        vectors = base_chart.mark_rule(strokeWidth=3, opacity=0.8).encode(
            x=alt.X('center_x:Q'), y=alt.Y('center_y:Q'), x2='vector_end_x:Q', y2='vector_end_y:Q',
            color=alt.Color('region:N').scale(domain=regions, range=region_colors).title('Region'),
            tooltip=[
                'city_name:N', 'region:N', 'geographic_feature:N',
                alt.Tooltip('avg_wind_speed_kmh:Q', title='Wind Speed (km/h)', format='.1f'),
                alt.Tooltip('avg_wind_direction_deg:Q', title='Wind Direction (°)', format='.0f')
            ]
        )
        
        # Add arrow calculations
        arrow_scale = vector_scale * 0.8
        df_vectors['arrow_dx'] = -arrow_scale * np.cos(df_vectors['direction_rad'] + np.pi * 5/6)
        df_vectors['arrow_dy'] = arrow_scale * np.sin(df_vectors['direction_rad'] + np.pi * 5/6)
        df_vectors['arrow_dx2'] = -arrow_scale * np.cos(df_vectors['direction_rad'] - np.pi * 5/6)
        df_vectors['arrow_dy2'] = arrow_scale * np.sin(df_vectors['direction_rad'] - np.pi * 5/6)
        df_vectors['arrow1_end_x'] = df_vectors['vector_end_x'] + df_vectors['arrow_dx']
        df_vectors['arrow1_end_y'] = df_vectors['vector_end_y'] + df_vectors['arrow_dy']
        df_vectors['arrow2_end_x'] = df_vectors['vector_end_x'] + df_vectors['arrow_dx2']
        df_vectors['arrow2_end_y'] = df_vectors['vector_end_y'] + df_vectors['arrow_dy2']
        
        arrow1 = base_chart.mark_rule(strokeWidth=2, opacity=0.8).encode(
            x=alt.X('vector_end_x:Q'), y=alt.Y('vector_end_y:Q'), x2='arrow1_end_x:Q', y2='arrow1_end_y:Q',
            color=alt.Color('region:N').scale(domain=regions, range=region_colors)
        )
        
        arrow2 = base_chart.mark_rule(strokeWidth=2, opacity=0.8).encode(
            x=alt.X('vector_end_x:Q'), y=alt.Y('vector_end_y:Q'), x2='arrow2_end_x:Q', y2='arrow2_end_y:Q',
            color=alt.Color('region:N').scale(domain=regions, range=region_colors)
        )
        
        vector_endpoints = base_chart.mark_circle(size=60).encode(
            x=alt.X('vector_end_x:Q'), y=alt.Y('vector_end_y:Q'),
            color=alt.Color('region:N').scale(domain=regions, range=region_colors),
            stroke=alt.value('white'), strokeWidth=alt.value(2),
            tooltip=[
                'city_name:N', 'region:N', 'geographic_feature:N',
                alt.Tooltip('avg_wind_speed_kmh:Q', title='Wind Speed (km/h)', format='.1f'),
                alt.Tooltip('avg_wind_direction_deg:Q', title='Wind Direction (°)', format='.0f')
            ]
        )
        
        return (outer_circle + center_point + vectors + arrow1 + arrow2 + vector_endpoints).resolve_scale(
            color='shared'
        ).properties(width=width, height=height, title='Inversed Direction Vectors: Direction Wind is Traveling to (Magnitude = Speed)')

    # Create a complete filterable version with arrows and endpoints
    wind_vector_filterable = create_filterable_wind_vector_chart(wind_df_enhanced, width=600, height=600)

    # Define selections first
    geo_selector = alt.selection_point(fields=['geographic_feature'], empty=True)
    speed_brush = alt.selection_interval(encodings=['x'])
    geographic_brush = alt.selection_interval()

    # Create std_chart with conditional styling instead of filtering
    std_chart = alt.Chart(wind_df_enhanced).mark_circle(
        size=60, stroke='white', strokeWidth=1
    ).encode(
        x=alt.X('avg_wind_speed_kmh:Q', title='Average Wind Speed (km/h)', scale=alt.Scale(domain=[0, 25])),
        y=alt.Y('wind_speed_std_kmh:Q', title='Wind Speed Standard Deviation (km/h)', scale=alt.Scale(domain=[0, 12])),
        color=alt.condition(
            geo_selector,
            alt.Color('region:N', title='Region', scale=alt.Scale(domain=regions, range=region_colors)),
            alt.value('lightgray')
        ),
        opacity=alt.condition(
            geo_selector,
            alt.value(0.8),
            alt.value(0.3)
        ),
        tooltip=[
            alt.Tooltip('city_name:N', title='City'), alt.Tooltip('region:N', title='Region'),
            alt.Tooltip('geographic_feature:N', title='Geographic Feature'),
            alt.Tooltip('avg_wind_speed_kmh:Q', title='Avg Wind Speed (km/h)', format='.1f'),
            alt.Tooltip('wind_speed_std_kmh:Q', title='Standard Deviation (km/h)', format='.1f')
        ]
    ).add_params(speed_brush).transform_filter(geographic_brush).properties(width=600, height=300, title="Average Wind Speed vs Standard Deviation")

    geographic_selector_chart = (
        alt.Chart(wind_df_enhanced)
        .mark_rect(height=50)
        .encode(
            x=alt.X('geographic_feature:N').sort(geo_features).title(None).axis(labelAngle=-15, labelPadding=10),
            color=alt.condition(
                geo_selector,
                alt.Color('geographic_feature:N').scale(domain=geo_features, range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']).legend(None),
                alt.value('lightgray')
            ),
            opacity=alt.condition(geo_selector, alt.value(1.0), alt.value(0.4)),
            stroke=alt.condition(geo_selector, alt.value('black'), alt.value('white')),
            strokeWidth=alt.condition(geo_selector, alt.value(2), alt.value(1))
        )
        .add_params(geo_selector)
        .transform_filter(speed_brush)
        .properties(width=600, height=75, title='Sort by Geographic Feature')
        )

    lat_min, lat_max = wind_df_enhanced['latitude'].min(), wind_df_enhanced['latitude'].max()
    lat_range = lat_max - lat_min
    lat_domain = [lat_min - lat_range * 0.05, lat_max + lat_range * 0.05]

    lon_min, lon_max = wind_df_enhanced['longitude'].min(), wind_df_enhanced['longitude'].max()
    lon_range = lon_max - lon_min
    lon_domain = [lon_min - lon_range * 0.05, lon_max + lon_range * 0.05]

    geographic_background = (
        alt.Chart(wind_df_enhanced)
        .mark_circle(size=60, opacity=0.3)
        .encode(
            x=alt.X('longitude:Q').scale(domain=lon_domain).title('Longitude (degrees)').axis(grid=True),
            y=alt.Y('latitude:Q').scale(domain=lat_domain).title('Latitude (degrees)').axis(grid=True),
            color=alt.value('lightgrey'),
            tooltip=['city_name:N', alt.Tooltip('longitude:Q', title='Longitude', format='.2f'), 
                    alt.Tooltip('latitude:Q', title='Latitude', format='.2f'), 'region:N', 'geographic_feature:N',
                    alt.Tooltip('avg_wind_speed_kmh:Q', title='Wind Speed (km/h)', format='.1f')],
            stroke=alt.value('white'), strokeWidth=alt.value(0.5)
        )
    )

    geographic_foreground = (
        alt.Chart(wind_df_enhanced)
        .mark_circle(size=60)
        .encode(
            x=alt.X('longitude:Q'), y=alt.Y('latitude:Q'),
            color=alt.Color('region:N').scale(domain=regions, range=region_colors).title('Region'),
            tooltip=['city_name:N', alt.Tooltip('longitude:Q', title='Longitude', format='.2f'),
                    alt.Tooltip('latitude:Q', title='Latitude', format='.2f'), 'region:N', 'geographic_feature:N',
                    alt.Tooltip('avg_wind_speed_kmh:Q', title='Wind Speed (km/h)', format='.1f')],
            stroke=alt.value('white'), strokeWidth=alt.value(1)
        )
        .transform_filter(geo_selector)
    )

    geographic_map_chart = (geographic_background + geographic_foreground).add_params(geographic_brush).properties(
        width=600, height=300, title='City Selector (Brush to Select)'
    )

    # Create wind vector chart with conditional styling (rebuild with selections applied)
    def create_interactive_wind_vector_chart(df, width=600, height=600):
        df_vectors = df.copy()
        df_vectors['direction_rad'] = np.radians((df_vectors['avg_wind_direction_deg'] - 90) % 360)
        
        max_wind_speed = df['avg_wind_speed_kmh'].max()
        vector_scale = 1
        
        df_vectors['dx'] = -df_vectors['avg_wind_speed_kmh'] * np.cos(df_vectors['direction_rad']) * vector_scale
        df_vectors['dy'] = df_vectors['avg_wind_speed_kmh'] * np.sin(df_vectors['direction_rad']) * vector_scale
        df_vectors['center_x'] = 0
        df_vectors['center_y'] = 0
        df_vectors['vector_end_x'] = df_vectors['dx']
        df_vectors['vector_end_y'] = df_vectors['dy']
        
        circle_radius = max_wind_speed
        domain_size = circle_radius * 2.4
        domain_min = -circle_radius * 1.2
        domain_max = circle_radius * 1.2
        
        # Static elements
        chart_pixel_size = width
        pixels_per_coord_unit = chart_pixel_size / domain_size
        circle_radius_pixels = circle_radius * pixels_per_coord_unit
        circle_area = np.pi * (circle_radius_pixels ** 2)
        circle_size = circle_area * (4 / np.pi)
        
        outer_circle = alt.Chart(pd.DataFrame({'x': [0], 'y': [0]})).mark_circle(
            size=circle_size, stroke='black', strokeWidth=3, fill=None, opacity=1.0
        ).encode(
            x=alt.X('x:Q').scale(domain=[domain_min, domain_max]).title('').axis(labels=False),
            y=alt.Y('y:Q').scale(domain=[domain_min, domain_max]).title('').axis(labels=False)
        )
        
        center_point = alt.Chart(pd.DataFrame({'x': [0], 'y': [0]})).mark_circle(
            size=200, stroke='black', strokeWidth=3, fill='white'
        ).encode(
            x=alt.X('x:Q').scale(domain=[domain_min, domain_max]),
            y=alt.Y('y:Q').scale(domain=[domain_min, domain_max])
        )
        
        # Arrows calculations
        arrow_scale = vector_scale * 0.8
        df_vectors['arrow_dx'] = -arrow_scale * np.cos(df_vectors['direction_rad'] + np.pi * 5/6)
        df_vectors['arrow_dy'] = arrow_scale * np.sin(df_vectors['direction_rad'] + np.pi * 5/6)
        df_vectors['arrow_dx2'] = -arrow_scale * np.cos(df_vectors['direction_rad'] - np.pi * 5/6)
        df_vectors['arrow_dy2'] = arrow_scale * np.sin(df_vectors['direction_rad'] - np.pi * 5/6)
        df_vectors['arrow1_end_x'] = df_vectors['vector_end_x'] + df_vectors['arrow_dx']
        df_vectors['arrow1_end_y'] = df_vectors['vector_end_y'] + df_vectors['arrow_dy']
        df_vectors['arrow2_end_x'] = df_vectors['vector_end_x'] + df_vectors['arrow_dx2']
        df_vectors['arrow2_end_y'] = df_vectors['vector_end_y'] + df_vectors['arrow_dy2']
        
        # Interactive vector elements with conditional styling
        vectors = alt.Chart(df_vectors).mark_rule(strokeWidth=3).encode(
            x=alt.X('center_x:Q'), y=alt.Y('center_y:Q'), x2='vector_end_x:Q', y2='vector_end_y:Q',
            color=alt.condition(
                geo_selector,
                alt.Color('region:N').scale(domain=regions, range=region_colors),
                alt.value('lightgray')
            ),
            opacity=alt.condition(
                geo_selector,
                alt.value(0.8),
                alt.value(0.3)
            ),
            tooltip=[
                'city_name:N', 'region:N', 'geographic_feature:N',
                alt.Tooltip('avg_wind_speed_kmh:Q', title='Wind Speed (km/h)', format='.1f'),
                alt.Tooltip('avg_wind_direction_deg:Q', title='Wind Direction (°)', format='.0f')
            ]
        ).transform_filter(geographic_brush).transform_filter(speed_brush)
        
        arrow1 = alt.Chart(df_vectors).mark_rule(strokeWidth=2).encode(
            x=alt.X('vector_end_x:Q'), y=alt.Y('vector_end_y:Q'), x2='arrow1_end_x:Q', y2='arrow1_end_y:Q',
            color=alt.condition(
                geo_selector,
                alt.Color('region:N').scale(domain=regions, range=region_colors),
                alt.value('lightgray')
            ),
            opacity=alt.condition(
                geo_selector,
                alt.value(0.8),
                alt.value(0.3)
            )
        ).transform_filter(geographic_brush).transform_filter(speed_brush)
        
        arrow2 = alt.Chart(df_vectors).mark_rule(strokeWidth=2).encode(
            x=alt.X('vector_end_x:Q'), y=alt.Y('vector_end_y:Q'), x2='arrow2_end_x:Q', y2='arrow2_end_y:Q',
            color=alt.condition(
                geo_selector,
                alt.Color('region:N').scale(domain=regions, range=region_colors),
                alt.value('lightgray')
            ),
            opacity=alt.condition(
                geo_selector,
                alt.value(0.8),
                alt.value(0.3)
            )
        ).transform_filter(geographic_brush).transform_filter(speed_brush)
        
        vector_endpoints = alt.Chart(df_vectors).mark_circle(size=60).encode(
            x=alt.X('vector_end_x:Q'), y=alt.Y('vector_end_y:Q'),
            color=alt.condition(
                geo_selector,
                alt.Color('region:N').scale(domain=regions, range=region_colors),
                alt.value('lightgray')
            ),
            opacity=alt.condition(
                geo_selector,
                alt.value(0.8),
                alt.value(0.3)
            ),
            stroke=alt.value('white'), strokeWidth=alt.value(2),
            tooltip=[
                'city_name:N', 'region:N', 'geographic_feature:N',
                alt.Tooltip('avg_wind_speed_kmh:Q', title='Wind Speed (km/h)', format='.1f'),
                alt.Tooltip('avg_wind_direction_deg:Q', title='Wind Direction (°)', format='.0f')
            ]
        ).transform_filter(geographic_brush).transform_filter(speed_brush)
        
        return (outer_circle + center_point + vectors + arrow1 + arrow2 + vector_endpoints).resolve_scale(
            color='shared'
        ).properties(width=width, height=height, title='Inversed Direction Vectors: Direction Wind is Traveling to (Magnitude = Speed)')

    wind_vector_interactive = create_interactive_wind_vector_chart(wind_df_enhanced, width=600, height=600)

    complete_dashboard_std = (
        geographic_map_chart &
        geographic_selector_chart &
        std_chart &
        wind_vector_interactive
    ).resolve_scale(color='independent')

    return complete_dashboard_std