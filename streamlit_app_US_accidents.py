
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
import pydeck as pdk
import altair as alt

# ✅ 页面配置放在最顶端
st.set_page_config(
    page_title="US Traffic Accident",
    layout="wide"
)

# ========== Prediction 页面逻辑 ==========
def run_prediction_sidebar():
    st.sidebar.title("Prediction Settings")
    # 原 app.py 中 sidebar 相关控件保留在 app_code 内定义

def run_prediction_main():

    # import warnings
    # warnings.filterwarnings('ignore')

    # default coordinates
    DEFAULT_LAT = 39.0
    DEFAULT_LNG = -84.0

    # location
    LOCATION_EXAMPLES = [
        # East coast
        "New York, NY",
        "Boston, MA",
        "Philadelphia, PA",
        "Washington, DC",
        "Miami, FL",
        "Atlanta, GA",
        "Charlotte, NC",
        "Baltimore, MD",

        # West coast
        "Los Angeles, CA",
        "San Francisco, CA",
        "Seattle, WA",
        "Portland, OR",
        "San Diego, CA",
        "Las Vegas, NV",
        "Phoenix, AZ",

        # Midwest
        "Chicago, IL",
        "Detroit, MI",
        "Minneapolis, MN",
        "St. Louis, MO",
        "Kansas City, MO",
        "Denver, CO",
        "Dallas, TX",
        "Houston, TX",

        # South
        "New Orleans, LA",
        "Nashville, TN",
        "Memphis, TN",
        "Austin, TX",
        "San Antonio, TX",
        "Tampa, FL",
        "Orlando, FL",

        # Northeast
        "Pittsburgh, PA",
        "Cleveland, OH",
        "Cincinnati, OH",
        "Buffalo, NY",
        "Hartford, CT",
        "Providence, RI"
    ]

    # session state initialization
    if 'coordinates' not in st.session_state:
        st.session_state.coordinates = (DEFAULT_LAT, DEFAULT_LNG)
    if 'location_name' not in st.session_state:
        st.session_state.location_name = "New York, USA"

    # # Add map data to session state
    if 'map_data' not in st.session_state:
        st.session_state.map_data = pd.DataFrame({
            'lat': [DEFAULT_LAT],
            'lon': [DEFAULT_LNG]
        })

    st.title('US Traffic Accident Severity Prediction')
    st.markdown("*Machine Learning Model Based on US Traffic Accident Data*")


    # Load model
    @st.cache_resource
    def load_model():
        try:
            model = joblib.load("histgradientboost_model.pkl")
            return model
        except Exception as e:
            st.error(f'Model loading failed: {str(e)}')
            return None


    model = load_model()


    # Initialize the geocoder
    @st.cache_resource
    def get_geocoder():
        return Nominatim(user_agent="traffic_accident_prediction")


    geolocator = get_geocoder()


    # Geocoding function
    @st.cache_data(ttl=3600)  # Cache 1 hour
    def geocode_location(location_name):
        try:
            location = geolocator.geocode(location_name)
            if location:
                return location.latitude, location.longitude
            return None
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            st.error(f"Geocoding service temporarily unavailable. Please try again later.")
            return None
        except Exception as e:
            st.error(f"Error during geocoding: {str(e)}")
            return None


    # Add an automatic geocoding function
    def auto_geocode(location_name):
        with st.spinner('Processing location...'):
            coordinates = geocode_location(location_name)
            if coordinates:
                st.session_state.coordinates = coordinates
                # Update map data
                st.session_state.map_data = pd.DataFrame({
                    'lat': [coordinates[0]],
                    'lon': [coordinates[1]]
                })
                st.success(f"Location found: {location_name}")
                st.write(f"Coordinates: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
            else:
                st.error("""
                Location not found. Please try:
                1. Using a more specific location name (e.g., 'New York, NY' instead of just 'New York')
                2. Including the state abbreviation
                3. Using a complete address
                """)
                st.session_state.coordinates = (DEFAULT_LAT, DEFAULT_LNG)
                # Reset map data
                st.session_state.map_data = pd.DataFrame({
                    'lat': [DEFAULT_LAT],
                    'lon': [DEFAULT_LNG]
                })


    # Create a sidebar
    app_mode = st.sidebar.selectbox(
        "Select Input Mode",
        ["Manual Input", "Use Sample Data"]
    )

    # Define sample data
    sample_data = {
        "Sunny Day - Minor Accident": {
            "location": "San Francisco, CA",
            "hour": 12, "weekday": 2, "month": 6,
            "temp": 75.0, "humidity": 45.0, "pressure": 30.05,
            "visibility": 10.0, "wind_speed": 8.0, "precipitation": 0.0,
            "weather": ["Clear", "Fair"]
        },
        "Rainy Day - Severe Accident": {
            "location": "New York, NY",
            "hour": 18, "weekday": 5, "month": 11,
            "temp": 45.0, "humidity": 90.0, "pressure": 29.50,
            "visibility": 2.0, "wind_speed": 15.0, "precipitation": 0.5,
            "weather": ["Rain", "Heavy"]
        },
        "Snow Day - Extremely Severe Accident": {
            "location": "Chicago, IL",
            "hour": 8, "weekday": 1, "month": 1,
            "temp": 20.0, "humidity": 85.0, "pressure": 29.80,
            "visibility": 0.5, "wind_speed": 20.0, "precipitation": 1.2,
            "weather": ["Snow", "Heavy"]
        }
    }

    # Define weather options
    weather_cols = [
        'Clear', 'Cloudy', 'Drifting', 'Drizzle', 'Fair',
        'Fog', 'Freezing', 'Funnel', 'Hail', 'Haze',
        'Heavy', 'Ice', 'Light', 'Mist', 'Mostly',
        'N', 'Overcast', 'Partial', 'Partly', 'Patches',
        'Rain', 'Sand', 'Scattered', 'Shallow', 'Showers',
        'Sleet', 'Small', 'Smoke', 'Snow', 'Squalls',
        'T', 'Thunder', 'Thunderstorm', 'Thunderstorms', 'Tornado',
        'Widespread', 'Wintry'
    ]

    # Displays different content according to the mode selected
    if app_mode == "Use Sample Data":
        st.sidebar.subheader("Select Sample Data")
        selected_sample = st.sidebar.selectbox("Choose a sample case", list(sample_data.keys()))

        # Get selected sample data
        selected_data = sample_data[selected_sample]

        # Set form values to sample data
        hour = selected_data["hour"]
        weekday = selected_data["weekday"]
        month = selected_data["month"]
        location_name = selected_data["location"]
        temperature = selected_data["temp"]
        humidity = selected_data["humidity"]
        pressure = selected_data["pressure"]
        visibility = selected_data["visibility"]
        wind_speed = selected_data["wind_speed"]
        precipitation = selected_data["precipitation"]
        selected_weather = selected_data["weather"]

        # Geocoding automatically
        with st.spinner('Processing location...'):
            coordinates = geocode_location(location_name)
            if coordinates:
                st.session_state.coordinates = coordinates
                # Update map data
                st.session_state.map_data = pd.DataFrame({
                    'lat': [coordinates[0]],
                    'lon': [coordinates[1]]
                })
                st.success(f"Location found: {location_name}")
            else:
                st.error(f"Could not find coordinates for {location_name}")
                st.session_state.coordinates = (DEFAULT_LAT, DEFAULT_LNG)
                # Reset map data
                st.session_state.map_data = pd.DataFrame({
                    'lat': [DEFAULT_LAT],
                    'lon': [DEFAULT_LNG]
                })

        # Display selected sample data information
        st.info(f"Sample data loaded: {selected_sample}")

        # Make predictions automatically
        predict_button = True

    else:  # Manual input mode
        st.sidebar.subheader('Please Enter Accident Information')

        date = st.sidebar.date_input("Select Date")
        time_str = st.sidebar.text_input("Enter Time (HH:MM)", value="12:00")
        try:
            hour = datetime.strptime(time_str, "%H:%M").hour
        except:
            hour = 0
        weekday = date.weekday()
        month = date.month

        # Location input
        location_name = st.sidebar.text_input("Enter Location", value=st.session_state.location_name)
        if st.sidebar.button("Geocode Location"):
            auto_geocode(location_name)

        st.sidebar.map(st.session_state.map_data, zoom=6)

        # Weather input
        temperature = st.sidebar.number_input("Change Temperature (°F)", value=70.0)
        humidity = st.sidebar.number_input("Change Humidity (%)", value=50.0, min_value=0.0, max_value=100.0)
        pressure = st.sidebar.number_input("Change Pressure (in)", value=29.92)
        visibility = st.sidebar.number_input("Change Visibility (mi)", value=10.0)
        wind_speed = st.sidebar.number_input("Change Wind Speed (mph)", value=5.0)
        precipitation = st.sidebar.number_input("Change Precipitation (in)", value=0.0)

        selected_weather = st.sidebar.multiselect("Select Weather Conditions", weather_cols)


    # Process prediction logic
    if st.sidebar.button("Make Prediction"):
        if model is not None:
            try:
                # Use the coordinates in session state
                start_lat, start_lng = st.session_state.coordinates

                # Create weather condition characteristics
                weather_features = {f'Weather_Main_{w}': 1 if w in selected_weather else 0
                                    for w in weather_cols}

                # Prepare for input data
                input_data = {
                    'Hour': hour,
                    'Weekday': weekday,
                    'Month': month,
                    'Start_Lat': start_lat,
                    'Start_Lng': start_lng,
                    'Temperature(F)': temperature,
                    'Humidity(%)': humidity,
                    'Pressure(in)': pressure,
                    'Visibility(mi)': visibility,
                    'Wind_Speed(mph)': wind_speed,
                    'Precipitation(in)': precipitation,
                    **weather_features
                }

                print(input_data)

                # Convert to a DataFrame
                input_df = pd.DataFrame([input_data])

                # Make predictions
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)

                # Results display area
                st.markdown("---")
                st.subheader("🔍 Prediction Results")

                # Display the results using a column layout
                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    # Show forecast results
                    severity_level = int(prediction[0])

                    # Different colors and ICONS are set according to the severity level
                    if severity_level == 1:
                        st.success("Predicted Accident Severity Level: 1 - Minor")
                    elif severity_level == 2:
                        st.info("Predicted Accident Severity Level: 2 - Moderate")
                    elif severity_level == 3:
                        st.warning("Predicted Accident Severity Level: 3 - Severe")
                    elif severity_level == 4:
                        st.error("Predicted Accident Severity Level: 4 - Extremely Severe")

                    # Display a summary of input features
                    st.subheader("Input Feature Summary")
                    st.write(f"📅 Date & Time: Month {month}, Weekday {weekday}, Hour {hour}")
                    st.write(f"📍 Location: Latitude {start_lat:.4f}, Longitude {start_lng:.4f}")
                    st.write(f"🌤️ Weather: Temperature {temperature}°F, Humidity {humidity}%, Visibility {visibility}mi")
                    if selected_weather:
                        st.write(f"☁️ Weather Conditions: {', '.join(selected_weather)}")

                with res_col2:
                    # Display prediction probability
                    st.subheader('Probability Distribution by Severity Level:')

                    # Create a DataFrame for the bar chart
                    prob_df = pd.DataFrame({
                        'Severity Level': ['1 - Minor', '2 - Moderate', '3 - Severe', '4 - Extremely Severe'],
                        'Probability': probability[0]
                    })

                    # Use the native bar chart of Streamlit
                    st.bar_chart(prob_df.set_index('Severity Level'))

                    # Highlight the highest probability
                    max_prob_idx = np.argmax(probability[0])
                    max_prob = probability[0][max_prob_idx]
                    st.info(f"Highest probability prediction: Level {max_prob_idx + 1} ({max_prob:.2%})")

            except Exception as e:
                st.error(f'Prediction error: {str(e)}')
                st.exception(e)
        else:
            st.error('Please ensure the model file is correctly loaded')

    # Add instructions
    with st.expander('📋 User Guide'):
        st.write('''
        #### How to Use:
        1. Select input mode (Manual Input or Use Sample Data)
        2. For Manual Input mode:
           - Select the date and time of the accident
           - Enter the location name in one of these ways:
             * Select a US city from the dropdown
             * Enter a custom US location (e.g., "New York, NY" or "123 Main St, Los Angeles, CA")
           - Click "Geocode Location" to convert the name to coordinates
           - Enter weather-related information
           - Select the weather conditions (multiple selections allowed)
           - Click the "Make Prediction" button to get results
        3. For Sample Data mode:
           - Choose a predefined sample case from the sidebar
           - The system will automatically load and predict results

        #### Location Input Tips:
        - Use the format: "City, State" (e.g., "New York, NY")
        - For better results, include the state abbreviation
        - You can also use specific addresses
        - If a location is not found, try being more specific
        - This system is optimized for US locations only

        #### Prediction Result Explanation:
        - Level 1 (Minor): Minor property damage, no injuries
        - Level 2 (Moderate): Moderate property damage, possible minor injuries
        - Level 3 (Severe): Serious property damage, possible severe injuries
        - Level 4 (Extremely Severe): Catastrophic property damage and casualties

        #### Data Notes:
        - Temperature, humidity, pressure, etc. use US units
        - Multiple weather conditions can be selected, the system will consider all selected factors
        - Location names should be as specific as possible
        - This model is trained on US traffic accident data
        ''')

    # Add footer
    st.markdown("---")
    st.markdown(
        "© 2025 Traffic Accident Severity Prediction System | Based on Histgradientboost Model | Data Source: US_Accidents")

# ========== UI 页面逻辑 ==========
def run_ui_sidebar():
    st.sidebar.title("Viewer Settings")
    # 原 ui.py 中 sidebar 控件保留在 ui_code 内定义

def run_ui_main():

    st.title("US Historical Accident Viewer")

    @st.cache_data
    def load_data():
        df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
        df = df.dropna(subset=["Start_Lat", "Start_Lng", "Severity"])
        df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors='coerce')
        df = df.dropna(subset=["Start_Time"])
        return df

    df = load_data()

    # Sidebar input
    st.sidebar.header("Select Location")

# 用户输入地址
    address_input = st.sidebar.text_input("Enter address (e.g. '2134 Western Ave, Seattle, WA')")

# 实时 geocode 转换为 lat/lng
    lat_input, lng_input = None, None
    if address_input:
        try:
            geolocator = Nominatim(user_agent="accident_viewer")
            location = geolocator.geocode(address_input)
            if location:
                lat_input = location.latitude
                lng_input = location.longitude
                st.sidebar.success(f"Found location: {lat_input:.4f}, {lng_input:.4f}")
            else:
                st.sidebar.error("Address not found.")
        except Exception as e:
            st.sidebar.error(f"Geocoding error: {str(e)}")

    radius_km = st.sidebar.slider("Search Radius (km)", 1, 100, 10)

    # 用户输入过去多少天
    st.sidebar.markdown("### Select Lookback Days")
    days_ago = st.sidebar.number_input("How many days in the past?", min_value=1, max_value=1000, value=365)

    # 图层控制
    show_heatmap = st.sidebar.checkbox("Show Heatmap", True)
    show_points = st.sidebar.checkbox("Show Points by Severity", True)

    # 主逻辑
    if lat_input and lng_input:
        try:
            lat = float(lat_input)
            lng = float(lng_input)

            now = pd.Timestamp("2023-03-15")  # 假设当前时间
            df_recent = df[df["Start_Time"] >= now - timedelta(days=int(days_ago))]

            def haversine_np(lat1, lon1, lat2, lon2):
                R = 6371.0
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
                return R * 2 * np.arcsin(np.sqrt(a))

            df_recent["distance_km"] = haversine_np(lat, lng, df_recent["Start_Lat"], df_recent["Start_Lng"])
            nearby_df = df_recent[df_recent["distance_km"] <= radius_km]
            num_accidents = len(nearby_df)

            st.subheader(f"Accident Map (Past {int(days_ago)} Days)")
            zoom_level = 12 if radius_km <= 10 else 10 if radius_km <= 30 else 8

            # 构造图层
            layers = []

            if show_heatmap:
                heatmap_layer = pdk.Layer(
                    "HeatmapLayer",
                    data=nearby_df,
                    get_position='[Start_Lng, Start_Lat]',
                    get_weight=1,
                    radiusPixels=40,
                    intensity=2,
                    threshold=0.05,
                )
                layers.append(heatmap_layer)

            if show_points:
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=nearby_df,
                    get_position='[Start_Lng, Start_Lat]',
                    get_fill_color="""
                        Severity == 1 ? [0, 200, 0, 160] :
                        Severity == 2 ? [255, 255, 0, 160] :
                        Severity == 3 ? [255, 165, 0, 160] :
                        [255, 0, 0, 160]
                    """,
                    get_radius=20,
                    pickable=True,
                    opacity=0.8,
                )
                layers.append(scatter_layer)

            # ✅ 用户位置 Marker 图标（IconLayer）
            icon_data = pd.DataFrame([{
                "name": "Your Location",
                "lat": lat,
                "lon": lng,
                "icon_data": {
                    "url": "https://cdn-icons-png.flaticon.com/512/684/684908.png",
                    "width": 512,
                    "height": 512,
                    "anchorY": 512
                }
            }])

            icon_layer = pdk.Layer(
                type="IconLayer",
                data=icon_data,
                get_icon="icon_data",
                get_position='[lon, lat]',
                get_size=3,
                size_scale=12,
                pickable=True,
            )
            layers.append(icon_layer)

            # 展示地图
            if layers:
                st.pydeck_chart(pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=pdk.ViewState(
                        latitude=lat,
                        longitude=lng,
                        zoom=zoom_level,
                        pitch=40,
                    ),
                    layers=layers,
                    tooltip={"text": "{name}"}
                ))
            else:
                st.info("Please select at least one layer to display.")

            # 图例说明
            st.markdown("### Color Meaning (By Severity)")
            st.markdown("""
            - 🟢 **Green (Severity 1)**: Minor accidents
            - 🟡 **Yellow (Severity 2)**: Moderate severity
            - 🟠 **Orange (Severity 3)**: Serious accidents
            - 🔴 **Red (Severity 4)**: Very severe accidents
            """)

            # 数据概览
            st.markdown("### Accident Summary")
            st.markdown(f"- In the **last {int(days_ago)} days**, within **{radius_km} km** of **({lat}, {lng})**:")
            st.markdown(f"- Total accidents: **{num_accidents}**")

            if num_accidents > 0:
                avg_density = num_accidents / (3.14 * radius_km ** 2)
                top_city = nearby_df["City"].value_counts().idxmax() if "City" in nearby_df.columns else "Unknown"
                st.markdown(f"- Average density: **{avg_density:.2f} per km²**")
                st.markdown(f"- Most frequent city: **{top_city}**")

                # 图表
                st.markdown("### Monthly Trend")
                nearby_df["Month"] = nearby_df["Start_Time"].dt.to_period("M").astype(str)
                trend_df = nearby_df.groupby("Month").size().reset_index(name="Accidents")
                peak_month = trend_df.loc[trend_df["Accidents"].idxmax()]
                low_month = trend_df.loc[trend_df["Accidents"].idxmin()]
                st.markdown(
                    f"""
                    This chart shows how accidents evolved over time in the selected region.

                    - 📈 **Highest number of accidents:** {peak_month['Month']} with **{peak_month['Accidents']}** reported incidents.
                    - 📉 **Lowest number of accidents:** {low_month['Month']} with only **{low_month['Accidents']}** reported.

                    This trend may reflect seasonal traffic patterns, weather impacts, or local events.
                    """
                )
                line_chart = alt.Chart(trend_df).mark_line(point=True).encode(
                    x="Month:T", y="Accidents:Q"
                ).properties(width=700)
                st.altair_chart(line_chart)

        except ValueError:
            st.error("Invalid input: Please enter valid coordinates.")
    else:
        st.info("Please input location to begin.")

    st.markdown("---")
    st.markdown(
        "© 2025 US Historical Traffic Accident Viewer | Data Source: US_Accidents")


# ========== 主入口 ==========
def main():
    page = st.sidebar.radio("Select View", ["Prediction System", "Accident Viewer"])

    if page == "Prediction System":
        run_prediction_sidebar()
        run_prediction_main()
    else:
        run_ui_sidebar()
        run_ui_main()

if __name__ == "__main__":
    main()
