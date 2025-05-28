import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import requests
import xml.etree.ElementTree as ET
import folium
import streamlit as st
from streamlit_folium import folium_static

# Load the imputer, scaler, and Random Forest model
imputer = joblib.load('imputer.pkl')  # SimpleImputer for handling missing values
scaler = joblib.load('scaler.pkl')  # StandardScaler
rf_model = joblib.load('rf_model.pkl')  # Random Forest model

# Manually defined coordinates for stations
station_coordinates = {
    "Bandra Kurla Complex, Mumbai - IITM": [19.0700, 72.8770],
    "Kandivali East, Mumbai - MPCB": [19.2120, 72.8560],
    "Mulund West, Mumbai - MPCB": [19.1800, 72.9500],
    "Borivali East, Mumbai - MPCB": [19.2840, 72.8580],
    "Chembur, Mumbai - MPCB": [19.0560, 72.8830],
    "Chhatrapati Shivaji Intl. Airport (T2), Mumbai - MPCB": [19.0886, 72.8656],
    "Kherwadi_Bandra East, Mumbai - MPCB": [19.0800, 72.8500],
    "Khindipada-Bhandup West, Mumbai - IITM": [19.2000, 72.9400],
    "Kurla, Mumbai - MPCB": [19.0730, 72.8760],
    "Malad West, Mumbai - IITM": [19.1770, 72.8280],
    "Mazgaon, Mumbai - IITM": [18.9470, 72.8310],
    "Mindspace-Malad West, Mumbai - MPCB": [19.1640, 72.8500],
    "Navy Nagar-Colaba, Mumbai - IITM": [18.9360, 72.8170],
    "Powai, Mumbai - MPCB": [19.1230, 72.9110],
    "Siddharth Nagar-Worli, Mumbai - IITM": [19.0300, 72.8200],
    "Vasai West, Mumbai - MPCB": [19.4300, 72.8220],
    "Vile Parle West, Mumbai - MPCB": [19.0900, 72.8470],
    "Worli, Mumbai - MPCB": [18.9860, 72.8180],
}

# Function to get latitude and longitude from predefined dictionary
def get_lat_lon(station_name):
    station_name = station_name.strip().lower()
    for name, coords in station_coordinates.items():
        if station_name in name.lower():
            return coords
    print(f"Coordinates not found for station: {station_name}")
    return None, None

# Function to return color based on AQI
def get_aqi_color(aqi):
    try:
        aqi = int(aqi)
    except (ValueError, TypeError):
        return 'gray'
    if aqi <= 50:
        return 'green'
    elif aqi <= 100:
        return 'lightgreen'
    elif aqi <= 200:
        return 'orange'
    elif aqi <= 300:
        return 'red'
    elif aqi <= 400:
        return 'purple'
    else:
        return 'maroon'

# Function to predict AQI from pollutants using the trained Random Forest model
def predict_aqi_from_pollutants(pollutant_dict):
    # Define pollutant order matching your model input
    pollutant_order = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

    # Get average values, fill missing or invalid ones with 0
    values = []
    for key in pollutant_order:
        val = pollutant_dict.get(key, {}).get('avg', '0')
        try:
            values.append(float(val))
        except ValueError:
            values.append(0.0)

    # Convert to numpy array and reshape
    X = np.array(values).reshape(1, -1)

    # Apply SimpleImputer to handle missing values
    X_imputed = imputer.transform(X)  # Impute missing values

    # Apply StandardScaler for scaling
    X_scaled = scaler.transform(X_imputed)  # Scale the values

    # Predict AQI using the Random Forest model
    prediction = rf_model.predict(X_scaled)
    return float(prediction[0])

# Streamlit UI
st.title('Air Quality Index (AQI) Data for Mumbai')
st.subheader('Fetch and Display AQI Data')

# Define the URL for the RSS feed
url = "https://airquality.cpcb.gov.in/caaqms/rss_feed"

# Add a button to fetch and display data
if st.button('Fetch AQI Data'):
    headers = {'accept': 'application/xml'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            root = ET.fromstring(response.content)
            mpcb_iitm_data = []

            for country in root.findall("Country"):
                for state in country.findall("State"):
                    if state.get("id") == "Maharashtra":
                        for city in state.findall("City"):
                            if city.get("id") == "Mumbai":
                                for station in city.findall("Station"):
                                    station_name = station.get("id").strip()
                                    if "MPCB" in station_name or "IITM" in station_name:
                                        last_update = station.get("lastupdate")
                                        pollutants = {}
                                        for pollutant in station.findall("Pollutant_Index"):
                                            pid = pollutant.get("id")
                                            pollutants[pid] = {
                                                "min": pollutant.get("Min"),
                                                "max": pollutant.get("Max"),
                                                "avg": pollutant.get("Avg")
                                            }

                                        air_quality_index = station.find("Air_Quality_Index")
                                        if air_quality_index is not None:
                                            aqi_value = air_quality_index.get("Value")
                                            predominant_parameter = air_quality_index.get("Predominant_Parameter")
                                            mpcb_iitm_data.append({
                                                "station_name": station_name,
                                                "last_update": last_update,
                                                "air_quality_index": aqi_value,
                                                "predominant_parameter": predominant_parameter,
                                                "pollutants": pollutants
                                            })
                                        else:
                                            st.warning(f"No Air Quality Index data for station: {station_name}")

            if mpcb_iitm_data:
                station_info = []

                for data in mpcb_iitm_data:
                    station_name = data['station_name']
                    aqi = data['air_quality_index']
                    last_update = data['last_update']
                    predominant_parameter = data['predominant_parameter']

                    # Use model prediction if AQI is missing or invalid
                    if not aqi or aqi.strip() == '':
                        try:
                            aqi = round(predict_aqi_from_pollutants(data['pollutants']))
                        except Exception as e:
                            aqi = "N/A"

                    station_row = {
                        "Station Name": station_name,
                        "Last Update": last_update,
                        "Air Quality Index": aqi,
                        "Predominant Parameter": predominant_parameter
                    }

                    # Add pollutants as individual columns
                    for pollutant, values in data['pollutants'].items():
                        station_row[f"{pollutant} Min"] = values['min']
                        station_row[f"{pollutant} Max"] = values['max']
                        station_row[f"{pollutant} Avg"] = values['avg']

                    station_info.append(station_row)

                # Convert to DataFrame and show in interactive table
                df = pd.DataFrame(station_info)
                df.index = df.index + 1
                st.dataframe(df)

                # Create and show map
                mumbai_map = folium.Map(location=[19.0760, 72.8777], zoom_start=12)

                for data in mpcb_iitm_data:
                    station_name = data['station_name']
                    aqi = data['air_quality_index']
                    lat, lon = get_lat_lon(station_name)
                    if lat and lon:
                        folium.Marker(
                            location=[lat, lon],
                            popup=f"{station_name}<br>AQI: {aqi}",
                            icon=folium.Icon(color=get_aqi_color(aqi))
                        ).add_to(mumbai_map)

                st.markdown("### AQI Station Locations")
                folium_static(mumbai_map, width=900, height=700)

            else:
                st.write("No data available for MPCB or IITM stations in Mumbai.")

        except ET.ParseError as e:
            st.error(f"Error parsing XML: {e}")
    else:
        st.error("Failed to retrieve data.")
