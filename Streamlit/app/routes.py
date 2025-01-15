# app/routes.py

from app.db import connect_db
from src.repository_data_water_quality import get_data_water_quality
from src.SeagrassPredict import run_prediction
from src.llm_analysis import get_analysis
from src.llm_predict import get_llm_response
import pandas as pd

def get_data(longitude1, latitude1, longitude2, latitude2):
    try:
        conn = connect_db()
        data = get_data_water_quality(conn, longitude1, longitude2, latitude1, latitude2)
        conn.close()

        predictions = run_prediction(data)
        
        if not data or len(data) != len(predictions):
            raise ValueError("Data and predictions length mismatch or no data found.")
        
        for i in range(len(data)):
            data[i]['prediction'] = predictions[i]

        analysis_response = get_analysis(data)

        response_data = {
            "analysis": analysis_response,
            "data": data
        }
        
        return response_data  # Return the modified data list with predictions
    
    except Exception as e:
        # Log the error or print it for debugging purposes
        print(f"Error occurred: {str(e)}")  # Or use a logging framework
        return []  # Return an empty list on error

def get_data_llm(user_query):
    try:
        response, coor = get_llm_response(user_query)

        if coor:
            bbox = response.get("location", {}).get("bbox", {})
            try:
                longitude1 = bbox.get('lon1')
                latitude1 = bbox.get('lat1')
                longitude2 = bbox.get('lon2')
                latitude2 = bbox.get('lat2')

                # Connect to the database and get data
                conn = connect_db()
                data = get_data_water_quality(conn, longitude1, longitude2, latitude1, latitude2)
                conn.close()

                # Run the prediction directly on the retrieved data
                predictions = run_prediction(data)
        
                if not data or len(data) != len(predictions):
                    raise ValueError("Data and predictions length mismatch or no data found.")
                
                for i in range(len(data)):
                    data[i]['prediction'] = predictions[i]

                analysis_response = get_analysis(data)

                response_data = {
                    "query": response.get("query"),
                    "response": response.get("response"),
                    "analysis": analysis_response,  # Add the analysis
                    "data": data  # Merge water quality data and predictions
                }
                
                return response_data, coor

            except Exception as e:
                print(f"Error occurred: {str(e)}")  # Or use a logging framework
                return []  # Return an empty list on error

        else:
            response_data = {
                    "query": response.get("query"),
                    "response": response.get("response"),
                    "analysis": "No analysis",  # Add the analysis
                    "data": ""  # Merge water quality data and predictions
            }

            return response_data, coor

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Or use a logging framework
        return []  # Return an empty list on error

def upload_csv(file):
    try:
        # Read the CSV into a DataFrame
        df = pd.read_csv(file)

        # Normalize column names
        column_mapping = {
            'latitude': 'latitude', 
            'longitude': 'longitude',
            'temp': 'temp',
            'do': 'do', 'DO': 'do', '"do"': 'do',
            'ph': 'ph', 'pH': 'ph', '"ph"': 'ph',
            'salinity': 'salinity',
            'tss': 'tss',
            'bathy': 'bathy', 'bathymetry': 'bathy'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Ensure required columns are present
        required_columns = ['latitude', 'longitude', 'temp', 'salinity', 'do', 'ph', 'tss', 'bathy']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        # Reorder the DataFrame
        df = df[required_columns]

        # Process the cleaned data (implement your cleaning logic here)
        cleaned_data = df.copy()  # For example, just using the cleaned DataFrame directly

        # Optionally: Run predictions on the cleaned data
        new_data = cleaned_data.to_dict('records')

        predictions = run_prediction(new_data)
        for i, row in enumerate(new_data):
            row['prediction'] = predictions[i]

        analysis_response = get_analysis(new_data)

        response_data = {
                "analysis": analysis_response,  # Add the analysis
                "data": new_data  # Return all water quality data and predictions
            }

        data_points = response_data['data']

        # Calculate min and max for longitude and latitude
        min_longitude = min(point['longitude'] for point in data_points)
        max_longitude = max(point['longitude'] for point in data_points)
        min_latitude = min(point['latitude'] for point in data_points)
        max_latitude = max(point['latitude'] for point in data_points)

        return response_data, min_longitude, max_longitude, min_latitude, max_latitude

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Or use a logging framework
        return []  # Return an empty list on error