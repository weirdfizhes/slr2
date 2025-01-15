import streamlit as st
from streamlit_folium import st_folium
import folium
from datetime import datetime
from app.routes import get_data, get_data_llm, upload_csv

# Set the layout to wide
st.set_page_config(layout="wide")

# Create three columns with custom widths: sidebar : map : chatbot
sidebar_col, map_col, chat_col = st.columns([1, 3, 1])

# Sidebar content (left side)
with sidebar_col:
    st.header("Location Input")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV Location", type="csv")
    st.caption("Limit 200MB per file • CSV")

    # Submit button for CSV file upload
    if st.button("Submit CSV", type="primary"):
        if uploaded_file is not None:
            try:
                prediction_data, min_longitude, max_longitude, min_latitude, max_latitude = upload_csv(uploaded_file)
                st.session_state.submitted = True
                st.session_state.coords = {
                    'point1': (min_latitude, min_longitude),
                    'point2': (max_latitude, max_longitude)
                }
                
                if prediction_data:  # Only store in session state if data is not empty
                    st.session_state.prediction_data = prediction_data
                else:
                    st.session_state.prediction_data = []  # Ensure it's a list even if empty

                st.success("CSV uploaded and processed successfully!")

            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.error("Please upload a valid CSV file.")
    
    # Coordinate inputs
    st.subheader("Manual Coordinates")

    latitude1 = st.number_input("Latitude 1", value=None, format="%.10f", step=0.000001)
    longitude1 = st.number_input("Longitude 1", value=None, format="%.10f", step=0.000001)

    latitude2 = st.number_input("Latitude 2", value=None, format="%.10f", step=0.000001)
    longitude2 = st.number_input("Longitude 2", value=None, format="%.10f", step=0.000001)

    # Add submit button with styling
    if st.button("Submit Coordinates", 
                 type="primary",  # Makes the button more prominent
                 help="Click to plot coordinates on map"):
        st.session_state.submitted = True
        st.session_state.coords = {
            'point1': (latitude1, longitude1),
            'point2': (latitude2, longitude2)
        }
        try:
            prediction_data = get_data(longitude1, latitude1, longitude2, latitude2)
            if prediction_data:  # Only store in session state if data is not empty
                st.session_state.prediction_data = prediction_data
            else:
                st.session_state.prediction_data = []  # Ensure it's a list even if empty
        except Exception as e:
            st.error(f"Error getting data: {e}")
            st.session_state.prediction_data = [] 
    else:
        if 'submitted' not in st.session_state:
            st.session_state.submitted = False
    
    # Footer
    st.markdown("---")
    st.markdown("Made with passion by RnD PLABS.ID")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

# Chatbot content (right side)
# Chatbot content (right side)
coor = False
with chat_col:
    st.header("Chat Bot")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Initialize response_data to ensure it exists
    if 'response_data' not in st.session_state:
        st.session_state.response_data = None

    # Chat message container with fixed height and scrolling
    chat_height = 400  # Adjust this value as needed

    # Create a container for chat messages
    messages_container = st.container()
    
    with messages_container:
        # Scrollable chat message area
        st.markdown(f'<div style="height: {chat_height}px; overflow-y: scroll;">', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close the scrollable div

    # Placeholder for chat input
    input_placeholder = st.empty()

    # Check if there's a prompt to process
    if 'input_index' not in st.session_state:
        st.session_state.input_index = 0  # Initialize input index

    # Chat input always at the bottom
    with input_placeholder:
        prompt = st.chat_input("Say something...", key=f"input_{st.session_state.input_index}")

    # If there's a prompt to process
    if prompt:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call the get_data_llm function directly
        response_data, coor = get_data_llm(prompt)  # Pass the user prompt to the function
        
        if coor:
            # Extracting the list of data points
            data_points = response_data['data']

            # Calculate min and max for longitude and latitude
            min_longitude = min(point['longitude'] for point in data_points)
            max_longitude = max(point['longitude'] for point in data_points)
            min_latitude = min(point['latitude'] for point in data_points)
            max_latitude = max(point['latitude'] for point in data_points)

            # Store response_data in the session state
            st.session_state.response_data = response_data

            st.session_state.submitted = True
            st.session_state.coords = {
                'point1': (min_latitude, min_longitude),
                'point2': (max_latitude, max_longitude)
            }

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            if response_data:  # Check if response_data is not empty
                assistant_response = response_data.get("response", "No response received.")
                st.markdown(assistant_response)  # Display the response directly
            else:
                st.markdown("**Error:** No data returned from the function.")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # Increment input index to ensure the next input field has a unique key
        st.session_state.input_index += 1
        
        # Clear the input placeholder to keep the input field at the bottom
        input_placeholder.empty()
        
        # Add the input field again with a new unique key
        with input_placeholder:
            st.chat_input("Say something...", key=f"input_{st.session_state.input_index}")

# Main content (map) in the middle column
with map_col:
    st.title("PLANT - PLABS Seagrass Location Analysis Network Tool")
    
    # Create a Folium map
    m = folium.Map(location=[-3, 120.393], zoom_start=5)
    
    # Add markers if coordinates have been submitted
    if hasattr(st.session_state, 'submitted') and st.session_state.submitted:
        # Define bounds for fitting the map later
        bounds = []

        # Add first marker
        point1_coords = [st.session_state.coords['point1'][0], st.session_state.coords['point1'][1]]
        folium.Marker(
            location=point1_coords,
            popup='Point 1',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        bounds.append(point1_coords)  # Add to bounds
        
        # Add second marker
        point2_coords = [st.session_state.coords['point2'][0], st.session_state.coords['point2'][1]]
        folium.Marker(
            location=point2_coords,
            popup='Point 2',
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        bounds.append(point2_coords)  # Add to bounds

        # Draw a line between the points
        folium.PolyLine(
            locations=[point1_coords, point2_coords],
            weight=2,
            color='red',
            opacity=0.8
        ).add_to(m)

        # Add circles and determine bounds for zoom
        if 'prediction_data' in st.session_state and st.session_state.prediction_data:
            for idx, prediction in enumerate(st.session_state.prediction_data['data']):
                lat = prediction['latitude']  # Adjust based on your prediction data structure
                lon = prediction['longitude']  # Adjust based on your prediction data structure
                bathy = prediction['bathy']
                pred_value = prediction['prediction']  # Access the prediction value

                # Determine the color and radius of the circle based on the prediction value
                if bathy > 0:
                    continue  # Skip this iteration if bathy > 0

                if pred_value == 0:
                    color = 'red'
                    radius = 5  # Radius in meters
                    fill_opacity = 0.6  # Visible
                elif 0 < pred_value <= 2:
                    color = 'green'
                    radius = 5  # Radius in meters
                    fill_opacity = 0.6  # Visible
                else:
                    color = 'darkgreen'  # Different shade for higher values
                    radius = 5  # Radius in meters
                    fill_opacity = 0.6  # Visible

                circle_marker = folium.CircleMarker(
                    location=(lat, lon),
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_opacity=fill_opacity,
                    popup=f'Prediction: {pred_value}<br>Bathy: {bathy}',  # Include bathy in the popup
                    key=f"circle_{idx}"  # Unique key for each circle
                ).add_to(m)

                bounds.append((lat, lon))  # Add to bounds for zooming

        # Set the view to fit the bounds of the markers
        if bounds:
            m.fit_bounds(bounds)

    # Add a Google Maps tile layer with your API key
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=True,
        control=True
    ).add_to(m)
    
    # Add a layer control
    folium.LayerControl().add_to(m)
    
    # Display the map
    st_data = st_folium(m, height=600, width=None)

    # Display analysis results at the bottom of the map
    if 'prediction_data' in st.session_state and st.session_state.prediction_data:
        analysis_result = st.session_state.prediction_data['analysis']['analysis']  # Get the analysis result
        # Render the analysis result using st.markdown
        st.markdown("### Seagrass Transplantation Suitability Analysis")
        st.markdown(analysis_result)
