import streamlit as st
import pandas as pd
import google.generativeai as genai
import fitz
import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import io
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import time

# Configure page
st.set_page_config(
    page_title="Scientific Paper Analyzer",
    page_icon="📚",
    layout="wide",
)

# Initialize session state
if 'papers_metadata' not in st.session_state:
    st.session_state.papers_metadata = []

# Initialize Google AI
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.0-pro")

# Load SpaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.warning("Installing required language model...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def extract_text_from_pdf(pdf_file):
    """Extract text from a single PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
        return ""

def extract_research_sites(text):
    """Extract research sites with focus on natural/geographical locations."""
    doc = nlp(text)
    locations = []
    
    # Enhanced patterns focusing on field sites
    site_patterns = [
        r"(?:study|research|field|sample|sampling)[\s]+(?:site|area|location|region|island)s?[\s]+(?:was|were|is|are)[\s]+(?:in|at|near|on)[\s]+(?P<loc>[^\.]+)",
        r"(?:conducted|performed|carried out|took place)[\s]+(?:in|at|near|on)[\s]+(?P<loc>[^\.]+)",
        r"(?:collected|gathered|obtained)[\s]+(?:from|in|at|near|on)[\s]+(?P<loc>[^\.]+)",
        r"(?:located|situated)[\s]+(?:in|at|on)[\s]+(?P<loc>[^\.]+)",
        r"(?:the|our)[\s]+(?:study|research|field)[\s]+(?:was|were)[\s]+(?:in|at|near|on)[\s]+(?P<loc>[^\.]+)",
        r"(?:national[\s]+park|forest|mountain|island|reserve|protected[\s]+area)[\s]+(?:of|called|named)?[\s]+(?P<loc>[^\.]+)"
    ]
    
    # Extract locations from patterns
    for pattern in site_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            loc = match.group('loc').strip()
            # Filter out institutional locations and common words
            if not any(word in loc.lower() for word in [
                'university', 'laboratory', 'lab', 'institute', 'department', 
                'faculty', 'center', 'centre', 'building', 'campus'
            ]):
                locations.append(loc)
    
    # Extract geographical entities
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            # Filter out institutional locations
            if not any(word in ent.text.lower() for word in [
                'university', 'laboratory', 'lab', 'institute', 'department', 
                'faculty', 'center', 'centre', 'building', 'campus'
            ]):
                locations.append(ent.text)
    
    return list(set(locations))

def extract_coordinates_from_text(text):
    """Extract coordinates from text if explicitly mentioned."""
    # Pattern for decimal degrees
    dd_pattern = r'(?P<lat>-?\d+\.?\d*)[°\s]*[NS][,\s]+(?P<lon>-?\d+\.?\d*)[°\s]*[EW]'
    
    # Pattern for degrees, minutes, seconds
    dms_pattern = r'(\d+)°\s*(\d+)\'\s*(\d+(?:\.\d+)?)"([NS])[,\s]+(\d+)°\s*(\d+)\'\s*(\d+(?:\.\d+)?)"([EW])'
    
    # Check for decimal degrees
    dd_match = re.search(dd_pattern, text, re.IGNORECASE)
    if dd_match:
        lat = float(dd_match.group('lat'))
        lon = float(dd_match.group('lon'))
        return {'latitude': lat, 'longitude': lon}
    
    # Check for DMS
    dms_match = re.search(dms_pattern, text)
    if dms_match:
        lat = convert_dms_to_dd(
            float(dms_match.group(1)),
            float(dms_match.group(2)),
            float(dms_match.group(3)),
            dms_match.group(4)
        )
        lon = convert_dms_to_dd(
            float(dms_match.group(5)),
            float(dms_match.group(6)),
            float(dms_match.group(7)),
            dms_match.group(8)
        )
        return {'latitude': lat, 'longitude': lon}
    
    return None

def convert_dms_to_dd(degrees, minutes, seconds, direction):
    """Convert degrees, minutes, seconds to decimal degrees."""
    dd = degrees + minutes/60 + seconds/3600
    if direction in ['S', 'W']:
        dd = -dd
    return dd

def safe_geocode(location_name, max_retries=3):
    """Safely geocode a location with retries and error handling."""
    geolocator = Nominatim(user_agent="scientific_paper_analyzer")
    
    # Try with the original location name
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(location_name, exactly_one=True)
            if location:
                return {
                    'name': location_name,
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'address': location.address
                }
            time.sleep(1)  # Respect rate limits
            
            # If no result, try with "National Park" or other common suffixes
            for suffix in ['National Park', 'Forest', 'Mountain', 'Island', 'Nature Reserve']:
                modified_name = f"{location_name} {suffix}"
                location = geolocator.geocode(modified_name, exactly_one=True)
                if location:
                    return {
                        'name': location_name,
                        'latitude': location.latitude,
                        'longitude': location.longitude,
                        'address': location.address
                    }
                time.sleep(1)
            
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(2)  # Wait longer between retries
        except Exception as e:
            st.warning(f"Error geocoding {location_name}: {str(e)}")
    
    # Try to extract coordinates from the location name as a last resort
    coords = extract_coordinates_from_text(location_name)
    if coords:
        return {
            'name': location_name,
            'latitude': coords['latitude'],
            'longitude': coords['longitude'],
            'address': location_name
        }
    
    return None

def parse_metadata_response(response_text):
    """Parse the AI response into a structured metadata dictionary."""
    metadata = {
        'Title': '',
        'Authors': '',
        'Year': '',
        'Research Location': '',
        'Latitude': None,
        'Longitude': None,
        'Methods Used': [],
        'Key Findings': '',
        'Keywords': []
    }
    
    current_field = None
    for line in response_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is a field header
        for field in metadata.keys():
            if line.lower().startswith(f"{field.lower()}:"):
                current_field = field
                value = line[len(field)+1:].strip()
                if field in ['Methods Used', 'Keywords']:
                    metadata[field] = [v.strip() for v in value.split(',') if v.strip()]
                else:
                    metadata[field] = value
                break
        else:
            if current_field and line:
                # Append to current field
                if current_field in ['Methods Used', 'Keywords']:
                    metadata[current_field].extend([v.strip() for v in line.split(',') if v.strip()])
                else:
                    metadata[current_field] += ' ' + line
    
    return metadata

def extract_metadata(text, filename):
    """Enhanced metadata extraction with improved location detection."""
    # Extract potential research sites
    locations = extract_research_sites(text)
    
    # Use AI to help identify the most relevant research site
    location_prompt = f"""
    From these potential research locations found in the paper:
    {', '.join(locations)}
    
    Which one is most likely the main research site? Consider:
    1. Must be a geographical location (island, forest, mountain, etc.)
    2. Cannot be an institution (university, laboratory, etc.)
    3. Should be the most specific location mentioned
    4. Should be where the actual field work or data collection occurred
    
    Return ONLY the name of the most likely main research site.
    If none are valid research sites, return "Unknown".
    """
    
    try:
        location_response = model.generate_content(location_prompt)
        main_location = location_response.text.strip()
    except:
        main_location = locations[0] if locations else "Unknown"
    
    # Get coordinates using multiple methods
    location_data = safe_geocode(main_location) if main_location != "Unknown" else None
    
    # Regular metadata extraction
    prompt = f"""
    Analyze this scientific paper and extract the following metadata:
    1. Title
    2. Authors
    3. Year
    4. Research Location (use this location: {main_location})
    5. Methods Used
    6. Key Findings
    7. Keywords

    For the Research Location, focus ONLY on the geographical location where the actual research was conducted (forest, island, mountain, etc.), NOT institutional locations.

    Please format your response exactly like this:
    Title: [paper title]
    Authors: [authors]
    Year: [year]
    Research Location: {main_location}
    Methods Used: [method1], [method2], [method3]
    Key Findings: [key findings]
    Keywords: [keyword1], [keyword2], [keyword3]

    Paper text:
    {text[:8000]}
    """
    
    response = model.generate_content(prompt)
    metadata = parse_metadata_response(response.text)
    
    # Add location data if available
    if location_data:
        metadata['Research Location'] = location_data['name']
        metadata['Latitude'] = location_data['latitude']
        metadata['Longitude'] = location_data['longitude']
        metadata['Full Address'] = location_data['address']
    
    return metadata

def create_connection_graph(papers_metadata):
    """Create a network graph showing connections between papers."""
    G = nx.Graph()
    
    # Create nodes for each paper
    for paper in papers_metadata:
        title = paper.get('Title', 'Untitled Paper')
        G.add_node(title, type='paper')
        
        # Add method nodes and connections
        for method in paper.get('Methods Used', []):
            method = method.strip()
            if method:
                G.add_node(method, type='method')
                G.add_edge(title, method)
        
        # Add location nodes and connections
        location = paper.get('Research Location')
        if location:
            G.add_node(location, type='location')
            G.add_edge(title, location)
            
    return G

def generate_synthesis(papers_metadata):
    """Generate a synthesis of all papers using Google AI."""
    papers_summary = "\n\n".join([ 
        f"Paper: {paper.get('Title', 'Untitled')}\nYear: {paper.get('Year', 'N/A')}\nKey Findings: {paper.get('Key Findings', 'N/A')}"
        for paper in papers_metadata
    ])
    
    prompt = f"""
    Based on these scientific papers, write a comprehensive synthesis that:
    1. Identifies common themes and patterns
    2. Discusses methodological approaches
    3. Summarizes key findings
    4. Suggests future research directions

    Papers information:
    {papers_summary}
    """
    
    response = model.generate_content(prompt)
    return response.text

def generate_new_paper(papers_metadata):
    """Generate a new research paper based on analyzed papers."""
    papers_summary = "\n\n".join([ 
        f"Paper: {paper.get('Title', 'Untitled')}\n"
        f"Methods: {', '.join(paper.get('Methods Used', []))}\n"
        f"Findings: {paper.get('Key Findings', 'N/A')}"
        for paper in papers_metadata
    ])
    
    prompt = f"""
    Based on the analysis of these papers, generate a new research paper that:
    1. Identifies a research gap
    2. Proposes a novel methodology
    3. Predicts potential findings
    4. Discusses implications

    Format the paper with these sections:
    - Title
    - Abstract
    - Introduction
    - Literature Review
    - Methodology
    - Expected Results
    - Discussion
    - Conclusion
    
    Reference Papers:
    {papers_summary}
    """
    
    response = model.generate_content(prompt)
    return response.text

# Main UI
st.title("📚 Scientific Paper Analyzer")

# File upload
uploaded_files = st.file_uploader(
    "Upload scientific papers (PDF)",
    type=['pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    # Process button
    if st.button("Process Papers"):
        # Clear previous results
        st.session_state.papers_metadata = []
        
        # Process each paper
        with st.spinner("Analyzing papers..."):
            for pdf_file in uploaded_files:
                text = extract_text_from_pdf(pdf_file)
                metadata = extract_metadata(text, pdf_file.name)
                st.session_state.papers_metadata.append(metadata)
        
        # Display success message
        st.success(f"Processed {len(uploaded_files)} papers successfully!")

# If we have processed papers, show the analysis
if st.session_state.papers_metadata:
    # Display paper connections
    st.subheader("Paper Connections")
    G = create_connection_graph(st.session_state.papers_metadata)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=8, font_weight='bold')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    st.image(buf)
    plt.close()
    
    # Generate synthesis
    st.subheader("Research Synthesis")
    if 'synthesis' not in st.session_state:
        st.session_state.synthesis = generate_synthesis(st.session_state.papers_metadata)
    st.write(st.session_state.synthesis)
    
    # Generate new paper button
    if st.button("Generate New Paper"):
        with st.spinner("Generating new paper..."):
            new_paper = generate_new_paper(st.session_state.papers_metadata)
            st.subheader("Generated Research Paper")
            st.write(new_paper)
            
            # Download buttons
            st.download_button(
                label="Download Generated Paper",
                data=new_paper,
                file_name="generated_paper.txt",
                mime="text/plain"
            )
    
    # Download synthesis button
    st.download_button(
        label="Download Synthesis",
        data=st.session_state.synthesis,
        file_name="research_synthesis.txt",
        mime="text/plain"
    )

    # Display map with research locations
    st.subheader("Research Locations")
    
    # Create a DataFrame for the map
    map_data = []
    for paper in st.session_state.papers_metadata:
        lat = paper.get('Latitude')
        lon = paper.get('Longitude')
        if lat and lon:
            map_data.append({
                'lat': lat,
                'lon': lon,
                'title': paper.get('Title', 'Untitled'),
                'location': paper.get('Research Location', 'Unknown'),
                'address': paper.get('Full Address', 'Not available')
            })
    
    if map_data:
        map_df = pd.DataFrame(map_data)
        st.map(map_df, latitude='lat', longitude='lon')
        
        # Display location details in an expanded table
        st.subheader("Research Location Details")
        location_df = pd.DataFrame(map_data)
        st.dataframe(
            location_df,
            column_config={
                "title": "Paper Title",
                "location": "Research Site",
                "address": "Full Address",
                "lat": "Latitude",
                "lon": "Longitude"
            },
            hide_index=True
        )
    else:
        st.error("No research locations could be identified and geocoded from the papers.")
        st.write("Extracted location mentions:")
        for paper in st.session_state.papers_metadata:
            st.write(f"- {paper.get('Title', 'Untitled')}: {paper.get('Research Location', 'No location found')}")