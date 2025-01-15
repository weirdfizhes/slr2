import streamlit as st
import pandas as pd
import google.generativeai as genai
import pymupdf
import networkx as nx
import matplotlib.pyplot as plt
import io
from opencage.geocoder import OpenCageGeocode
import spacy
import requests
import seaborn as sns
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from collections import defaultdict
import plotly.graph_objects as go



# Configure page
st.set_page_config(
    page_title="Scientific Paper Analyzer",
    page_icon="📚",
    layout="wide",
)

# Initialize session state
if 'papers_metadata' not in st.session_state:
    st.session_state.papers_metadata = []

# Initialize Google AI and OpenCage
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENCAGE_API_KEY = st.secrets["OPENCAGE_API_KEY"]
SCOPUS_API_KEY = st.secrets["SCOPUS_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.0-pro")
geocoder = OpenCageGeocode(OPENCAGE_API_KEY)

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:  # noqa: E722
        st.warning("Installing required language model...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
        return ""

def safe_geocode(location_str):
    """Safely geocode a location string using OpenCage."""
    if not location_str or location_str.lower() in ['unknown', 'various', 'multiple']:
        return None
        
    try:
        time.sleep(1)  # Rate limiting
        result = geocoder.geocode(location_str)
        
        if result and len(result):
            # Get the first result
            location = result[0]
            return {
                'latitude': location['geometry']['lat'],
                'longitude': location['geometry']['lng'],
                'address': location['formatted'],
                'country': location['components'].get('country'),
                'confidence': location['confidence']
            }
    except Exception as e:
        st.warning(f"Geocoding error for {location_str}: {e}")
    return None

def get_paper_category(text):
    """Determine paper's research field."""
    prompt = """
    Analyze this scientific paper and determine its main research field/category.
    Return ONLY the most specific appropriate category name, no explanation.
    """
    try:
        response = model.generate_content(prompt + f"\n\nPaper text:\n{text[:3000]}")
        return response.text.strip()
    except Exception:
        return "Uncategorized"

def extract_metadata(text, filename):
    """Extract paper metadata."""
    category = get_paper_category(text)
    
    prompt = f"""
    Analyze this scientific paper and extract:
    1. Title
    2. Authors
    3. Year
    4. Research Field: {category}
    5. Research Location (only geographical location where research was conducted)
    6. Methods Used
    7. Key Findings
    8. Keywords

    Format as:
    Title: [paper title]
    Authors: [authors]
    Year: [year]
    Research Field: {category}
    Research Location: [location]
    Methods Used: [method1], [method2], [method3]
    Key Findings: [key findings]
    Keywords: [keyword1], [keyword2], [keyword3]
    """
    
    response = model.generate_content(prompt + f"\n\nPaper text:\n{text[:8000]}")
    metadata = {}
    
    for line in response.text.split('\n'):
        if ': ' in line:
            key, value = line.split(': ', 1)
            if key in ['Methods Used', 'Keywords']:
                metadata[key] = [v.strip() for v in value.split(',')]
            else:
                metadata[key] = value.strip()
    
    # Get coordinates if location is available
    if 'Research Location' in metadata:
        location_data = safe_geocode(metadata['Research Location'])
        if location_data:
            metadata.update(location_data)
    
    return metadata


def calculate_similarity(paper1, paper2):
    """Calculate similarity between papers."""
    score = 0
    weights = {
        'field': 0.3,
        'methods': 0.3,
        'keywords': 0.2,
        'year': 0.2
    }
    
    if paper1.get('Research Field') == paper2.get('Research Field'):
        score += weights['field']
    
    methods1 = set(paper1.get('Methods Used', []))
    methods2 = set(paper2.get('Methods Used', []))
    if methods1 or methods2:
        score += weights['methods'] * len(methods1.intersection(methods2)) / len(methods1.union(methods2))
    
    keywords1 = set(paper1.get('Keywords', []))
    keywords2 = set(paper2.get('Keywords', []))
    if keywords1 or keywords2:
        score += weights['keywords'] * len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
    
    try:
        year1 = int(paper1.get('Year', 0))
        year2 = int(paper2.get('Year', 0))
        if year1 and year2:
            year_diff = abs(year1 - year2)
            if year_diff == 0:
                score += weights['year']
            elif year_diff <= 5:
                score += weights['year'] * (1 - year_diff/5)
    except ValueError:
        pass
    
    return score

def create_paper_network(papers_metadata):
    """Create network visualization of paper connections."""
    G = nx.Graph()

    # Add nodes
    for paper in papers_metadata:
        title = paper.get('Title', 'Untitled')
        field = paper.get('Research Field', 'Unknown')
        G.add_node(title, field=field)

    # Add edges (if applicable)
    for i, paper1 in enumerate(papers_metadata):
        for paper2 in papers_metadata[i+1:]:
            similarity = calculate_similarity(paper1, paper2)
            if similarity > 0:
                G.add_edge(paper1.get('Title', 'Untitled'), paper2.get('Title', 'Untitled'), weight=similarity)

    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, with_labels=True, node_color='lightblue',
        edge_color='gray', node_size=3000, font_size=10
    )
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def create_interactive_paper_network(papers_metadata):
    """Create interactive network visualization of paper connections."""
    G = nx.Graph()
    
    # Create a mapping of papers for easy lookup
    paper_lookup = {paper.get('Title', 'Untitled'): paper for paper in papers_metadata}
    
    # Add nodes
    for paper in papers_metadata:
        title = paper.get('Title', 'Untitled')
        field = paper.get('Research Field', 'Unknown')
        G.add_node(title, 
                  field=field,
                  year=paper.get('Year', 'Unknown'),
                  methods=', '.join(paper.get('Methods Used', [])),
                  keywords=', '.join(paper.get('Keywords', [])))
    
    # Add edges
    for i, paper1 in enumerate(papers_metadata):
        for paper2 in papers_metadata[i+1:]:
            similarity = calculate_similarity(paper1, paper2)
            if similarity > 0:
                G.add_edge(
                    paper1.get('Title', 'Untitled'),
                    paper2.get('Title', 'Untitled'),
                    weight=similarity
                )
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1.5)
    
    # Prepare node traces by field
    fields = list(set(nx.get_node_attributes(G, 'field').values()))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(fields)))
    field_colors = dict(zip(fields, colors))
    
    node_traces = []
    
    for field in fields:
        field_nodes = [node for node, attr in G.nodes(data=True) 
                      if attr.get('field') == field]
        
        x_coords = [pos[node][0] for node in field_nodes]
        y_coords = [pos[node][1] for node in field_nodes]
        
        # Create hover text
        hover_texts = []
        for node in field_nodes:
            paper = paper_lookup[node]
            hover_text = f"""
            Title: {node}<br>
            Field: {paper.get('Research Field', 'Unknown')}<br>
            Year: {paper.get('Year', 'Unknown')}<br>
            Methods: {paper.get('Methods Used', [])}<br>
            Keywords: {paper.get('Keywords', [])}
            """
            hover_texts.append(hover_text)
        
        node_trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            hoverinfo='text',
            text=[node[:20] + '...' if len(node) > 20 else node for node in field_nodes],
            textposition="top center",
            hovertext=hover_texts,
            marker=dict(
                size=30,
                color=f'rgba({int(field_colors[field][0]*255)},{int(field_colors[field][1]*255)},{int(field_colors[field][2]*255)},0.8)',
                line=dict(width=2),
                symbol='circle'
            ),
            name=field,
            showlegend=True
        )
        node_traces.append(node_trace)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_texts = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        similarity = edge[2]['weight']
        hover_text = f"""
        Connection between:<br>
        {edge[0]}<br>
        {edge[1]}<br>
        Similarity: {similarity:.2f}
        """
        edge_texts.extend([hover_text, hover_text, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        hovertext=edge_texts,
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title='Interactive Paper Similarity Network',
            titlefont=dict(size=16),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                title='Research Fields',
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Reset View",
                            method="relayout",
                            args=[{"xaxis.range": None, 
                                  "yaxis.range": None}]
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    x=0,
                    xanchor="left",
                    y=0,
                    yanchor="top"
                )
            ]
        )
    )
    
    return fig

def save_interactive_visualization(fig, format='html'):
    """Save the interactive visualization in specified format."""
    if format == 'html':
        return fig.to_html()
    else:
        buf = io.BytesIO()
        fig.write_image(buf, format=format)
        buf.seek(0)
        return buf

def create_keyword_heatmap(papers_metadata):
    """Create heatmap of keyword co-occurrences."""
    # Get all unique keywords
    all_keywords = set()
    for paper in papers_metadata:
        keywords = paper.get('Keywords', [])
        all_keywords.update([k.strip() for k in keywords if k.strip()])
    
    # Create co-occurrence matrix
    keyword_list = list(all_keywords)
    matrix = np.zeros((len(keyword_list), len(keyword_list)))
    
    for paper in papers_metadata:
        keywords = [k.strip() for k in paper.get('Keywords', []) if k.strip()]
        for i, kw1 in enumerate(keyword_list):
            for j, kw2 in enumerate(keyword_list):
                if i != j and kw1 in keywords and kw2 in keywords:
                    matrix[i][j] += 1
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, 
                xticklabels=keyword_list,
                yticklabels=keyword_list,
                cmap='YlOrRd',
                annot=True,
                fmt='g')
    
    plt.title('Keyword Co-occurrence Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_synthesis(papers_metadata):
    """Generate research synthesis."""
    papers_summary = "\n\n".join([
        f"Title: {paper.get('Title', 'Untitled')}\n"
        f"Field: {paper.get('Research Field', 'Unknown')}\n"
        f"Key Findings: {paper.get('Key Findings', 'Not available')}"
        for paper in papers_metadata
    ])
    
    prompt = f"""
    Generate a comprehensive research synthesis for these papers:
    {papers_summary}
    
    Include:
    1. Common themes and patterns
    2. Major findings and implications
    3. Research gaps
    4. Methodological insights
    
    Format with clear paragraphs and headings.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating synthesis: {str(e)}"

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

def get_summary_table(papers_metadata):
        """Return a DataFrame summarizing key metadata for each paper."""
        data = []
        for paper in papers_metadata:
            data.append({
                'Title': paper.get('Title', ''),
                'Authors': paper.get('Authors', ''),
                'Year': paper.get('Year', ''),
                'Field': paper.get('Research Field', ''),
                'Methods': ", ".join(paper.get('Methods Used', [])),
                'Keywords': ", ".join(paper.get('Keywords', []))
            })
        return pd.DataFrame(data)

def create_year_timeline(papers_metadata):
    """Create a timeline visualization showing papers by year with authors."""
    # Extract years and convert to integers
    years_data = []
    for paper in papers_metadata:
        try:
            year = int(paper.get('Year', 0))
            if year > 0:
                years_data.append({
                    'Year': year,
                    'Authors': paper.get('Authors', 'Unknown Authors'),
                    'Field': paper.get('Research Field', 'Unknown')
                })
        except ValueError:
            continue
    
    if not years_data:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(years_data)
    
    # Create visualization with larger figure size and adjusted margins
    plt.figure(figsize=(15, 8))  # Increased height
    
    # Create scatter plot with different colors for different fields
    fields = df['Field'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(fields)))
    
    # Add some padding to the year range
    year_min = df['Year'].min() - 1
    year_max = df['Year'].max() + 1
    plt.xlim(year_min, year_max)
    
    for field, color in zip(fields, colors):
        field_data = df[df['Field'] == field]
        plt.scatter(field_data['Year'], [1]*len(field_data), 
                   label=field, c=[color], s=100)
    
    # Add author names as annotations with improved spacing
    for _, row in df.iterrows():
        # Shorten author list if too long
        authors = row['Authors']
        if len(authors) > 50:  # Truncate if too long
            authors = authors[:47] + "..."
            
        plt.annotate(authors, 
                    (row['Year'], 1),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    rotation=45,
                    fontsize=8)  # Reduced font size
    
    plt.title('Paper Timeline by Authors')
    plt.xlabel('Year')
    plt.ylabel('')
    plt.yticks([])
    
    # Adjust legend position and layout
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Adjust layout with specific padding
    plt.subplots_adjust(bottom=0.2, right=0.85, top=0.95)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close()
    return buf

def create_method_distribution(papers_metadata):
    """Create a bar chart showing the distribution of research methods."""
    method_counts = defaultdict(int)
    for paper in papers_metadata:
        methods = paper.get('Methods Used', [])
        for method in methods:
            method = method.strip()
            if method:
                method_counts[method] += 1
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    methods = list(method_counts.keys())
    counts = list(method_counts.values())
    
    plt.bar(methods, counts, color='lightgreen')
    plt.title('Distribution of Research Methods')
    plt.xlabel('Method')
    plt.ylabel('Number of Papers')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def extract_info_from_abstract(abstract_text):
    """
    Given an abstract, use the LLM to extract:
      - Methods Used
      - Keywords
      - Research Field
    """
    prompt = f"""
    You have an abstract from a scientific paper. Extract the following:
    - Methods Used (list them, comma-separated)
    - Keywords (list them, comma-separated)
    - Research Field (one short phrase)

    Format them as:
    Methods: [method1], [method2]
    Keywords: [keyword1], [keyword2]
    Research Field: [field]

    Abstract:
    {abstract_text}
    """
    try:
        response = model.generate_content(prompt)
        lines = response.text.splitlines()
        info = {"Methods": [], "Keywords": [], "Research Field": ""}

        for line in lines:
            line = line.strip()
            if line.lower().startswith("methods:"):
                methods_str = line.split(":", 1)[1].strip()
                info["Methods"] = [m.strip() for m in methods_str.split(",") if m.strip()]
            elif line.lower().startswith("keywords:"):
                keywords_str = line.split(":", 1)[1].strip()
                info["Keywords"] = [k.strip() for k in keywords_str.split(",") if k.strip()]
            elif line.lower().startswith("research field:"):
                info["Research Field"] = line.split(":", 1)[1].strip()
        
        return info
    except Exception as e:
        st.warning(f"Error extracting info from abstract: {str(e)}")
        return {
            "Methods": [],
            "Keywords": [],
            "Research Field": ""
        }
        
def search_scopus_by_keywords(metadata):
    """
    Search for similar papers using the extracted keywords from a single paper's metadata.
    1) Build query from metadata['Keywords']
    2) Fetch the abstract from Scopus results
    3) Use extract_info_from_abstract() to parse each abstract for methods, keywords, and field

    Returns a list of dictionaries, each containing:
      - Title
      - Authors
      - Year
      - DOI
      - Abstract
      - Extracted: { Methods, Keywords, Research Field }
    """
    # Gather keywords from this paper's metadata
    paper_keywords = metadata.get("Keywords", [])
    if not paper_keywords:
        return []

    # Build query string from keywords
    # Example: for keywords ["machine learning", "classification"] => TITLE-ABS-KEY("machine learning") AND TITLE-ABS-KEY("classification")
    query_parts = [f'TITLE-ABS-KEY("{kw}")' for kw in paper_keywords]
    combined_query = " AND ".join(query_parts)
    
    # Prepare request
    base_url = "https://api.elsevier.com/content/search/scopus"
    headers = {"X-ELS-APIKey": SCOPUS_API_KEY}
    params = {
        "query": f"({combined_query}) AND openaccess(1)",
        "count": 5,  # adjust how many results you want
        "field": "dc:title,dc:creator,prism:coverDate,prism:doi,dc:description"
    }

    similar_papers = []
    try:
        resp = requests.get(base_url, headers=headers, params=params)
        if resp.ok:
            entries = resp.json().get("search-results", {}).get("entry", [])
            for entry in entries:
                title = entry.get("dc:title", "")
                authors = entry.get("dc:creator", "")
                year = entry.get("prism:coverDate", "")[:4]
                doi = entry.get("prism:doi", "")
                abstract = entry.get("dc:description", "")

                # Extract methods, keywords, field from the abstract
                extracted_info = extract_info_from_abstract(abstract)

                similar_papers.append({
                    "Title": title,
                    "Authors": authors,
                    "Year": year,
                    "DOI": doi,
                    "Abstract": abstract,
                    "Extracted Methods": extracted_info["Methods"],
                    "Extracted Keywords": extracted_info["Keywords"],
                    "Extracted Field": extracted_info["Research Field"],
                })
        else:
            st.warning(f"Scopus request failed with status code {resp.status_code}")
    except Exception as e:
        st.error(f"Error searching Scopus with keywords: {str(e)}")

    return similar_papers

# Main UI
st.title("📚 Scientific Paper Analyzer")

uploaded_files = st.file_uploader(
    "Upload scientific papers (PDF)",
    type=['pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Papers"):
        st.session_state.papers_metadata = []
        
        with st.spinner("Analyzing papers..."):
            for pdf_file in uploaded_files:
                text = extract_text_from_pdf(pdf_file)
                metadata = extract_metadata(text, pdf_file.name)
                st.session_state.papers_metadata.append(metadata)
        
        st.success(f"Processed {len(uploaded_files)} papers successfully!")

if st.session_state.papers_metadata:

    # Update analyzer with processed papers
    analyzer = st.session_state.papers_metadata

    st.subheader("Summary Table of Papers")
    summary_df = get_summary_table(analyzer)
    st.dataframe(summary_df)

    st.subheader("Visualizations")
    viz_option = st.selectbox(
        "Choose visualization type",
        ["Paper Connections", "Timeline", "Keyword Heatmap", "Method Distribution"]
    )

    if viz_option == "Paper Connections":
        # Paper network
        st.subheader("Paper Similarity Network")
        st.write("Node colors represent research fields. Line thickness shows similarity based on:")
        st.write("- Research field (30%)")
        st.write("- Shared methods (30%)")
        st.write("- Shared keywords (20%)")
        st.write("- Temporal proximity (20%)")
        st.image(create_paper_network(st.session_state.papers_metadata), use_column_width=True)


    elif viz_option == "Timeline":
        st.image(create_year_timeline(st.session_state.papers_metadata))

    elif viz_option == "Keyword Heatmap":
        # Keyword heatmap
        st.subheader("Keyword Analysis")
        st.write("This heatmap shows how frequently keywords appear together across papers.")
        st.image(create_keyword_heatmap(st.session_state.papers_metadata))

    elif viz_option == "Method Distribution":
        st.image(create_method_distribution(st.session_state.papers_metadata))
    
    # Place the slider in the main page
    st.subheader("Configuration")
    confidence_threshold = st.slider(
        "Geocoding Confidence Threshold",
        min_value=0,
        max_value=10,
        value=5,
        help="Filter out locations with confidence score below this threshold"
    )

    # Research locations map
    st.subheader("Research Locations")
    map_data = []
    for paper in st.session_state.papers_metadata:
        if ('latitude' in paper and 'longitude' in paper and 
            paper.get('confidence', 0) >= confidence_threshold):
            map_data.append({
                'lat': paper['latitude'],
                'lon': paper['longitude'],
                'title': paper.get('Title', 'Untitled'),
                'location': paper.get('Research Location', 'Unknown'),
                'country': paper.get('country', 'Unknown')
            })
    
    if map_data:
        map_df = pd.DataFrame(map_data)
        st.map(map_df, latitude='lat', longitude='lon')
        
        # Show location details
        st.write("Research Location Details:")
        for loc in map_data:
            st.write(f"- **{loc['title']}**: {loc['location']} ({loc['country']})")
    else:
        st.info("No location data available or all locations below confidence threshold.")
    
    
    # Research synthesis
    st.subheader("Research Synthesis")
    synthesis = generate_synthesis(st.session_state.papers_metadata)
    st.write(synthesis)

    
    # Download button
    st.download_button(
        label="Download Synthesis",
        data=synthesis,
        file_name="research_synthesis.txt",
        mime="text/plain"
    )

    if st.button("Generate New Paper"):
        with st.spinner("Generating new paper..."):
            new_paper = generate_new_paper(st.session_state.papers_metadata)
            st.subheader("Generated Research Paper")
            st.write(new_paper)
            st.download_button("Download Generated Paper", data=new_paper, file_name="generated_paper.txt", mime="text/plain")

def calculate_similarity_detailed(paper1, paper2):
    """Calculate detailed similarity between papers with individual scores."""
    scores = {
        'field': 0,
        'methods': 0,
        'keywords': 0,
        'year': 0
    }
    
    weights = {
        'field': 0.3,
        'methods': 0.3,
        'keywords': 0.2,
        'year': 0.2
    }
    
    # Field similarity
    if paper1.get('Research Field') == paper2.get('Research Field'):
        scores['field'] = 1.0
    
    # Methods similarity
    methods1 = set(paper1.get('Methods Used', []))
    methods2 = set(paper2.get('Methods Used', []))
    if methods1 or methods2:
        scores['methods'] = len(methods1.intersection(methods2)) / len(methods1.union(methods2))
    
    # Keywords similarity
    keywords1 = set(paper1.get('Keywords', []))
    keywords2 = set(paper2.get('Keywords', []))
    if keywords1 or keywords2:
        scores['keywords'] = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
    
    # Year proximity
    try:
        year1 = int(paper1.get('Year', 0))
        year2 = int(paper2.get('Year', 0))
        if year1 and year2:
            year_diff = abs(year1 - year2)
            if year_diff == 0:
                scores['year'] = 1.0
            elif year_diff <= 5:
                scores['year'] = 1 - year_diff/5
    except ValueError:
        scores['year'] = 0
    
    # Calculate weighted total
    total_score = sum(scores[key] * weights[key] for key in scores)
    
    return {
        'total_score': total_score,
        'individual_scores': scores,
        'common_methods': methods1.intersection(methods2),
        'common_keywords': keywords1.intersection(keywords2)
    }

@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


def format_citation(title, authors, year):
    """Format citation in the requested style: Title (Author, Year)"""
    # Extract first author's last name
    if isinstance(authors, str):
        author = authors.split(',')[0].split()[-1]
    else:
        author = "Unknown"
    
    year = str(year) if year else "n.d."
    return f"{title} ({author}, {year})"

def format_citation_with_link(title, authors, year, doi):
    """Format citation with link in the requested style: Title (Author, Year) link: URL"""
    if isinstance(authors, str):
        author = authors.split(',')[0].split()[-1]
    else:
        author = "Unknown"
    
    year = str(year) if year else "n.d."
    citation = f"{title} ({author}, {year})"
    
    if doi:
        link = f"https://doi.org/{doi}"
    else:
        link = "No link available"
        
    return f"{citation} link: {link}"

def search_scopus_categorized(metadata):
    """Search for similar papers in Scopus based on three categories."""
    # Initialize results dictionary with empty lists
    results = {
        'method_based': [],
        'keyword_based': [],
        'field_based': []
    }
    
    # If no metadata is provided, return empty results
    if not metadata:
        return results
    
    methods = metadata.get('Methods Used', [])
    keywords = metadata.get('Keywords', [])
    field = metadata.get('Research Field', '')
    
    try:
        headers = {"X-ELS-APIKey": SCOPUS_API_KEY}
        base_url = "https://api.elsevier.com/content/search/scopus"
        field_params = "dc:title,dc:creator,prism:coverDate,prism:doi"
        
        # Search by methods
        if methods:
            method_query = ' OR '.join(f'TITLE-ABS-KEY("{method}")' for method in methods)
            params = {
                "query": f"({method_query}) AND openaccess(1)",
                "count": 5,
                "field": field_params
            }
            try:
                response = requests.get(base_url, headers=headers, params=params)
                if response.ok:
                    entries = response.json().get('search-results', {}).get('entry', [])
                    for entry in entries:
                        citation = format_citation_with_link(
                            entry.get('dc:title', 'Untitled'),
                            entry.get('dc:creator', 'Unknown'),
                            entry.get('prism:coverDate', '')[:4],
                            entry.get('prism:doi', '')
                        )
                        results['method_based'].append(citation)
            except Exception as e:
                st.warning(f"Error in method-based search: {str(e)}")
        
        # Search by keywords
        if keywords:
            keyword_query = ' OR '.join(f'TITLE-ABS-KEY("{keyword}")' for keyword in keywords)
            params = {
                "query": f"({keyword_query}) AND openaccess(1)",
                "count": 5,
                "field": field_params
            }
            try:
                response = requests.get(base_url, headers=headers, params=params)
                if response.ok:
                    entries = response.json().get('search-results', {}).get('entry', [])
                    for entry in entries:
                        citation = format_citation_with_link(
                            entry.get('dc:title', 'Untitled'),
                            entry.get('dc:creator', 'Unknown'),
                            entry.get('prism:coverDate', '')[:4],
                            entry.get('prism:doi', '')
                        )
                        results['keyword_based'].append(citation)
            except Exception as e:
                st.warning(f"Error in keyword-based search: {str(e)}")
        
        # Search by research field
        if field:
            params = {
                "query": f'TITLE-ABS-KEY("{field}") AND openaccess(1)',
                "count": 5,
                "field": field_params
            }
            try:
                response = requests.get(base_url, headers=headers, params=params)
                if response.ok:
                    entries = response.json().get('search-results', {}).get('entry', [])
                    for entry in entries:
                        citation = format_citation_with_link(
                            entry.get('dc:title', 'Untitled'),
                            entry.get('dc:creator', 'Unknown'),
                            entry.get('prism:coverDate', '')[:4],
                            entry.get('prism:doi', '')
                        )
                        results['field_based'].append(citation)
            except Exception as e:
                st.warning(f"Error in field-based search: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in Scopus search: {str(e)}")
    
    # Always return results, even if empty
    return results


def search_scopus_combined_criteria(metadata):
    """Search for papers that match multiple criteria (methods, keywords, and field) simultaneously."""
    if not metadata:
        return []
    
    try:
        headers = {"X-ELS-APIKey": SCOPUS_API_KEY}
        base_url = "https://api.elsevier.com/content/search/scopus"
        field_params = "dc:title,dc:creator,prism:coverDate,prism:doi,citedby-count"
        
        # Get search criteria
        methods = metadata.get('Methods Used', [])
        keywords = metadata.get('Keywords', [])
        field = metadata.get('Research Field', '')
        
        # Build combined query
        query_parts = []
        
        # Add methods to query
        if methods:
            methods_query = ' AND '.join(f'TITLE-ABS-KEY("{method}")' for method in methods[:2])  # Limit to top 2 methods
            query_parts.append(f"({methods_query})")
            
        # Add keywords to query
        if keywords:
            keywords_query = ' AND '.join(f'TITLE-ABS-KEY("{keyword}")' for keyword in keywords[:2])  # Limit to top 2 keywords
            query_parts.append(f"({keywords_query})")
            
        # Add research field to query
        if field:
            query_parts.append(f'TITLE-ABS-KEY("{field}")')
        
        # Combine all parts with AND
        if query_parts:
            combined_query = ' AND '.join(query_parts)
            
            params = {
                "query": f"({combined_query}) AND openaccess(1)",
                "count": 10,  # Increased count since we're being more specific
                "sort": "-citedby-count",  # Sort by citation count
                "field": field_params
            }
            
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.ok:
                results = []
                entries = response.json().get('search-results', {}).get('entry', [])
                
                for entry in entries:
                    # Calculate relevance score based on matching criteria
                    title = entry.get('dc:title', '').lower()
                    relevance_score = 0
                    
                    # Check for method matches in title
                    for method in methods:
                        if method.lower() in title:
                            relevance_score += 1
                            
                    # Check for keyword matches in title
                    for keyword in keywords:
                        if keyword.lower() in title:
                            relevance_score += 1
                    
                    # Get citation count
                    citation_count = int(entry.get('citedby-count', 0))
                    
                    # Create result entry
                    result = {
                        'citation': format_citation_with_link(
                            entry.get('dc:title', 'Untitled'),
                            entry.get('dc:creator', 'Unknown'),
                            entry.get('prism:coverDate', '')[:4],
                            entry.get('prism:doi', '')
                        ),
                        'relevance_score': relevance_score,
                        'citation_count': citation_count
                    }
                    results.append(result)
                
                # Sort results by relevance score and citation count
                results.sort(key=lambda x: (x['relevance_score'], x['citation_count']), reverse=True)
                return results
                
    except Exception as e:
        st.error(f"Error in Scopus search: {str(e)}")
        return []
    
# Add download functionality for categorized similar papers report
def generate_categorized_papers_report():
        report = "# Categorized Similar Papers Report\n\n"
        
        for paper in st.session_state.papers_metadata:
            report += f"## Original Paper: {paper.get('Title', 'Untitled')}\n\n"
            similar_papers = search_scopus_categorized(paper)
            
            report += "### Papers with Similar Methods\n"
            if similar_papers['method_based']:
                for citation in similar_papers['method_based']:
                    report += f"- {citation}\n"
            else:
                report += "No papers found with similar methods.\n"
            
            report += "\n### Papers with Similar Keywords\n"
            if similar_papers['keyword_based']:
                for citation in similar_papers['keyword_based']:
                    report += f"- {citation}\n"
            else:
                report += "No papers found with similar keywords.\n"
            
            report += "\n### Papers in the Same Research Field\n"
            if similar_papers['field_based']:
                for citation in similar_papers['field_based']:
                    report += f"- {citation}\n"
            else:
                report += "No papers found in the same field.\n"
            
            report += "\n---\n\n"
            
        return report

def search_scopus_similar(metadata):
    """Search for similar papers in Scopus based on paper metadata."""
    similar_papers = []
    
    # Construct search query from metadata
    keywords = metadata.get('Keywords', [])
    methods = metadata.get('Methods Used', [])
    field = metadata.get('Research Field', '')
    
    # Combine search terms
    search_terms = []
    if keywords:
        search_terms.extend(keywords[:3])  # Use top 3 keywords
    if methods:
        search_terms.extend(methods[:2])   # Use top 2 methods
    if field:
        search_terms.append(field)
        
    query = ' AND '.join(f'"{term}"' for term in search_terms)
    
    # Search Scopus
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {"X-ELS-APIKey": SCOPUS_API_KEY}
    params = {
        "query": f"{query} AND openaccess(1)",
        "count": 5,  # Limit to top 5 results per paper
        "field": "dc:title,dc:creator,prism:coverDate,prism:doi,dc:description,authkeywords"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        results = response.json().get('search-results', {}).get('entry', [])
        
        # Process results
        for result in results:
            similar_paper = {
                'Title': result.get('dc:title', 'No title available'),
                'Authors': result.get('dc:creator', 'No creator available'),
                'Year': result.get('prism:coverDate', '')[:4],
                'DOI': result.get('prism:doi', ''),
                'Abstract': result.get('dc:description', ''),
                'Keywords': result.get('authkeywords', '').split('|') if result.get('authkeywords') else []
            }
            similar_papers.append(similar_paper)
            
        return similar_papers
    except Exception as e:
        st.error(f"Error searching Scopus: {str(e)}")
        return []

# Add this to your main UI section, after processing uploaded papers
if st.session_state.papers_metadata:
    st.subheader("Similar Papers from Scopus")
    
    for i, paper in enumerate(st.session_state.papers_metadata):
        st.markdown(f"### Paper {i+1}: {paper.get('Title', 'Untitled')}")
        
        similar_papers = search_scopus_categorized(paper)
        
        # Display method-based results
        col1, col2, col3 = st.columns(3)
            
        with col1:
            st.markdown("### 📊 Similar Methods")
            if similar_papers['method_based']:
                for citation in similar_papers['method_based']:
                    st.write(citation)
            else:
                st.write("No papers found.")
            
        with col2:
            st.markdown("### 🔑 Similar Keywords")
            if similar_papers['keyword_based']:
                for citation in similar_papers['keyword_based']:
                    st.write(citation)
            else:
                st.write("No papers found.")
            
        with col3:
            st.markdown("### 📚 Same Field")
            if similar_papers['field_based']:
                for citation in similar_papers['field_based']:
                    st.write(citation)
            else:
                st.write("No papers found.")

    # Add download functionality for similar papers report
    def generate_similar_papers_report():
        report = "# Similar Papers Report\n\n"
        
        for paper in st.session_state.papers_metadata:
            report += f"## Original Paper: {paper.get('Title', 'Untitled')}\n\n"
            similar_papers = search_scopus_similar(paper)
            
            if similar_papers:
                for similar in similar_papers:
                    report += f"### {similar['Title']}\n"
                    report += f"- Authors: {similar['Authors']}\n"
                    report += f"- Year: {similar['Year']}\n"
                    report += f"- Keywords: {', '.join(similar['Keywords'])}\n"
                    if similar['DOI']:
                        report += f"- DOI: https://doi.org/{similar['DOI']}\n"
                    report += "\n"
            else:
                report += "No similar papers found in Scopus.\n\n"
            
            report += "---\n\n"
                   
        return report

    report = generate_categorized_papers_report()
    st.download_button(
        label="Download Similar Papers Report",
        data=report,
        file_name="similar_papers_report.txt",
        mime="text/markdown"
    )

if st.session_state.papers_metadata:
    # st.subheader("Similar Papers from Scopus")
    
    for i, paper in enumerate(st.session_state.papers_metadata):
        st.markdown(f"### Paper {i+1}: {paper.get('Title', 'Untitled')}")
        
        # Get similar papers using combined criteria
        similar_papers = search_scopus_combined_criteria(paper)
        
        if similar_papers:
            st.markdown("#### Most Similar Papers (matching multiple criteria)")
            for idx, result in enumerate(similar_papers, 1):
                citation = result['citation']
                relevance = result['relevance_score']
                citations = result['citation_count']
                
                # Create expandable section for each paper
                with st.expander(f"Similar Paper {idx} (Relevance Score: {relevance}, Citations: {citations})"):
                    st.write(citation)
                    
                    # Show matching criteria
                    matches = []
                    if relevance > 0:
                        matches.append("✓ Matching methods/keywords in title")
                    if citations > 0:
                        matches.append(f"✓ Cited {citations} times")
                    
                    if matches:
                        st.markdown("**Matching Criteria:**")
                        for match in matches:
                            st.markdown(match)
        else:
            st.info("No papers found matching multiple criteria. Try adjusting the search parameters.")
        
        st.markdown("---")

    # Add download functionality for similar papers report
    def generate_combined_papers_report():
        report = "# Similar Papers Report (Combined Criteria)\n\n"
        
        for paper in st.session_state.papers_metadata:
            report += f"## Original Paper: {paper.get('Title', 'Untitled')}\n\n"
            similar_papers = search_scopus_combined_criteria(paper)
            
            if similar_papers:
                report += "### Similar Papers:\n\n"
                for idx, result in enumerate(similar_papers, 1):
                    report += f"#### {idx}. {result['citation']}\n"
                    report += f"- Relevance Score: {result['relevance_score']}\n"
                    report += f"- Citations: {result['citation_count']}\n\n"
            else:
                report += "No similar papers found matching multiple criteria.\n\n"
            
            report += "---\n\n"
                   
        return report

    report = generate_combined_papers_report()
    st.download_button(
        label="Download Similar Combined Papers Report",
        data=report,
        file_name="similar_papers_combined_report.txt",
        mime="text/markdown"
    )