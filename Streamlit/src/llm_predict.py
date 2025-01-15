# src/llm_predict.py

import textwrap
# import google.generativeai as genai
import atexit
# import vertexai
# from vertexai.generative_models import GenerativeModel
# import google.auth
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import base64
import json
import os
import re
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

def decrypt_json_file(encrypted_file, secret_key):
    # Convert the secret key to bytes and ensure it's 16 bytes long
    key = secret_key.encode('utf-8')
    key = key[:16].ljust(16, b'\0')

    # Read the encrypted data
    with open(encrypted_file, 'rb') as f:
        encrypted_data = base64.b64decode(f.read())

    # Extract the IV and encrypted data
    iv = encrypted_data[:16]
    encrypted_data = encrypted_data[16:]

    # Create AES cipher
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt the data
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

    # Convert decrypted data to JSON
    json_data = decrypted_data.decode('utf-8')
    return json.loads(json_data)

def delete_temp_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Register cleanup to delete the temp_decrypted_key.json on exit
atexit.register(delete_temp_file, 'temp_decrypted_key.json')

load_dotenv()

# Set up Google API key and configure Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
secret_key = os.getenv("secret_key")

current_directory = os.getcwd()

encrypted_file_path = os.path.join(current_directory, 'gdc_encrypted.json')

decrypted_json = decrypt_json_file(encrypted_file_path, secret_key)

# decrypted_json = decrypt_json_file('D:\\G\\UNPAD\\Plabs\\RnD\\Gitlab Projects\\PLANT\\Backend\\backend\\gdc_encrypted.json', secret_key)

temp_key_path = 'temp_decrypted_key.json'
with open(temp_key_path, 'w') as f:
    json.dump(decrypted_json, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_path

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# credentials, project_id = google.auth.default()

# genai.configure(api_key=GOOGLE_API_KEY)
# vertexai.init(project=project_id, location="us-central1")

# # Define model and configure
# generation_model = "gemini-1.5-flash"
# model_generation = GenerativeModel(generation_model)

# Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees (DD)
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = float(degrees) + (float(minutes) / 60) + (float(seconds) / 3600)
    if direction in ['S', 'W']:
        decimal = -decimal
    return round(decimal, 6)

# Generate the prompt based on user query
def make_prompt(query):
    location_keywords = ["where", "location", "latitude", "longitude", "coordinates", "island", "place"]
    oceanography_keywords = ["seagrass", "oceanography", "water", "quality", "marine", "coastal", "seaweed", 
                           "meadows", "carbon", "ecosystem", "habitat", "sediment", "photosynthesis", 
                           "aquatic", "zostera", "posidonia", "underwater", "vegetation", "biodiversity", 
                           "nutrient", "conservation", "protected", "climate", "eutrophication", "hypoxia",
                           "acidification", "dredging", "estuarine", "anthropogenic"]

    expertise_statement = """
    Primary Domain: Marine and Coastal Sciences

    Core Specializations:
    1. Seagrass Ecology and Blue Carbon Systems
       - Meadow ecosystem dynamics
       - Carbon sequestration processes
       - Habitat productivity assessment
    
    2. Marine Environmental Analysis
       - Water quality parameters
       - Sediment dynamics
       - Nutrient cycling
    
    3. Coastal Ecosystem Management
       - Conservation strategies
       - Restoration methodologies
       - Protected area planning
    
    4. Climate Change Impact Assessment
       - Coastal vulnerability analysis
       - Adaptation strategies
       - Mitigation measures
    """

    # Check if the query is location-based
    if any(keyword in query.lower() for keyword in location_keywords):
        prompt = textwrap.dedent(f"""
        You are a world-class geographic assistant. I will share a location-related query: '{query}'.
        Explain about the location very concise and detail, and then 
        please provide the most accurate location response, including the coordinates 
        (latitude and longitude) in DMS (Degrees, Minutes, Seconds) format.
        Format the coordinates as: Coordinates: X°Y'Z.Z"N, A°B'C.C"E.
        Also, convert and provide these coordinates in decimal degrees (DD) format. Additionally, give bounding box coordinates in DD format:
        [lat1, lon1] : [lat2, lon2].
        """)

    elif any(keyword in query.lower() for keyword in oceanography_keywords):
        # Oceanography query handling remains the same
        prompt = f"""
        As a Research Specialist in Marine Sciences, I will address your inquiry regarding: '{query}'

        Response Protocol:
        - Analysis will be based on peer-reviewed research
        - Information will be presented with scientific accuracy
        - Conclusions will be supported by current environmental data

        Relevant expertise:
        {expertise_statement}

        Proceeding with detailed analysis...
        """

    else:
        # Modified specialization response
        if "specialization" in query.lower() or "expertise" in query.lower():
            prompt = f"""
            Formal Statement of Specialization:

            This research entity specializes in marine and coastal sciences, with particular emphasis on seagrass ecosystems and their role in environmental processes. The scope of expertise encompasses:

            {expertise_statement}

            All analyses and assessments are conducted within these defined domains, utilizing established scientific methodologies and peer-reviewed research frameworks.
            """
        else:
            prompt = f"""
            Formal Acknowledgment:
            
            Regarding your query: '{query}'

            This research entity maintains specific expertise in marine and coastal sciences, as defined below:

            {expertise_statement}

            Analysis will proceed within these established parameters of expertise.
            """

    return prompt.strip()

# Function to get the response from LLM
def get_llm_response(user_query):
    prompt = make_prompt(user_query)
    coor = False
    try:
        # answer = model_generation.generate_content(prompt)
        ai_response = conversation.predict(input=prompt)
        # print(memory)
        # ai_response = answer.text.strip()

        # Extract coordinates and bbox
        if any(keyword in user_query.lower() for keyword in ["where", "location", "latitude", "longitude", "coordinates"]):
            # Extract DMS coordinates from AI response using regex
            dms_pattern = r"([0-9]+)°([0-9]+)'([0-9\.]+)\"([NS]),\s*([0-9]+)°([0-9]+)'([0-9\.]+)\"([EW])"
            match = re.search(dms_pattern, ai_response)
            
            if match:
                # Convert extracted DMS to Decimal Degrees
                lat_dms = (match.group(1), match.group(2), match.group(3), match.group(4))
                lon_dms = (match.group(5), match.group(6), match.group(7), match.group(8))
                
                latitude_dd = dms_to_decimal(*lat_dms)
                longitude_dd = dms_to_decimal(*lon_dms)
                
                # Generate a bounding box (bbox) for the location
                bbox = {
                    "lat1": round(latitude_dd - 0.01, 2),
                    "lon1": round(longitude_dd - 0.01, 2),
                    "lat2": round(latitude_dd + 0.01, 2),
                    "lon2": round(longitude_dd + 0.01, 2),
                }

                # Prepare final response
                location_data = {
                    "query": user_query,
                    "response": ai_response,
                    "location": {
                        "DMS": {
                            "latitude": lat_dms,
                            "longitude": lon_dms
                        },
                        "DD": {
                            "latitude": latitude_dd,
                            "longitude": longitude_dd
                        },
                        "bbox": bbox
                    }
                }
                coor = True
                return location_data, coor

            else:
                response = {
                            "response": ai_response,
                            "query": user_query
                            }
                return response, coor
        else:
            response = {
                        "response": ai_response,
                        "query": user_query
                        }
            return response, coor

    except Exception as e:
        return {"error": str(e)}
