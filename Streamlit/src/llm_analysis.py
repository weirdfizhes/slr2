# src/llm_analysis.py

import google.generativeai as genai
import os
from dotenv import load_dotenv
import atexit
import vertexai
from vertexai.generative_models import GenerativeModel
import google.auth
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import base64
import json

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
credentials, project_id = google.auth.default()

genai.configure(api_key=GOOGLE_API_KEY)
vertexai.init(project=project_id, location="us-central1")

# Define model and configure
generation_model = "gemini-1.5-pro"
model_generation = GenerativeModel(generation_model)

# Define a function to create a prompt based on the predictions data
def make_analysis_prompt(data):
    prompt = """
    Based on the following prediction data, determine whether the area is suitable for seagrass transplantation. 
    The prediction values range from 0 to 2, where:
    - 0 means unsuitable for seagrass transplantation
    - 1 means moderately suitable
    - 2 means highly suitable

    Please analyze the prediction data, provide a summary of your analysis, and summarize also
    the water data quality for each coordinate, and explain the reasons for your assessment.

    Also, note that each coordinate scale represents 100 square meters.

    Water data quality obtained from prediction data includes these properties:
    - bathy: depth of sea water
    - do: dissolved oxygen
    - latitude: latitude coordinate
    - longitude: longitude coordinate
    - ph: potential of hydrogen
    - prediction: prediction result
    - salinity: salinity value
    - temp: temperature
    - tss: total suspended solids

    For bathy values:
    - Positive or above 0 values indicate land, making seagrass transplantation impossible.
    - Values less than -4 (deeper below sea level) indicate high depths, which make transplantation impractical due to effort and feasibility limitations.

    ### Carbon Capture Calculation
    - Every point with a prediction value more than 0 or more than 1 (highly suitable) is considered a "green point."
    - Each green point represents an area of 100 square meters capable of capturing 400 kg of carbon annually.
    - Additionally, each green point can accumulate a long-term sediment storage of 2,500 kg of carbon.
    - Calculate the total carbon capture potential for all green points by multiplying the number of green points by 400 kg (annual) and 2,500 kg (long-term sediment storage).

    ### Example Calculation
    If there are 10 green points:
    - Annual carbon capture = 10 x 400 kg = 4,000 kg of carbon per year.
    - Long-term sediment storage = 10 x 2,500 kg = 25,000 kg of carbon.

    Prediction Data:
    {}
    """.format(data)
    return prompt

# Function to get the analysis from LLM
def get_analysis(data):
    prompt = make_analysis_prompt(data)
    try:
        answer = model_generation.generate_content(prompt)
        ai_response = answer.text.strip()
        return {"analysis": ai_response}
    except Exception as e:
        return {"error": str(e)}
