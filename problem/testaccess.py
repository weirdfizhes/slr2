import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/contacts.readonly'
]
SERVICE_ACCOUNT_FILE = 'F:/slr-project/problem/keys/cloud-sheets-448704-dcb563489f7c.json'

def test_people_api():
    try:
        # Authenticate with the service account
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('people', 'v1', credentials=creds)

        # Test fetching a profile photo
        email = 'maulana@plabs.id'  # Replace with an email in your domain
        person = service.people().get(resourceName=f'people/{email}', personFields='photos').execute()
        photos = person.get('photos', [])
        if photos:
            print("Profile photo URL:", photos[0]['url'])
        else:
            print("No profile photo found for this email.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_people_api()

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SERVICE_ACCOUNT_FILE = 'F:/slr-project/problem/keys/cloud-sheets-448704-dcb563489f7c.json'

def test_google_sheets():
    try:
        # Set up credentials
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        # Authorize and access the sheet
        client = gspread.authorize(creds)
        sheet = client.open_by_key('1zSCwsTdOcy0P-3Utq8NapJ_vMRWCmcOqprBbU4CM7vI').worksheet('Sheet2')

        # Fetch data
        data = sheet.get_all_records()
        print("Access successful! Data fetched:")
        print(data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_google_sheets()
