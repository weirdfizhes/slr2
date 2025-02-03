import pygame
import random
import sys
from pygame.locals import *
import os
import requests
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import gspread

class FloatingImage:
    def __init__(self, image_path, author, title, screen_width, screen_height):
        self.original_image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.original_image, (150, 150))
        self.rect = self.image.get_rect()

        self.rect.x = random.randint(0, screen_width - self.rect.width)
        self.rect.y = random.randint(0, screen_height - self.rect.height)

        self.speed_x = random.uniform(-2, 2)
        self.speed_y = random.uniform(-2, 2)

        self.author = author
        self.title = title

        self.font_author = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 36)

        self.author_surface = self.font_author.render(author, True, (200, 200, 200))
        self.title_surface = self.font_title.render(title, True, (255, 255, 255))

        self.screen_width = screen_width
        self.screen_height = screen_height

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        if self.rect.left < 0 or self.rect.right > self.screen_width:
            self.speed_x *= -1
        if self.rect.top < 0 or self.rect.bottom > self.screen_height:
            self.speed_y *= -1

    def draw(self, screen):
        screen.blit(self.image, self.rect)

        # Draw author
        author_rect = self.author_surface.get_rect(
            centerx=self.rect.centerx,
            top=self.rect.bottom + 5
        )
        screen.blit(self.author_surface, author_rect)

        # Draw title
        title_rect = self.title_surface.get_rect(
            centerx=self.rect.centerx,
            top=author_rect.bottom + 5
        )
        screen.blit(self.title_surface, title_rect)

class App:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Google Sheets Floating Images")

        self.clock = pygame.time.Clock()
        self.floating_images = []
        self.running = True

        self.sheet_data = []
        self.credentials = self.setup_google_credentials()
        self.setup_google_sheets()

    def setup_google_credentials(self):
        """Set up Google API credentials."""
        SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/contacts.readonly'
]
        creds = Credentials.from_service_account_file(
            'F:/slr-project/problem/keys/cloud-sheets-448704-dcb563489f7c.json', 
            scopes=SCOPES
        )
        return creds

    def setup_google_sheets(self):
        """Fetch data from Google Sheets."""
        try:
            client = gspread.authorize(self.credentials)
            self.sheet = client.open_by_key('1zSCwsTdOcy0P-3Utq8NapJ_vMRWCmcOqprBbU4CM7vI').worksheet('Sheet2')
            self.sheet_data = self.sheet.get_all_records()
        except Exception as e:
            print(f"Error setting up Google Sheets: {e}")
            sys.exit(1)

    def fetch_profile_photo(self, email):
        """Fetch the Google profile photo using People API."""
        try:
            service = build('people', 'v1', credentials=self.credentials)
            person = service.people().get(resourceName=f'people/{email}', personFields='photos').execute()
            photos = person.get('photos', [])
            if photos:
                return photos[0]['url']
        except Exception as e:
            print(f"Error fetching profile photo for {email}: {e}")
        return None

    def add_images_from_sheet(self):
        """Add floating images using profile photos."""
        print("Fetching data from Google Sheet...")
        self.floating_images = []  # Clear existing images
        for row in self.sheet_data:
            # Extract required columns
            author = row.get('Author', 'Unknown')
            title = row.get('Title', 'Untitled')
            email = row.get('Email', '')

            # Skip rows without an email
            if not email:
                print("Skipping row without an email:", row)
                continue

            # Fetch profile photo
            photo_url = self.fetch_profile_photo(email)
            if not photo_url:
                print(f"No profile photo found for {email}")
                continue

            try:
                response = requests.get(photo_url, stream=True, timeout=10)
                if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                    # Save the image temporarily
                    with open('temp_image.jpg', 'wb') as f:
                        f.write(response.content)

                    # Create a floating image object
                    floating_image = FloatingImage(
                        'temp_image.jpg',
                        author,
                        title,
                        self.screen_width,
                        self.screen_height
                    )
                    self.floating_images.append(floating_image)
                    print(f"Added profile photo for author: {author}")
                else:
                    print(f"Failed to download profile photo for {email}")
            except Exception as e:
                print(f"Error loading profile photo for email '{email}': {e}")

    def add_images_from_sheet(self):
        """Add floating images using profile photos."""
        print("Fetching data from Google Sheet...")
        self.floating_images = []  # Clear existing images
        for row in self.sheet_data:
            # Extract required columns
            author = row.get('Author', 'Unknown')
            title = row.get('Title', 'Untitled')
            email = row.get('Email', '')

            # Skip rows without an email
            if not email:
                print(f"Skipping row without an email: {row}")
                continue

            # Fetch profile photo
            photo_url = self.fetch_profile_photo(email)
            if not photo_url:
                print(f"No profile photo found for {email}")
                photo_path = 'placeholder_image.png'  # Use placeholder image
            else:
                try:
                    response = requests.get(photo_url, stream=True, timeout=10)
                    if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                        # Save the image temporarily
                        photo_path = 'temp_image.jpg'
                        with open(photo_path, 'wb') as f:
                            f.write(response.content)
                    else:
                        print(f"Failed to download profile photo for {email}")
                        photo_path = 'placeholder_image.png'  # Use placeholder image
                except Exception as e:
                    print(f"Error loading profile photo for email '{email}': {e}")
                    photo_path = 'placeholder_image.png'  # Use placeholder image

            # Create a floating image object even without an image
            floating_image = FloatingImage(
                photo_path,
                author,
                title,
                self.screen_width,
                self.screen_height
            )
            self.floating_images.append(floating_image)
            print(f"Added floating item for author: {author}")

        print(f"Total floating images added: {len(self.floating_images)}")

    def run(self):
        """Run the application."""
        print("Initializing the app and loading images...")
        self.add_images_from_sheet()
        print(f"Number of images loaded: {len(self.floating_images)}")

        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_r:  # Press 'R' to refresh
                        print("Refreshing data from Google Sheet...")
                        self.setup_google_sheets()
                        self.add_images_from_sheet()

            # Update
            for floating_image in self.floating_images:
                floating_image.update()

            # Draw
            self.screen.fill((0, 0, 0))
            for floating_image in self.floating_images:
                floating_image.draw(self.screen)

            # Draw refresh instruction
            font = pygame.font.Font(None, 24)
            refresh_text = font.render("Press 'R' to Refresh", True, (255, 255, 255))
            self.screen.blit(refresh_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = App()
    app.run()
