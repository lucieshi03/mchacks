import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import json

# Dynamically resolve the path to the .env file in the parent folder
env_path = Path(__file__).resolve().parent.parent / ".env"

# Load the .env file
load_dotenv(dotenv_path=env_path)

# Access the environment variable
API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=API_KEY, transport="rest") 

# Define the path to the JSON file
file_path = Path(__file__).resolve().parent.parent.parent / 'counters.json'

# Read and parse the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Accessing values (example for a dictionary structure)
var_name2= data.get('bad_counter')
var_name1= data.get('blink_bad_counter')

def get_message(caught_slouching=10, caught_not_blinking=10):

    # True: positive result, False: negative result
    tone = caught_slouching < 10 and caught_not_blinking < 10

    # Assumes the user had bad habits
    prompt = f"Given this data: user was caught slouching {caught_slouching} times and caught not blinking enough {caught_not_blinking} times, write a unique aggressive roast about this person for practicing bad desk habits. Make it at most 100 words."

    if tone:
        if not bool(caught_not_blinking and caught_not_blinking):
            prompt = "The user was caught slouching 0 times and caught not blinking enough  0 times. Write a unique message encouraging them to keep up with the great habits.Make it at most 100 words."
        else:
            prompt = f"Given this data: the user was caught slouching {caught_slouching} times and caught not blinking enough {caught_not_blinking} times, write a unique message encouraging the user to keep up with the great habits despite their mishaps. Make it tacky and use words like slay and kween. Make it at most 100 words."

    model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction="Depending on the prompt, you will either aggressively roast the user or be super bubbly, positive and encouraging.Make it at most 100 words.")
    response = model.generate_content(prompt)

    return (response.text)