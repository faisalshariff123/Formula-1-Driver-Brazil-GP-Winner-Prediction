import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import requests
import fastf1
import fastf1.plotting
from fastf1.events import get_event_schedule
import os
from dotenv import load_dotenv
from fastf1.core import DataNotLoadedError

load_dotenv()
WATSON_API_KEY = os.getenv("WATSON_API_KEY")

def get_iam_token(api_key):
    """Exchange IBM Cloud API key for IAM access token"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    
    response = requests.post(url, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(f"Failed to get IAM token: {response.text}")
    
    return response.json()["access_token"]


if not WATSON_API_KEY:
    raise ValueError("WATSON_API_KEY environment variable not set")

print(f"✓ API Key loaded: {WATSON_API_KEY[:10]}...")

# Get IAM token
try:
    iam_token = get_iam_token(WATSON_API_KEY)
    print("✓ IAM token obtained successfully")
except Exception as e:
    print(f"✗ Failed to get IAM token: {e}")
    raise

# Call watsonx
url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"

body = {
    "messages": [{"role":"system","content":"You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.\n\nAny HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.\n\nWhen returning code blocks, specify language.\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."}],
    "project_id": "7f5c0fb7-e612-4e34-9192-3b54405bf6b8",
    "model_id": "meta-llama/llama-3-3-70b-instruct",
    "frequency_penalty": 0,
    "max_tokens": 2000,
    "presence_penalty": 0,
    "temperature": 0,
    "top_p": 1
}

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {iam_token}"  # Use IAM token, not API key
}

response = requests.post(url, headers=headers, json=body)

if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text))

data = response.json()
print("✓ Watson API call successful")

# Create cache directory if it doesn't exist
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

year = 2025 
season = get_event_schedule(year)
race_names = season['EventName'].unique()
df = pd.DataFrame()

for race in race_names[:]:
    session = fastf1.get_session(2025, race, 'R')
    session.load()
    try:
        laps = session.laps
    except DataNotLoadedError:
        print(f"⚠️ Skipping {race}: lap data not available")
        continue

    if laps is None or laps.empty:
        print(f"⚠️ Skipping {race}: no lap rows")
        continue

    df = pd.concat([df, laps])

df_driver = df['Driver'].unique()
print(df_driver)