from openai import OpenAI
import numpy as np
import os

from dotenv import load_dotenv
import json
import pandas as pd
import joblib

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# json_data_path = r"C:\Users\Maan\Desktop\9th Semester\AI Lab\Project1\changed_sensor_data.json"

# def load_json_data(json_path):
#     if not os.path.exists(json_path):
#         raise FileNotFoundError(f"JSON file not found: {json_path}")
#     print(f"[INFO] Loading JSON data from {json_path}")
#     with open(json_path, "r") as file:
#         return json.load(file)
# ata = load_json_data(json_data_path)




client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
    )

def fault_api(ftype, features_text, data):
    
    prompt = f"""
    You are an AI assistant for predictive maintenance.

    The SHAP feature impacts are:
    {features_text}

    Based on these values, predict:
    1. The most likely fault type (e.g., 'Minor Lubrication Issue', 'Major Lubrication Issue', 'Bearing Wear').
    2. A recommended predictive maintenance action (e.g., 'Lubricate motor bearings', 'Schedule inspection', 'Replace bearing').

    Give your answer in JSON format with two fields: 'fault_type' and 'maintenance_action'. just write the fault type and 
    maintenance action without any explanation.
    """
    deepseek_resp = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3.1:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return deepseek_resp.choices[0].message.content

