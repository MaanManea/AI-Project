from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
import pandas as pd


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
    )

def fault_api(ftype, features_text):
    
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

