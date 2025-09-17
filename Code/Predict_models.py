import joblib
import json
import numpy as np
import os
import pandas as pd
import itertools

# from Fault_rul import fault_rul
from Fault_api import fault_api

current_dir = os.path.dirname(__file__)
models_folder = os.path.join(current_dir, r"../Models")

FAULT_LABEL_MODEL_PATH = os.path.join(models_folder, r"fault_label_model.pkl")
FAULT_TYPE_MODEL_PATH = os.path.join(models_folder, r"fault_type_model.pkl")
FAULT_RUL_MODEL_PATH = os.path.join(models_folder, r"fault_rul_model.pkl")
FAULT_EXPLAINER_MODEL_PATH = os.path.join(models_folder, r"fault_label_explainer.pkl")

json_data_path = r"C:\Users\Maan\Desktop\9th Semester\AI Lab\Project1\changed_sensor_data.json"

class ModelLoader:
    def __init__(self):
        self.fault_label_model = None
        self.fault_type_model = None
        self.fault_rul_model = None
        self.fault_explainer_model = None
        self.fault_api_model = None
        self.load_all_models()

    def load_fault_label_model(self):
        if os.path.exists(FAULT_LABEL_MODEL_PATH):
            self.fault_label_model = joblib.load(FAULT_LABEL_MODEL_PATH)
            print("✅ Fault Label Model loaded successfully.")
        else:
            try:
                from Fault_label import fault_label
                self.fault_label_model = fault_label().model
                print("✅ Fault Label Model trained and loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to train/load Fault Label Model: {e}")
            raise FileNotFoundError(f"Fault Label Model not found at {FAULT_LABEL_MODEL_PATH}")

    def load_fault_type_model(self):
        if os.path.exists(FAULT_TYPE_MODEL_PATH):
            self.fault_type_model = joblib.load(FAULT_TYPE_MODEL_PATH)
            print("✅ Fault Type Model loaded successfully.")
        else:
            try:
                from Fault_type import fault_type
                self.fault_type_model = fault_type().model
                print("✅ Fault Type Model trained and loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to train/load Fault Type Model: {e}")
            raise FileNotFoundError(f"Fault Type Model not found at {FAULT_TYPE_MODEL_PATH}")

    # def load_fault_rul_model(self):
    #     if os.path.exists(FAULT_RUL_MODEL_PATH):
    #         self.fault_rul_model = joblib.load(FAULT_RUL_MODEL_PATH)
    #         print("✅ Fault RUL Model loaded successfully.")
    #     else:
    #         try:
    #             self.fault_rul_model = fault_rul().model
    #             print("✅ Fault Label Model trained and loaded successfully.")
    #         except Exception as e:
    #             print(f"[ERROR] Failed to train/load Fault Label Model: {e}")
    #         raise FileNotFoundError(f"Fault RUL Model not found at {FAULT_RUL_MODEL_PATH}")

    def load_fault_explainer_model(self):
        if os.path.exists(FAULT_EXPLAINER_MODEL_PATH):
            self.fault_explainer_model = joblib.load(FAULT_EXPLAINER_MODEL_PATH)
            print("✅ Fault Explainer Model loaded successfully.")
        else:
            try:
                from Fault_label import fault_label
                self.fault_explainer_model = fault_label().explainer
                print("✅ Fault Label Model trained and loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to train/load Fault Label Model: {e}")
            raise FileNotFoundError(f"Fault Explainer Model not found at {FAULT_EXPLAINER_MODEL_PATH}")

    def load_all_models(self):
        self.load_fault_label_model()
        self.load_fault_type_model()
        # self.load_fault_rul_model()
        self.load_fault_explainer_model()

def predict(model, data):
    try:
        # Ensure the data is in the correct format (e.g., a 2D array)
        X = np.array([list(row.values()) for row in data])
        predictions = model.predict(X)
        print("[INFO] Predictions:")
        if len(predictions) == 1:
            return predictions
        else:
            return predictions[0]
        
    except Exception as e:
        print(f"[ERROR] Failed to make predictions: {e}")


def load_json_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    print(f"[INFO] Loading JSON data from {json_path}")
    with open(json_path, "r") as file:
        return json.load(file)

def on_label(label, model_loader, data):
    if label == 0:
        rul = 0 #predict(model_loader.fault_rul_model, data)
        ftype = "Normal"
        explain = "All sensors reading are normal"
        api = None
    else:
        ftype = predict(model_loader.fault_type_model, data)
        df = pd.read_json(json_data_path)
        explain = model_loader.fault_explainer_model.shap_values(df)
        row_shap = explain[0]
        total = np.sum(row_shap)
        row_sums = np.sum(row_shap, axis=1)
        # feature_names = list(data[0].keys())
        clmns = list(data[0].keys())
        row_percent = (float(sum/total) for sum in row_sums)
        zipped = zip(clmns, row_percent)
        features_affect = dict(sorted(zipped, key=lambda x: x[1], reverse=True))
        first_three = dict(itertools.islice(features_affect.items(), 3))
        features_text = ", ".join([f"{k}: {v}" for k, v in first_three.items()])
        rul = 0
        api = fault_api(ftype, features_text, data)
    return ftype, rul, features_text, api

def predict_models():
    model_loader = ModelLoader()
    data = load_json_data(json_data_path)
    label = predict(model_loader.fault_label_model, data)
    results = on_label(label, model_loader, data)
    return results
    
    
