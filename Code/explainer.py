import os
import pandas as pd
from Fault_label import fault_label
import shap
from joblib import dump, load

current_dir = os.path.dirname(__file__)
models_folder = os.path.join(current_dir, r"../Models")
FAULT_LABEL_MODEL_PATH = os.path.join(models_folder, r"fault_label_model.pkl")
FAULT_RUL_MODEL_PATH = os.path.join(models_folder, r"fault_rul_model.pkl")
FAULT_TYPE_MODEL_PATH = os.path.join(models_folder, r"fault_type_model.pkl")
FAULT_LABEL_EXPLAINER_PATH = os.path.join(models_folder, r"fault_label_explainer.pkl")
FAULT_TYPE_EXPLAINER_PATH = os.path.join(models_folder, r"fault_type_explainer.pkl")
FAULT_RUL_EXPLAINER_PATH = os.path.join(models_folder, r"fault_rul_explainer.pkl")

data_folder = os.path.join(current_dir, r"../Data")
DATA = os.path.join(data_folder, r"conveyor_data.csv")
# average = 
class explainer:
    def __init__(self):
        self.label = None
        self.type = None
        self.rul = None
        self.average = None
        try:
            df = pd.read_csv(DATA)
            skipped = "Fault_Type","RUL_hours","Fault_Label"
            X = df.drop(columns=skipped, axis=1)
            self.average = X.sample(1000, random_state=42)
        except FileNotFoundError:
            raise ValueError("Cannot open the data file")

    def label_explainer(self):
        try:
            self.label = load(FAULT_LABEL_MODEL_PATH)
            explainer = shap.TreeExplainer(self.label, self.average)
            dump(explainer, FAULT_LABEL_EXPLAINER_PATH)
        except FileNotFoundError:
            self.label = fault_label().model
        
    def type_explainer(self):
        try:
            self.type = load(FAULT_TYPE_MODEL_PATH)
            explainer = shap.TreeExplainer(self.type, self.average)
            dump(explainer, FAULT_TYPE_EXPLAINER_PATH)
        except FileNotFoundError:
            raise ValueError("Cannot open the data file")
    
    def rul_explainer(self):
        try:
            self.rul = load(FAULT_RUL_MODEL_PATH)
            explainer = shap.TreeExplainer(self.rul, self.average)
            dump(explainer, FAULT_RUL_EXPLAINER_PATH)
        except FileNotFoundError:
            raise ValueError("Cannot open the data file")
        



