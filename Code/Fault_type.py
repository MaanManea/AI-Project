import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
current_dir = os.path.dirname(__file__)
models_folder = os.path.join(current_dir, r"../Models")
FAULT_TYPE_MODEL_PATH = os.path.join(models_folder, r"fault_type_model.pkl")

csv_path  = r"C:\Users\Maan\Desktop\9th Semester\AI Lab\selected_conveyor_data.csv"
class fault_type:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        try:
            # Load dataset
            data = pd.read_csv(csv_path)

            # Filter out Fault_Label == 0 (if this is desired behavior)
            data = data[data['Fault_Label'] != 0]

            # Drop unused and target columns as per your instructions
            data = data.drop(columns=['RUL_hours', 'Fault_Label'])

            target_column = 'Fault_Type'
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.train()
        except FileNotFoundError:
            raise ValueError("Cannot open the data file")
    def train(self):
        if self.model is not None:
            self.model.fit(self.X_train, self.y_train)
            joblib.dump(self.model, FAULT_TYPE_MODEL_PATH)

    def evaluate(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Model hasn't been trained or data not available.")

        self.y_pred = self.model.predict(self.X_test)

        print("✅ Confusion Matrix:")
        print(confusion_matrix(self.y_test, self.y_pred))

        print("\n✅ Classification Report:")
        print(classification_report(self.y_test, self.y_pred))

        print("\n✅ Accuracy:")
        print(accuracy_score(self.y_test, self.y_pred))

