import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class FaultClassifier:
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

    def load_and_preprocess_data(self, csv_path):
        # Load dataset
        data = pd.read_csv(csv_path)

        # Filter out Fault_Label == 0 (if this is desired behavior)
        data = data[data['Fault_Label'] != 0]

        # Drop unused and target columns as per your instructions
        data = data.drop(columns=['RUL_hours', 'Timestamp', 'Unit_ID', 'Fault_Label'])

        # Show feature names used for prediction (optional)
        print("📌 Feature names used for prediction:")
        print(list(data.columns))

        target_column = 'Fault_Type'
        X = data.drop(columns=[target_column])
        y = data[target_column]

        return X, y

    def train(self, csv_path):
        X, y = self.load_and_preprocess_data(csv_path)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

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

    def predict(self, input_data):
        return self.model.predict(input_data)

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")


if __name__ == "__main__":
    csv_path = "M:/PYTHON/AI_project/conveyor.csv"  # Update this path accordingly

    # Instantiate the classifier
    classifier = FaultClassifier()

    # Train model
    classifier.train(csv_path)

    # Evaluate model
    classifier.evaluate()

    # Save model
    classifier.save_model("trained_model.pkl")

    # You can also load and predict later by uncommenting below:

    # classifier.load_model("trained_model.pkl")
    # Example prediction (make sure columns match training features):
    # import pandas as pd
    # new_data = pd.DataFrame([[4, 892, 12.5, 0.34, 0.12, 0.08, 55.4, 102.3, 72.1, 48.6]],
    #                         columns=['Lifecycle_ID', 'Repair_Flag', 'Cycles_Completed', 'Motor_Current_mean (A)', 'Vibration_max (G)', 'Vibration_std (G)', 'Vibration_high_freq_mean (G)', 'Temperature_mean (°C)', 'Pressure_mean (kPa)', 'Acoustic_mean (dB)', 'Humidity_mean (%)'])
    # prediction = classifier.predict(new_data)
    # print("Prediction:", prediction)
