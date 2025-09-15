import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import shap


class fault_label:
    def __init__(self, data, target, skipped):
        try:
            df = pd.read_csv(data)
            self.target_column = target  
            if skipped is None:
                self.skipped_clms = [target]
            else:
                self.skipped_clms = list(skipped) + [target]
            self.label_encoders = {}
            for col in df.columns:
                if df[col].dtype == "object":  
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le  
            try:
                X = df.drop(columns= self.skipped_clms, axis=1)
                y = df[self.target_column]  
                
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42   
                ) 
                num_classes = len(np.unique(y))
                self.model = XGBClassifier(
                            n_estimators=1,
                            max_depth=5,
                            learning_rate=0.2,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=1.0,
                            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
                            eval_metric="mlogloss" if num_classes > 2 else "logloss",
                            tree_method="hist"
                        )
                self.fit_model()
                
            except KeyError:
                raise ValueError("Target or skipped columns are not in the file")
        except FileNotFoundError:
            raise ValueError("Cannot open the data file")
        
    def fit_model(self):
        if self.model:
            self.model.fit(self.X_train, self.y_train)
            background = self.X_train.sample(1000, random_state=42)
            self.explainer = shap.TreeExplainer(self.model, background)
            
    def show_accuracy(self):
        if self.model:
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)  # for AUC
            acc = accuracy_score(self.y_test, y_pred)
            print(f"Accuracy: {acc:.4f}")
            prec = precision_score(self.y_test, y_pred, average='macro')  
            print(f"Precision (macro): {prec:.4f}")
            auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr')  
            print(f"AUC: {auc:.4f}")
            return f"AC: {acc}\nPrecision: {prec}\nAUC: {auc}"
        
class fault_label_predict:
    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer

    def shap_explain(self, features, row_index=0):
        self.shap_values = self.explainer.shap_values(features)
        row_shap = self.shap_values[row_index]
        total = np.sum(row_shap)
        row_sums = np.sum(row_shap, axis=1)
        clmns = features.columns
        row_percent = (float(sum/total) for sum in row_sums)
        zipped = zip(clmns, row_percent)
        features_affect = dict(sorted(zipped, key=lambda x: x[1], reverse=True))

        # for feature, value in features_affect.items():
        #     print(f"{feature}: {value:.4f}")

        return f"The data affect of the row {row_index} is:\n{features_affect}"

    def predict_model(self, testdatafile, target, skipped):
        try:
            if skipped is None:
                self.skipped_clms = [target]
            else:
                self.skipped_clms = list(skipped) + [target]
            self.test_df = pd.read_csv(testdatafile)
            test_df = test_df.drop(columns=self.skipped_clms errors='ignore',axis=1)
            predicted_fault = self.model.predict(test_df)
            return predicted_fault
            
        except FileNotFoundError:
            raise ValueError("Cannot open the data file")