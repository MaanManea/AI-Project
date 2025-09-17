import sys 
import os

from Predict_models import predict_models




print("Welcome to the Fault Diagnosis System")
results = predict_models()
print(results)
# data = r"C:\Users\Maan\Desktop\9th Semester\AI Lab\Project1\conveyor_pdm_multilifecycle_kpa_with_belt_RUL.csv"
# target = "Fault_Label"
# skipped = "Fault_Type","RUL_hours"

# faultlabel = fault_label()
# faultlabel.fit_model()
# features = faultlabel.X_test
# faultlabel.shap_explain(testdata)
# print(faultlabel.shap_values)
# model = faultlabel.model
# explainer = faultlabel.explainer
# predic = fault_label_predict(model, explainer)
# testdata = r"C:\Users\Maan\Desktop\9th Semester\AI Lab\tt\conveyor_test.csv"
# results = predic.predict_model(testdata, target, skipped)
# print(results)

