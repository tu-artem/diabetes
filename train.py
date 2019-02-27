import pickle

import pandas as pd
# import numpy as np

from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

data = pd.read_csv("data/raw/Chapter_3_Diabetes_data.csv", low_memory=False)

categorical = [
    "encounter_id",
    "patient_nbr",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]
for variable in categorical:
    data[variable] = pd.Categorical(data[variable])

data.drop_duplicates(inplace=True)

rows_per_patient = data.groupby('patient_nbr')["encounter_id"].count()
data = data.merge(rows_per_patient, on='patient_nbr', suffixes=("", "_count"))

data["target"] = data["readmitted"] != "NO"

data.set_index("encounter_id", inplace=True)

not_train_columns = [
           "patient_nbr", 
           "payer_code", 
           "medical_specialty",
           "weight",
           "diag_1",
           "diag_2",
           "diag_3",
           "change",
           "diabetesMed",
           "readmitted",
           "encounter_id_count",
           "target"]

data["race"] = data["race"].fillna("NA")

x_train, x_test, y_train, y_test = train_test_split(
    data.drop(not_train_columns, axis=1), 
    data["target"],
    test_size=0.20)       


ohe = OneHotEncoder(handle_unknown="ignore")

x_train = ohe.fit_transform(x_train)
x_test = ohe.transform(x_test)

clf = LogisticRegression(random_state=42, solver="lbfgs")
clf.fit(x_train, y_train)

print(f"Train accuracy:{accuracy_score(y_train.to_list(), clf.predict(x_train))}")
print(f"Test accuracy: {accuracy_score(y_test.to_list(), clf.predict(x_test))}")


with open("models/ohe.pcl", "wb") as f:
    pickle.dump(ohe, f)

with open("models/model.pcl", "wb") as f:
    pickle.dump(clf, f)
