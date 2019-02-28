import pickle

import pandas as pd
# import numpy as np

from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer

from catboost import CatBoostClassifier

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
data["running_count"] = data.groupby("patient_nbr").cumcount()
data.set_index("encounter_id", inplace=True)

not_train_columns = [
           "patient_nbr", 
           "payer_code", 
      #     "medical_specialty",
           "weight",
           "diag_1",
           "diag_2",
           "diag_3",
       #    "change",
       #    "diabetesMed",
           "readmitted",
     #      "running_count",
           "encounter_id_count",
           "target"]

data["race"] = data["race"].fillna("NA")
data["medical_specialty"] = data["medical_specialty"].fillna("NA")

data = data.sort_values(by="encounter_id")
       
x_train, x_test, y_train, y_test = train_test_split(
    data.drop(not_train_columns, axis=1),
    data["target"],
    shuffle=False,
    test_size=0.10)

cat = [t != "int64" for t in x_train.dtypes]
num = [t == "int64" for t in x_train.dtypes]

transformer = ColumnTransformer(
    [("cat",  OneHotEncoder(handle_unknown="ignore"), cat),
     ("num", Normalizer(), num)]
)


x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

# clf = LogisticRegression(random_state=42, solver="lbfgs")
# clf.fit(x_train, y_train)

# print(f"Train accuracy:{accuracy_score(y_train.to_list(), clf.predict(x_train))}")
# print(f"Test accuracy: {accuracy_score(y_test.to_list(), clf.predict(x_test))}")


# with open("models/ohe.pcl", "wb") as f:
#     pickle.dump(ohe, f)

# with open("models/model.pcl", "wb") as f:
#     pickle.dump(clf, f)

model = CatBoostClassifier(
    iterations=10,
    depth=10,
    learning_rate=0.5,
    loss_function='Logloss',
    logging_level='Verbose'
    )

y_train_int = [int(x) for x in y_train.to_list()]
y_test_int = [int(x) for x in y_test.to_list()]

model.fit(x_train.toarray(), y_train_int)


print(accuracy_score(y_train_int, model.predict(x_train.toarray())))
print(accuracy_score(y_test_int, model.predict(x_test.toarray())))
