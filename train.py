import pickle
import json

import pandas as pd
import numpy as np


from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from utils import evaluate_model

SAVE_MODELS = True

data = pd.read_csv("data/raw/Chapter_3_Diabetes_data.csv", low_memory=False)

data["race"] = data["race"].fillna("Other")
data["medical_specialty"] = data["medical_specialty"].fillna("NA")
data["payer_code"] = data["payer_code"].fillna("NA")

age_mapping = {
    "[0-10)": 10,
    "[10-20)": 20,
    "[20-30)": 30,
    "[30-40)": 40,
    "[40-50)": 50,
    "[50-60)": 60,
    "[60-70)": 70,
    "[70-80)": 80,
    "[80-90)": 90,
    "[90-100)": 100
}
data["age"] = data["age"].replace(age_mapping)

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
data["running_count"] = data.groupby("patient_nbr").cumcount()

data["target"] = data["readmitted"] == "<30"
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
           "examide",
           "citoglipton",
           "diabetesMed",
           "readmitted",
           "running_count",
           "encounter_id_count",
           "target"]


data = data.sort_values(by="encounter_id")

x_train, x_test, y_train, y_test = train_test_split(
    data.drop(not_train_columns, axis=1),
    data["target"],
    shuffle=False,
    test_size=0.20)

# categorical_features = np.where(x_train.dtypes != "int64")[0]

cat = [t != "int64" for t in x_train.dtypes]
num = [t == "int64" for t in x_train.dtypes]

cat_names = x_train.columns[cat]
num_names = x_train.columns[num]


transformer = ColumnTransformer(
     [("num", Normalizer(), num),
      ("cat",  OneHotEncoder(handle_unknown="ignore"), cat)],
)

x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

cat_names = transformer.transformers_[1][1].get_feature_names(cat_names)

all_feature_names = list(num_names)
all_feature_names.extend(cat_names)

model = XGBClassifier(
    max_depth=5,
    early_stopping_rounds=10,
    scale_pos_weight=3,
    min_child_weight=1)


# y_train_int = [int(x) for x in y_train.to_list()]
# y_test_int = [int(x) for x in y_test.to_list()]

model.fit(x_train,
          y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=True)

# y_train_int = [int(x) for x in y_train.to_list()]
# y_test_int = [int(x) for x in y_test.to_list()]

# model.fit(x_train.toarray(), y_train_int)

print(evaluate_model(y_test, model.predict_proba(x_test)[:, 1]))

# print(accuracy_score(y_train_int, model.predict(x_train.toarray())))
# print(accuracy_score(y_test_int, model.predict(x_test.toarray())))

if SAVE_MODELS:
    #model.save_model("models/model.xgb")

    with open("models/model.pcl", "wb") as f:
        pickle.dump(model, f)

    with open("models/transformer.pcl", "wb") as f:
        pickle.dump(transformer, f)

    np.save("data/processed/x_train.np", x_train)

    with open("data/processed/features.json", "w") as f:
        json.dump(all_feature_names, f)
