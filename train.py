import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from catboost import CatBoostClassifier

SAVE_MODELS = False

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
data["running_count"] = data.groupby("patient_nbr").cumcount()

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
           "running_count",
           "readmitted",
           "encounter_id_count",
           "target"]

data["race"] = data["race"].fillna("NA")
data["medical_specialty"] = data["medical_specialty"].fillna("NA")

data = data.sort_values(by="encounter_id")

x_train, x_test, y_train, y_test = train_test_split(
    data.drop(not_train_columns, axis=1),
    data["target"],
    shuffle=False,
    test_size=0.30)

categorical_features = np.where(x_train.dtypes != "int64")[0]

model = CatBoostClassifier(
    iterations=100,
    depth=3,
    learning_rate=0.1,
    loss_function='Logloss',
    logging_level='Verbose',
    cat_features=categorical_features,
    eval_metric="Accuracy",
    early_stopping_rounds=10,
    l2_leaf_reg=0.1
    )


y_train_int = [int(x) for x in y_train.to_list()]
y_test_int = [int(x) for x in y_test.to_list()]

model.fit(x_train,
          y_train_int,
          eval_set=(x_test, y_test_int)
          )

# y_train_int = [int(x) for x in y_train.to_list()]
# y_test_int = [int(x) for x in y_test.to_list()]

# model.fit(x_train.toarray(), y_train_int)

print(roc_auc_score(y_test.to_list(), model.predict_proba(x_test)[:, 1]))
print(accuracy_score(y_test.to_list(), model.predict(x_test)))
# print(accuracy_score(y_train_int, model.predict(x_train.toarray())))
# print(accuracy_score(y_test_int, model.predict(x_test.toarray())))
