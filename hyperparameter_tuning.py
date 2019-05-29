# import pickle
# import json
# import logging
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer

from xgboost import XGBClassifier

from utils import evaluate_model

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# logger = logging.getLogger('diabetes')
# logger.setLevel(logging.DEBUG)
# # create file handler which logs even debug messages
# fh = logging.FileHandler('tuning.log')
# fh.setLevel(logging.DEBUG)
# logger.addHandler(fh)

roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)


# logger.info("Starting data preprocessing")
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
           # "change",
           "examide",
           "citoglipton",
           # "diabetesMed",
           "readmitted",
           # "running_count",
           "encounter_id_count",
           "target"]


data = data.sort_values(by="encounter_id")

x_train, _, y_train,_ = train_test_split(
    data.drop(not_train_columns, axis=1),
    data["target"],
    shuffle=False,
    test_size=0.20)


cat = [t != "int64" for t in x_train.dtypes]
num = [t == "int64" for t in x_train.dtypes]

cat_names = x_train.columns[cat]
num_names = x_train.columns[num]


transformer = ColumnTransformer(
     [("num", StandardScaler(), num),
      ("cat",  OneHotEncoder(handle_unknown="ignore"), cat)],
)

x_train = transformer.fit_transform(x_train)


cat_names = transformer.transformers_[1][1].get_feature_names(cat_names)

all_feature_names = list(num_names)
all_feature_names.extend(cat_names)

# logger.info("Starting cross-validation")



# logger.info("Parameters: %s", params)



# logger.info("Cross-validation ROC-AUC %s", cv)

def objective(space):

    model = XGBClassifier(**space)

    cv = cross_val_score(
        model,
        x_train,
        y_train,
        scoring=roc_auc_scorer,
        cv=3
    )

    avg_score = np.mean(cv)
    
    print(f"SCORE: {avg_score}")

    return{'loss': 1 - avg_score, 'status': STATUS_OK}


params = {
    "max_depth": 5,
    "n_estimators": 10,
    "min_child_weight": 3,
    "colsample_bytree": 0.68,
    "subsample": 0.63
}

space = {
    'max_depth': hp.choice('max_depth', range(12, 3, -2)),
    'min_child_weight': hp.quniform('min_child', 1, 5, 1),
    'subsample': hp.uniform('subsample', 0.5, 0.7)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials)

print(best)
