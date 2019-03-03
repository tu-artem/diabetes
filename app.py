import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from xgboost import XGBClassifier
import pickle
import json
from lime import lime_tabular

app = Flask(__name__)

# Load the model
model = XGBClassifier()
model.load_model("models/model.xgb")
x_train = np.load("data/processed/x_train.npy")
with open("data/processed/features.json") as f:
    all_feature_names = json.load(f)


with open("models/transformer.pcl", "rb") as f:
    transformer = pickle.load(f)


explainer = lime_tabular.LimeTabularExplainer(
    x_train,
    feature_names=all_feature_names,
    class_names=["NO", "YES"],
    discretize_continuous=True
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # app.logger.info(data)
    df = pd.DataFrame(data, index=[0])
    data_array = transformer.transform(df).toarray()

    prediction = model.predict_proba(data_array)
    # Take the first value of prediction
    output = prediction[0].tolist()
    return jsonify(output)


@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json(force=True)
    # app.logger.info(data)
    df = pd.DataFrame(data, index=[0])
    data_array = transformer.transform(df)
    exp = explainer.explain_instance(
        data_array.toarray()[0],
        model.predict_proba,
        num_features=5,
        top_labels=2
        )

    output = exp.as_list()
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
