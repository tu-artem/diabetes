import pandas as pd
# import numpy as np
from flask import Flask, request, jsonify
# from xgboost import XGBClassifier
import pickle
import json
from eli5 import explain_prediction, format_as_dataframe

app = Flask(__name__)

# Load the model
# model = XGBClassifier()
# model.load_model("models/model.xgb")
with open("models/model.pcl", "rb") as f:
    model = pickle.load(f)

with open("data/processed/features.json") as f:
    all_feature_names = json.load(f)


with open("models/transformer.pcl", "rb") as f:
    transformer = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # app.logger.info(data)
    df = pd.DataFrame(data, index=[0])
    app.logger.info(df.iloc[0])
    data_array = transformer.transform(df)

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

    exp = explain_prediction(
        model,
        data_array[0],
        feature_names=all_feature_names,
        top=(5, 5),
        targets=[True])

    output = format_as_dataframe(exp).to_dict()
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
