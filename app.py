import pandas as pd
from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
import pickle
import json
app = Flask(__name__)

# Load the model
model = CatBoostClassifier()
model.load_model("models/model.cbm")

with open("models/transformer.pcl", "rb") as f:
    transformer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    print(request)
    data = request.get_json(force=True)
    app.logger.info(data)
    # Make prediction using model loaded from disk as per the data.
    #df = pd.DataFrame(json.loads(data), index=[0])
    df = pd.DataFrame(data, index=[0])

    data_array = transformer.transform(df).toarray()

    prediction = model.predict_proba(data_array)
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(list(output))


if __name__ == '__main__':
    app.run(port=5000, debug=True)