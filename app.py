import pandas as pd
from flask import Flask, request, jsonify
import pickle
import json
app = Flask(__name__)

# Load the model
with open("models/ohe.pcl", "rb") as f:
    ohe = pickle.load(f)

with open("models/model.pcl", "rb") as f:
    model = pickle.load(f)

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    df = pd.DataFrame(json.loads(data), index=[0])
    enc = ohe.transform(df)
    prediction = model.predict_proba(enc)
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(list(output))


if __name__ == '__main__':
    app.run(port=5000, debug=True)