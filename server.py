from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model from the shared volume
model = joblib.load('/app/models/iris_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input array from the request body
    get_json = request.get_json()
    iris_input = get_json['input']

    # Make prediction using the model
    prediction = model.predict(np.array(iris_input).reshape(1, -1))

    # Return the prediction as a response
    return jsonify({"prediction": prediction[0]})


@app.route('/')
def hello():
    return 'Welcome to Docker Lab'


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')