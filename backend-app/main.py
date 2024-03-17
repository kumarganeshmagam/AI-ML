from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained models
model_weight = joblib.load("decision_tree_weight.pkl")
model_length = joblib.load("decision_tree_length.pkl")
model_hw = joblib.load("decision_tree_hw.pkl")

# Function to predict species based on input weight
def predict_species_weight(weight):
    weight_array = np.array([weight]).reshape(1, -1)
    predicted_species = model_weight.predict(weight_array)[0]
    return predicted_species

# Function to predict species based on input length
def predict_species_length(length1, length2, length3):
    length_data = np.array([[length1, length2, length3]])
    predicted_species_length = model_length.predict(length_data)[0]
    return predicted_species_length

# Function to predict species based on height and width
def predict_species_hw(height, width):
    hw_data = np.array([[height, width]])
    predicted_species = model_hw.predict(hw_data)[0]
    return predicted_species

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    category = data['category'].lower()

    if category == 'weight':
        weight = float(data['weight'])
        predicted_species = predict_species_weight(weight)
    elif category == 'length':
        length1 = float(data['length1'])
        length2 = float(data['length2'])
        length3 = float(data['length3'])
        predicted_species = predict_species_length(length1, length2, length3)
    elif category == 'hw':
        height = float(data['height'])
        width = float(data['width'])
        predicted_species = predict_species_hw(height, width)
    else:
        return jsonify({"error": "Invalid category selected. Please choose 'weight', 'length', or 'hw'."}), 400

    return jsonify({"predicted_species": predicted_species}), 200

if __name__ == '__main__':
    app.run(debug=True)
