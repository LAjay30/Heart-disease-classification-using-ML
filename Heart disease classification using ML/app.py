from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open("rf.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract numerical inputs
        numerical_features = [float(request.form[key]) for key in [
            'age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
            'exang', 'oldpeak', 'ca'
        ]]

        # One-hot encode categorical inputs
        cp = int(request.form['cp'])  # Values: 0, 1, 2, 3
        thal = int(request.form['thal'])  # Values: 0, 1, 2, 3
        slope = int(request.form['slope'])  # Values: 0, 1, 2

        # One-hot encoding transformation
        cp_encoded = [1 if i == cp else 0 for i in range(4)]
        thal_encoded = [1 if i == thal else 0 for i in range(4)]
        slope_encoded = [1 if i == slope else 0 for i in range(3)]

        # Combine all features
        features = np.array(numerical_features + cp_encoded + thal_encoded + slope_encoded).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction=f"Predicted Class: {prediction}")
    
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
