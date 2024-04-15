from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('tunned_kidney_Cancer_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Extract form data dynamically
            form_data = {key: float(value) for key, value in request.form.items()}
            # Prepare input data for prediction
            values = np.array([list(form_data.values())])

            # Make prediction using the loaded model
            classification_prediction = model.predict(values)
            ckd_stage_prediction_prob = model.predict_proba(values)[0]  # Probabilities for all CKD stages

            # Scale the probability to CKD stage 1-5
            ckd_stage_prediction = int(np.argmax(ckd_stage_prediction_prob)) + 1

            # Convert classification prediction to numeric format
            classification = 1 if classification_prediction[0] == 'yes' else 0

            # Construct JSON response
            response = {
                "classification": classification,
                "ckd_stage": ckd_stage_prediction,
                "probabilities": ckd_stage_prediction_prob.tolist()  # Convert probabilities to list for JSON serialization
            }

            return jsonify(response)
        except Exception as e:
            # Handle errors gracefully
            return jsonify({"error": str(e)}), 400

    # Render the initial form template for GET requests
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
