from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        
        print("Form data received:", data)  # Debugging line to print form data
        
        input_data = np.array(data).reshape(1, -1)
        prediction = model.predict(input_data)

        result = "has Heart Disease" if prediction[0] == 1 else "does NOT have Heart Disease"
        print("Prediction result:", result)  # Debugging line to print prediction result

        return render_template('index.html', prediction_text=f'The patient {result}.')  # Pass result to HTML

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')  # In case of any error


if __name__ == '__main__':
    app.run(debug=True)
