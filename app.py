from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')

# Feature names based on the breast cancer dataset
features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_method = request.form['input_method']

        try:
            if input_method == 'comma':
                input_data = request.form['features_comma']
                input_data = [float(i.strip()) for i in input_data.split(',')]
            else:
                input_data = [float(request.form[f'feature{i}']) for i in range(30)]

            input_data_np = np.asarray(input_data).reshape(1, -1)
            prediction = model.predict(input_data_np)
            result = "✅ The Breast Cancer is Benign" if prediction[0] == 1 else "⚠️ The Breast Cancer is Malignant"
            return render_template('index.html', result=result)
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
