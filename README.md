### 📊 Breast Cancer Classification - Flask Web App

This project is a **Machine Learning-based web application** built using **Flask** that classifies whether a breast tumor is benign or malignant using a trained model. The prediction is based on user-provided data (from `data.csv`) and a logistic regression model trained with `breast_cancer_data`.

---

## 📁 Folder Structure

```
ML3/
│
├── app.py                         # Flask backend server
├── model.py                       # Model training and saving script
├── breast_cancer_model.pkl        # Trained model file
├── Breast_cancer_classification.pkl # Model
│
├── dataset/
│   └── data.csv                   # Breast cancer dataset
│
├── templates/
│   └── index.html                 # Web interface for input
│
├── requirements.txt              # All required Python packages
└── README.md                     # Project documentation
```

---

## 🚀 Features

* Uploads form data through a simple web interface.
* Predicts whether the tumor is **benign** or **malignant**.
* Built using Flask and scikit-learn.
* Includes pre-trained model (`breast_cancer_model.pkl`).

---

## 🧠 Model Training

The `model.py` script uses the breast cancer dataset (`data.csv`) to train a **Logistic Regression** model and saves it as a `.pkl` file.

```bash
python model.py
```

---

## 🖥️ Run the App

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Start the Flask server
python app.py
```

Then open your browser and go to:
**[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 🧾 Requirements

All Python dependencies are listed in `requirements.txt`.

---

## 📷 Screenshot

![screenshot](screenshot.png)

---

## 🔍 Dataset Info

* Name: `data.csv`
* Type: Supervised binary classification
* Target labels: `Malignant`, `Benign`

---

## 📌 License

This project is licensed under the MIT License.

---
