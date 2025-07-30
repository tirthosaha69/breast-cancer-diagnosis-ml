# model.py

import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
df = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
df['label'] = breast_cancer_dataset.target

# Features and labels
X = df.drop(columns='label', axis=1)
y = df['label']

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)

# Model
model = LogisticRegression(max_iter=10000)
model.fit(xtrain, ytrain)

# Save the model
joblib.dump(model, 'breast_cancer_model.pkl')

print("Model trained and saved as 'breast_cancer_model.pkl'")
