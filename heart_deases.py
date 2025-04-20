# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle   # using pickle5 for saving/loading

# Load dataset
db = pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\heart deases\heart_disease_data.csv')

# Check dataset
print("First 5 rows:\n", db.head())
print("Dataset shape:", db.shape)
print("Missing values:\n", db.isnull().sum())
print("Target value counts:\n", db['target'].value_counts())

# Split data into features and label
x = db.drop(columns='target', axis=1)
y = db['target']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

print("Shapes:")
print("x:", x.shape)
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)

# Model training
model = LogisticRegression()
model.fit(x_train, y_train)

# Accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Training Accuracy:", training_data_accuracy)

# Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print("Test Accuracy:", test_data_accuracy)

# Save the model using pickle5
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as heart_disease_model.pkl")

# Example prediction using saved model
input_data = (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2)
input_array = np.asarray(input_data).reshape(1, -1)

# Load the model again (to demonstrate)
with open('heart_disease_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Prediction from loaded model
prediction = loaded_model.predict(input_array)
print("Prediction (0 = No disease, 1 = Disease):", prediction[0])
