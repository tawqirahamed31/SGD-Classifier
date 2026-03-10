# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1️. Load the Iris dataset and separate the features (X) and target labels (y).

2️. Split the dataset into training and testing sets using train_test_split().

3️. Standardize the features using StandardScaler to improve model performance.

4️. Train the SGD Classifier using the training data with model.fit().

5️. Predict the species for test data or new flower data using model.predict() and evaluate using accuracy or classification report.

## Program:
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris Dataset
iris = load_iris()

# Features and Target
X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create SGD Classifier
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Train Model
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Output:
<img width="550" height="335" alt="image" src="https://github.com/user-attachments/assets/ffeacb3c-8e46-46ee-b4e6-81f98b8df66c" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
