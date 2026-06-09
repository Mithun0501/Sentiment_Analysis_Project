import pandas as pd
import joblib

from sklearn.metrics import accuracy_score

# Load test data
test_data = pd.read_csv("test_data.csv")

test_data = test_data.dropna()

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Prepare test data
X_test = test_data["sentence"]
y_test = test_data["sentiment"]

X_test_vectorized = vectorizer.transform(X_test)

# Predict
predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy * 100, 2), "%")
