import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# dataset load
data = pd.read_csv("phishing_dataset.csv")

# features and label
X = data.drop("label", axis=1)
y = data["label"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
model = LogisticRegression(max_iter=1000)

# train
model.fit(X_train, y_train)

# accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# save model
joblib.dump(model, "model.pkl")

print("Model saved successfully")