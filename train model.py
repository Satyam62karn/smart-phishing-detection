

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

df = pd.read_csv(r"C:\Users\satya\OneDrive\Desktop\Phishing_Email.csv")
df.dropna(inplace=True)

safe = df[df["Email Type"] == "Safe Email"]
phish = df[df["Email Type"] == "Phishing Email"]

safe = safe.sample(phish.shape[0], random_state=42)
data = pd.concat([safe, phish], ignore_index=True)

data["Email Type"] = data["Email Type"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})

X = data["Email Text"]
y = data["Email Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("model", LogisticRegression())
])

param_grids = {

    "Logistic Regression": {
        "model": [LogisticRegression(max_iter=1000)],
        "model__C": [0.1, 1, 10]
    },

    "SVM (RBF)": {
        "model": [SVC(probability=True)],
        "model__C": [1, 10],
        "model__gamma": ["scale", "auto"]
    },

    "KNN": {
        "model": [KNeighborsClassifier()],
        "model__n_neighbors": [3, 5, 7]
    },

    "Decision Tree": {
        "model": [DecisionTreeClassifier()],
        "model__max_depth": [None, 10, 20]
    },

    "Random Forest": {
        "model": [RandomForestClassifier()],
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 20]
    },

    "MLP": {
        "model": [MLPClassifier(max_iter=300)],
        "model__hidden_layer_sizes": [(100,), (100, 50)],
        "model__alpha": [0.0001, 0.001]
    },

    "XGBoost": {
        "model": [XGBClassifier(eval_metric="logloss")],
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.01, 0.1]
    }
}

results = []
auc_scores = {}
conf_matrices = {}
roc_curves = {}
best_models = {}

for name, params in param_grids.items():

    print(f"\nTraining {name} with GridSearchCV...")

    grid = GridSearchCV(
        pipeline,
        params,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)

    if hasattr(best_model.named_steps["model"], "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append([name, acc, prec, rec, f1])
    auc_scores[name] = auc
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[name] = (fpr, tpr)

metrics_df = pd.DataFrame(
    results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]
)

auc_df = pd.DataFrame(
    list(auc_scores.items()), columns=["Model", "AUC Score"]
)

print("\n=== PERFORMANCE METRICS ===")
print(metrics_df)

print("\n=== AUC SCORES ===")
print(auc_df)

for name, cm in conf_matrices.items():
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plt.figure(figsize=(8,6))
for name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

metrics_df.set_index("Model").plot(kind="bar", figsize=(10,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.show()

for name, model in best_models.items():

    clf = model.named_steps["model"]
    tfidf = model.named_steps["tfidf"]
    features = tfidf.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_

    elif hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_[0])

    else:
        continue

    top = np.argsort(importance)[-20:]

    plt.figure(figsize=(6,6))
    plt.barh(range(20), importance[top])
    plt.yticks(range(20), features[top])
    plt.title(f"Top Features - {name}")
    plt.show()

rf_model = best_models["Random Forest"]

X_test_vec = rf_model.named_steps["tfidf"].transform(X_test)
rf_clf = rf_model.named_steps["model"]

explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_test_vec)

shap.summary_plot(
    shap_values,
    X_test_vec,
    feature_names=rf_model.named_steps["tfidf"].get_feature_names_out()
)