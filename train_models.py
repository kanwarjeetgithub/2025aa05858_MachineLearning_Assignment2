import os
import pickle
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create model folder
os.makedirs("model", exist_ok=True)

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "logistic.pkl": LogisticRegression(max_iter=500),
    "decision_tree.pkl": DecisionTreeClassifier(),
    "knn.pkl": KNeighborsClassifier(),
    "naive_bayes.pkl": GaussianNB(),
    "random_forest.pkl": RandomForestClassifier(),
    "xgboost.pkl": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train & Save
for file_name, model in models.items():
    model.fit(X_train, y_train)
    with open(f"model/{file_name}", "wb") as f:
        pickle.dump(model, f)

print("All models saved in /model folder")
