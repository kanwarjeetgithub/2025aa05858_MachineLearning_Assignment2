import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title("Machine Learning Model Comparison App")

# Model dictionary
model_paths = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# Sidebar
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

model_name = st.sidebar.selectbox("Select Model", list(model_paths.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Load model
    model_path = model_paths[model_name]

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_pred = model.predict(X)

        st.subheader("Model Selected:")
        st.write(model_name)

        # Metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        st.subheader("Evaluation Metrics")
        st.write(f"Accuracy: {acc:.3f}")
        st.write(f"Precision: {prec:.3f}")
        st.write(f"Recall: {rec:.3f}")
        st.write(f"F1 Score: {f1:.3f}")

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y, y_pred))
    else:
        st.error("Model file not found! Run train_models.py first.")
