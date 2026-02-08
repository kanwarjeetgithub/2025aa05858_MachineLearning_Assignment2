import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    matthews_corrcoef
)

st.title("ML Classification Model Comparison App")
# -----------------------------
# 📥 Download Sample Test Data
# -----------------------------
st.subheader("Download Sample Test File")
sample_path = "data/breast_cancer_dataset.csv"

if os.path.exists(sample_path):
    with open(sample_path, "rb") as f:
        st.download_button(
            label="Download Sample CSV",
            data=f,
            file_name="breast_cancer_dataset.csv",
            mime="text/csv"
        )
else:
    st.info("Sample file not found in repo. Please add data/breast_cancer_dataset.csv")

# Model paths
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
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

model_name = st.sidebar.selectbox("Select Model", list(model_paths.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())
    # Load feature names used during training
    with open("model/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
        
    # If target column exists
    if df.shape[1] > len(feature_names):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df.copy()
        y = None
    
    # Add missing columns (if any)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    
    # Remove extra columns & reorder correctly
    X = X[feature_names]

    # Load model
    model_path = model_paths[model_name]

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        if y is not None:
            try:
                    y_pred = model.predict(X)
                    # Metrics
                    acc = accuracy_score(y, y_pred)
                    prec = precision_score(y, y_pred)
                    rec = recall_score(y, y_pred)
                    f1 = f1_score(y, y_pred)
                    mcc = matthews_corrcoef(y, y_pred)
                    
                    try:
                        auc = roc_auc_score(y, model.predict_proba(X)[:,1])
                    except:
                        auc = 0
                    
                    st.subheader(f"Model Selected: {model_name}")
                    
                    # 📊 Metrics Table
                    metrics_df = pd.DataFrame({
                        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score", "MCC Score"],
                        "Value": [acc, prec, rec, f1, auc, mcc]
                    })
                    st.write("### Evaluation Metrics Table")
                    st.dataframe(metrics_df, use_container_width=True)
                    # Confusion Matrix Heatmap
                    st.write("### Confusion Matrix")
                    cm = confusion_matrix(y, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    st.pyplot(fig)

                    # Metrics Bar Chart
                    st.write("### Metrics Visualization")

                    metrics = [acc, prec, rec, f1, auc, mcc]
                    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]

                    fig2, ax2 = plt.subplots()
                    sns.barplot(x=labels, y=metrics, ax=ax2)

                    plt.xticks(rotation=30)
                    plt.title("Model Performance Metrics")

                    st.pyplot(fig2)

    
            except:
                st.warning("Target column format not compatible for metric calculation.")
        else:
            st.info("No target column found. Showing predictions only.")
    else:
        st.error("Model file not found! Run train_models.py first.")

        
        
        

        

        

        

        
