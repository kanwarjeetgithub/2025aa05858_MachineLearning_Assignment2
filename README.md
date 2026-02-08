
📌 **Problem Statement**

The objective of this project is to build, evaluate, and deploy multiple machine learning classification models using a real-world dataset. The solution demonstrates an end-to-end workflow including model training, evaluation, saving trained models, and building an interactive Streamlit web application for predictions.

📊 **Dataset Description**

This project uses the Breast Cancer Wisconsin Dataset (from Scikit-learn).  
Key characteristics:  
Total samples: 569  
Total features: 30 numerical features  
Target classes: 2 (0 = Malignant, 1 = Benign)  
Type: Binary classification problem  
Why this dataset?  
Meets assignment requirement of >500 instances  
Contains >12 features  
Well-suited for classification model comparison  
Target column:  
target → 0 or 1  

🤖 **Machine Learning Models Implemented**

The following six models were trained on the same dataset:  
--Logistic Regression  
--Decision Tree Classifier  
--K-Nearest Neighbors (KNN)  
--Naive Bayes (GaussianNB)  
--Random Forest (Ensemble)  
--XGBoost (Ensemble)  
All trained models are saved as .pkl files and loaded in the Streamlit application for predictions.  

📈 **Comparison Table with the evaluation metrics**

<img width="590" height="141" alt="image" src="https://github.com/user-attachments/assets/318d6a8d-0d2e-4b2e-ac61-465024268193" />  
  
🔎 **Observations on Model Performance**    
        **Model	Observation**    
          **Logistic Regression** - Good baseline performance and stable results.  
          **Decision Tree**	 - Easy to interpret but may overfit.  
          **KNN**	 - Works well but sensitive to feature scaling.  
          **Naive Bayes**	 - Fast and simple; performs well on independent features.  
          **Random Forest**	 - Strong performance and reduces overfitting.  
          **XGBoost**	 - Often achieves the best accuracy due to boosting technique.  

