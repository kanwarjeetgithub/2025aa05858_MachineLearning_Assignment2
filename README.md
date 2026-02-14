
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
          
🔹 **Project Structure**

      project-folder/
        │-- app.py
        │-- requirements.txt
        │-- README.md
        │-- train_models.py
        │-- model/
        │   │-- decision_tree.pkl
        │   │-- feature_names.pkl
        │   │-- knn.pkl
        │   │-- logistic.pkl
        │   │-- naive_bayes.pkl
        │   │-- random_forest.pkl
        │   │-- xgboost.pkl
        │-- data/
        │   │--breast_cancer_dataset.csv

🔹 Streamlit App Features  

  *Upload test dataset (CSV)      
  *Select model from dropdown      
    *Display:       
      *Evaluation metrics        
      *Confusion matrix        
      *Predictions      
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/c08885ad-2768-4678-8233-fda655b18988" /> 
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/4444054d-42ec-4264-bf4d-fd60220b6598" />


  🔹 How to Run the Project       
      
        **Install dependencies    
          pip install -r requirements.txt 
          
        **Run Python program to create models which will save models in model folder as pkl files    
          python train_models.py    
          
        **Run Streamlit app on local    
          python -m streamlit run app.py    
          
        **Deploy on Streamlit Community Cloud       
          1. Go to https://streamlit.io/cloud    
          2. Sign in using GitHub account    
          3. Click “New App”    
          4. Select your repository    
          5. Choose branch (usually main)    
          6. Select app.py    
          7. Click Deploy      

## 🔹 Deployment

**Live App:**  
[https://2025aa05858machinelearningassignment2-j5yj6n6kd6nuzgh6ynrsht.streamlit.app/](https://2025aa05858machinelearningassignment2-j5yj6n6kd6nuzgh6ynrsht.streamlit.app/)

**GitHub Repository:**  
[https://github.com/kanwarjeetgithub/2025aa05858_MachineLearning_Assignment2](https://github.com/kanwarjeetgithub/2025aa05858_MachineLearning_Assignment2)


           



  

