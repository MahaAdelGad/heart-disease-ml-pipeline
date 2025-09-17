Heart Disease Prediction - Machine Learning Pipeline


Project Description:
This project analyzes and predicts the risk of heart disease using the Heart Disease UCI dataset.
It follows a comprehensive machine learning pipeline, including preprocessing, dimensionality reduction (PCA), feature selection, supervised/unsupervised models, and hyperparameter tuning.
The best model was exported as a .pkl file for future use.


Steps Implemented:

1- Data Preprocessing & Cleaning:
Loaded dataset
Checked and handled missing values
One-hot encoded categorical features (cp, thal, slope)
Standardized numerical features

2- Exploratory Data Analysis (EDA)
Correlation heatmap
Target distribution

3- Dimensionality Reduction (PCA)
Reduced dataset dimensions while preserving ~95% variance
Visualized explained variance

4- Feature Selection
Random Forest Feature Importance
Recursive Feature Elimination (RFE)
Chi-Square Test

5- Supervised Learning Models
Logistic Regression
Random Forest
KNN
SVM
Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC & AUC

6- Unsupervised Learning Models
K-Means clustering (Elbow method)
Hierarchical clustering (Dendrogram)

7- Hyperparameter Tuning
GridSearchCV for Random Forest
Selected best parameters and improved accuracy

8- Model Export
Saved best Random Forest model with scaler and feature columns as heart_disease_model.pkl


How to Run:
1- Clone or download this repository
2- Install dependencies: pip install -r requirements.txt
3- Run the project: python project.py
4- To load the saved model later:
import joblib
model, scaler, columns = joblib.load("heart_disease_model.pkl")


Results
Best Model: Random Forest (after tuning)
Best Accuracy: ~X% (replace with your result)
ROC curves plotted for all models
Feature importance and clustering visualizations


Files:
project.py → main code
heart.csv → dataset
heart_disease_model.pkl → saved model (binary file, not human-readable)
requirements.txt → dependencies list
README.md → project documentation
