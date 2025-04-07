                                               🩺 Heart Disease Prediction using Machine Learning



📌 Objective
To develop a machine learning model that accurately predicts the presence of heart disease in patients based on clinical features, with a strong emphasis on recall to minimize false negatives in diagnosis.



📊 Dataset
Source: heart.csv (from a Kaggle dataset archive)
Total Instances: 303
Features: 13 clinical attributes including:
Age, Sex, Chest Pain Type (cp), Resting Blood Pressure (trestbps)
Cholesterol (chol), Fasting Blood Sugar (fbs), Max Heart Rate (thalach)
Exercise-Induced Angina (exang), ST Depression (oldpeak), etc.
Target: 0 (no heart disease), 1 (has heart disease)


🔍 Data Cleaning & Preprocessing
Removed outliers:
Rows with ca > 3
Rows where thal == 0
Scaled features using StandardScaler for appropriate models
No missing values found


📈 Exploratory Data Analysis (EDA)
Visualized target distribution and its relationship with age, sex, chest pain, cholesterol, heart rate, etc.
Correlation heatmap revealed strongest positive correlation with cp, thalach and strongest negative with oldpeak, exang.


🤖 Models Used
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Classifier (SVC)
Random Forest Classifier


🧪 Evaluation Metrics
Accuracy
Recall (Primary Focus)
Precision
Also used:
Cross-validation (5-fold) for accuracy and recall


🔧 Hyperparameter Tuning
Logistic Regression: GridSearchCV (for C, penalty)
Random Forest: RandomizedSearchCV (for n_estimators, max_depth, min_samples_split, min_samples_leaf)


✅ Best Model
Logistic Regression with tuned hyperparameters (C=0.1, penalty='l2') provided the best balance of recall and interpretability.


📌 Conclusion
This project successfully demonstrates the use of multiple machine learning models to predict heart disease. After evaluating accuracy and recall across models and tuning hyperparameters, Logistic Regression was selected as the final model due to its strong recall score, making it effective for medical diagnosis tasks.


🚀 How to Run
Clone the repo:
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
Install requirements:
pip install -r requirements.txt
Launch notebooks:
jupyter notebook
🛠️ Tech Stack
Python
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn
Jupyter Notebook
