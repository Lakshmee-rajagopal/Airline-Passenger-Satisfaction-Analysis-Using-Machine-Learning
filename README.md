# **Airline-Passenger-Satisfaction-Analysis-Using-Machine-Learning**

The primary objective of this project is to build an efficient machine learning pipeline to predict airline customer satisfaction. The output will categorize passengers as either "Satisfied" or "Neutral or Dissatisfied." Performance will be evaluated using various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

**Overview of the Project**

This project involves the following steps:

1) Dataset Preprocessing:
Handle missing values using median imputation.
Remove outliers using the Interquartile Range (IQR) method.
Ensure proper data types for categorical and numerical columns.
Check for duplicate entries and clean the dataset.

2) Exploratory Data Analysis (EDA):
Visualize the distribution of the target variable.
Analyze correlations among features.
Generate visual insights on features like Age, Flight Distance, Travel Class, and In-flight WiFi Service.

3) Addressing Class Imbalance:
Check for imbalanced data in the target variable.
Apply Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset.

4) Feature Selection:
Use SelectKBest with mutual information to identify the top features influencing passenger satisfaction.
Data Splitting and Preprocessing:

5) Split the dataset into training and testing sets.
Apply scaling to numerical features and one-hot encoding to categorical features.

6) Model Training:
Train multiple machine learning models: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and Gradient Boosting.
Evaluate each model using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
Plot confusion matrices and ROC curves for each model.

7) Hyperparameter Tuning:
Use GridSearchCV to optimize hyperparameters for all models.

8) Selecting the Best Model:
Identify the best-performing model based on the weighted average of all evaluation metrics.
Display the best model along with its evaluation metrics.

9) Saving the Model:
Save the best model as a pipeline for future predictions.

10) Prediction Interface:
A prediction interface is developed to help users in testing unseen data.

**DATA STORY**

[Airline Passenger Satisfaction Dataset](https://drive.google.com/file/d/1VtYC86HrBZNrX3-4E-wtQ6ntz469A0IF/view?usp=sharing)

The dataset is sourced from Maven Analytics which represents real-world data collected from airline passengers. The information has been aggregated from passenger surveys and feedback forms administered by airlines to understand customer satisfaction better.

The dataset contains real-world feedback from airline passengers, offering a comprehensive view of customer satisfaction levels and factors influencing them. It provides information about demographic details, flight characteristics, and in-flight service ratings. The data structure includes:
* Demographics: Gender, Age
* Flight Details: Customer Type, Type of Travel, Class, Flight Distance, Departure Delay, Arrival Delay
* Service Ratings: In-flight WiFi Service, Gate Location, Food and Drink, Cleanliness, etc.
* Target Variable: Satisfaction (Satisfied or Neutral/Dissatisfied)

This project successfully predicts passenger satisfaction using various machine learning models. It highlights critical factors affecting satisfaction, offering airlines actionable insights to enhance customer experiences.
