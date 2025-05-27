# Machine-learning-laptop-price-predictor-model
# Project Objective
The goal of this project is to build a robust machine learning model that predicts laptop prices accurately. As the laptop market continues to expand with various brands and specifications, having a precise pricing model becomes crucial for both consumers and manufacturers. The project also has one user friendly interface that helps the user to enter their desired configurations and predict the price of their dream laptops.
# Overview of Project
This is a end-to-end Machine Learning Project that uses a Supervised Learning Regression Model which has been trained upon laptop dataset using a series of regression algorithms. The project EDA and Model Training part is conducted using Jupyter Notebook and the Application is generated upon Visual Studio Code. Libraries like Sklearn and Streamlit have been used in the process. The web-application is being hosted upon Streamlit Share.
# About Dataset
The underlying dataset consist around 1303 records and 13 features. Among the features are the likes of Company, Type, Inches, ScreenResolution, Cpu, Ram, Memory, Gpu, OpSys, Weight, Price and few unnecessary features. The dataset is very raw and requires alot of cleaning and preprocessing.
# Approach Used
**Data Understanding and Cleaning:**
Import the dataset and explore its shape, sample, and presence of NULL values, duplicate records, missing values, rubbish values.
**Feature Engineering:**
Extract relevant features from the dataset and changing datatypes of features.
Transform and create new features to enhance model performance.
**Data Visualization:**
Visualizing the data and the patterns that it create by performing Univariate, Bivariate Analysis using various various types of charts.
**Feature Selection:**
Selecting the appropriate features that affect the price of the laptop by checking their correlation with the dependent variable.
**ML Model Development:**
Training different regression algorithms like (linear regression, decision trees, gradient boost etc.).
Evaluate model performance using metrics like (R2_Score, Mean Absolute Error, Mean Squared Error ).
Selecting the best model.
**Web App Development:**
Develop a web application using Streamlit where users can input laptop configurations and predict prices for their configuration.
**Deployment:**
Deploy the trained model and its integrated api on Streamlit Share.
# User Interface
![image](https://github.com/user-attachments/assets/3888c52a-080c-4167-ba7c-4d8794d8ca56)
# Conclusions
This project is a machine learning-based Laptop Price Predictor that uses regression models to estimate laptop prices based on their specifications. The key steps involve data preprocessing, feature engineering, model training, and evaluation.

Categorical features such as Company, TypeName, CPU brand, GPU brand, and Operating System are transformed using OneHotEncoding via a ColumnTransformer, which converts them into numerical form suitable for machine learning. These transformed features, along with the original numerical ones, are passed to a Random Forest Regressor, an ensemble learning algorithm that builds multiple decision trees and averages their outputs for robust prediction.

The entire process — from encoding to model training — is managed through a Pipeline to streamline data handling and ensure consistency. The model is trained using the fit() method and evaluated on a test set using R² score (to measure prediction quality) and Mean Absolute Error (MAE) (to assess average error).

