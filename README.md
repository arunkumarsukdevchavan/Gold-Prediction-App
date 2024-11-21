# Gold-Prediction-App
## Problem Statement :
Predicting gold prices is a challenging task due to the dynamic and multifactorial nature of the market. This application aims to build a machine learning-based tool to predict future gold prices using historical gold price data and key economic indicators such as inflation rates, interest rates, GDP, and tariff rates. The ultimate goal is to provide accurate predictions for financial decision-making and investment strategies.
## Algorithm :
1)Import Libraries:
* Streamlit for the web app interface.
* Pandas for data manipulation.
* Scikit-learn for machine learning models and data preprocessing.
* Matplotlib for data visualization.

2)Initialize Session State:Define and manage session states for user login/logout using Streamlit's session state feature.

3)Login System:
* Create a secure login system with predefined credentials.
* Verify username and password combinations and provide feedback for invalid login attempts.
  
4)Data Upload: Allow users to upload two CSV files(One containing historical gold prices and One containing economic indicators (e.g., inflation rate, GDP, etc.)

5)Data Preprocessing:
* Merge datasets on the Date column.
* Convert Date to a datetime format and sort the data.
* Handle missing values by filling them with the mean of the respective column.
* Standardize numeric features using StandardScaler.
  
6)Feature Selection:
* Define input features: Inflation Rate, Interest Rate, GDP, and Tariff Rate.
* Set the target variable: Average Closing Price.

7)Model Training:
* Split the data into training and testing sets (80% train, 20% test).
* Initialize three regression models: Random Forest Regressor, Gradient Boosting Regressor, Decision Tree Regressor.
* Train each model using the training data.

8)Evaluate Model Performance:
* Test each model on the test dataset.
* Calculate the R² score for each model to assess performance.
* Store the scores in a dictionary for comparison.
  
9)Visualize Results:
* Display the R² scores of all models in text and as a bar chart.
* Annotate the chart with precise R² values.
  
10)Future Predictions:
* Create input fields for users to provide future economic indicator values.
* Scale the inputs and make predictions using all trained models.
* Display the predicted gold prices for each model.

11)Logout System:Include a logout button that resets the session state, returning the user to the login page.

## OUTPUT:
![WhatsApp Image 2024-11-12 at 15 18 04_33d3b2ad](https://github.com/user-attachments/assets/720c4713-c5c5-4b1e-85f5-83015c89c6f5)
![WhatsApp Image 2024-11-12 at 15 18 41_99c8dd34](https://github.com/user-attachments/assets/23b9f228-51ff-4636-8ca5-ca2a215b91a1)

## RESULT:
The app successfully predicts gold prices using machine learning models (Random Forest, Gradient Boosting, and Decision Tree), achieving competitive R² scores for each model. It provides a user-friendly interface for uploading data, visualizing model performance, and generating future gold price predictions based on economic indicators. This makes it a practical tool for financial decision-making and investment planning.

