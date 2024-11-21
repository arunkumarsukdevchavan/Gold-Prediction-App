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

## CODE:
```
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login function
def login():
    if st.session_state.username == "arun" and st.session_state.password == "mastermind":
        st.session_state.logged_in = True
    elif st.session_state.username == "tejaswini" and st.session_state.password == "teja321":
        st.session_state.logged_in = True
    elif st.session_state.username == "thirisha" and st.session_state.password == "3sha321":
        st.session_state.logged_in = True
    else:
        st.session_state.login_error = "Invalid username or password"

# Logout function
def logout():
    st.session_state.logged_in = False

# Login page
def login_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://e1.pxfuel.com/desktop-wallpaper/581/154/desktop-wallpaper-backgrounds-for-login-page-login-page.jpg');
            background-size: cover;
            background-position: center;
            padding: 5rem;
            border-radius: 10px;
            color: #ffffff;
            background-size: 100% 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='color: white;'>Login Page</h1>", unsafe_allow_html=True)

    st.session_state.username = st.text_input("Username")
    st.session_state.password = st.text_input("Password", type="password")

    if st.button("Login"):
        login()

    if 'login_error' in st.session_state:
        st.error(st.session_state.login_error)

# Data processing and model training function
def process_and_train(gold_prices, economic_data):
    # Merge datasets on Date
    data = pd.merge(gold_prices, economic_data, on='Date', how='left')

    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)

    # Fill missing values with the mean
    data['Inflation Rate '].fillna(data['Inflation Rate '].mean(), inplace=True)
    data['Interest Rate'].fillna(data['Interest Rate'].mean(), inplace=True)
    data['GDP'].fillna(data['GDP'].mean(), inplace=True)
    data['Tariff Rate'].fillna(data['Tariff Rate'].mean(), inplace=True)

    # Encode and scale features
    label_encoder = LabelEncoder()
    data['Inflation Rate '] = label_encoder.fit_transform(data['Inflation Rate '])

    scaler = StandardScaler()
    data[['Inflation Rate ', 'Interest Rate', 'GDP', 'Tariff Rate']] = scaler.fit_transform(
        data[['Inflation Rate ', 'Interest Rate', 'GDP', 'Tariff Rate']])

    # Define features and target
    features = ['Inflation Rate ', 'Interest Rate', 'GDP', 'Tariff Rate']
    X = data[features]
    y = data['Average Closing Price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
    }

    # Dictionary to store R² scores
    r2_scores = {}

    # Train each model and calculate R² scores
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores[model_name] = r2

    # Display R² scores for each model
    st.write("## Model R² Scores")
    for model_name, r2 in r2_scores.items():
        st.write(f"{model_name}: R² Score = {r2:.4f}")

    # Plot advanced comparison of models with R² values on the bars
    st.write("## Model Comparison Plot")
    model_names = list(r2_scores.keys())
    r2_values = list(r2_scores.values())

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors and create the bar chart
    bar_colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars = ax.bar(model_names, r2_values, color=bar_colors)

    # Add a grid for clarity
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Annotate bars with the R² scores
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Customize axis labels and title
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of R² Scores by Model', fontsize=16, fontweight='bold')

    # Customize ticks and axis
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=12, rotation=45, ha='right')
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # Adjust Y-axis to better visualize R² scores

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Inputs for future predictions
    st.write("### Enter Future Economic Data for Prediction:")
    future_Inflation_Rate = st.number_input("Future Inflation Rate", format="%.2f")
    future_Interest_Rate = st.number_input("Future Interest Rate", format="%.2f")
    future_GDP = st.number_input("Future GDP", format="%.2f")
    future_Applied = st.number_input("Future Applied Rate", format="%.2f")

    if st.button("Predict Future Gold Price"):
        # Prepare future data for prediction
        future_economic_data = np.array([[future_Inflation_Rate, future_Interest_Rate, future_GDP, future_Applied]])
        future_economic_data_scaled = scaler.transform(future_economic_data)

        # Display future predictions for each model
        st.write("## Future Gold Price Predictions")
        for model_name, model in models.items():
            future_gold_price = model.predict(future_economic_data_scaled)
            st.write(f"{model_name}: Predicted Future Gold Price = {future_gold_price[0]:.2f}")

# Main application page
def app_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://t4.ftcdn.net/jpg/03/09/30/11/360_F_309301133_FeVFkJxwrgZmjSWQ0HWEu1nF3l6ZMCqS.jpg');
            background-size: fit;
            background-position: center;
            padding: 5rem;
            border-radius: 10px;
            color: #ffffff;
            background-size: 100% 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='color: black;'>Gold Prediction App</h1>", unsafe_allow_html=True)
    st.header("Welcome to the app!")

    # Add two CSV file uploaders
    st.write("Upload CSV files")
    uploaded_file1 = st.file_uploader("Select gold price CSV file", type=["csv"], key="file1")
    uploaded_file2 = st.file_uploader("Select economic data CSV file", type=["csv"], key="file2")

    if uploaded_file1 is not None and uploaded_file2 is not None:
        # Read the uploaded CSV files
        gold_prices = pd.read_csv(uploaded_file1)
        economic_data = pd.read_csv(uploaded_file2)

        # Process data and train the model
        process_and_train(gold_prices, economic_data)

    if st.button("Logout"):
        logout()

# Display appropriate page
if st.session_state.logged_in:
    app_page()
else:
    login_page()
```
## OUTPUT:
![WhatsApp Image 2024-11-12 at 15 18 04_33d3b2ad](https://github.com/user-attachments/assets/720c4713-c5c5-4b1e-85f5-83015c89c6f5)
![WhatsApp Image 2024-11-12 at 15 18 41_99c8dd34](https://github.com/user-attachments/assets/23b9f228-51ff-4636-8ca5-ca2a215b91a1)

## RESULT:
The app successfully predicts gold prices using machine learning models (Random Forest, Gradient Boosting, and Decision Tree), achieving competitive R² scores for each model. It provides a user-friendly interface for uploading data, visualizing model performance, and generating future gold price predictions based on economic indicators. This makes it a practical tool for financial decision-making and investment planning.

