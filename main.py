import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
def logout():
    st.session_state.logged_in = False

# Login page
def login_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://getwallpapers.com/wallpaper/full/b/e/0/1183587-gold-background-wallpaper-1920x1080-hd-for-mobile.jpg');
            background-size: cover;
            background-position: center;
            padding: 5rem;
            border-radius: 10px;
            color: #ffffff; /* Text color for better visibility on wallpaper */
            background-size: 100% 100%;
        }
        .login-form {
            max-width: 300px;
            margin: 40px auto;
            padding: 30px;
            background-color: #f9f9f9;
            border: 1px solid #ccc;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='color: white;'>Login Page</h1>", unsafe_allow_html=True)

    st.markdown("<p style='color: white;'>Username</p>", unsafe_allow_html=True)
    st.session_state.username = st.text_input("")

    st.markdown("<p style='color: white;'>Password</p>", unsafe_allow_html=True)
    st.session_state.password = st.text_input("", type="password")


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
    data['Unemployment Rate'].fillna(data['Unemployment Rate'].mean(), inplace=True)
    data['GDP'].fillna(data['GDP'].mean(), inplace=True)
    data['Applied'].fillna(data['Applied'].mean(), inplace=True)

    # Encode and scale features
    label_encoder = LabelEncoder()
    data['Inflation Rate '] = label_encoder.fit_transform(data['Inflation Rate '])

    scaler = StandardScaler()
    data[['Inflation Rate ', 'Unemployment Rate', 'GDP', 'Applied']] = scaler.fit_transform(data[['Inflation Rate ', 'Unemployment Rate', 'GDP', 'Applied']])

    # Define features and target
    features = ['Inflation Rate ', 'Unemployment Rate', 'GDP', 'Applied']
    X = data[features]
    y = data['Average Closing Price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plot actual vs. predicted gold prices
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], y, label='Actual Gold Prices')
    plt.plot(data['Date'].iloc[y_test.index], y_pred, label='Predicted Gold Prices', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.legend()

    future_Inflation_Rate = st.number_input("Enter future inflation rate")
    future_Unemployment_Rate = st.number_input("Enter future Unemployment Rate")
    future_GDP = st.number_input("Enter future GDP")
    future_Applied = st.number_input("Enter future Tariff Rate")

    # Predict future gold price based on new economic data
    future_economic_data = np.array([[future_Inflation_Rate, future_Unemployment_Rate, future_GDP, future_Applied]])
    future_gold_price = model.predict(future_economic_data)
    st.write(f'Predicted Gold Price: {future_gold_price[0]}')

# Main application page
def app_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://api.deepai.org/job-view-file/778477cf-1da2-4329-a340-2d8654190053/outputs/output.jpg?art-image=true');
            background-size: fit;
            background-position: center;
            padding: 5rem;
            border-radius: 10px;
            color: #ffffff; /* Text color for better visibility on wallpaper */
            background-size: 100% 100%;
        }
        
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Gold Predicton App")
    st.write("Welcome to the app!")

    # Add two CSV file uploaders
    st.header("Upload CSV files")
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
