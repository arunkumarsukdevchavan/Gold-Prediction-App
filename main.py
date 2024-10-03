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
    future_Unemployment_Rate = st.number_input("Future Unemployment Rate", format="%.2f")
    future_GDP = st.number_input("Future GDP", format="%.2f")
    future_Applied = st.number_input("Future Applied Rate", format="%.2f")

    if st.button("Predict Future Gold Price"):
        # Prepare future data for prediction
        future_economic_data = np.array([[future_Inflation_Rate, future_Unemployment_Rate, future_GDP, future_Applied]])
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
