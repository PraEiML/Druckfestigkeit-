import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to load the dataset
def load_data():
    data = pd.read_csv("Concrete_Data_Yeh.csv")
    return data

# Function to train a model and predict strength
def train_and_predict(data):
    # Assume that the target variable is "CompressiveStrength"
    X = data.drop(columns=["CompressiveStrength"])
    y = data["CompressiveStrength"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate performance (e.g., Mean Squared Error)
    mse = mean_squared_error(y_test, predictions)

    return model, mse, X_test, y_test, predictions

# Streamlit app layout
def main():
    st.title("Concrete Strength Prediction")

    # Load data
    st.write("### Data Preview:")
    data = load_data()
    st.dataframe(data.head())

    # Show data info
    st.write("### Data Description:")
    st.write(data.describe())

    # Train model and predict
    model, mse, X_test, y_test, predictions = train_and_predict(data)

    # Display performance metrics
    st.write("### Model Performance:")
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Display prediction vs true values
    st.write("### Predictions vs Actual Values:")
    comparison_df = pd.DataFrame({
        "True Values": y_test,
        "Predictions": predictions
    })
    st.dataframe(comparison_df)

    # Option to input new data for prediction
    st.write("### Predict Concrete Strength:")
    with st.form("predict_form"):
        st.write("Enter values for the following features:")
        cement = st.number_input("Cement (kg)", min_value=0.0, max_value=1000.0)
        slag = st.number_input("Slag (kg)", min_value=0.0, max_value=1000.0)
        ash = st.number_input("Ash (kg)", min_value=0.0, max_value=1000.0)
        water = st.number_input("Water (kg)", min_value=0.0, max_value=1000.0)
        superplastic = st.number_input("Superplasticizer (kg)", min_value=0.0, max_value=1000.0)
        coarseagg = st.number_input("Coarse Aggregate (kg)", min_value=0.0, max_value=1000.0)
        fineagg = st.number_input("Fine Aggregate (kg)", min_value=0.0, max_value=1000.0)
        
        submit_button = st.form_submit_button(label="Predict")
        
        if submit_button:
            input_data = np.array([[cement, slag, ash, water, superplastic, coarseagg, fineagg]])
            prediction = model.predict(input_data)
            st.write(f"Predicted Concrete Compressive Strength: {prediction[0]:.2f} MPa")

if __name__ == "__main__":
    main()
