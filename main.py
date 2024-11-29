import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Upload Concrete Data CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Clean column names by stripping leading/trailing spaces and converting to lowercase
        data.columns = data.columns.str.strip().str.lower()
        
        # Display columns in the dataset for debugging
        st.write("Columns in the dataset:", data.columns.tolist())
        return data
    else:
        st.error("Please upload a CSV file.")
        return None

# Function to train a machine learning model and predict
def train_and_predict(data):
    # Ensure the column names are clean (no leading/trailing spaces)
    data.columns = data.columns.str.strip().str.lower()

    # Assuming the model should predict 'compressive_strength' (we'll generate synthetic labels here)
    # Generate 'compressive_strength' as an example using some of the features.
    if "cement" in data.columns and "slag" in data.columns and "water" in data.columns:
        # We'll make a synthetic target (compressive strength) by a weighted sum of some columns.
        st.write("Generating synthetic compressive strength for training...")
        data['compressive_strength'] = (0.5 * data['cement'] + 0.3 * data['slag'] + 0.2 * data['water'])
    else:
        st.error("Necessary columns like 'cement', 'slag', or 'water' are missing.")
        return None, None, None, None, None

    # Prepare data for training
    X = data.drop(columns=['compressive_strength'])  # Features (everything except compressive strength)
    y = data['compressive_strength']  # Target (compressive strength)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features (important for many models, especially if the features have different scales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the model - you can use LinearRegression or RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict values on the test set
    predictions = model.predict(X_test_scaled)

    # Calculate performance (Mean Squared Error)
    mse = mean_squared_error(y_test, predictions)

    return model, mse, X_test, y_test, predictions

# Streamlit app layout
def main():
    st.title("Concrete Strength Prediction (Betonfestigkeit)")

    # Load the dataset
    data = load_data()

    if data is not None:
        # Train the model and make predictions
        model, mse, X_test, y_test, predictions = train_and_predict(data)

        if model is not None:
            # Display performance metrics
            st.write("### Model Performance:")
            st.write(f"Mean Squared Error: {mse:.2f}")

            # Display comparison between true and predicted values
            st.write("### Predictions vs Actual Values:")
            comparison_df = pd.DataFrame({
                "True Values": y_test,
                "Predictions": predictions
            })
            st.dataframe(comparison_df)

            # Option to input new data for prediction
            st.write("### Predict Concrete Strength:")
            with st.form("predict_form"):
                # Inputs for concrete mix parameters
                cement = st.number_input("Cement (kg)", min_value=0.0, max_value=1000.0)
                slag = st.number_input("Slag (kg)", min_value=0.0, max_value=1000.0)
                water = st.number_input("Water (kg)", min_value=0.0, max_value=1000.0)
                superplastic = st.number_input("Superplasticizer (kg)", min_value=0.0, max_value=1000.0)
                coarseagg = st.number_input("Coarse Aggregate (kg)", min_value=0.0, max_value=1000.0)
                fineagg = st.number_input("Fine Aggregate (kg)", min_value=0.0, max_value=1000.0)
                
                submit_button = st.form_submit_button(label="Predict")
                
                if submit_button:
                    # New data for prediction (after scaling)
                    input_data = np.array([[cement, slag, water, superplastic, coarseagg, fineagg]])
                    input_data_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_data_scaled)
                    st.write(f"Predicted Concrete Compressive Strength: {prediction[0]:.2f} MPa")

if __name__ == "__main__":
    main()
