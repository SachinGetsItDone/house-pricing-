# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

# Load the dataset to infer features
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('HousePricePrediction.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main app
def main():
    st.title("🏠 House Price Prediction App")
    st.write("Enter the house features below to predict the sale price")
    
    # Load data and model
    df = load_dataset()
    model = load_model()
    
    if df is None or model is None:
        st.stop()
    
    # Identify target column and features
    target_col = 'SalePrice'
    if target_col not in df.columns:
        # Try to identify target column
        possible_targets = ['SalePrice', 'saleprice', 'price', 'Price']
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            # Assume last column is target
            target_col = df.columns[-1]
    
    # Get feature columns (all except target)
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Separate numerical and categorical features
    numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    st.sidebar.header("Input Features")
    
    # Dictionary to store user inputs
    user_input = {}
    
    # Create input widgets for numerical features
    if numerical_cols:
        st.sidebar.subheader("Numerical Features")
        for col in numerical_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            
            # Handle NaN values
            if pd.isna(min_val):
                min_val = 0.0
            if pd.isna(max_val):
                max_val = 100.0
            if pd.isna(mean_val):
                mean_val = (min_val + max_val) / 2
            
            user_input[col] = st.sidebar.number_input(
                f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                help=f"Range: {min_val:.2f} - {max_val:.2f}"
            )
    
    # Create input widgets for categorical features
    if categorical_cols:
        st.sidebar.subheader("Categorical Features")
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) > 0:
                user_input[col] = st.sidebar.selectbox(
                    f"{col}",
                    options=unique_vals,
                    index=0
                )
            else:
                user_input[col] = st.sidebar.text_input(f"{col}", value="")
    
    # Predict button
    if st.sidebar.button("🔮 Predict Price", type="primary"):
        try:
            # Create DataFrame with user input in exact column order
            input_df = pd.DataFrame([user_input], columns=feature_cols)
            
            # Make prediction
            prediction = model.predict(input_df)
            
            # Display results
            st.success("### Prediction Results")
            col1, col2, col3 = st.columns(3)
            
            with col2:
                st.metric(
                    label="Predicted House Price",
                    value=f"${prediction[0]:,.2f}"
                )
            
            # Show input summary
            with st.expander("📋 View Input Summary"):
                st.dataframe(input_df.T, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("Please ensure all inputs are valid and the model is compatible with the dataset.")
    
    # Display dataset info
    with st.expander("ℹ️ Dataset Information"):
        st.write(f"**Total Features:** {len(feature_cols)}")
        st.write(f"**Numerical Features:** {len(numerical_cols)}")
        st.write(f"**Categorical Features:** {len(categorical_cols)}")
        st.write(f"**Dataset Shape:** {df.shape}")

if __name__ == "__main__":
    main()

scikit-learn
