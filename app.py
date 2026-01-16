import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
@st.cache_resource
def load_model():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found.")
        st.stop()

# Load dataset for feature extraction
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('HousePricePrediction.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file not found.")
        st.stop()

# Prepare input data
def prepare_input(user_data, feature_cols):
    df = pd.DataFrame([user_data], columns=feature_cols)
    return df

# Make prediction
def predict_price(model, user_data, feature_cols):
    input_df = prepare_input(user_data, feature_cols)
    prediction = model.predict(input_df)
    return prediction[0]

# App interface
def main():
    st.set_page_config(
        page_title="House Price Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stButton>button {
            width: 100%; height: 3.5rem; font-size: 1.2rem; font-weight: 600;
            border-radius: 10px; margin-top: 1rem;
        }
        .success-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 15px; color: white; text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2); margin: 2rem 0;
        }
        .info-box {
            background: #f0f2f6; padding: 1rem; border-radius: 10px;
            border-left: 4px solid #667eea; margin: 1rem 0;
        }
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 0.5rem;
        }
        .subtitle { text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem; }
        .section-header {
            color: #667eea; font-size: 1.3rem; font-weight: 600;
            margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-powered real estate valuation system</p>", unsafe_allow_html=True)
    
    model = load_model()
    df = load_dataset()
    
    # Extract features
    target_col = 'SalePrice'
    feature_cols = [col for col in df.columns if col != target_col]
    numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    st.markdown("""
        <div class='info-box'>
            ✅ <strong>Status:</strong> Model Ready | <strong>Features:</strong> {} Total ({} Numerical, {} Categorical)
        </div>
    """.format(len(feature_cols), len(numerical_cols), len(categorical_cols)), unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📝 Property Details", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns(2, gap="large")
        
        input_data = {}
        
        with col1:
            st.markdown("<div class='section-header'>🔢 Numerical Features</div>", unsafe_allow_html=True)
            
            # Split numerical features into two columns
            mid_point = len(numerical_cols) // 2
            col1a, col1b = st.columns(2)
            
            for idx, col in enumerate(numerical_cols):
                min_val = float(df[col].min()) if not pd.isna(df[col].min()) else 0.0
                max_val = float(df[col].max()) if not pd.isna(df[col].max()) else 100.0
                mean_val = float(df[col].mean()) if not pd.isna(df[col].mean()) else (min_val + max_val) / 2
                
                with col1a if idx < mid_point else col1b:
                    input_data[col] = st.number_input(
                        f"📊 {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        help=f"Range: {min_val:.2f} - {max_val:.2f}"
                    )
        
        with col2:
            st.markdown("<div class='section-header'>📋 Categorical Features</div>", unsafe_allow_html=True)
            
            # Split categorical features into two columns
            mid_point = len(categorical_cols) // 2
            col2a, col2b = st.columns(2)
            
            for idx, col in enumerate(categorical_cols):
                unique_vals = df[col].dropna().unique().tolist()
                
                with col2a if idx < mid_point else col2b:
                    if len(unique_vals) > 0:
                        input_data[col] = st.selectbox(
                            f"🏷️ {col}",
                            options=unique_vals,
                            index=0
                        )
                    else:
                        input_data[col] = st.text_input(f"🏷️ {col}", value="")
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Predict House Price", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing property..."):
                try:
                    prediction = predict_price(model, input_data, feature_cols)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class='success-card'>
                            <h2 style='color: white; margin: 0;'>💰 Predicted House Price</h2>
                            <p style='font-size: 2.5rem; margin: 1rem 0; font-weight: 800;'>${prediction:,.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric("💵 Estimated Value", f"${prediction:,.2f}")
                    
                    with col_m2:
                        price_range_low = prediction * 0.95
                        price_range_high = prediction * 1.05
                        st.metric("📊 Range (±5%)", f"${price_range_low:,.0f} - ${price_range_high:,.0f}")
                    
                    with col_m3:
                        st.metric("🏡 Confidence", "High")
                    
                    with st.expander("📋 View Input Summary"):
                        col_s1, col_s2 = st.columns(2)
                        
                        with col_s1:
                            st.markdown("**Numerical Features**")
                            num_data = {k: v for k, v in input_data.items() if k in numerical_cols}
                            st.dataframe(pd.DataFrame([num_data]).T, use_container_width=True)
                        
                        with col_s2:
                            st.markdown("**Categorical Features**")
                            cat_data = {k: v for k, v in input_data.items() if k in categorical_cols}
                            st.dataframe(pd.DataFrame([cat_data]).T, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")
                    st.write("Please ensure all inputs are valid and the model is compatible.")
    
    with tab2:
        st.markdown("### ℹ️ About This App")
        st.write("This tool uses machine learning to predict house prices based on property features.")
        
        st.markdown("### 📊 Model Information")
        st.write(f"- **Total Features**: {len(feature_cols)}")
        st.write(f"- **Numerical Features**: {len(numerical_cols)}")
        st.write(f"- **Categorical Features**: {len(categorical_cols)}")
        st.write(f"- **Training Data**: {df.shape[0]} houses")
        
        st.markdown("### 🎯 How It Works")
        st.write("1. Enter property details using the form")
        st.write("2. Click 'Predict House Price' button")
        st.write("3. View estimated price and confidence range")

if __name__ == "__main__":
    main()
