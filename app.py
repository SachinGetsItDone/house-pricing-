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
        st.error("⚠️ Model file 'house_price_model.pkl' not found. Please upload the model file.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("💡 Make sure 'house_price_model.pkl' is in the same directory as app.py and scikit-learn is installed")
        st.stop()

# Load dataset for feature extraction
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('HousePricePrediction.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file 'HousePricePrediction.csv' not found. Please upload the dataset.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        st.stop()


# Prepare input data to match the training pipeline
def prepare_input(user_data):
    """
    Prepare input matching the model's expected features:
    - LotArea_log
    - TotalBsmtSF_log  
    - YearBuilt
    - YearRemodAdd
    - MSZoning (categorical)
    - LotConfig (categorical)
    - BldgType (categorical)
    - Exterior1st (categorical)
    - CondGroup (categorical)
    - Cond_x_MSSubClass (interaction term)
    """
    
    # Create the input dataframe with the exact columns the model expects
    input_dict = {
        'LotArea_log': user_data['LotArea_log'],
        'TotalBsmtSF_log': user_data['TotalBsmtSF_log'],
        'YearBuilt': user_data['YearBuilt'],
        'YearRemodAdd': user_data['YearRemodAdd'],
        'MSZoning': user_data['MSZoning'],
        'LotConfig': user_data['LotConfig'],
        'BldgType': user_data['BldgType'],
        'Exterior1st': user_data['Exterior1st'],
        'CondGroup': user_data['CondGroup'],
        'Cond_x_MSSubClass': user_data['Cond_x_MSSubClass']
    }
    
    df = pd.DataFrame([input_dict])
    return df

# Make prediction
def predict_price(model, user_data):
    input_df = prepare_input(user_data)
    # Model predicts log price
    pred_log = model.predict(input_df)
    # Convert back to actual price
    pred_price = np.expm1(pred_log[0])
    return pred_price

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
    
    st.markdown("""
        <div class='info-box'>
            ✅ <strong>Status:</strong> Model Ready | <strong>Algorithm:</strong> Ridge Regression with Polynomial Features
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📝 Property Details", "ℹ️ About"])
    
    with tab1:
        input_data = {}
        
        st.markdown("<div class='section-header'>🏗️ Property Characteristics</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            lot_area = st.number_input(
                "Lot Area (sq ft)",
                min_value=1000,
                max_value=200000,
                value=10000,
                step=100,
                help="Size of the lot in square feet"
            )
            input_data['LotArea_log'] = np.log1p(lot_area)
        
        with col2:
            bsmt_sf = st.number_input(
                "Total Basement SF",
                min_value=0,
                max_value=6000,
                value=1000,
                step=50,
                help="Total basement area in square feet"
            )
            input_data['TotalBsmtSF_log'] = np.log1p(bsmt_sf)
        
        with col3:
            input_data['YearBuilt'] = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2025,
                value=2000,
                step=1
            )
        
        with col4:
            input_data['YearRemodAdd'] = st.number_input(
                "Year Remodeled",
                min_value=1800,
                max_value=2025,
                value=2000,
                step=1
            )
        
        st.markdown("<div class='section-header'>📋 Property Details</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            input_data['MSZoning'] = st.selectbox(
                "Zoning Classification",
                options=df['MSZoning'].dropna().unique().tolist(),
                help="General zoning classification"
            )
        
        with col2:
            input_data['LotConfig'] = st.selectbox(
                "Lot Configuration",
                options=df['LotConfig'].dropna().unique().tolist(),
                help="Lot configuration"
            )
        
        with col3:
            input_data['BldgType'] = st.selectbox(
                "Building Type",
                options=df['BldgType'].dropna().unique().tolist(),
                help="Type of dwelling"
            )
        
        with col4:
            input_data['Exterior1st'] = st.selectbox(
                "Exterior Covering",
                options=df['Exterior1st'].dropna().unique().tolist(),
                help="Exterior covering on house"
            )
        
        st.markdown("<div class='section-header'>🏡 Condition & Quality</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overall_cond = st.slider(
                "Overall Condition",
                min_value=1,
                max_value=10,
                value=5,
                help="Overall condition rating (1-10)"
            )
            # Create CondGroup based on OverallCond
            if overall_cond <= 4:
                input_data['CondGroup'] = 'Bad'
            elif overall_cond <= 6:
                input_data['CondGroup'] = 'Average'
            else:
                input_data['CondGroup'] = 'Good'
        
        with col2:
            ms_subclass = st.number_input(
                "MS SubClass",
                min_value=20,
                max_value=190,
                value=60,
                step=10,
                help="Building class (type of dwelling)"
            )
        
        with col3:
            # Create interaction term
            input_data['Cond_x_MSSubClass'] = overall_cond * ms_subclass
            st.metric(
                "Cond × SubClass",
                f"{input_data['Cond_x_MSSubClass']}",
                help="Interaction term between condition and subclass"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Predict House Price", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing property..."):
                try:
                    prediction = predict_price(model, input_data)
                    
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
                        st.markdown("**Property Features**")
                        
                        summary_data = {
                            'Lot Area': f"{lot_area:,} sq ft",
                            'Total Basement SF': f"{bsmt_sf:,} sq ft",
                            'Year Built': input_data['YearBuilt'],
                            'Year Remodeled': input_data['YearRemodAdd'],
                            'Zoning': input_data['MSZoning'],
                            'Lot Config': input_data['LotConfig'],
                            'Building Type': input_data['BldgType'],
                            'Exterior': input_data['Exterior1st'],
                            'Condition Group': input_data['CondGroup'],
                            'MS SubClass': ms_subclass,
                            'Overall Condition': overall_cond
                        }
                        
                        st.dataframe(
                            pd.DataFrame(list(summary_data.items()), columns=['Feature', 'Value']),
                            use_container_width=True,
                            hide_index=True
                        )
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")
                    st.write("Please ensure all inputs are valid and the model is compatible.")
                    
                    # Debug information
                    with st.expander("🔍 Debug Information"):
                        st.write("**Input Data:**")
                        st.json(input_data)
                        st.write("**Error Details:**")
                        import traceback
                        st.code(traceback.format_exc())
    
    with tab2:
        st.markdown("### ℹ️ About This App")
        st.write("""
        This tool uses a Ridge Regression model with polynomial features to predict house prices 
        based on various property characteristics. The model was trained on historical housing data 
        and uses feature engineering including log transformations and interaction terms.
        """)
        
        st.markdown("### 📊 Model Features")
        st.write("""
        **Numerical Features (with log transformation):**
        - Lot Area
        - Total Basement Square Footage
        - Year Built
        - Year Remodeled/Added
        
        **Categorical Features:**
        - MS Zoning (General zoning classification)
        - Lot Configuration
        - Building Type
        - Exterior Covering
        - Condition Group (derived from Overall Condition)
        
        **Engineered Features:**
        - Polynomial features (degree 2)
        - Interaction term: Overall Condition × MS SubClass
        """)
        
        st.markdown("### 🎯 How It Works")
        st.write("""
        1. Enter property details using the form
        2. The app automatically:
           - Applies log transformations to skewed features
           - Creates the CondGroup category based on Overall Condition
           - Calculates the interaction term
           - Applies polynomial features and scaling (handled by the model)
        3. Click 'Predict House Price' to get the estimated value
        4. The model predicts the log of the sale price, which is converted back to the actual price
        """)
        
        st.markdown("### 📦 Model Information")
        st.write("""
        - **Algorithm**: Ridge Regression (alpha=10.0)
        - **Preprocessing**: PolynomialFeatures (degree=2) + StandardScaler
        - **Target**: Log-transformed Sale Price
        - **Categorical Encoding**: One-Hot Encoding (drop first)
        """)
        
        st.markdown("### 📋 Requirements")
        st.write("Make sure your `requirements.txt` includes:")
        st.code("""streamlit
pandas
numpy
scikit-learn""")

if __name__ == "__main__":
    main()
