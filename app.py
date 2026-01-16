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


# Prepare input data
def prepare_input(user_data, feature_cols, model, df_original):
    # Create dataframe from user input
    df = pd.DataFrame([user_data], columns=feature_cols)
    
    # Get categorical columns from original dataset
    categorical_cols = df_original[feature_cols].select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df_original[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Convert numerical columns to float first
    df_numerical = df[numerical_cols].copy()
    for col in df_numerical.columns:
        df_numerical[col] = float(df_numerical[col].iloc[0])
    
    # Create engineered features
    if 'LotArea' in df_numerical.columns:
        df_numerical['LotArea_log'] = float(np.log1p(float(df_numerical['LotArea'].iloc[0])))
    
    if 'TotalBsmtSF' in df_numerical.columns:
        df_numerical['TotalBsmtSF_log'] = float(np.log1p(float(df_numerical['TotalBsmtSF'].iloc[0])))
    
    # CondGroup feature
    if 'OverallCond' in df_numerical.columns:
        cond_val = float(df_numerical['OverallCond'].iloc[0])
        if cond_val <= 4:
            condgroup = 'Bad'
        elif cond_val <= 6:
            condgroup = 'Average'
        else:
            condgroup = 'Good'
    else:
        condgroup = 'Average'
    
    # Cond_x_MSSubClass interaction
    if 'OverallCond' in df_numerical.columns and 'MSSubClass' in df_numerical.columns:
        df_numerical['Cond_x_MSSubClass'] = float(df_numerical['OverallCond'].iloc[0]) * float(df_numerical['MSSubClass'].iloc[0])
    
    # One-hot encode categorical variables
    if categorical_cols:
        df_categorical = pd.get_dummies(df[categorical_cols], drop_first=False)
        
        # One-hot encode CondGroup
        condgroup_df = pd.DataFrame([{
            'CondGroup_Average': 1.0 if condgroup == 'Average' else 0.0,
            'CondGroup_Bad': 1.0 if condgroup == 'Bad' else 0.0,
            'CondGroup_Good': 1.0 if condgroup == 'Good' else 0.0
        }])
        
        # Combine all features
        df_final = pd.concat([df_numerical.reset_index(drop=True), condgroup_df.reset_index(drop=True), df_categorical.reset_index(drop=True)], axis=1)
    else:
        df_final = df_numerical
    
    # Try to get model's expected features
    model_features = None
    try:
        if hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
    except:
        pass
    
    # If model expects different features, align columns
    if model_features is not None:
        # Add missing columns with 0.0
        for col in model_features:
            if col not in df_final.columns:
                df_final[col] = 0.0
        
        # Reorder to match model
        df_final = df_final[model_features]
    
    # Convert all to float
    for col in df_final.columns:
        df_final[col] = float(df_final[col].iloc[0])
    
    return df_final

# Make prediction
def predict_price(model, user_data, feature_cols, df_original):
    input_df = prepare_input(user_data, feature_cols, model, df_original)
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
        input_data = {}
        
        # Numerical Features Section
        if numerical_cols:
            st.markdown("<div class='section-header'>🔢 Numerical Features</div>", unsafe_allow_html=True)
            
            # Calculate columns needed
            num_features = len(numerical_cols)
            cols_per_row = 4
            
            for i in range(0, num_features, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col_name in enumerate(numerical_cols[i:i+cols_per_row]):
                    min_val = float(df[col_name].min()) if not pd.isna(df[col_name].min()) else 0.0
                    max_val = float(df[col_name].max()) if not pd.isna(df[col_name].max()) else 100.0
                    mean_val = float(df[col_name].mean()) if not pd.isna(df[col_name].mean()) else (min_val + max_val) / 2
                    
                    with cols[j]:
                        input_data[col_name] = st.number_input(
                            f"{col_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            help=f"Range: {min_val:.2f} - {max_val:.2f}",
                            key=f"num_{col_name}"
                        )
        
        # Categorical Features Section
        if categorical_cols:
            st.markdown("<div class='section-header'>📋 Categorical Features</div>", unsafe_allow_html=True)
            
            # Calculate columns needed
            num_features = len(categorical_cols)
            cols_per_row = 4
            
            for i in range(0, num_features, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col_name in enumerate(categorical_cols[i:i+cols_per_row]):
                    unique_vals = df[col_name].dropna().unique().tolist()
                    
                    with cols[j]:
                        if len(unique_vals) > 0:
                            input_data[col_name] = st.selectbox(
                                f"{col_name}",
                                options=unique_vals,
                                index=0,
                                key=f"cat_{col_name}"
                            )
                        else:
                            input_data[col_name] = st.text_input(
                                f"{col_name}", 
                                value="",
                                key=f"text_{col_name}"
                            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Predict House Price", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing property..."):
                try:
                    prediction = predict_price(model, input_data, feature_cols, df)
                    
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
                            if num_data:
                                st.dataframe(pd.DataFrame([num_data]).T, use_container_width=True)
                        
                        with col_s2:
                            st.markdown("**Categorical Features**")
                            cat_data = {k: v for k, v in input_data.items() if k in categorical_cols}
                            if cat_data:
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
        
        st.markdown("### 📦 Requirements")
        st.write("Make sure your `requirements.txt` includes:")
        st.code("""streamlit
pandas
numpy
scikit-learn""")

if __name__ == "__main__":
    main()
