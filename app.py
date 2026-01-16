import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_model():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'house_price_model.pkl' not found. Please upload the model file.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('HousePricePrediction.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file 'HousePricePrediction.csv' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

def prepare_input(user_data):
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

def predict_price(model, user_data):
    input_df = prepare_input(user_data)
    pred_log = model.predict(input_df)
    pred_price = np.expm1(pred_log[0])
    return pred_price

def main():
    st.set_page_config(
        page_title="House Price Predictor",
        page_icon="🏠",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #fdfbfb 0%, #f6d5b8 50%, #ffecd2 100%);
            background-attachment: fixed;
        }
        
        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 3rem 2rem;
        }
        
        .hero-section {
            text-align: center;
            margin-bottom: 4rem;
            padding: 3rem 2rem;
        }
        
        .hero-title {
            font-size: 4.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            letter-spacing: -2px;
        }
        
        .hero-subtitle {
            font-size: 1.5rem;
            color: #666;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            padding: 2.5rem;
            margin-bottom: 2rem;
        }
        
        .glass-card-small {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .section-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: #333;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        }
        
        .result-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 20px;
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.25);
            padding: 3rem;
            text-align: center;
            margin: 2rem 0;
        }
        
        .price-display {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 1.25rem;
            font-weight: 600;
            padding: 1rem 3rem;
            border: none;
            border-radius: 50px;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 2rem;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(102, 126, 234, 0.5);
        }
        
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 0.75rem;
            font-size: 1rem;
            color: #333;
        }
        
        .stSlider > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        label {
            font-weight: 600;
            color: #444;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }
        
        .footer {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            padding: 2rem;
            text-align: center;
            margin-top: 4rem;
            color: #666;
        }
        
        .footer-text {
            font-size: 1rem;
            font-weight: 500;
        }
        
        .footer-links {
            margin-top: 1rem;
            font-size: 0.9rem;
        }
        
        .footer-links a {
            color: #667eea;
            text-decoration: none;
            margin: 0 1rem;
            font-weight: 600;
        }
        
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 700;
            color: #333;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 12px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #444;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    model = load_model()
    df = load_dataset()
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">House Price Predictor</h1>
            <p class="hero-subtitle">AI-powered real estate valuation with advanced machine learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Property Details</h2>', unsafe_allow_html=True)
    
    input_data = {}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=1000,
            max_value=200000,
            value=10000,
            step=100
        )
        input_data['LotArea_log'] = np.log1p(lot_area)
    
    with col2:
        bsmt_sf = st.number_input(
            "Total Basement SF",
            min_value=0,
            max_value=6000,
            value=1000,
            step=50
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
    
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        input_data['MSZoning'] = st.selectbox(
            "Zoning Classification",
            options=df['MSZoning'].dropna().unique().tolist()
        )
    
    with col2:
        input_data['LotConfig'] = st.selectbox(
            "Lot Configuration",
            options=df['LotConfig'].dropna().unique().tolist()
        )
    
    with col3:
        input_data['BldgType'] = st.selectbox(
            "Building Type",
            options=df['BldgType'].dropna().unique().tolist()
        )
    
    with col4:
        input_data['Exterior1st'] = st.selectbox(
            "Exterior Covering",
            options=df['Exterior1st'].dropna().unique().tolist()
        )
    
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overall_cond = st.slider(
            "Overall Condition",
            min_value=1,
            max_value=10,
            value=5
        )
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
            step=10
        )
    
    with col3:
        input_data['Cond_x_MSSubClass'] = overall_cond * ms_subclass
        st.metric(
            "Condition × SubClass",
            f"{input_data['Cond_x_MSSubClass']}"
        )
    
    predict_button = st.button("Predict House Price")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_button:
        with st.spinner("Analyzing property features..."):
            try:
                prediction = predict_price(model, input_data)
                
                st.markdown(f"""
                    <div class="result-card">
                        <h2 style="color: #333; font-weight: 700; margin: 0;">Predicted House Price</h2>
                        <div class="price-display">${prediction:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">${prediction:,.0f}</div>
                            <div class="metric-label">Estimated Value</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    price_range_low = prediction * 0.95
                    price_range_high = prediction * 1.05
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">${price_range_low:,.0f} - ${price_range_high:,.0f}</div>
                            <div class="metric-label">Price Range (±5%)</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">High</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    st.markdown("""
        <div class="footer">
            <div class="footer-text">Built by Sachin</div>
            <div class="footer-links">
                <a href="#" target="_blank">GitHub</a>
                <a href="#" target="_blank">LinkedIn</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
