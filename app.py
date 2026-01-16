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
        st.error("Model file 'house_price_model.pkl' not found.")
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

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-image: url('https://github.com/SachinGetsItDone/house-pricing-/raw/main/house%201.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container {
    max-width: 1400px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

h1 {
    font-size: 4rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.subtitle {
    text-align: center;
    font-size: 1.3rem;
    color: #ffffff;
    margin-bottom: 3rem;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
}

.glass-container {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    padding: 2.5rem;
    margin-bottom: 2rem;
}

.result-glass {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 20px;
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.25);
    padding: 3rem;
    text-align: center;
    margin: 2rem 0;
}

.price-text {
    font-size: 3.5rem;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    margin: 1rem 0;
}

.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid rgba(255, 255, 255, 0.4);
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border: none !important;
    border-radius: 50px !important;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
    width: 100% !important;
    margin-top: 2rem !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.5) !important;
}

.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 12px !important;
    color: #333 !important;
}

label {
    font-weight: 600 !important;
    color: #ffffff !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}

.metric-glass {
    background: rgba(255, 255, 255, 0.3);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.footer-glass {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    padding: 2rem;
    text-align: center;
    margin-top: 3rem;
}

.footer-text {
    color: #666;
    font-size: 1rem;
    font-weight: 500;
}

.footer-links {
    margin-top: 1rem;
}

.footer-links a {
    color: #667eea;
    text-decoration: none;
    margin: 0 1rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

model = load_model()
df = load_dataset()

st.markdown("<h1>House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered real estate valuation with advanced machine learning</p>", unsafe_allow_html=True)

st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Property Details</div>", unsafe_allow_html=True)

input_data = {}

col1, col2, col3, col4 = st.columns(4)

with col1:
    lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=200000, value=10000, step=100)
    input_data['LotArea_log'] = np.log1p(lot_area)

with col2:
    bsmt_sf = st.number_input("Total Basement SF", min_value=0, max_value=6000, value=1000, step=50)
    input_data['TotalBsmtSF_log'] = np.log1p(bsmt_sf)

with col3:
    input_data['YearBuilt'] = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000, step=1)

with col4:
    input_data['YearRemodAdd'] = st.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2000, step=1)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    input_data['MSZoning'] = st.selectbox("Zoning Classification", options=df['MSZoning'].dropna().unique().tolist())

with col2:
    input_data['LotConfig'] = st.selectbox("Lot Configuration", options=df['LotConfig'].dropna().unique().tolist())

with col3:
    input_data['BldgType'] = st.selectbox("Building Type", options=df['BldgType'].dropna().unique().tolist())

with col4:
    input_data['Exterior1st'] = st.selectbox("Exterior Covering", options=df['Exterior1st'].dropna().unique().tolist())

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    overall_cond = st.slider("Overall Condition", min_value=1, max_value=10, value=5)
    if overall_cond <= 4:
        input_data['CondGroup'] = 'Bad'
    elif overall_cond <= 6:
        input_data['CondGroup'] = 'Average'
    else:
        input_data['CondGroup'] = 'Good'

with col2:
    ms_subclass = st.number_input("MS SubClass", min_value=20, max_value=190, value=60, step=10)

with col3:
    input_data['Cond_x_MSSubClass'] = overall_cond * ms_subclass
    st.metric("Condition × SubClass", f"{input_data['Cond_x_MSSubClass']}")

predict_button = st.button("Predict House Price")

st.markdown("</div>", unsafe_allow_html=True)

if predict_button:
    with st.spinner("Analyzing property features..."):
        try:
            prediction = predict_price(model, input_data)
            
            st.markdown(f"""
                <div class='result-glass'>
                    <h2 style='color: #ffffff; font-weight: 700; margin: 0; text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);'>Predicted House Price</h2>
                    <div class='price-text'>${prediction:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            price_range_low = prediction * 0.95
            price_range_high = prediction * 1.05
            
            with col1:
                st.markdown(f"""
                    <div class='metric-glass'>
                        <div style='font-size: 1.5rem; font-weight: 700; color: #ffffff; text-shadow: 0 1px 5px rgba(0, 0, 0, 0.5);'>${prediction:,.0f}</div>
                        <div style='font-size: 0.9rem; color: #ffffff; margin-top: 0.5rem; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);'>ESTIMATED VALUE</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class='metric-glass'>
                        <div style='font-size: 1.5rem; font-weight: 700; color: #ffffff; text-shadow: 0 1px 5px rgba(0, 0, 0, 0.5);'>${price_range_low:,.0f} - ${price_range_high:,.0f}</div>
                        <div style='font-size: 0.9rem; color: #ffffff; margin-top: 0.5rem; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);'>PRICE RANGE (±5%)</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class='metric-glass'>
                        <div style='font-size: 1.5rem; font-weight: 700; color: #ffffff; text-shadow: 0 1px 5px rgba(0, 0, 0, 0.5);'>High</div>
                        <div style='font-size: 0.9rem; color: #ffffff; margin-top: 0.5rem; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);'>CONFIDENCE</div>
                    </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")

st.markdown("""
    <div class='footer-glass'>
        <div class='footer-text'>Built by Hitesh Kumar and Sachin Sharma</div>
        <div class='footer-links'>
            <a href='https://github.com/SachinGetsItDone' target='_blank'>GitHub</a>
            <a href='https://www.linkedin.com/in/sachin-sharma-898504340' target='_blank'>LinkedIn</a>
        </div>
    </div>
""", unsafe_allow_html=True)
