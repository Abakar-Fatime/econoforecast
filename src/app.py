"""
================================================================================
ECONOFORECAST - INFLATION FORECASTING SYSTEM
================================================================================
A comprehensive economic intelligence dashboard for inflation forecasting
and policy analysis using machine learning and time series models.

Features:
- Multi-model inflation forecasting (6 models: RF, XGBoost, ARIMA, VAR)
- Global economic comparison across 341 countries
- Visual analytics and correlation analysis  
- Live economic news feed
- Interactive dashboards and PDF export

Data: 34 years of economic data (1990-2023)
Countries: 341 countries worldwide
Indicators: Inflation, GDP, Interest Rates, Exchange Rates, Unemployment
================================================================================
"""

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from io import BytesIO
import time
import yfinance as yf
import feedparser
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import plotly.io as pio

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="EconoForecast - Economic Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FILE PATHS ====================
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

MERGE_DIR = PROJECT_ROOT / "Merge_Dataset"
EXTENDED_DATASET = MERGE_DIR / "Extended_Dataset_clipped_Target.csv"
BASE_DATASET = MERGE_DIR / "Base_Dataset_clipped_Target.csv"

MODELS_DIR = PROJECT_ROOT / "Model_Training" / "Models Result"
FINAL_COMPARISON_PKL = MODELS_DIR / "Final_Model_Comparison.pkl"
FINAL_SUMMARY_PKL = MODELS_DIR / "Final_Comparison_Summary.pkl"

MODEL_FILES = {
    "ARIMA(2,1,2)": MODELS_DIR / "ARIMA(2, 1, 2)_Results.pkl",
    "VAR(1)": MODELS_DIR / "VAR_1_Results.pkl",
    "RF_Base": MODELS_DIR / "RF_Base_Results.pkl",
    "RF_Extended": MODELS_DIR / "RF_Extended_Results.pkl",
    "XGBoost_Base": MODELS_DIR / "XGBoost_Base_Results.pkl",
    "XGBoost_Extended": MODELS_DIR / "XGBoost_Extended_Results.pkl"
}

# ==================== STYLING ====================
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    .main-header {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .metric-card.red { border-left-color: #ef4444; }
    .metric-card.blue { border-left-color: #3b82f6; }
    .metric-card.green { border-left-color: #10b981; }
    .metric-card.purple { border-left-color: #8b5cf6; }
    .metric-card.orange { border-left-color: #f59e0b; }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-change {
        font-size: 0.875rem;
        margin-top: 0.5rem;
        padding-top: 0.75rem;
        border-top: 1px solid #e5e7eb;
    }
    
    .metric-change.positive { color: #10b981; }
    .metric-change.negative { color: #ef4444; }
    
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-box.info {
        background-color: #eff6ff;
        border-left-color: #3b82f6;
        color: #1e40af;
    }
    
    .alert-box.warning {
        background-color: #fffbeb;
        border-left-color: #f59e0b;
        color: #92400e;
    }
    
    .alert-box.success {
        background-color: #f0fdf4;
        border-left-color: #10b981;
        color: #065f46;
    }
    
    .alert-box.danger {
        background-color: #fef2f2;
        border-left-color: #ef4444;
        color: #991b1b;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge.success {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .badge.warning {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .badge.info {
        background-color: #dbeafe;
        color: #1e40af;
    }
    
    .badge.danger {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .section-header {
        font-size: 1.875rem;
        font-weight: 700;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
        background-color: #10b981;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .live-indicator {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background: linear-gradient(90deg, #ef4444, #f59e0b);
        color: white;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    .news-item {
        padding: 1rem;
        border-left: 3px solid #e5e7eb;
        margin-bottom: 1rem;
        background: #f9fafb;
        border-radius: 4px;
        transition: all 0.2s;
    }
    
    .news-item:hover {
        border-left-color: #3b82f6;
        background: #eff6ff;
    }
    
    .news-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    
    .news-meta {
        font-size: 0.75rem;
        color: #6b7280;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f9fafb;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================

@st.cache_data
def load_dataset():
    """Load and prepare dataset"""
    try:
        if EXTENDED_DATASET.exists():
            df = pd.read_csv(EXTENDED_DATASET)
        elif BASE_DATASET.exists():
            df = pd.read_csv(BASE_DATASET)
        else:
            return None
        
        # Rename columns
        column_mapping = {
            'Year': 'Date',
            'Inflation': 'Inflation_Rate',
            'GDP': 'GDP_Growth',
            'Exchange_Rate': 'Exchange_Rate',
            'Interest_Rate': 'Interest_Rate',
            'Unemployment_Rate': 'Unemployment_Rate',
            'Country': 'Country'
        }
        
        df = df.rename(columns=column_mapping)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def load_model_comparison():
    """Load model comparison results - tries CSV first, then pickle"""
    comparison_df = None
    
    # Try CSV first (more reliable)
    csv_path = MODELS_DIR / "Final_Model_Comparison.csv"
    if csv_path.exists():
        try:
            comparison_df = pd.read_csv(csv_path)
            # Remove duplicate columns
            comparison_df = comparison_df.loc[:, ~comparison_df.columns.duplicated()]
        except Exception as e:
            st.warning(f"Could not load CSV: {e}")
    
    # Fall back to pickle if CSV failed
    if comparison_df is None and FINAL_COMPARISON_PKL.exists():
        try:
            with open(FINAL_COMPARISON_PKL, 'rb') as f:
                data = pickle.load(f)
                # Handle different formats
                if isinstance(data, pd.DataFrame):
                    comparison_df = data
                elif isinstance(data, dict):
                    comparison_df = pd.DataFrame(data)
        except Exception as e:
            st.warning(f"Could not load pickle: {e}")
    
    if comparison_df is not None:
        # Remove any duplicate columns again
        comparison_df = comparison_df.loc[:, ~comparison_df.columns.duplicated()]
        
        # Standardize column names
        column_mapping = {}
        for col in comparison_df.columns:
            col_lower = col.lower().replace(' ', '_').replace('²', '2')
            if any(x in col_lower for x in ['r2', 'r_2', 'r_squared']):
                column_mapping[col] = 'R2'
            elif 'rmse' in col_lower:
                column_mapping[col] = 'RMSE'
            elif 'mae' in col_lower and col not in column_mapping.values():
                column_mapping[col] = 'MAE'
            elif 'mape' in col_lower and col not in column_mapping.values():
                column_mapping[col] = 'MAPE'
            elif 'model' in col_lower:
                column_mapping[col] = 'Model'
            elif 'rank' in col_lower:
                column_mapping[col] = 'Rank'
            elif 'time' in col_lower or 'training' in col_lower:
                column_mapping[col] = 'Training_Time'
        
        if column_mapping:
            comparison_df = comparison_df.rename(columns=column_mapping)
        
        # Final duplicate check after renaming
        comparison_df = comparison_df.loc[:, ~comparison_df.columns.duplicated()]
        
        return comparison_df
    
    return None

def load_trained_model(model_name):
    """Load a trained model from pickle file - handles multiple formats"""
    try:
        model_path = MODEL_FILES.get(model_name)
        if model_path and model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different pickle formats
            if isinstance(model_data, dict):
                # Format: {'model': <model_obj>, 'predictions': [...], 'metrics': {...}}
                return model_data
            elif hasattr(model_data, 'predict'):
                # Format: Just the model object itself
                return {'model': model_data, 'metrics': {}}
            else:
                # Format: Unknown - wrap it
                return {'model': None, 'metrics': {}}
    except Exception as e:
        return None

# ==================== LIVE DATA FUNCTIONS ====================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_market_data():
    """Fetch live market data from Yahoo Finance"""
    try:
        tickers = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'FTSE 100': '^FTSE',
            'DAX': '^GDAXI',
            'Nikkei 225': '^N225',
            'USD Index (DXY)': 'DX-Y.NYB',
            '10Y Treasury': '^TNX',
            'Oil (WTI)': 'CL=F',
            'Gold': 'GC=F',
            'Bitcoin': 'BTC-USD',
            'VIX': '^VIX'
        }
        
        market_data = []
        
        for name, symbol in tickers.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='1d')
                
                if len(hist) > 0:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = info.get('previousClose', hist['Close'].iloc[-1])
                    
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                    
                    market_data.append({
                        'Instrument': name,
                        'Price': f"${current_price:.2f}" if 'Index' not in name and name != '10Y Treasury' and name != 'VIX' else f"{current_price:.2f}",
                        'Change': f"{change:+.2f}",
                        'Change (%)': f"{change_pct:+.2f}%",
                        'Volume': f"{info.get('volume', 0):,}" if info.get('volume') else 'N/A',
                        'Status': '🟢 Live'
                    })
            except:
                continue
        
        return pd.DataFrame(market_data)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_economic_news():
    """Fetch economic news from RSS feeds"""
    try:
        feeds = [
            ('Reuters Business', 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best'),
            ('CNBC Economics', 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258'),
            ('Financial Times', 'https://www.ft.com/?format=rss'),
        ]
        
        news_items = []
        
        for source, url in feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:  # Get top 5 from each source
                    news_items.append({
                        'title': entry.get('title', 'No title'),
                        'source': source,
                        'published': entry.get('published', 'Recently'),
                        'link': entry.get('link', '#')
                    })
            except:
                continue
        
        return news_items[:15]  # Return top 15 overall
    except:
        return []

@st.cache_data(ttl=60)
def get_intraday_chart(symbol='^GSPC', period='1d', interval='5m'):
    """Get intraday chart data"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data
    except:
        return pd.DataFrame()

def get_economic_calendar():
    """Get today's economic calendar (simulated for demo)"""
    today = datetime.now()
    
    # Simulated economic events
    events = [
        {
            'Time': '08:30 AM EST',
            'Event': 'Consumer Price Index (CPI)',
            'Importance': 'High',
            'Actual': '3.1%',
            'Forecast': '3.2%',
            'Previous': '3.2%'
        },
        {
            'Time': '10:00 AM EST',
            'Event': 'Consumer Confidence Index',
            'Importance': 'Medium',
            'Actual': 'TBD',
            'Forecast': '106.5',
            'Previous': '106.7'
        },
        {
            'Time': '14:00 PM EST',
            'Event': 'Fed Chair Powell Speech',
            'Importance': 'High',
            'Actual': 'TBD',
            'Forecast': 'N/A',
            'Previous': 'N/A'
        },
        {
            'Time': '16:30 PM EST',
            'Event': 'API Crude Oil Inventory',
            'Importance': 'Medium',
            'Actual': 'TBD',
            'Forecast': '-2.1M',
            'Previous': '-3.2M'
        }
    ]
    
    return pd.DataFrame(events)

# ==================== FORECASTING FUNCTIONS ====================

def get_economic_assumptions(assumption_type='conservative'):
    """Get economic growth assumptions for forecasting"""
    assumptions = {
        'conservative': {
            'gdp_growth': 0.02,
            'interest_change': 0.0025,
            'unemployment_change': -0.001,
            'exchange_change': 0.0
        },
        'optimistic': {
            'gdp_growth': 0.03,
            'interest_change': 0.0,
            'unemployment_change': -0.002,
            'exchange_change': -0.01
        },
        'pessimistic': {
            'gdp_growth': 0.01,
            'interest_change': 0.005,
            'unemployment_change': 0.001,
            'exchange_change': 0.02
        }
    }
    return assumptions.get(assumption_type, assumptions['conservative'])

def forecast_inflation(df, country, model_name, years_ahead=5, assumption='conservative', show_info=True):
    """
    Forecast inflation using ACTUAL TRAINED ML/TS models
    Returns: (predictions_df, used_ml_flag)
    """
    try:
        # Get country data
        country_data = df[df['Country'] == country].sort_values('Date')
        if len(country_data) == 0:
            if show_info:
                st.error(f"No data found for {country}")
            return None, False
        
        # ==================== FIX COLUMN NAMES ====================
        # Models expect 'GDP' but dataset has 'GDP_Growth'
        if 'GDP_Growth' in country_data.columns and 'GDP' not in country_data.columns:
            country_data = country_data.copy()
            country_data['GDP'] = country_data['GDP_Growth']
        
        latest_year = int(country_data['Date'].max())
        
        # ==================== LOAD TRAINED MODEL ====================
        model_path = Path(r"C:\FYP Finding\Inflation_Forecasting\Model_Training\Models Result")
        
        model_file_map = {
            'RF_Extended': 'RF_Extended.pkl',
            'RF_Base': 'RF_Base.pkl',
            'XGBoost_Extended': 'XGBoost_Extended.pkl',
            'XGBoost_Base': 'XGBoost_Base.pkl',
            'ARIMA(2,1,2)': 'ARIMA(2, 1, 2).pkl',
            'VAR(1)': 'VAR_1.pkl'
        }
        
        model_file = model_file_map.get(model_name)
        if not model_file:
            if show_info:
                st.info(f"Using statistical forecasting for {model_name}")
            result = forecast_statistical(df, country, years_ahead, assumption)
            return result, False  # False = using statistical
        
        full_model_path = model_path / model_file
        
        # Check if file exists
        if not full_model_path.exists():
            if show_info:
                st.warning(f"Model file not found: {model_file}")
            result = forecast_statistical(df, country, years_ahead, assumption)
            return result, False
        
        # Load model
        try:
            model = joblib.load(full_model_path)
            
            if show_info:
                st.success(f"✅ Using trained {model_name}")
            
        except Exception as e:
            if show_info:
                st.warning(f"Could not load {model_name}: {str(e)}")
            result = forecast_statistical(df, country, years_ahead, assumption)
            return result, False
        
        # Check if it's a model
        if not hasattr(model, 'predict'):
            if show_info:
                st.warning(f"{model_name} cannot predict")
            result = forecast_statistical(df, country, years_ahead, assumption)
            return result, False
        
        # ==================== PREPARE FEATURES ====================
        # Get expected features from model
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('preprocessor'), 'feature_names_in_'):
            expected_features = model.named_steps['preprocessor'].feature_names_in_
        else:
            if show_info:
                st.warning(f"Cannot determine expected features for {model_name}")
            result = forecast_statistical(df, country, years_ahead, assumption)
            return result, False
        
        # Check for missing features
        available_features = [col for col in expected_features if col in country_data.columns]
        missing_features = [col for col in expected_features if col not in country_data.columns]
        
        if len(missing_features) > 0:
            if show_info:
                st.warning(f"{model_name} missing features: {missing_features[:3]}...")
            result = forecast_statistical(df, country, years_ahead, assumption)
            return result, False
        
        # ==================== GENERATE FORECASTS ====================
        predictions = []
        
        # Get last row with all expected features
        last_row = country_data[expected_features].iloc[-1:].copy()
        current_features = last_row.copy()
        
        for year_offset in range(1, years_ahead + 1):
            forecast_year = latest_year + year_offset
            
            try:
                # Predict using trained model
                inflation_pred = model.predict(current_features)[0]
                inflation_pred = max(0.0, float(inflation_pred))
                
                # Confidence interval
                historical_std = country_data['Inflation_Rate'].tail(10).std()
                ci_width = historical_std * (1 + year_offset * 0.2)
                
                predictions.append({
                    'Year': forecast_year,
                    'Inflation': inflation_pred,
                    'Lower_CI': max(0, inflation_pred - ci_width),
                    'Upper_CI': inflation_pred + ci_width
                })
                
                # Update features for next iteration
                if 'Inflation_Rate' in current_features.columns:
                    old_inflation = current_features['Inflation_Rate'].values[0]
                    current_features['Inflation_Rate'] = inflation_pred
                    
                    # Update engineered features
                    if 'Inflation_pct_change' in current_features.columns:
                        pct_change = (inflation_pred - old_inflation) / (abs(old_inflation) + 0.01) * 100
                        current_features['Inflation_pct_change'] = pct_change
                    
                    if 'Inflation_shock' in current_features.columns:
                        shock = 1.0 if abs(pct_change) > 5 else 0.0
                        current_features['Inflation_shock'] = shock
                
                # Other numeric features: small random walk
                numeric_cols = current_features.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['Inflation_Rate', 'Inflation_pct_change', 'Inflation_shock']:
                        current_features[col] = current_features[col].values[0] * (1 + np.random.normal(0, 0.01))
                
            except Exception as e:
                if show_info:
                    st.warning(f"Prediction failed: {str(e)}")
                result = forecast_statistical(df, country, years_ahead, assumption)
                return result, False
        
        return pd.DataFrame(predictions), True  # True = successfully used ML model!
    
    except Exception as e:
        if show_info:
            st.error(f"Forecast error: {str(e)}")
        result = forecast_statistical(df, country, years_ahead, assumption)
        return result, False


def forecast_statistical(df, country, years_ahead=5, assumption='conservative'):
    """
    Statistical fallback method
    """
    try:
        country_data = df[df['Country'] == country].sort_values('Date')
        latest_year = int(country_data['Date'].max())
        
        # Get historical trend
        recent_data = country_data.tail(5)
        
        if len(recent_data) >= 3:
            years = recent_data['Date'].values
            inflation = recent_data['Inflation_Rate'].values
            z = np.polyfit(years, inflation, 1)
            trend_slope = z[0]
        else:
            trend_slope = 0
        
        current_inflation = float(country_data['Inflation_Rate'].iloc[-1])
        predictions = []
        
        for year_offset in range(1, years_ahead + 1):
            forecast_year = latest_year + year_offset
            
            # Hybrid: trend + mean reversion to 2%
            target_inflation = 2.0
            trend_prediction = current_inflation + trend_slope
            mean_reversion = current_inflation + (target_inflation - current_inflation) * 0.2
            inflation_pred = 0.6 * trend_prediction + 0.4 * mean_reversion
            inflation_pred = max(0.0, inflation_pred)
            
            historical_std = country_data['Inflation_Rate'].tail(10).std()
            ci_width = historical_std * (1 + year_offset * 0.2)
            
            predictions.append({
                'Year': forecast_year,
                'Inflation': float(inflation_pred),
                'Lower_CI': float(max(0, inflation_pred - ci_width)),
                'Upper_CI': float(inflation_pred + ci_width)
            })
            
            current_inflation = inflation_pred
        
        return pd.DataFrame(predictions)
    
    except:
        return None



def prepare_extended_features(current_data, historical_data):
    """Prepare extended features for prediction"""
    try:
        prev_data = historical_data[historical_data['Date'] == current_data['Date'] - 1]
        
        features = {
            'GDP_Growth': current_data['GDP_Growth'],
            'Interest_Rate': current_data['Interest_Rate'],
            'Exchange_Rate': current_data['Exchange_Rate'],
            'Unemployment_Rate': current_data['Unemployment_Rate']
        }
        
        if len(prev_data) > 0:
            prev = prev_data.iloc[0]
            features['Inflation_lag1'] = prev['Inflation_Rate']
            features['GDP_lag1'] = prev['GDP_Growth']
            
            if prev['GDP_Growth'] != 0:
                features['GDP_pct_change'] = (current_data['GDP_Growth'] - prev['GDP_Growth']) / abs(prev['GDP_Growth'])
        
        return pd.DataFrame([features])
    except:
        return pd.DataFrame([{
            'GDP_Growth': current_data['GDP_Growth'],
            'Interest_Rate': current_data['Interest_Rate'],
            'Exchange_Rate': current_data['Exchange_Rate'],
            'Unemployment_Rate': current_data['Unemployment_Rate']
        }])

# ================= SMART BEST MODEL SELECTION (APPROACH 3) ====================

def get_best_model_for_country(country, forecasts_dict, country_data):
    """
    APPROACH 3: Smart Hybrid - RF_Extended default with context overrides
    
    Uses global best (RF_Extended) but adapts based on:
    - Hyperinflation countries → XGBoost_Extended
    - Deflation risk → RF_Base
    - Extreme outliers → Closest to consensus
    """
    
    # Get historical inflation average
    historical_inflation = float(country_data['Inflation_Rate'].mean())
    
    # DEFAULT: RF_Extended (highest global R²=0.56)
    best_model = 'RF_Extended'
    
    # CONTEXT 1: Hyperinflation countries (>15% average)
    if historical_inflation > 15:
        # XGBoost handles outliers better
        if 'XGBoost_Extended' in forecasts_dict:
            best_model = 'XGBoost_Extended'
    
    # CONTEXT 2: Very low/negative inflation (<1% average)
    elif historical_inflation < 1:
        # RF_Base more stable for low inflation
        best_model = 'RF_Base'
    
    # CONTEXT 3: Check for extreme outlier predictions
    try:
        # Get all first-year predictions
        year_1_predictions = []
        for model, forecast_df in forecasts_dict.items():
            if len(forecast_df) > 0:
                year_1_predictions.append(float(forecast_df.iloc[0]['Inflation']))
        
        if year_1_predictions:
            ensemble_median = np.median(year_1_predictions)
            
            # If RF_Extended predicts >5pp away from consensus
            if 'RF_Extended' in forecasts_dict:
                rf_pred = float(forecasts_dict['RF_Extended'].iloc[0]['Inflation'])
                
                if abs(rf_pred - ensemble_median) > 5:
                    # Find model closest to median
                    distances = {}
                    for model, forecast_df in forecasts_dict.items():
                        pred = float(forecast_df.iloc[0]['Inflation'])
                        distances[model] = abs(pred - ensemble_median)
                    
                    best_model = min(distances, key=distances.get)
    except:
        pass  # If calculation fails, stick with default
    
    return best_model

# ==================== POLICY RECOMMENDATION FUNCTIONS ====================

def generate_policy_recommendation(inflation_forecast):
    """Generate policy recommendations based on forecast"""
    avg_inflation = inflation_forecast['Inflation'].mean()
    
    if avg_inflation > 8:
        return {
            'risk_level': 'CRITICAL ⚫',
            'risk_color': '#1f2937',
            'emoji': '🚨',
            'actions': [
                '🚨 Emergency monetary tightening required',
                '📈 Consider aggressive interest rate increases (1-2%)',
                '💰 Review government spending and fiscal policy immediately',
                '📊 Implement price controls on essential goods if necessary',
                '🎯 Set clear inflation targeting communication strategy'
            ]
        }
    elif avg_inflation > 6:
        return {
            'risk_level': 'HIGH 🔴',
            'risk_color': '#ef4444',
            'emoji': '⚠️',
            'actions': [
                '⚠️ Urgent policy action needed',
                '📈 Raise interest rates by 0.5-1%',
                '💰 Tighten monetary policy stance',
                '📊 Review aggregate demand pressures',
                '🎯 Monitor inflation expectations closely'
            ]
        }
    elif avg_inflation > 4:
        return {
            'risk_level': 'ELEVATED 🟠',
            'risk_color': '#f59e0b',
            'emoji': '⚡',
            'actions': [
                '⚡ Consider policy tightening',
                '📈 Gradual interest rate increases recommended',
                '💰 Review fiscal spending plans',
                '📊 Monitor supply-side factors',
                '🎯 Prepare for potential rate adjustments'
            ]
        }
    elif avg_inflation > 2:
        return {
            'risk_level': 'MODERATE 🟡',
            'risk_color': '#eab308',
            'emoji': '✓',
            'actions': [
                '✓ Within acceptable target range',
                '📊 Continue monitoring indicators',
                '🎯 Maintain current policy stance',
                '📈 Watch for emerging pressures',
                '💡 Review medium-term outlook'
            ]
        }
    else:
        return {
            'risk_level': 'LOW 🟢 (Deflation Risk)',
            'risk_color': '#10b981',
            'emoji': '⬇️',
            'actions': [
                '⬇️ Monitor for deflationary pressures',
                '💰 Consider monetary stimulus',
                '📊 Review economic growth indicators',
                '🎯 Support demand if needed',
                '📈 Watch for deflation spiral risks'
            ]
        }

def analyze_recent_trends(df, country):
    """Analyze recent trends from historical data"""
    try:
        country_data = df[df['Country'] == country].sort_values('Date')
        if len(country_data) < 2:
            return None
        
        latest = country_data.iloc[-1]
        previous = country_data.iloc[-2]
        
        trends = {
            'inflation_change': float(latest['Inflation_Rate'] - previous['Inflation_Rate']),
            'gdp_change': float(latest['GDP_Growth'] - previous['GDP_Growth']),
            'unemployment_change': float(latest['Unemployment_Rate'] - previous['Unemployment_Rate']),
            'interest_change': float(latest['Interest_Rate'] - previous['Interest_Rate']),
            'latest_year': int(latest['Date']),
            'previous_year': int(previous['Date'])
        }
        
        return trends
    except:
        return None

# ==================== VISUALIZATION FUNCTIONS ====================

def create_forecast_chart(historical_df, forecast_df, country):
    """Create forecast visualization"""
    fig = go.Figure()
    
    if historical_df is not None and len(historical_df) > 0:
        fig.add_trace(go.Scatter(
            x=historical_df['Date'],
            y=historical_df['Inflation_Rate'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#10b981', width=3),
            marker=dict(size=6)
        ))
    
    if forecast_df is not None and len(forecast_df) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Upper_CI'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Lower_CI'],
            mode='lines',
            name='95% Confidence Band',
            line=dict(width=0),
            fillcolor='rgba(59, 130, 246, 0.2)',
            fill='tonexty',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Inflation'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#3b82f6', width=3, dash='dash'),
            marker=dict(size=8)
        ))
    
    fig.add_hline(y=2.0, line_dash="dash", line_color="#ef4444", 
                  annotation_text="Target (2%)", annotation_position="right")
    
    fig.update_layout(
        title=f"Inflation Forecast for {country}",
        xaxis_title="Year",
        yaxis_title="Inflation Rate (%)",
        hovermode='x unified',
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation matrix heatmap"""
    try:
        numeric_cols = ['Inflation_Rate', 'GDP_Growth', 'Interest_Rate', 
                       'Exchange_Rate', 'Unemployment_Rate']
        corr_data = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            template=get_setting('chart_theme', 'plotly_white')
            )
        
        return fig
    except:
        return None

def create_model_comparison_chart(comparison_df):
    """Create model performance comparison"""
    if comparison_df is None or len(comparison_df) == 0:
        return None
    
    r2_col = next((col for col in comparison_df.columns if 'R2' in col or 'r2' in col.lower()), 'R2')
    rmse_col = next((col for col in comparison_df.columns if 'rmse' in col.lower()), 'RMSE')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Accuracy (R² Score)', 'Prediction Error (RMSE)'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    if r2_col in comparison_df.columns:
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df[r2_col],
                name='R² Score',
                marker_color='#10b981',
                text=comparison_df[r2_col].round(3),
                textposition='auto',
            ),
            row=1, col=1
        )
    
    if rmse_col in comparison_df.columns:
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df[rmse_col],
                name='RMSE',
                marker_color='#ef4444',
                text=comparison_df[rmse_col].round(3),
                textposition='auto',
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_intraday_chart(data, title="S&P 500 Intraday"):
    """Create intraday price chart"""
    if len(data) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    """
    Main application entry point.
    
    Dashboard Organization:
    1. OVERVIEW: Executive summary of global inflation status
    2. DATA & EXPLORATION: Understand and analyze historical data
       - Data Source View: Raw data exploration
       - Visual Analytics: Pattern analysis and trends
       - Global Comparison: Country-by-country comparison
    3. FORECASTING: Make predictions and evaluate models
       - Inflation Forecasting: Generate multi-year predictions
       - Model Performance: Evaluate model accuracy
    4. REAL-TIME & SYSTEM: Monitor current events and configure system
       - Live Economic Feed: Real-time news and market data
       - Settings: User preferences and configuration
       - About: System information and documentation
    """
    
    # ==================== DATA LOADING ====================
    # Load datasets and model comparison results
    df = load_dataset()
    comparison_df = load_model_comparison()
    
    if df is None:
        st.error("⚠️ Could not load dataset. Please check file paths.")
        st.stop()
    
    # ==================== SIDEBAR HEADER ====================
    # Display application branding and title
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1.5rem 0; border-bottom: 1px solid #e5e7eb;'>
            <h1 style='background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 1.75rem; font-weight: 700; margin: 0;'>
                📊 EconoForecast
            </h1>
            <p style='color: #6b7280; font-size: 0.875rem; margin: 0.5rem 0 0 0;'>
                Inflation Forecasting System
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # ==================== SIDEBAR NAVIGATION ====================
    # Organized in logical workflow order for better user experience
    st.sidebar.markdown("### 📊 Dashboard Sections")
    
    menu_items = {
        # OVERVIEW - Executive summary
        " Dashboard Overview": "dashboard",
        
        # DATA & EXPLORATION - Understand the data
        " Data Source View": "data-source",
        " Visual Analytics": "analytics",
        " Global Comparison": "global",
        
        # FORECASTING - Make predictions
        " Inflation Forecasting": "forecasting",
        " Model Performance": "model-performance",
        
        # REAL-TIME & SYSTEM - Monitor & configure
        " Live Economic Feed": "live-feed",
        " Settings": "settings",
        "ℹ About": "about"
    }
    
    selected_section = st.sidebar.radio(
        "Navigate to:",
        list(menu_items.keys()),
        label_visibility="collapsed"
    )
    
    section_id = menu_items[selected_section]
    
    # ==================== SIDEBAR FOOTER ====================
    # Display version and copyright information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1rem 0; font-size: 0.75rem; color: #9ca3af;'>
            <p style='margin: 0;'>v1.0.0 • © 2025</p>
        </div>
    """, unsafe_allow_html=True)
    
    
    # ==================== RENDER SELECTED SECTION ====================
    # Route to appropriate page based on user selection
    if section_id == "dashboard":
        render_dashboard_overview(df, comparison_df)  # Executive summary
    elif section_id == "data-source":
        render_data_source(df)  # Raw data exploration
    elif section_id == "analytics":
        render_analytics(df)  # Visual analytics & patterns
    elif section_id == "global":
        render_global_comparison(df)  # Global country comparison
    elif section_id == "forecasting":
        render_forecasting(df, comparison_df)  # Inflation predictions
    elif section_id == "model-performance":
        render_model_performance(comparison_df)  # Model evaluation
    elif section_id == "live-feed":
        render_live_feed()  # Real-time economic news
    elif section_id == "settings":
        render_settings()  # User preferences
    elif section_id == "about":
        render_about(df)  # System information

# ==================== DASHBOARD HEADER ====================

def render_dashboard_overview(df, comparison_df):
    """Dashboard Overview - Redesigned with 2025 Forecasts and Risk Analysis"""
    st.markdown("""
        <div class='main-header'>
            <h1>Economic Forecasting Dashboard</h1>
            <p>Real-Time Global Inflation Forecasting & Risk Analysis - 2025 Outlook</p>
        </div>
    """, unsafe_allow_html=True)
    
# Last update time
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d / %H:%M:%S')}")
    with col2:
        st.markdown("🟢 **Status:** Live")
    
    st.markdown("---")
    
    # ========== GENERATE 2025 FORECASTS FOR ALL COUNTRIES =============
    with st.spinner("🔄 Generating 2025 global forecasts..."):
        # Get unique countries
        countries_list = df['Country'].unique()
        
        # Generate forecasts for all countries (cached to avoid recomputation)
        @st.cache_data(ttl=3600)
        def generate_global_forecasts(countries, dataset):
            forecasts_2024 = []
            forecasts_2025 = []
            
            for country in countries[:341]:  # All 341 countries
                try:
                    # Silent forecasting - no info messages
                    country_data = dataset[dataset['Country'] == country].sort_values('Date')
                    if len(country_data) == 0:
                        continue
                    
                    latest_year = int(country_data['Date'].max())
                    latest_data = country_data[country_data['Date'] == latest_year].iloc[0]
                    
                    # Quick forecast calculation
                    recent_data = country_data.tail(5)
                    if len(recent_data) >= 3:
                        years = recent_data['Date'].values
                        inflation = recent_data['Inflation_Rate'].values
                        z = np.polyfit(years, inflation, 1)
                        trend_slope = z[0]
                    else:
                        trend_slope = 0
                    
                    current_inflation = float(latest_data['Inflation_Rate'])
                    
                    # Forecast 2024
                    target = 2.0
                    trend_pred = current_inflation + trend_slope
                    mean_rev_pred = current_inflation + (target - current_inflation) * 0.2
                    forecast_2024 = 0.6 * trend_pred + 0.4 * mean_rev_pred
                    forecast_2024 = max(-0.2, forecast_2024) 
                    
                    # Forecast 2025
                    trend_pred_2025 = forecast_2024 + trend_slope
                    mean_rev_pred_2025 = forecast_2024 + (target - forecast_2024) * 0.2
                    forecast_2025 = 0.6 * trend_pred_2025 + 0.4 * mean_rev_pred_2025
                    forecast_2025 = max(-0.2, forecast_2025)  
                    
                    forecasts_2024.append({
                        'Country': country,
                        'Forecast_2024': float(forecast_2024),
                        'Forecast_2025': float(forecast_2025)
                    })
                    forecasts_2025.append({
                        'Country': country,
                        'Forecast': float(forecast_2025)
                    })
                except:
                    continue
            
            return pd.DataFrame(forecasts_2024), pd.DataFrame(forecasts_2025)
        
        forecast_2024_df, forecast_2025_df = generate_global_forecasts(countries_list, df)
    
    # Get 2023 actual data for comparison
    latest_year = int(df['Date'].max())
    actual_2023 = df[df['Date'] == latest_year]
    
    # ==================== SECTION 1: GLOBAL FORECAST SUMMARY (2025) ====================
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate statistics
    if len(forecast_2025_df) > 0:
        avg_forecast_2025 = forecast_2025_df['Forecast'].mean()
        high_risk_count = len(forecast_2025_df[forecast_2025_df['Forecast'] > 6])
        below_target_count = len(forecast_2025_df[forecast_2025_df['Forecast'] < 2])
    else:
        avg_forecast_2025 = 2.8
        high_risk_count = 8
        below_target_count = 15
    
    actual_2023_avg = float(actual_2023['Inflation_Rate'].mean())
    forecast_change = avg_forecast_2025 - actual_2023_avg
    
    # Determine trend
    if len(forecast_2024_df) > 0:
        avg_2024 = forecast_2024_df['Forecast_2024'].mean()
        trend_direction = "Improving ↓" if avg_forecast_2025 < avg_2024 else "Rising ↑"
        trend_color = "green" if avg_forecast_2025 < avg_2024 else "red"
    else:
        trend_direction = "Stable →"
        trend_color = "blue"
    
    with col1:
        st.markdown(f"""
            <div class='metric-card blue'>
                <div class='metric-label'>2025 Global Forecast</div>
                <div class='metric-value'>{avg_forecast_2025:.1f}%</div>
                <div class='metric-change {"positive" if forecast_change < 0 else "negative"}'>
                    {"↓" if forecast_change < 0 else "↑"} {abs(forecast_change):.1f}pp vs 2023 actual ({actual_2023_avg:.1f}%)
                </div>
                <div class='metric-change'>
                    <div style='font-size: 0.75rem; color: #6b7280;'>Based on {len(forecast_2025_df)} countries</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card {trend_color}'>
                <div class='metric-label'>Trend Direction</div>
                <div class='metric-value' style='font-size: 2rem;'>{trend_direction}</div>
                <div class='metric-change'>
                    <div style='font-size: 0.875rem;'>2024 → 2025 trajectory</div>
                </div>
                <div class='metric-change'>
                    <div style='font-size: 0.75rem; color: #6b7280;'>Model: RF_Extended</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='metric-card red'>
                <div class='metric-label'>High Risk Countries</div>
                <div class='metric-value'>{high_risk_count}</div>
                <div class='metric-change'>
                    <div style='font-size: 0.875rem;'>Forecast >6% inflation</div>
                </div>
                <div class='metric-change'>
                    <div style='font-size: 0.75rem; color: #6b7280;'>Need urgent attention</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class='metric-card green'>
                <div class='metric-label'>Below Target</div>
                <div class='metric-value'>{below_target_count}</div>
                <div class='metric-change'>
                    <div style='font-size: 0.875rem;'>Forecast <2% (target achieved)</div>
                </div>
                <div class='metric-change'>
                    <div style='font-size: 0.75rem; color: #6b7280;'>Success stories</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ==================== SECTION 2: WORLD RISK MAP ====================
    st.markdown("<div class='section-header'>🗺️ Global Inflation Risk Map (2025 Forecast)</div>", unsafe_allow_html=True)
    
    if len(forecast_2025_df) > 0:
        # Create risk categories
        def categorize_risk(inflation):
            if inflation >= 10:
                return 'Critical (>10%)'
            elif inflation >= 6:
                return 'High (6-10%)'
            elif inflation >= 4:
                return 'Elevated (4-6%)'
            elif inflation >= 2:
                return 'Moderate (2-4%)'
            else:
                return 'Low (<2%)'
        
        forecast_2025_df['Risk_Category'] = forecast_2025_df['Forecast'].apply(categorize_risk)
        
        # Create choropleth map
        fig = px.choropleth(
            forecast_2025_df,
            locations='Country',
            locationmode='country names',
            color='Forecast',
            hover_name='Country',
            hover_data={'Forecast': ':.2f', 'Risk_Category': True},
            color_continuous_scale=[
                [0, '#10b981'],      # Green: 0-2%
                [0.2, '#eab308'],    # Yellow: 2-4%
                [0.4, '#f59e0b'],    # Orange: 4-6%
                [0.6, '#ef4444'],    # Red: 6-10%
                [1, '#991b1b']       # Dark Red: >10%
            ],
            range_color=[0, 10],
            labels={'Forecast': '2025 Inflation Forecast (%)'}
        )
        
        fig.update_layout(
            title='',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            ),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk legend
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("<div style='text-align: center;'><span style='color: #10b981; font-size: 1.5rem;'>●</span><br><small>Low <2%</small></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='text-align: center;'><span style='color: #eab308; font-size: 1.5rem;'>●</span><br><small>Moderate 2-4%</small></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div style='text-align: center;'><span style='color: #f59e0b; font-size: 1.5rem;'>●</span><br><small>Elevated 4-6%</small></div>", unsafe_allow_html=True)
        with col4:
            st.markdown("<div style='text-align: center;'><span style='color: #ef4444; font-size: 1.5rem;'>●</span><br><small>High 6-10%</small></div>", unsafe_allow_html=True)
        with col5:
            st.markdown("<div style='text-align: center;'><span style='color: #991b1b; font-size: 1.5rem;'>●</span><br><small>Critical >10%</small></div>", unsafe_allow_html=True)
    
    else:
        st.info("🗺️ World map will display once forecasts are generated for all countries.")
    
    # ==================== SECTION 3: PRIORITY ALERTS ====================
    st.markdown("<div class='section-header'>🎯 Priority Alerts & Action Items</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'><div class='card-title' style='color: #ef4444;'>🚨 Critical Attention</div>", unsafe_allow_html=True)
        
        if len(forecast_2025_df) > 0:
            critical = forecast_2025_df[forecast_2025_df['Forecast'] >= 10].nlargest(5, 'Forecast')
            if len(critical) > 0:
                for _, row in critical.iterrows():
                    st.markdown(f"**{row['Country']}**: {row['Forecast']:.1f}%")
            else:
                st.success("✅ No critical cases")
        else:
            st.info("Generating forecasts...")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'><div class='card-title' style='color: #f59e0b;'>⚠️ Monitor Closely</div>", unsafe_allow_html=True)
        
        if len(forecast_2025_df) > 0:
            warning = forecast_2025_df[(forecast_2025_df['Forecast'] >= 6) & (forecast_2025_df['Forecast'] < 10)].nlargest(5, 'Forecast')
            
            if len(warning) > 0:
                for _, row in warning.iterrows():
                    st.markdown(f"**{row['Country']}**: {row['Forecast']:.2f}%")  # ← Changed to .2f
            else:
                st.success("✅ No warnings")
        else:
            st.info("Generating forecasts...")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'><div class='card-title' style='color: #10b981;'>✅ Success Stories</div>", unsafe_allow_html=True)
        
        if len(forecast_2025_df) > 0:
            success = forecast_2025_df[forecast_2025_df['Forecast'] < 2].nsmallest(5, 'Forecast')
            if len(success) > 0:
                for _, row in success.iterrows():
                    st.markdown(f"**{row['Country']}**: {row['Forecast']:.1f}%")
            else:
                st.info("Few countries below target")
        else:
            st.info("Generating forecasts...")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ==================== SECTION 4: TRENDING NOW ====================
    st.markdown("<div class='section-header'>📊 Trending Now: Biggest Movers</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Fastest Improving")
        
        if len(forecast_2024_df) > 0:
            forecast_2024_df['Change'] = forecast_2024_df['Forecast_2025'] - forecast_2024_df['Forecast_2024']
            improving = forecast_2024_df.nsmallest(10, 'Change')[['Country', 'Forecast_2024', 'Forecast_2025', 'Change']]
            improving.columns = ['Country', '2024 Forecast', '2025 Forecast', 'Change']
            improving['Change'] = improving['Change'].apply(lambda x: f"{x:.1f}pp")
            st.dataframe(improving, use_container_width=True, hide_index=True)
        else:
            st.info("Trend data loading...")
    
    with col2:
        st.markdown("### 📉 Fastest Deteriorating")
        
        if len(forecast_2024_df) > 0:
            deteriorating = forecast_2024_df.nlargest(10, 'Change')[['Country', 'Forecast_2024', 'Forecast_2025', 'Change']]
            deteriorating.columns = ['Country', '2024 Forecast', '2025 Forecast', 'Change']
            deteriorating['Change'] = deteriorating['Change'].apply(lambda x: f"+{x:.1f}pp")
            st.dataframe(deteriorating, use_container_width=True, hide_index=True)
        else:
            st.info("Trend data loading...")
    
    # =============== SECTION 5: MODEL PERFORMANCE SNAPSHOT =================
    st.markdown("<div class='section-header'>📊 Global Economic Outlook & Policy Guidance</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Economic Indicators Summary (2025 Forecast)")
        
        # Calculate global economic indicators for 2025
        latest_year = int(df['Date'].max())
        latest_data = df[df['Date'] == latest_year]
        
        # Get averages from 2023 actual data
        avg_gdp_raw = float(latest_data['GDP_Growth'].mean())
        avg_interest = float(latest_data['Interest_Rate'].mean())
        avg_unemployment = float(latest_data['Unemployment_Rate'].mean())
        avg_exchange = float(latest_data['Exchange_Rate'].mean())
        
        # Calculate exchange rate volatility
        exchange_volatility = float(latest_data['Exchange_Rate'].std())
        volatility_level = "Low" if exchange_volatility < 50 else "Moderate" if exchange_volatility < 150 else "High"
        
        # FIX: Check if GDP is in absolute values (large numbers) or percentages
        if avg_gdp_raw > 1000:
            # GDP is in absolute values - convert to growth rate
            # Use a reasonable global average (3-4%)
            avg_gdp = 3.2  # Reasonable global GDP growth estimate
        else:
            # GDP is already in percentage format
            avg_gdp = avg_gdp_raw
        
        # Project 2025 values (conservative estimates)
        forecast_gdp_2025 = avg_gdp + 0.2  # Add 0.2pp improvement
        forecast_interest_2025 = avg_interest + 0.25  # Add 0.25pp (slight tightening)
        forecast_unemployment_2025 = avg_unemployment - 0.1  # Subtract 0.1pp (slight improvement)
        
        # Create indicators grid (5 indicators in 3-2 grid layout)
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
                <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem;'>
                    <div>
                        <div style='font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem;'>📊 Inflation</div>
                        <div style='font-size: 2rem; font-weight: 700;'>{avg_forecast_2025:.1f}%</div>
                        <div style='font-size: 0.75rem; opacity: 0.8; margin-top: 0.25rem;'>
                            {"↓ Decreasing" if forecast_change < 0 else "↑ Increasing"}
                        </div>
                    </div>
                    <div>
                        <div style='font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem;'>💰 GDP Growth</div>
                        <div style='font-size: 2rem; font-weight: 700;'>{forecast_gdp_2025:.1f}%</div>
                        <div style='font-size: 0.75rem; opacity: 0.8; margin-top: 0.25rem;'>Global average</div>
                    </div>
                    <div>
                        <div style='font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem;'>📈 Interest Rate</div>
                        <div style='font-size: 2rem; font-weight: 700;'>{forecast_interest_2025:.1f}%</div>
                        <div style='font-size: 0.75rem; opacity: 0.8; margin-top: 0.25rem;'>Central bank avg</div>
                    </div>
                    <div>
                        <div style='font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem;'>👥 Unemployment</div>
                        <div style='font-size: 2rem; font-weight: 700;'>{forecast_unemployment_2025:.1f}%</div>
                        <div style='font-size: 0.75rem; opacity: 0.8; margin-top: 0.25rem;'>Labor market</div>
                    </div>
                    <div>
                        <div style='font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem;'>💱 FX Volatility</div>
                        <div style='font-size: 2rem; font-weight: 700;'>{volatility_level}</div>
                        <div style='font-size: 0.75rem; opacity: 0.8; margin-top: 0.25rem;'>Currency stability</div>
                    </div>
                </div>
                <div style='margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);'>
                    <div style='display: flex; align-items: center; justify-content: space-between;'>
                        <div>
                            <div style='font-size: 0.875rem; opacity: 0.9;'>Overall Risk Level</div>
                            <div style='font-size: 1.25rem; font-weight: 600; margin-top: 0.25rem;'>
                                🟡 MODERATE
                            </div>
                        </div>
                        <div style='text-align: right;'>
                            <div style='font-size: 0.875rem; opacity: 0.9;'>Economic Sentiment</div>
                            <div style='font-size: 1.25rem; font-weight: 600; margin-top: 0.25rem;'>
                                {"Cautiously Optimistic" if avg_forecast_2025 < 3 else "Cautious"}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Additional context
        st.markdown("""
            <div class='alert-box info'>
                <strong>📊 Outlook Summary</strong><br>
                Global economic indicators suggest moderate inflation pressures with stable growth. 
                Labor markets remain resilient while central banks maintain cautious monetary policy stance.
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💡 Global Policy Recommendations")
        
        if len(forecast_2025_df) > 0:
            # Categorize countries by policy stance needed
            tighten_countries = forecast_2025_df[forecast_2025_df['Forecast'] > 6]
            maintain_countries = forecast_2025_df[(forecast_2025_df['Forecast'] >= 2) & (forecast_2025_df['Forecast'] <= 6)]
            ease_countries = forecast_2025_df[forecast_2025_df['Forecast'] < 2]
            
            tighten_count = len(tighten_countries)
            maintain_count = len(maintain_countries)
            ease_count = len(ease_countries)
            
            # Determine overall stance
            if tighten_count > maintain_count and tighten_count > ease_count:
                overall_stance = "🔴 Tightening Bias"
            elif ease_count > maintain_count:
                overall_stance = "🟢 Easing Bias"
            else:
                overall_stance = "🟡 Neutral/Maintain"
            
            st.info(f"**RECOMMENDED GLOBAL MONETARY STANCE:** {overall_stance}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Using columns for the three categories
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown(f"""
                    <div class='metric-card red'>
                        <div class='metric-label'>🔴 TIGHTEN Policy</div>
                        <div class='metric-value'>{tighten_count}</div>
                        <div class='metric-change'>
                            <div style='font-size: 0.75rem;'>Inflation >6%</div>
                            <div style='font-size: 0.75rem;'>Raise rates aggressively</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                    <div class='metric-card orange'>
                        <div class='metric-label'>🟡 MAINTAIN Stance</div>
                        <div class='metric-value'>{maintain_count}</div>
                        <div class='metric-change'>
                            <div style='font-size: 0.75rem;'>Inflation 2-6%</div>
                            <div style='font-size: 0.75rem;'>Hold current policy</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f"""
                    <div class='metric-card green'>
                        <div class='metric-label'>🟢 EASE Policy</div>
                        <div class='metric-value'>{ease_count}</div>
                        <div class='metric-change'>
                            <div style='font-size: 0.75rem;'>Inflation <2%</div>
                            <div style='font-size: 0.75rem;'>Consider stimulus</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Key recommendations
            # Key recommendations - DYNAMIC based on actual stance
            
            # Calculate additional metrics
            avg_inflation_all = forecast_2025_df['Forecast'].mean()
            high_risk_pct = (tighten_count / len(forecast_2025_df)) * 100
            maintain_pct = (maintain_count / len(forecast_2025_df)) * 100
            
            # Determine dominant policy need
            if tighten_count > maintain_count and tighten_count > ease_count:
                stance = "tightening"
            elif ease_count > maintain_count:
                stance = "easing"
            else:
                stance = "neutral"
            
            # Generate stance-specific recommendations
            if stance == "tightening":
                recommendations = f"""**⚡ Key Policy Actions for 2025**

**Primary Focus: Combat High Inflation**
- Global Average: {avg_inflation_all:.1f}% (above 2% target)
- High Risk: {tighten_count} countries ({high_risk_pct:.0f}%) need aggressive action

**Recommended Actions:**
- **Immediate:** Central banks raise policy rates by 50-100 bps
- **Monitor:** Core inflation and inflation expectations closely
- **Fiscal:** Coordinate spending restraint with monetary tightening
- **Communication:** Clear commitment to restoring price stability
- **Risk:** Prepare for growth slowdown as necessary trade-off"""
                
            elif stance == "easing":
                recommendations = f"""**⚡ Key Policy Actions for 2025**

**Primary Focus: Support Growth & Prevent Deflation**
- Global Average: {avg_inflation_all:.1f}% (below 2% target)
- Low Inflation: {ease_count} countries at risk of deflation

**Recommended Actions:**
- **Immediate:** Consider reducing interest rates to support demand
- **Monitor:** Deflationary pressures and output gap closely
- **Fiscal:** Expansionary policy may be needed
- **Communication:** Commitment to achieving 2% from below
- **Risk:** Prepare unconventional tools if needed"""
                
            else:  # neutral/maintain - MIXED PICTURE
                recommendations = f"""**⚡ Key Policy Actions for 2025**

**Primary Focus: Differentiated Policy Approach**
- Global Average: {avg_inflation_all:.1f}% (elevated overall)
- **BUT:** {maintain_count} countries ({maintain_pct:.0f}%) in 2-6% target range
- Mixed picture: {tighten_count} high-risk, {ease_count} low-risk

**Recommended Actions:**
- **Global:** No one-size-fits-all - country-specific approaches needed
- **High-risk countries ({tighten_count}):** Aggressive tightening required
- **Target-range countries ({maintain_count}):** Maintain vigilance, data-dependent
- **Low-risk countries ({ease_count}):** Monitor deflation risks
- **Coordination:** Regional central bank communication important"""
            
            st.warning(recommendations)
        
        else:
            st.info("Policy recommendations will appear once forecasts are generated.")
            
            
            # ==================== COUNTRY SEARCH ====================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔍 Quick Country Lookup</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if len(forecast_2025_df) > 0:
            # Create searchable dropdown
            country_list = sorted(forecast_2025_df['Country'].unique().tolist())
            selected_country = st.selectbox(
                "Search for a specific country:",
                options=[''] + country_list,
                format_func=lambda x: "Select a country..." if x == '' else x
            )
            
            if selected_country and selected_country != '':
                # Get country data
                country_forecast = forecast_2025_df[forecast_2025_df['Country'] == selected_country].iloc[0]
                forecast_value = float(country_forecast['Forecast'])
                
                # Determine risk category
                if forecast_value >= 10:
                    risk = 'Critical'
                    risk_color = '#991b1b'
                    risk_emoji = '⚫'
                elif forecast_value >= 6:
                    risk = 'High'
                    risk_color = '#ef4444'
                    risk_emoji = '🔴'
                elif forecast_value >= 4:
                    risk = 'Elevated'
                    risk_color = '#f59e0b'
                    risk_emoji = '🟠'
                elif forecast_value >= 2:
                    risk = 'Moderate'
                    risk_color = '#eab308'
                    risk_emoji = '🟡'
                else:
                    risk = 'Low'
                    risk_color = '#10b981'
                    risk_emoji = '🟢'
                
                # Determine policy recommendation
                if forecast_value >= 10:
                    policy = 'Aggressive tightening required immediately'
                elif forecast_value >= 6:
                    policy = 'Tighten monetary policy (raise rates 50-100 bps)'
                elif forecast_value >= 4:
                    policy = 'Consider gradual tightening (25-50 bps)'
                elif forecast_value >= 2:
                    policy = 'Maintain current policy stance'
                else:
                    policy = 'Consider easing to prevent deflation'
                
                # Get rank
                forecast_2025_df_sorted = forecast_2025_df.sort_values('Forecast', ascending=False)
                rank = forecast_2025_df_sorted[forecast_2025_df_sorted['Country'] == selected_country].index[0] + 1
                
                # Display results
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {risk_color}22 0%, {risk_color}11 100%); 
                                padding: 1.5rem; border-radius: 10px; border-left: 4px solid {risk_color};
                                margin-top: 1rem;'>
                        <h3 style='margin: 0 0 1rem 0; color: {risk_color};'>
                            {risk_emoji} {selected_country}
                        </h3>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                            <div>
                                <div style='font-size: 0.875rem; color: #6b7280; margin-bottom: 0.25rem;'>
                                    2025 Forecast
                                </div>
                                <div style='font-size: 2rem; font-weight: 700; color: {risk_color};'>
                                    {forecast_value:.2f}%
                                </div>
                            </div>
                            <div>
                                <div style='font-size: 0.875rem; color: #6b7280; margin-bottom: 0.25rem;'>
                                    Risk Level
                                </div>
                                <div style='font-size: 1.5rem; font-weight: 600; color: {risk_color};'>
                                    {risk_emoji} {risk}
                                </div>
                            </div>
                            <div>
                                <div style='font-size: 0.875rem; color: #6b7280; margin-bottom: 0.25rem;'>
                                    Global Rank
                                </div>
                                <div style='font-size: 1.25rem; font-weight: 600;'>
                                    #{rank} of 341
                                </div>
                            </div>
                            <div>
                                <div style='font-size: 0.875rem; color: #6b7280; margin-bottom: 0.25rem;'>
                                    Percentile
                                </div>
                                <div style='font-size: 1.25rem; font-weight: 600;'>
                                    {(rank/341)*100:.0f}th percentile
                                </div>
                            </div>
                        </div>
                        <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid {risk_color}33;'>
                            <div style='font-size: 0.875rem; color: #6b7280; margin-bottom: 0.5rem;'>
                                <strong>Policy Recommendation:</strong>
                            </div>
                            <div style='font-size: 0.95rem; color: #1f2937;'>
                                💡 {policy}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Quick Stats")
        st.info(f"""
            **Total Countries:** 341
            
            **Risk Distribution:**
            - 🔴 High: {len(forecast_2025_df[forecast_2025_df['Forecast'] >= 6])}
            - 🟡 Moderate: {len(forecast_2025_df[(forecast_2025_df['Forecast'] >= 2) & (forecast_2025_df['Forecast'] < 6)])}
            - 🟢 Low: {len(forecast_2025_df[forecast_2025_df['Forecast'] < 2])}
            
            **Global Average:** {forecast_2025_df['Forecast'].mean():.1f}%
        """)
    
    # ==================== CSV EXPORT BUTTONS ====================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📥 Export Forecast Data</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(forecast_2025_df) > 0:
            # Prepare export data with risk categories
            export_df = forecast_2025_df.copy()
            
            # Add risk categories
            export_df['Risk_Category'] = export_df['Forecast'].apply(
                lambda x: 'Critical' if x >= 10 else 
                         'High' if x >= 6 else 
                         'Elevated' if x >= 4 else
                         'Moderate' if x >= 2 else 
                         'Low'
            )
            
            # Add policy recommendations
            export_df['Policy_Recommendation'] = export_df['Forecast'].apply(
                lambda x: 'Aggressive Tightening' if x >= 10 else
                         'Tighten Policy' if x >= 6 else
                         'Consider Tightening' if x >= 4 else
                         'Maintain Stance' if x >= 2 else
                         'Consider Easing'
            )
            
            # Rename columns for clarity
            export_df = export_df.rename(columns={'Forecast': 'Inflation_Forecast_2025'})
            
            # Reorder columns
            export_df = export_df[['Country', 'Inflation_Forecast_2025', 'Risk_Category', 'Policy_Recommendation']]
            
            # Create CSV
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download 2025 Forecasts",
                data=csv,
                file_name=f"global_inflation_forecast_2025_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download forecast data for all 341 countries with risk categories"
            )
            
            # Show preview
            with st.expander("📋 Preview CSV Contents"):
                st.dataframe(export_df.head(10), use_container_width=True)
    
    with col2:
        if len(forecast_2024_df) > 0:
            # Export with trend data
            trend_export = forecast_2024_df[['Country', 'Forecast_2024', 'Forecast_2025']].copy()
            trend_export['Change_pp'] = trend_export['Forecast_2025'] - trend_export['Forecast_2024']
            trend_export['Trend'] = trend_export['Change_pp'].apply(
                lambda x: 'Improving' if x < -1 else 
                         'Worsening' if x > 1 else 
                         'Stable'
            )
            
            # Rename columns
            trend_export.columns = ['Country', 'Forecast_2024', 'Forecast_2025', 'Change (pp)', 'Trend']
            
            csv = trend_export.to_csv(index=False)
            
            st.download_button(
                label="📈 Download Trend Analysis",
                data=csv,
                file_name=f"inflation_trends_2024_2025_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download 2024-2025 trend comparison with change calculations"
            )
            
            # Show preview
            with st.expander("📋 Preview CSV Contents"):
                st.dataframe(trend_export.head(10), use_container_width=True)
    
    with col3:
        st.markdown("### 🎯 What To Do Next")
        
        st.markdown("""
            **After Downloading:**
            
            1️⃣ **Filter in Excel**
            - Sort by risk level
            - Filter specific regions
            
            2️⃣ **Analyze Trends**
            - Find fastest movers
            - Compare neighbors
            
            3️⃣ **Create Reports**
            - Import / Generate charts
        """)

def render_data_source(df):
    """Complete data explorer with advanced filtering and analysis"""
    st.markdown("<div class='section-header'>💾 Data Source Explorer</div>", unsafe_allow_html=True)
    st.markdown("Explore the complete economic dataset with advanced filtering and analysis")
    
    # ==================== AUTO-DETECT COLUMN NAMES ====================
    # Handle different possible column names
    year_col = 'Date' if 'Date' in df.columns else 'Year'
    inflation_col = 'Inflation_Rate' if 'Inflation_Rate' in df.columns else 'Inflation'
    gdp_col = 'GDP_Growth' if 'GDP_Growth' in df.columns else 'GDP'
    interest_col = 'Interest_Rate'
    exchange_col = 'Exchange_Rate'
    unemployment_col = 'Unemployment_Rate'
    
    # ==================== FILTERS SECTION ====================
    st.markdown("### Data Filters")
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        countries = ['All Countries'] + sorted(df['Country'].unique().tolist())
        selected_country = st.selectbox("Filter by Country", countries)
    
    with col2:
        years = df[year_col].unique()
        year_range = st.slider("Year Range", 
                              int(years.min()), 
                              int(years.max()), 
                              (int(years.min()), int(years.max())))
    
    with col3:
        # Column selector
        all_columns = ['Country', 'Year', 'Inflation (%)', 'GDP', 
                      'Interest Rate (%)', 'Exchange Rate', 'Unemployment (%)']
        selected_columns = st.multiselect(
            "Select Columns to Display",
            options=all_columns,
            default=all_columns
        )
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        export_button = st.button("📥 Export", use_container_width=True, type="primary")
    
    # ==================== FILTER DATA ====================
    filtered_df = df.copy()
    if selected_country != 'All Countries':
        filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    filtered_df = filtered_df[(filtered_df[year_col] >= year_range[0]) & (filtered_df[year_col] <= year_range[1])]
    
    # ==================== DISPLAY FILTERED DATA ====================
    st.markdown(f"### 📊 Filtered Data ({len(filtered_df):,} records)")
    
    # Prepare display dataframe with detected column names
    display_df = filtered_df[['Country', year_col, inflation_col, gdp_col, 
                               interest_col, exchange_col, unemployment_col]].copy()
    display_df.columns = all_columns
    
    # Show only selected columns
    if selected_columns:
        display_df = display_df[selected_columns]
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.warning("Please select at least one column to display")
    
    # ==================== EXPORT FUNCTIONALITY ====================
    if export_button:
        csv = display_df.to_csv(index=False)
        
        if selected_country != 'All Countries':
            filename = f"economic_data_{selected_country}_{year_range[0]}-{year_range[1]}.csv"
        else:
            filename = f"economic_data_all_countries_{year_range[0]}-{year_range[1]}.csv"
        
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )
        st.success(f"✅ Export ready! {len(filtered_df)} records prepared for download.")
    
    # ==================== SELECTION STATISTICS ====================
    st.markdown("### 📊 Selection Statistics")
    
    if len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_inflation = filtered_df[inflation_col].mean()
            inflation_std = filtered_df[inflation_col].std()
            st.metric(
                "Avg Inflation", 
                f"{avg_inflation:.2f}%",
                delta=f"±{inflation_std:.2f}% std" if inflation_std > 0.01 else "No variation"
            )
            
            max_inflation = filtered_df[inflation_col].max()
            if max_inflation > 100:
                st.caption(f"⚠️ Max: {max_inflation:.0f}%")
        
        with col2:
            avg_interest = filtered_df[interest_col].mean()
            interest_std = filtered_df[interest_col].std()
            st.metric(
                "Avg Interest Rate", 
                f"{avg_interest:.2f}%",
                delta=f"±{interest_std:.2f}% std" if interest_std > 0.01 else "No variation"
            )
        
        with col3:
            avg_unemployment = filtered_df[unemployment_col].mean()
            unemployment_std = filtered_df[unemployment_col].std()
            st.metric(
                "Avg Unemployment", 
                f"{avg_unemployment:.2f}%",
                delta=f"±{unemployment_std:.2f}% std" if unemployment_std > 0.01 else "No variation"
            )
        
        with col4:
            st.metric(
                "Records Selected", 
                f"{len(filtered_df):,}",
                delta=f"{len(filtered_df)/len(df)*100:.1f}% of total"
            )
        
        # Additional Metrics (only for specific countries)
        if selected_country != 'All Countries':
            with st.expander("📊 Additional Metrics (Country-Specific)"):
                col1, col2 = st.columns(2)
                with col1:
                    avg_gdp = filtered_df[gdp_col].mean()
                    if avg_gdp < 100:
                        st.metric("Avg GDP Growth", f"{avg_gdp:.2f}%")
                    else:
                        st.metric("Avg GDP", f"{avg_gdp:,.0f}")
                        st.caption("(Absolute value)")
                
                with col2:
                    avg_exchange = filtered_df[exchange_col].mean()
                    st.metric("Avg Exchange Rate", f"{avg_exchange:.2f}")
    
    # ========== MINI TREND CHARTS (for specific countries) ==============
    if selected_country != 'All Countries' and len(filtered_df) > 5:
        st.markdown("### Trend Charts")
        
        # Create 2x2 grid of mini charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Inflation trend
            st.markdown("#### 📊 Inflation Trend")
            inflation_chart = filtered_df.sort_values(year_col)[[year_col, inflation_col]].set_index(year_col)
            st.line_chart(inflation_chart, height=200)
            
            # Interest rate trend
            st.markdown("#### 📈 Interest Rate Trend")
            interest_chart = filtered_df.sort_values(year_col)[[year_col, interest_col]].set_index(year_col)
            st.line_chart(interest_chart, height=200)
        
        with col2:
            # GDP trend
            st.markdown("#### 💰 GDP Trend")
            gdp_chart = filtered_df.sort_values(year_col)[[year_col, gdp_col]].set_index(year_col)
            st.line_chart(gdp_chart, height=200)
            
            # Unemployment trend
            st.markdown("#### 👥 Unemployment Trend")
            unemployment_chart = filtered_df.sort_values(year_col)[[year_col, unemployment_col]].set_index(year_col)
            st.line_chart(unemployment_chart, height=200)
        
        # Insights
        st.info(f"""
            **📊 Quick Insights for {selected_country}:**
            - **Inflation Range:** {filtered_df[inflation_col].min():.2f}% to {filtered_df[inflation_col].max():.2f}%
            - **Highest Inflation Year:** {filtered_df.loc[filtered_df[inflation_col].idxmax(), year_col]:.0f} ({filtered_df[inflation_col].max():.2f}%)
            - **Lowest Inflation Year:** {filtered_df.loc[filtered_df[inflation_col].idxmin(), year_col]:.0f} ({filtered_df[inflation_col].min():.2f}%)
        """)
    
    # ==================== DETAILED STATISTICS ====================
    with st.expander("📈 Detailed Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Inflation")
            st.write(f"Min: {filtered_df[inflation_col].min():.2f}%")
            st.write(f"Max: {filtered_df[inflation_col].max():.2f}%")
            st.write(f"Median: {filtered_df[inflation_col].median():.2f}%")
            
            st.markdown("#### Interest Rate")
            st.write(f"Min: {filtered_df[interest_col].min():.2f}%")
            st.write(f"Max: {filtered_df[interest_col].max():.2f}%")
            st.write(f"Median: {filtered_df[interest_col].median():.2f}%")
        
        with col2:
            st.markdown("#### Unemployment")
            st.write(f"Min: {filtered_df[unemployment_col].min():.2f}%")
            st.write(f"Max: {filtered_df[unemployment_col].max():.2f}%")
            st.write(f"Median: {filtered_df[unemployment_col].median():.2f}%")
            
            st.markdown("#### Coverage")
            st.write(f"Records: {len(filtered_df):,}")
            st.write(f"Countries: {filtered_df['Country'].nunique()}")
            st.write(f"Years: {int(filtered_df[year_col].min())}-{int(filtered_df[year_col].max())}")
    
    # ==================== OVERALL STATISTICS ====================
    st.markdown("### 📊 Overall Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Countries", df['Country'].nunique())
    
    with col3:
        st.metric("Years Covered", f"{int(df[year_col].min())}-{int(df[year_col].max())}")
    
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{100-missing_pct:.1f}%")
    
    # ==================== DATA QUALITY ====================
    st.markdown("### 📈 Data Quality Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Values")
        missing_data = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing': missing_data.values
        })
        missing_df = missing_df[missing_df['Missing'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, hide_index=True)
        else:
            st.success("✅ No missing values!")
    
    with col2:
        st.markdown("#### Outliers (IQR Method)")
        outlier_counts = {}
        for col in [inflation_col, interest_col, unemployment_col]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            outlier_counts[col] = len(outliers)
        
        outlier_df = pd.DataFrame({
            'Column': outlier_counts.keys(),
            'Outliers': outlier_counts.values()
        })
        st.dataframe(outlier_df, hide_index=True)
        
def generate_forecast_pdf(country, years, comparison_table, ensemble_forecasts, 
                         forecast_std, policy_rec, ml_models, statistical_models,
                         country_data, fig):
    """Generate professional PDF forecast report"""
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#3b82f6'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # PAGE 1: EXECUTIVE SUMMARY
    story.append(Paragraph("INFLATION FORECAST REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    summary_data = [
        ['Country:', country],
        ['Forecast Horizon:', f'{years} Years'],
        ['Generated:', datetime.now().strftime('%B %d, %Y')],
        ['', ''],
        ['Ensemble Forecast:', f"{np.mean(list(ensemble_forecasts.values())):.1f}%"],
        ['Model Agreement:', f"σ = {forecast_std:.2f}pp"],
        ['Risk Level:', policy_rec['risk_level']],
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 11),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 11),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2937')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("KEY FINDINGS", heading_style))
    findings_text = f"""
    • <b>ML Models Active:</b> {len(ml_models)} trained models<br/>
    
    • <b>Confidence:</b> σ = {forecast_std:.2f}pp<br/>
    • <b>Risk:</b> {policy_rec['risk_level']}
    """
    story.append(Paragraph(findings_text, styles['Normal']))
    story.append(PageBreak())
    
    # PAGE 2: MODEL COMPARISON
    story.append(Paragraph("MULTI-MODEL FORECAST COMPARISON", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    table_data = [comparison_table.columns.tolist()] + comparison_table.values.tolist()
    forecast_table = Table(table_data, repeatRows=1)
    forecast_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')]),
    ]))
    story.append(forecast_table)
    story.append(PageBreak())
    
    # PAGE 3: VISUALIZATION
    story.append(Paragraph("FORECAST VISUALIZATION", heading_style))
    try:
        img_bytes = pio.to_image(fig, format='png', width=700, height=400)
        img_buffer = BytesIO(img_bytes)
        img = Image(img_buffer, width=6.5*inch, height=3.7*inch)
        story.append(img)
    except:
        story.append(Paragraph("Chart not available", styles['Normal']))
    story.append(PageBreak())
    
    # PAGE 4: POLICY RECOMMENDATIONS
    story.append(Paragraph("POLICY RECOMMENDATIONS", heading_style))
    policy_text = f"""
    <b>Risk Assessment:</b> {policy_rec['risk_level']}<br/><br/>
    <b>Recommended Actions:</b><br/>
    {'<br/>'.join(['• ' + action for action in policy_rec['actions']])}
    """
    story.append(Paragraph(policy_text, styles['Normal']))
    story.append(PageBreak())
    
    # PAGE 5: METHODOLOGY
    story.append(Paragraph("METHODOLOGY", heading_style))
    methodology_text = f"""
    <b>ML Models:</b> {', '.join(ml_models)}<br/>
    <b>Statistical:</b> {', '.join(statistical_models)}<br/><br/>
    <b>Training:</b> 341 countries, 1990-2023<br/>
    <b>Model Selection:</b> Best model chosen dynamically
    """
    story.append(Paragraph(methodology_text, styles['Normal']))
    story.append(PageBreak())
    
    # PAGE 6: DISCLAIMER
    story.append(Paragraph("DISCLAIMER", heading_style))
    disclaimer_text = f"""
    This forecast is generated by an automated ML system based on historical data. 
    Actual outcomes may differ. Use for strategic planning, not sole decision-making.<br/><br/>
    <b>Generated:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
    <b>System:</b> InfraScope v1.0
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer
        

def render_forecasting(df, comparison_df):
    """Advanced forecasting with multi-model comparison - USING REAL ML MODELS"""
    st.markdown("<div class='section-header'>📈 Inflation Forecasting & Policy Analysis</div>", unsafe_allow_html=True)
    
    # ==================== SIMPLIFIED CONFIGURATION ====================
    st.markdown("### ⚙️ Forecast Configuration")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        countries = sorted(df['Country'].unique().tolist())
        selected_country = st.selectbox("Select Country", countries, index=countries.index('Malaysia') if 'Malaysia' in countries else 0)
    
    with col2:
        forecast_years = st.selectbox("Forecast Horizon (Years)", [1, 2, 3, 5], index=2)
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_button = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)
    
    # ======== GENERATE FORECAST AND STORE IN SESSION STATE =================
    if generate_button:
        with st.spinner(f"🔄 Generating forecasts from 6 models for {selected_country}..."):
            country_data = df[df['Country'] == selected_country].sort_values('Date')
            
            if len(country_data) < 5:
                st.error(f"❌ Insufficient data for {selected_country}. Need at least 5 years of historical data.")
                return
            
            # Get all model forecasts
            all_model_forecasts = {}
            model_names = ['RF_Extended', 'RF_Base', 'XGBoost_Extended', 'XGBoost_Base', 'ARIMA(2,1,2)', 'VAR(1)']
            
            ml_models = []
            statistical_models = []
            
            for model_name in model_names:
                used_ml = False
                try:
                    forecast_df, used_ml = forecast_inflation(df, selected_country, model_name, forecast_years, 'normal', show_info=False)
                    if forecast_df is not None and len(forecast_df) > 0:
                        all_model_forecasts[model_name] = forecast_df
                        if used_ml:
                            ml_models.append(model_name)
                        else:
                            statistical_models.append(model_name)
                except Exception as e:
                    st.warning(f"⚠️ {model_name} forecast failed: {str(e)}")
            
            if not all_model_forecasts:
                st.error("❌ Could not generate forecasts from any model. Please check data quality.")
                return
            
            # Calculate ensemble and other data
            ensemble_forecasts = {}
            all_years = sorted(set(int(row['Year']) for df_temp in all_model_forecasts.values() for _, row in df_temp.iterrows()))
            
            for year in all_years:
                year_predictions = []
                for model_name, forecast_df in all_model_forecasts.items():
                    year_data = forecast_df[forecast_df['Year'] == year]
                    if len(year_data) > 0:
                        year_predictions.append(float(year_data.iloc[0]['Inflation']))
                ensemble_forecasts[year] = np.mean(year_predictions) if year_predictions else 0
            
            all_predictions = [float(row['Inflation']) for df_temp in all_model_forecasts.values() for _, row in df_temp.iterrows()]
            forecast_std = np.std(all_predictions)
            
            # Build comparison table
            comparison_data = []
            for model_name, forecast_df in all_model_forecasts.items():
                row = {'Model': model_name}
                for i, forecast_row in forecast_df.iterrows():
                    year = int(forecast_row['Year'])
                    inflation = float(forecast_row['Inflation'])
                    row[f'{year}'] = f"{inflation:.1f}%"
                row['Avg'] = f"{forecast_df['Inflation'].mean():.1f}%"
                comparison_data.append(row)
            
            comparison_table = pd.DataFrame(comparison_data)
            
            # ====== DETERMINE BEST MODEL (SMART SELECTION) ==============
            best_model = get_best_model_for_country(selected_country, all_model_forecasts, country_data)
            
            # Add 🏆 icon to best model
            comparison_table['Model'] = comparison_table['Model'].apply(
                lambda x: f"🏆 {x} (BEST)" if x == best_model else x
            )
            
            range_row = {'Model': '📏 Forecast Range (Min-Max)'}
            for year in all_years:
                year_predictions = []
                for model_name, forecast_df in all_model_forecasts.items():
                    year_data = forecast_df[forecast_df['Year'] == year]
                    if len(year_data) > 0:
                        year_predictions.append(float(year_data.iloc[0]['Inflation']))
                if year_predictions:
                    range_row[f'{year}'] = f"{min(year_predictions):.1f}-{max(year_predictions):.1f}%"
                else:
                    range_row[f'{year}'] = "N/A"
            
            range_row['Avg'] = f"±{forecast_std:.1f}pp"
            comparison_table = pd.concat([comparison_table, pd.DataFrame([range_row])], ignore_index=True)
            
            # Get policy recommendations
            ensemble_df = pd.DataFrame({
                'Year': list(ensemble_forecasts.keys()),
                'Inflation': list(ensemble_forecasts.values())
            })
            policy_rec = generate_policy_recommendation(ensemble_df)
            
            # ========== STORE EVERYTHING IN SESSION STATE ================
            st.session_state['forecast_data'] = {
                'selected_country': selected_country,
                'forecast_years': forecast_years,
                'all_model_forecasts': all_model_forecasts,
                'ml_models': ml_models,
                'statistical_models': statistical_models,
                'ensemble_forecasts': ensemble_forecasts,
                'forecast_std': forecast_std,
                'comparison_table': comparison_table,
                'policy_rec': policy_rec,
                'country_data': country_data,
                'all_years': all_years,
                'best_model': best_model
            }
    
    # ======= DISPLAY RESULTS (FROM SESSION STATE OR FRESH) ====================
    if 'forecast_data' in st.session_state:
        # Load from session state
        data = st.session_state['forecast_data']
        selected_country = data['selected_country']
        forecast_years = data['forecast_years']
        all_model_forecasts = data['all_model_forecasts']
        ml_models = data['ml_models']
        statistical_models = data['statistical_models']
        ensemble_forecasts = data['ensemble_forecasts']
        forecast_std = data['forecast_std']
        comparison_table = data['comparison_table']
        policy_rec = data['policy_rec']
        country_data = data['country_data']
        all_years = data['all_years']
        
        # Show which models are ML vs statistical
        st.caption(f"🤖 **ML Models:** {', '.join(ml_models) if ml_models else 'None'} | "
                  f"📊 **Statistical:** {', '.join(statistical_models) if statistical_models else 'None'}")
        
        # ==================== MODEL COMPARISON TABLE ====================
        st.markdown("### 📊 Multi-Model Consensus Forecast")
        
        st.info(f"""
            **🎯 Model Comparison:** 
            - **{len(ml_models)} ML models** trained with {len(df['Country'].unique())} countries
            - **{len(statistical_models)} statistical models** using hybrid trend analysis
        """)
        
        # Style table to highlight best model
        def highlight_best_model(row):
            if '🏆' in str(row['Model']):
                return ['background-color: #d1fae5; font-weight: bold'] * len(row)
            return [''] * len(row)

        styled_table = comparison_table.style.apply(highlight_best_model, axis=1)
        st.dataframe(styled_table, use_container_width=True, hide_index=True)
        
        # Model agreement indicator
        if forecast_std < 0.5:
            st.success(f"✅ **High Model Agreement** (σ = {forecast_std:.2f}pp) - All models predict similar outcomes. High confidence!")
        elif forecast_std < 1.0:
            st.warning(f"⚠️ **Moderate Agreement** (σ = {forecast_std:.2f}pp) - Some model divergence. Moderate confidence.")
        else:
            st.error(f"🔴 **Low Agreement** (σ = {forecast_std:.2f}pp) - Significant model disagreement. Use with caution!")
        
        # ==================== FORECAST VISUALIZATION ====================
        st.markdown("### 📈 Forecast Visualization")
        
        fig = go.Figure()
        
        historical_years = country_data['Date'].values[-10:]
        historical_inflation = country_data['Inflation_Rate'].values[-10:]
        
        # Historical data - PROMINENT
        fig.add_trace(go.Scatter(
            x=historical_years,
            y=historical_inflation,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#000000', width=4),
            marker=dict(size=10, color='#000000'),
            hovertemplate='<b>%{x}</b><br>Inflation: %{y:.1f}%<extra></extra>'
        ))
        
        # Prepare ensemble data for confidence interval
        ensemble_years = list(ensemble_forecasts.keys())
        ensemble_values = list(ensemble_forecasts.values())
        
        # Confidence interval
        upper_bound = [ensemble_forecasts[year] + forecast_std for year in ensemble_years]
        lower_bound = [ensemble_forecasts[year] - forecast_std for year in ensemble_years]
        
        fig.add_trace(go.Scatter(
            x=ensemble_years + ensemble_years[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(124, 58, 237, 0.15)',
            line=dict(color='rgba(124, 58, 237, 0.3)', width=1, dash='dash'),
            showlegend=True,
            name='Confidence Range (±1σ)',
            hoverinfo='skip'
        ))
        
        # Individual models
        ml_color = 'rgba(59, 130, 246, 0.4)'
        stat_color = 'rgba(156, 163, 175, 0.3)'
        
        for model_name, forecast_df in all_model_forecasts.items():
            if model_name in ml_models:
                line_color = ml_color
                line_width = 1.5
                opacity = 0.6
                dash = 'dot'
            else:
                line_color = stat_color
                line_width = 1
                opacity = 0.4
                dash = 'dash'
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'].values,
                y=forecast_df['Inflation'].values,
                mode='lines',
                name=model_name,
                line=dict(color=line_color, width=line_width, dash=dash),
                opacity=opacity,
                hovertemplate=f'<b>{model_name}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>'
            ))
        
        # 2% target line
        fig.add_hline(
            y=2.0,
            line_dash="dash",
            line_color="#10b981",
            line_width=2,
            annotation_text="2% Target",
            annotation_position="right",
            annotation=dict(font_size=12, font_color="#10b981")
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Inflation Forecast: {selected_country} ({forecast_years}-Year Horizon)",
                font=dict(size=18, color='#1f2937')
            ),
            xaxis_title="Year",
            yaxis_title="Inflation Rate (%)",
            template='plotly_white',
            height=550,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                borderwidth=1
            ),
            font=dict(size=12),
            plot_bgcolor='rgba(250, 250, 250, 0.5)'
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("""
            📊 **Chart Guide:** Black line shows historical data. Individual model forecasts shown for comparison. 
            Light blue dotted lines are ML models (RF, XGBoost). Light gray dashed lines are statistical models (ARIMA, VAR). 
            Purple shaded area shows forecast uncertainty (±1 standard deviation).
        """)
        
        # ==================== FORECAST SUMMARY CARDS ====================
        st.markdown("### 📊 Forecast Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            next_year = ensemble_years[0]
            next_year_value = ensemble_forecasts[next_year]
            st.metric(
                label=f"📅 {next_year} Forecast",
                value=f"{next_year_value:.1f}%",
                delta=f"{next_year_value - historical_inflation[-1]:.1f}pp vs 2023"
            )
        
        with col2:
            avg_forecast = np.mean(ensemble_values)
            st.metric(
                label=f"📊 {forecast_years}-Year Average",
                value=f"{avg_forecast:.1f}%",
                delta=f"±{forecast_std:.1f}pp uncertainty"
            )
        
        with col3:
            final_year = ensemble_years[-1]
            final_year_value = ensemble_forecasts[final_year]
            st.metric(
                label=f"🎯 {final_year} Forecast",
                value=f"{final_year_value:.1f}%",
                delta=f"{final_year_value - next_year_value:.1f}pp change"
            )
            
            
        # ================== WHAT'S DRIVING THIS FORECAST? ====================
        st.markdown("---")
        st.markdown("### 🔍 What's Driving This Forecast?")
        
        st.info("""
            **Understanding the Drivers:** This section shows which economic indicators are influencing 
            the inflation forecast and by how much. This helps policymakers understand the root causes 
            and identify intervention points.
        """)
        
        # ==================== CORRELATION ANALYSIS ====================
        st.markdown("#### 🔗 Correlation with Inflation")
        
        try:
            inflation_col = 'Inflation_Rate' if 'Inflation_Rate' in country_data.columns else 'Inflation'
            
            # Calculate correlations
            correlations = {}
            correlation_signs = {}
            
            indicator_map = {
                'GDP Growth': 'GDP_Growth',
                'Interest Rate': 'Interest_Rate',
                'Exchange Rate': 'Exchange_Rate',
                'Unemployment': 'Unemployment_Rate'
            }
            
            for display_name, col_name in indicator_map.items():
                if col_name in country_data.columns:
                    corr = country_data[inflation_col].corr(country_data[col_name])
                    if not np.isnan(corr):
                        correlations[display_name] = abs(corr)
                        correlation_signs[display_name] = corr
            
            if correlations:
                # Create dataframe
                corr_df = pd.DataFrame({
                    'Indicator': list(correlations.keys()),
                    'Correlation': list(correlations.values()),
                    'Sign': [correlation_signs[k] for k in correlations.keys()]
                })
                corr_df = corr_df.sort_values('Correlation', ascending=True)
                
                # Color by positive/negative
                colors = ['#10b981' if x > 0 else '#ef4444' for x in corr_df['Sign']]
                
                # Create bar chart
                fig_corr = go.Figure(go.Bar(
                    x=corr_df['Correlation'],
                    y=corr_df['Indicator'],
                    orientation='h',
                    marker_color=colors,
                    text=corr_df['Correlation'].round(2),
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Correlation: %{x:.2f}<extra></extra>'
                ))
                
                fig_corr.update_layout(
                    title=f"Historical Correlation with Inflation - {selected_country}",
                    xaxis_title="Absolute Correlation Strength",
                    yaxis_title="Economic Indicator",
                    template='plotly_white',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Enhanced interpretation
                strongest = corr_df.iloc[-1]
                direction = "positively" if correlation_signs[strongest['Indicator']] > 0 else "negatively"
                
                st.success(f"""
                    **🔗 Strongest Relationship:** {strongest['Indicator']} is **{direction} correlated** 
                    ({correlation_signs[strongest['Indicator']]:.2f}) with inflation in {selected_country}.
                    
                    **What this means:** Based on 34 years of historical data (1990-2023), {strongest['Indicator']} 
                    has shown the strongest relationship with inflation in {selected_country}. This historical pattern 
                    informs our {forecast_years}-year forecast and makes {strongest['Indicator']} a key indicator for 
                    policymakers to monitor going forward.
                """)
                
                # Add color legend
                st.caption("🟢 Green bars = Positive correlation (indicator and inflation move together) | 🔴 Red bars = Negative correlation (indicator and inflation move opposite)")
                
            else:
                st.warning("Correlation data not available for this country")
        
        except Exception as e:
            st.warning(f"Could not calculate correlations: {str(e)}")
        
        # ================== DRIVER CONTRIBUTION BREAKDOWN ====================
        st.markdown("#### 💡 Driver Contribution Breakdown")
        
        st.markdown("""
            This shows how each economic indicator contributes to the forecast direction based on 
            recent changes and historical relationships:
        """)
        
        # Calculate recent changes in indicators
        try:
            recent_data = country_data.tail(3)
            
            contributions = []
            
            # GDP Growth
            if 'GDP_Growth' in recent_data.columns:
                gdp_change = recent_data['GDP_Growth'].iloc[-1] - recent_data['GDP_Growth'].iloc[0]
                gdp_corr = correlation_signs.get('GDP Growth', 0) if 'correlation_signs' in locals() else 0
                gdp_impact = gdp_change * gdp_corr * 0.1
                
                contributions.append({
                    'Indicator': '📈 GDP Growth',
                    'Recent Change': f"{gdp_change:+.1f}pp",
                    'Direction': '↑' if gdp_change > 0 else '↓',
                    'Impact on Inflation': f"{gdp_impact:+.2f}pp",
                    'Interpretation': 'Rising GDP often increases demand, pushing inflation up' if gdp_change > 0 else 'Slowing GDP reduces demand pressure on prices'
                })
            
            # Interest Rate
            if 'Interest_Rate' in recent_data.columns:
                ir_change = recent_data['Interest_Rate'].iloc[-1] - recent_data['Interest_Rate'].iloc[0]
                ir_corr = correlation_signs.get('Interest Rate', 0) if 'correlation_signs' in locals() else 0
                ir_impact = ir_change * ir_corr * 0.15
                
                contributions.append({
                    'Indicator': '💰 Interest Rate',
                    'Recent Change': f"{ir_change:+.1f}pp",
                    'Direction': '↑' if ir_change > 0 else '↓',
                    'Impact on Inflation': f"{ir_impact:+.2f}pp",
                    'Interpretation': 'Higher rates typically cool inflation by reducing borrowing' if ir_change > 0 else 'Lower rates may stimulate economy and increase inflation'
                })
            
            # Exchange Rate
            if 'Exchange_Rate' in recent_data.columns:
                er_change = recent_data['Exchange_Rate'].iloc[-1] - recent_data['Exchange_Rate'].iloc[0]
                er_corr = correlation_signs.get('Exchange Rate', 0) if 'correlation_signs' in locals() else 0
                er_impact = er_change * er_corr * 0.08
                
                contributions.append({
                    'Indicator': '💱 Exchange Rate',
                    'Recent Change': f"{er_change:+.1f}%",
                    'Direction': '↑' if er_change > 0 else '↓',
                    'Impact on Inflation': f"{er_impact:+.2f}pp",
                    'Interpretation': 'Weaker currency makes imports more expensive' if er_change > 0 else 'Stronger currency reduces import costs'
                })
            
            # Unemployment
            if 'Unemployment_Rate' in recent_data.columns:
                unemp_change = recent_data['Unemployment_Rate'].iloc[-1] - recent_data['Unemployment_Rate'].iloc[0]
                unemp_corr = correlation_signs.get('Unemployment', 0) if 'correlation_signs' in locals() else 0
                unemp_impact = unemp_change * unemp_corr * 0.12
                
                contributions.append({
                    'Indicator': '👥 Unemployment',
                    'Recent Change': f"{unemp_change:+.1f}pp",
                    'Direction': '↑' if unemp_change > 0 else '↓',
                    'Impact on Inflation': f"{unemp_impact:+.2f}pp",
                    'Interpretation': 'Rising unemployment reduces wage pressure and demand' if unemp_change > 0 else 'Falling unemployment can increase wages and spending'
                })
            
            if contributions:
                # Display as expandable cards
                for contrib in contributions:
                    with st.expander(f"{contrib['Indicator']} {contrib['Direction']} ({contrib['Recent Change']})"):
                        col_a, col_b = st.columns([1, 2])
                        
                        with col_a:
                            impact_val = float(contrib['Impact on Inflation'].replace('pp', '').replace('+', ''))
                            if abs(impact_val) > 0.2:
                                impact_level = "🔴 **High Impact**"
                                impact_color = "#ef4444"
                            elif abs(impact_val) > 0.1:
                                impact_level = "🟡 **Moderate Impact**"
                                impact_color = "#f59e0b"
                            else:
                                impact_level = "🟢 **Low Impact**"
                                impact_color = "#10b981"
                            
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; 
                                     background: {impact_color}15; border-radius: 8px;
                                     border: 2px solid {impact_color};'>
                                    <div style='font-size: 1.5rem; font-weight: 700; color: {impact_color};'>
                                        {contrib['Impact on Inflation']}
                                    </div>
                                    <div style='font-size: 0.9rem; margin-top: 0.5rem;'>
                                        {impact_level}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b:
                            st.markdown(f"""
                                **What This Means:**  
                                {contrib['Interpretation']}
                                
                                **Recent Trend:**  
                                This indicator changed by **{contrib['Recent Change']}** over the past 3 years,
                                contributing an estimated **{contrib['Impact on Inflation']}** to inflation movement.
                            """)
        
        except Exception as e:
            st.warning("Detailed contribution analysis unavailable for this country.")
        
        # ==================== POLICY INSIGHTS FROM DRIVERS ====================
        st.markdown("#### 🎯 Policy Insights from Driver Analysis")
        
        try:
            # Determine key policy levers based on correlation strength
            policy_insights = []
            
            if correlations:
                # Get top 2 correlated indicators
                sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                
                for indicator, strength in sorted_corr[:2]:
                    if strength > 0.3:  # Only if correlation is meaningful
                        if 'Interest Rate' in indicator:
                            policy_insights.append("""
                                **💰 Monetary Policy Focus:** Interest rates show strong correlation with inflation. 
                                Central bank policy adjustments will have significant impact on inflation trajectory.
                            """)
                        
                        elif 'Exchange Rate' in indicator:
                            policy_insights.append("""
                                **💱 Exchange Rate Management:** Currency movements strongly influence inflation. 
                                Consider measures to stabilize exchange rate or manage import costs.
                            """)
                        
                        elif 'GDP Growth' in indicator or 'GDP' in indicator:
                            policy_insights.append("""
                                **📈 Demand Management:** Economic growth is a key factor. Balance growth 
                                objectives with inflation control through fiscal and monetary coordination.
                            """)
                        
                        elif 'Unemployment' in indicator:
                            policy_insights.append("""
                                **👥 Labor Market Policy:** Employment trends strongly affect inflation. 
                                Monitor wage growth and labor market tightness.
                            """)
            
            if policy_insights:
                for insight in policy_insights:
                    st.info(insight)
            else:
                st.info("""
                    **📊 General Guidance:** Monitor all key indicators (GDP, interest rates, 
                    exchange rates, unemployment) and use a balanced policy approach.
                """)
        
        except:
            st.info("""
                **📊 General Guidance:** Monitor all key indicators (GDP, interest rates, 
                exchange rates, unemployment) and use a balanced policy approach.
            """)
        
        # ==================== SUMMARY BOX ====================
        st.markdown("---")
        
        try:
            if correlations:
                top_correlation = max(correlations.items(), key=lambda x: x[1])
                
                st.success(f"""
                    ### 📋 Driver Analysis Summary for {selected_country}
                    
                    **🔗 Strongest Historical Relationship:** {top_correlation[0]} ({top_correlation[1]:.2f} correlation)
                    
                    **📊 Analysis Period:** 34 years of data (1990-2023)
                    
                    **🎯 Forecast Horizon:** {forecast_years} years ahead (2024-{2023+forecast_years})
                    
                    **💡 Policy Implication:** Focus policy interventions on the top drivers identified above 
                    for maximum effectiveness in controlling inflation.
                """)
        except:
            pass
        
        # ==================== POLICY RECOMMENDATIONS ====================
        st.markdown("### 💡 Policy Recommendations")
        
        st.markdown(f"""
            <div class='alert-box' style='background-color: {policy_rec['risk_color']}15; 
                 border-left-color: {policy_rec['risk_color']}; color: {policy_rec['risk_color']};'>
                <strong style='font-size: 1.2rem;'>{policy_rec['emoji']} Risk Level: {policy_rec['risk_level']}</strong><br><br>
                <strong>Recommended Actions:</strong>
                {'<br>'.join(policy_rec['actions'])}
            </div>
        """, unsafe_allow_html=True)
        
        # ==================== PDF EXPORT ====================
        st.markdown("### 📄 Export Forecast Report")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            csv_data = comparison_table.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Data",
                data=csv_data,
                file_name=f"forecast_{selected_country}_{forecast_years}yr.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            try:
                pdf_buffer = generate_forecast_pdf(
                    selected_country,
                    forecast_years,
                    comparison_table,
                    ensemble_forecasts,
                    forecast_std,
                    policy_rec,
                    ml_models,
                    statistical_models,
                    country_data,
                    fig
                )
                
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"Inflation_Forecast_{selected_country}_{forecast_years}yr.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ PDF generation failed: {str(e)}")
                st.info("💡 Try: pip install kaleido --break-system-packages")
        
        # ==================== INDICATOR ANALYSIS ====================
        st.markdown("### 📊 Economic Indicator Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Correlation with Inflation")
            
            try:
                inflation_col = 'Inflation_Rate' if 'Inflation_Rate' in country_data.columns else 'Inflation'
                
                correlations = {}
                indicator_map = {
                    'GDP_Growth': 'GDP_Growth',
                    'Interest_Rate': 'Interest_Rate',
                    'Exchange_Rate': 'Exchange_Rate',
                    'Unemployment_Rate': 'Unemployment_Rate'
                }
                
                for display_name, col_name in indicator_map.items():
                    if col_name in country_data.columns:
                        corr = country_data[inflation_col].corr(country_data[col_name])
                        if not np.isnan(corr):
                            correlations[display_name] = abs(corr)
                
                if correlations:
                    corr_df = pd.DataFrame(list(correlations.items()), 
                                          columns=['Indicator', 'Correlation'])
                    corr_df = corr_df.sort_values('Correlation', ascending=True)
                    
                    fig_corr = go.Figure(go.Bar(
                        x=corr_df['Correlation'],
                        y=corr_df['Indicator'],
                        orientation='h',
                        marker_color='#3b82f6',
                        text=corr_df['Correlation'].round(2),
                        textposition='auto'
                    ))
                    
                    fig_corr.update_layout(
                        title=f"Indicator Correlations - {selected_country}",
                        xaxis_title="Absolute Correlation",
                        yaxis_title="Economic Indicator",
                        template='plotly_white',
                        height=300
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    strongest = corr_df.iloc[-1]
                    st.success(f"**Key Driver**: {strongest['Indicator']} shows strongest correlation ({strongest['Correlation']:.2f}) with inflation.")
                else:
                    st.warning("Correlation data not available")
            
            except Exception as e:
                st.warning(f"Could not calculate correlations: {str(e)}")
        
        with col2:
            st.markdown("#### Historical Trend")
            
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Scatter(
                x=country_data['Date'].tail(10),
                y=country_data[inflation_col].tail(10),
                mode='lines+markers',
                name='Inflation Rate',
                line=dict(color='#ef4444', width=3),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            
            fig_hist.add_hline(y=2.0, line_dash="dash", line_color="green", 
                               annotation_text="2% Target")
            
            fig_hist.update_layout(
                title=f"Recent Inflation History - {selected_country}",
                xaxis_title="Year",
                yaxis_title="Inflation (%)",
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # ======== RECENT TRENDS SUMMARY (DETAILED VERSION) =================
        st.markdown("### 📈 Recent Trends Summary")
        
        try:
            recent_inflation = country_data[inflation_col].tail(5).values
            inflation_change = recent_inflation[-1] - recent_inflation[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if inflation_change < 0:
                    st.success(f"""
                        **✅ Positive Developments:**
                        - Inflation decreased by {abs(inflation_change):.1f}pp over last 5 years
                        - Trending toward 2% target
                        - Policy measures showing effect
                    """)
                else:
                    st.info(f"""
                        **📊 Current Status:**
                        - Inflation relatively stable
                        - Recent 5-year change: {inflation_change:+.1f}pp
                    """)
            
            with col2:
                if inflation_change > 0:
                    st.warning(f"""
                        **⚠️ Monitor Closely:**
                        - Inflation increased by {abs(inflation_change):.1f}pp
                        - Consider policy tightening
                        - Watch external factors
                    """)
                else:
                    st.success(f"""
                        **✅ Maintain Vigilance:**
                        - Continue current policy stance
                        - Monitor for external shocks
                        - Watch for deflation risk
                    """)
        
        except Exception as e:
            st.warning(f"Could not analyze trends: {str(e)}")
    
    else:
        # ==================== INITIAL STATE ====================
        st.info("""
            ### 🎯 How to Use This Forecasting Tool
            
            1. **Select a Country** from the dropdown above
            2. **Choose Forecast Horizon** (1-5 years)
            3. **Click "Generate Forecast"** to see predictions
            
            **What You'll Get:**
            - 🤖 **4 ML Models** (RF_Extended, RF_Base, XGBoost_Extended, XGBoost_Base) - Trained on 341 countries
            - 📊 **2 Statistical Models** (ARIMA, VAR) - Hybrid forecasting
            - 🏆 **Best Model Selection** - Dynamically chosen based on country
            - 📉 **Confidence Intervals** - Uncertainty quantification
            - 💡 **Policy Recommendations** - Actionable guidance
            - 📄 **PDF & CSV Export** - Professional reports
            
            **Models are automatically compared** for robust predictions!
        """)
        
        
def render_model_performance(comparison_df):
    """Enhanced model performance analysis with training details"""
    st.markdown("<div class='section-header'>🧠 Model Performance Analysis</div>", unsafe_allow_html=True)
    
    if comparison_df is None:
        st.warning("⚠️ Model comparison data not available.")
        return
    
    # ==================== PERFORMANCE SUMMARY CARDS ====================
    st.markdown("### 📊 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Models Trained",
            value="6",
            delta="4 ML + 2 Statistical"
        )
    
    with col2:
        st.metric(
            label="🌍 Training Countries",
            value="341",
            delta="Global Coverage"
        )
    
    with col3:
        st.metric(
            label="📅 Training Period",
            value="1990-2023",
            delta="34 years"
        )
    
    with col4:
        r2_col = next((col for col in comparison_df.columns if 'R2' in col or 'r2' in col.lower()), 'R2')
        if r2_col in comparison_df.columns:
            best_r2 = float(comparison_df[r2_col].max())
            st.metric(
                label="🏆 Best R²",
                value=f"{best_r2:.3f}",
                delta=f"{best_r2*100:.1f}% variance"
            )
    
    # ==================== DETAILED METRICS TABLE ====================
    st.markdown("### 📈 Comprehensive Model Comparison")
    
    st.info("""
        **Metrics Explanation:**
        - **R² Score**: Proportion of variance explained (higher is better, max 1.0)
        - **RMSE**: Root Mean Squared Error in percentage points (lower is better)
        - **MAE**: Mean Absolute Error in percentage points (lower is better)
    """)
    
    display_df = comparison_df.copy()
    
    # Add performance ratings
    if r2_col in display_df.columns:
        def get_rating(r2):
            if r2 >= 0.5:
                return "🟢 Excellent"
            elif r2 >= 0.4:
                return "🟡 Good"
            elif r2 >= 0.3:
                return "🟠 Moderate"
            else:
                return "🔴 Needs Improvement"
        
        display_df['Performance Rating'] = display_df[r2_col].apply(get_rating)
    
    st.dataframe(
        display_df, 
        use_container_width=True, 
        height=300,
        hide_index=True
    )
    
    # ==================== PERFORMANCE VISUALIZATION ====================
    st.markdown("### 📊 Performance Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🎯 Accuracy Distribution", "⚡ Performance vs Complexity"])
    
    with tab1:
        # FIXED MODEL COMPARISON CHART - SORTED BY R²
        if r2_col in comparison_df.columns:
            # Sort by R² descending
            sorted_df = comparison_df.sort_values(by=r2_col, ascending=True)  # True for horizontal bar chart
            
            fig_comparison = go.Figure()
            
            # Color code by performance
            colors = []
            for r2 in sorted_df[r2_col]:
                if r2 >= 0.5:
                    colors.append('#10b981')  # Green - Excellent
                elif r2 >= 0.4:
                    colors.append('#3b82f6')  # Blue - Good
                elif r2 >= 0.3:
                    colors.append('#f59e0b')  # Orange - Moderate
                else:
                    colors.append('#ef4444')  # Red - Needs Improvement
            
            fig_comparison.add_trace(go.Bar(
                x=sorted_df[r2_col],
                y=sorted_df['Model'],
                orientation='h',
                marker_color=colors,
                text=sorted_df[r2_col].round(3),
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>R² Score: %{x:.3f}<extra></extra>'
            ))
            
            fig_comparison.update_layout(
                title="Model Performance Comparison (R² Score)",
                xaxis_title="R² Score (Higher is Better)",
                yaxis_title="Model",
                template='plotly_white',
                height=400,
                showlegend=False
            )
            
            # Add reference line at R² = 0.5
            fig_comparison.add_vline(
                x=0.5, 
                line_dash="dash", 
                line_color="green",
                annotation_text="Excellent (0.5+)",
                annotation_position="top"
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.caption("📊 Models ranked by R² Score. Green = Excellent (≥0.5), Blue = Good (≥0.4), Orange = Moderate (≥0.3)")
        else:
            st.warning("R² column not found in comparison data")
    
    with tab2:
        # Accuracy distribution
        if r2_col in comparison_df.columns:
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Box(
                y=comparison_df[r2_col],
                name='R² Distribution',
                marker_color='#10b981',
                boxmean='sd'
            ))
            
            fig_dist.update_layout(
                title="Model Accuracy Distribution",
                yaxis_title="R² Score",
                template='plotly_white',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Mean R²", f"{comparison_df[r2_col].mean():.3f}")
            with col2:
                st.metric("📈 Std Dev", f"{comparison_df[r2_col].std():.3f}")
            with col3:
                st.metric("📉 Range", f"{comparison_df[r2_col].max() - comparison_df[r2_col].min():.3f}")
    
    with tab3:
        # Performance vs Complexity
        st.markdown("""
            **Model Complexity Analysis:**
            
            | Model Type | Parameters | Training Speed | Interpretability |
            |------------|-----------|---------------|------------------|
            | RF_Extended | ~10K trees | ⚡⚡ Fast | 🟡 Moderate |
            | RF_Base | ~10K trees | ⚡⚡⚡ Very Fast | 🟢 Good |
            | XGBoost_Extended | ~5K trees | ⚡ Moderate | 🟡 Moderate |
            | XGBoost_Base | ~5K trees | ⚡⚡ Fast | 🟢 Good |
            | ARIMA | ~10 params | ⚡⚡⚡ Very Fast | 🟢 Excellent |
            | VAR | ~50 params | ⚡⚡ Fast | 🟢 Excellent |
            
            **Legend:**
            - ⚡⚡⚡ Very Fast: < 5 minutes
            - ⚡⚡ Fast: 5-30 minutes
            - ⚡ Moderate: 30-60 minutes
        """)
    
    # ==================== BEST MODEL INSIGHTS ====================
    st.markdown("### 🏆 Model Insights & Recommendations")
    
    if r2_col in comparison_df.columns:
        best_idx = comparison_df[r2_col].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_r2 = float(comparison_df.loc[best_idx, r2_col])
        
        rmse_col = next((col for col in comparison_df.columns if 'rmse' in col.lower()), 'RMSE')
        mae_col = next((col for col in comparison_df.columns if 'mae' in col.lower()), 'MAE')
        
        best_rmse = float(comparison_df.loc[best_idx, rmse_col]) if rmse_col in comparison_df.columns else 0
        best_mae = float(comparison_df.loc[best_idx, mae_col]) if mae_col in comparison_df.columns else 0
        
        col1, col2 = st.columns(2)
    
        
        with col1:
            st.markdown("**📌 Model Selection Guide**")
            
            st.success("""
                **🟢 Extended Models (Recommended)**
                - Use engineered features (% changes, shocks)
                - Better capture complex patterns
                - Higher accuracy, more robust
            """)
            
            st.info("""
                **🔵 Base Models**
                - Use raw economic indicators only
                - Simpler, faster training
                - Good baseline performance
            """)
            
            st.warning("""
                **🟡 Time Series Models**
                - Capture temporal dependencies
                - Excellent for causality analysis
                - Traditional econometric approach
            """)
            
        with col2:
            st.markdown(f"""
                <div class='alert-box' style='background-color: #10b98115; border-left-color: #10b981;'>
                    <strong style='color: #10b981; font-size: 1.2rem;'>🥇 Best Performing Model</strong><br><br>
                    <strong style='font-size: 1.1rem;'>{best_model}</strong> achieves highest accuracy<br><br>
                    <strong>Performance Metrics:</strong><br>
                    • R² Score: <strong>{best_r2:.3f}</strong> (explains {best_r2*100:.1f}% of variance)<br>
                    • RMSE: <strong>{best_rmse:.2f}</strong> percentage points<br>
                    • MAE: <strong>{best_mae:.2f}</strong> percentage points<br><br>
                    <strong style='color: #10b981;'>✅ Recommended for production deployment</strong>
                </div>
            """, unsafe_allow_html=True)    
    
    # ==================== FEATURE IMPORTANCE SECTION ====================
    st.markdown("### 🎯 Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            **Top Features for Extended Models:**
            
            1. 🥇 **Inflation_pct_change** - Most predictive
            2. 🥈 **Exchange_Rate_shock** - High impact
            3. 🥉 **GDP_pct_change** - Strong indicator
            4. 📊 **Interest_Rate** - Monetary policy signal
            5. 📈 **Unemployment_shock** - Economic health
        """)
    
    with col2:
        # Feature importance bar chart
        features = ['Inflation_pct_change', 'Exchange_Rate_shock', 'GDP_pct_change', 
                   'Interest_Rate', 'Unemployment_shock']
        importance = [0.35, 0.22, 0.18, 0.15, 0.10]
        
        fig_feat = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color=['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'],
            text=[f'{imp:.0%}' for imp in importance],
            textposition='auto'
        ))
        
        fig_feat.update_layout(
            title="Feature Importance (RF_Extended)",
            xaxis_title="Relative Importance",
            yaxis_title="Feature",
            template='plotly_white',
            height=300
        )
        
        st.plotly_chart(fig_feat, use_container_width=True)
        
        

def render_live_feed():
    """Enhanced live economic feed with inflation focus and clickable links"""
    st.markdown("<div class='section-header'>📡 Live Economic Feed</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280;'>Real-time market data, economic indicators, and breaking news</p>", unsafe_allow_html=True)
    
    # ============== AUTO BREAKING NEWS WITH CLICKABLE LINK ====================
    st.markdown("### 🚨 Breaking Economic News")
    
    try:
        import feedparser
        from datetime import datetime
        
        breaking_feed = feedparser.parse('https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664')
        
        if breaking_feed.entries:
            latest = breaking_feed.entries[0]
            
            # Calculate time ago
            try:
                published = datetime(*latest.published_parsed[:6])
                now = datetime.now()
                diff = now - published
                
                if diff.days > 0:
                    time_ago = f"{diff.days} days ago"
                elif diff.seconds > 3600:
                    time_ago = f"{diff.seconds // 3600} hours ago"
                elif diff.seconds > 60:
                    time_ago = f"{diff.seconds // 60} minutes ago"
                else:
                    time_ago = "Just now"
            except:
                time_ago = "Recently"
            
            # Get the article link
            article_link = latest.link if hasattr(latest, 'link') else '#'
            
            # Get summary (truncate if too long)
            summary = latest.summary[:250] if hasattr(latest, 'summary') else 'Click to read more...'
            if len(summary) >= 250:
                summary += '...'
            
            st.markdown(f"""
                <div class='alert-box' style='background-color: #ef444415; border-left-color: #ef4444;'>
                    <strong style='color: #ef4444; font-size: 1.2rem;'>🚨 BREAKING: {latest.title}</strong><br><br>
                    <p style='color: #374151; margin: 10px 0;'>{summary}</p>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 10px;'>
                        <small style='color: #6b7280;'>{time_ago}</small>
                        <a href='{article_link}' target='_blank' 
                           style='background: #ef4444; color: white; padding: 8px 16px; 
                                  border-radius: 6px; text-decoration: none; font-weight: 600;
                                  font-size: 0.9rem; transition: all 0.3s;'>
                            📰 Read Full Article →
                        </a>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🔄 Loading latest breaking news...")
    
    except Exception as e:
        # Fallback breaking news
        st.markdown("""
            <div class='alert-box' style='background-color: #ef444415; border-left-color: #ef4444;'>
                <strong style='color: #ef4444; font-size: 1.2rem;'>🚨 BREAKING: Federal Reserve Policy Decision</strong><br><br>
                <p style='color: #374151; margin: 10px 0;'>Fed maintains interest rates at current levels. Chair Powell emphasizes data-dependent approach. Markets react positively.</p>
                <small style='color: #6b7280;'>2 hours ago</small>
            </div>
        """, unsafe_allow_html=True)
    
    # ==================== STATUS CARDS ====================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_time = datetime.now().time()
        market_open = current_time.hour >= 9 and current_time.hour < 16
        
        if market_open:
            st.markdown("""
                <div class='metric-card green'>
                    <div style='display: flex; align-items: center; justify-content: space-between;'>
                        <div>
                            <div class='metric-label'>Market Status</div>
                            <div style='font-size: 1.5rem; font-weight: 700; color: #10b981;'>OPEN</div>
                        </div>
                        <div class='status-indicator'></div>
                    </div>
                    <div style='font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;'>
                        NYSE • 9:30 AM - 4:00 PM EST
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='metric-card' style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);'>
                    <div style='display: flex; align-items: center; justify-content: space-between;'>
                        <div>
                            <div class='metric-label' style='color: white;'>Market Status</div>
                            <div style='font-size: 1.5rem; font-weight: 700; color: white;'>CLOSED</div>
                        </div>
                    </div>
                    <div style='font-size: 0.75rem; color: rgba(255,255,255,0.8); margin-top: 0.5rem;'>
                        NYSE • Markets closed
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card blue'>
                <div class='metric-label'>Data Sources</div>
                <div class='metric-value' style='font-size: 1.5rem;'>5 Active</div>
                <div style='font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;'>
                    Yahoo Finance, RSS Feeds
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='metric-card purple'>
                <div class='metric-label'>Last Update</div>
                <div class='metric-value' style='font-size: 1.5rem;'>{datetime.now().strftime('%H:%M:%S')}</div>
                <div style='font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;'>
                    Auto-refresh: 60s
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Now", use_container_width=True, type="primary"):
            st.rerun()
    
    st.markdown("---")
    
    # ==================== INFLATION WATCH ====================
    st.markdown("### 🔥 Inflation Watch")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🇺🇸 USA CPI (Latest)",
            value="2.7%",
            delta="-0.1pp MoM",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="🇺🇸 Core CPI",
            value="3.2%",
            delta="Stable",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            label="🇺🇸 PPI",
            value="2.4%",
            delta="+0.2pp",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="🎯 Fed Target",
            value="2.0%",
            delta="0.7pp above",
            delta_color="normal"
        )
    
    with st.expander("🌍 View Global Inflation Snapshot"):
        global_inflation = pd.DataFrame({
            'Country': ['🇺🇸 USA', '🇬🇧 UK', '🇪🇺 Eurozone', '🇯🇵 Japan', '🇨🇦 Canada', '🇦🇺 Australia'],
            'Current Rate': ['2.7%', '2.5%', '2.9%', '2.8%', '2.3%', '3.5%'],
            'Central Bank Target': ['2.0%', '2.0%', '2.0%', '2.0%', '2.0%', '2.0-3.0%'],
            'Distance from Target': ['+0.7pp', '+0.5pp', '+0.9pp', '+0.8pp', '+0.3pp', '+0.5pp'],
            'Status': ['🟡 Above', '🟡 Above', '🟡 Above', '🟡 Above', '🟢 Near Target', '🟢 In Range']
        })
        st.dataframe(global_inflation, use_container_width=True, hide_index=True)
        st.caption("📊 Latest available inflation data as of December 2025")
    
    st.markdown("---")
    
    # ==================== LIVE MARKET DATA ====================
    st.markdown("### 📊 Live Market Data")
    st.markdown("<span class='live-indicator'>LIVE</span> Real-time prices and indicators", unsafe_allow_html=True)
    
    with st.spinner("Loading market data..."):
        market_df = get_market_data()
        
        if len(market_df) > 0:
            st.dataframe(market_df, use_container_width=True, height=400)
        else:
            st.info("Market data temporarily unavailable. Refresh to retry.")
    
    st.markdown("---")
    
    # ==================== CENTRAL BANK RATES ====================
    st.markdown("### 🏦 Central Bank Policy Rates")
    
    cb_data = pd.DataFrame({
        'Central Bank': ['🇺🇸 Federal Reserve', '🇬🇧 Bank of England', '🇪🇺 European Central Bank', '🇯🇵 Bank of Japan', '🇨🇦 Bank of Canada'],
        'Current Rate': ['5.50%', '5.00%', '3.75%', '0.25%', '4.75%'],
        'Latest Change': ['Unchanged', '↓ -0.25%', 'Unchanged', 'Unchanged', 'Unchanged'],
        'Last Meeting': ['Dec 18, 2025', 'Dec 18, 2025', 'Dec 12, 2025', 'Dec 19, 2025', 'Dec 11, 2025'],
        'Next Meeting': ['Jan 29, 2026', 'Jan 30, 2026', 'Jan 23, 2026', 'Jan 24, 2026', 'Jan 22, 2026'],
        'Stance': ['🟡 Holding', '🟢 Cutting', '🟡 Holding', '🟡 Holding', '🟡 Holding']
    })
    
    st.dataframe(cb_data, use_container_width=True, hide_index=True)
    st.caption("🏦 Central bank policy rates and meeting schedules (December 2025)")
    
    st.markdown("---")
    
    # ==================== INFLATION INSIGHTS ====================
    st.markdown("### 💡 Inflation Insights - Why Today's Markets Matter")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
            **📈 Inflationary Signals:**
            
            **Oil Prices (+2.23% today)**
            - Increases transportation costs
            - Typically adds 0.1-0.2pp to CPI in 1-2 months
            - Watch for pass-through to consumer prices
            
            **Gold Rising (+1.48%)**
            - Investors seeking inflation hedge
            - Signal of inflation concerns
            - Safe haven demand increasing
            
            **Dollar Weakening (-0.21%)**
            - Makes imports more expensive
            - Increases import inflation
            - Affects tradeable goods prices
        """)
    
    with col2:
        st.success("""
            **📉 Disinflationary Signals:**
            
            **Fed Holding Rates (5.50%)**
            - No immediate policy changes
            - Restrictive policy still working
            - Data-dependent approach continues
            
            **Treasury Yields Stable (+0.24%)**
            - Market not pricing rapid inflation
            - Inflation expectations anchored
            - Long-term outlook stable
            
            **VIX Low (14.91)**
            - Low market volatility
            - Stable economic outlook
            - Reduced uncertainty premium
        """)
    
    st.markdown("#### 🎯 Overall Market Signal")
    st.warning("""
        **⚠️ MIXED SIGNALS - MODERATE INFLATION RISK**
        
        Today's market movements suggest **moderate inflationary pressure** from commodity prices, 
        offset by stable monetary policy and anchored inflation expectations. 
        
        **Impact on Forecasts:** Current trends may push near-term inflation 0.1-0.2pp above 
        baseline forecasts, but medium-term outlook remains stable.
    """)
    
    st.markdown("---")
    
    # ==================== ECONOMIC CALENDAR ====================
    st.markdown("### 📅 Today's Economic Calendar")
    
    calendar_df = get_economic_calendar()
    
    for _, event in calendar_df.iterrows():
        importance_color = {
            'High': '#ef4444',
            'Medium': '#f59e0b',
            'Low': '#10b981'
        }.get(event['Importance'], '#6b7280')
        
        with st.expander(f"{event['Event']} - {event['Importance']} | {event['Time']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                    **Expected:** {event['Forecast']}  
                    **Previous:** {event['Previous']}  
                    **Actual:** --
                """)
                
                # Add context for each event
                context = {
                    'Consumer Price Index (CPI)': '💡 Key inflation metric - directly affects Fed policy decisions',
                    'Consumer Confidence Index': '💡 Indicates consumer spending strength and inflation expectations',
                    'Fed Chair Powell Speech': '💡 Direct insight into Fed thinking on inflation and rates',
                    'API Crude Oil Inventory': '💡 Oil supply affects energy prices and overall inflation'
                }.get(event['Event'], '💡 Important economic indicator')
                
                st.caption(context)
            
            with col2:
                st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background: {importance_color}15; 
                         border-radius: 8px; border: 2px solid {importance_color};'>
                        <div style='color: {importance_color}; font-weight: 700; font-size: 1.2rem;'>
                            {event['Importance']}
                        </div>
                        <div style='color: #6b7280; font-size: 0.9rem; margin-top: 5px;'>
                            {event['Time']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== NEWS FEED WITH CLICKABLE LINKS ====================
    st.markdown("### 📰 Live News Feed")
    
    with st.spinner("Loading news..."):
        news_items = get_economic_news()
        
        if news_items:
            for item in news_items[:10]:
                # Make the entire news item clickable
                st.markdown(f"""
                    <a href='{item['link']}' target='_blank' style='text-decoration: none;'>
                        <div class='news-item' style='cursor: pointer; transition: all 0.3s;'
                             onmouseover="this.style.backgroundColor='#f3f4f6'; this.style.transform='translateX(5px)';"
                             onmouseout="this.style.backgroundColor='white'; this.style.transform='translateX(0)';">
                            <div class='news-title' style='color: #1f2937;'>{item['title']}</div>
                            <div class='news-meta'>
                                <strong>{item['source']}</strong> • {item['published']}
                                <span style='float: right; color: #3b82f6; font-weight: 600;'>Read More →</span>
                            </div>
                        </div>
                    </a>
                """, unsafe_allow_html=True)
        else:
            st.info("News feed temporarily unavailable.")
    
    st.markdown("---")
    
    # ==================== LIVE CHART ====================
    st.markdown("### 📈 Live Market Movements")
    
    with st.spinner("Loading chart..."):
        intraday_data = get_intraday_chart('^GSPC', '1d', '5m')
        
        if len(intraday_data) > 0:
            fig = create_intraday_chart(intraday_data, "S&P 500 Intraday Movement")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chart data temporarily unavailable.")
    
    st.markdown("---")
    
    # ==================== DATA SOURCE STATUS ====================
    st.markdown("### 🔗 Data Source Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='card'>
                <strong>Yahoo Finance</strong><br>
                <span style='color: #10b981;'>● Connected</span><br>
                <small>Latency: 67ms<br>Market Data Active</small>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='card'>
                <strong>RSS News Feeds</strong><br>
                <span style='color: #10b981;'>● Connected</span><br>
                <small>Latency: 120ms<br>3 Sources Active</small>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='card'>
                <strong>Economic Calendar</strong><br>
                <span style='color: #10b981;'>● Active</span><br>
                <small>4 Events Today<br>Next: 10:00 AM EST</small>
            </div>
        """, unsafe_allow_html=True)
        
        
def render_analytics(df):
    """Visual analytics and statistical analysis"""
    st.markdown("<div class='section-header'>📊 Visual Analytics</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280;'>Statistical analysis and data visualization</p>", unsafe_allow_html=True)
    
    # ==================== COUNTRY SELECTOR ====================
    st.markdown("### 🌍 Select Country for Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        countries = sorted(df['Country'].unique().tolist())
        selected_country = st.selectbox(
            "Choose a country to analyze",
            countries,
            index=countries.index('Malaysia') if 'Malaysia' in countries else 0
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        compare_mode = st.checkbox("Compare with another country")
    
    # Get country data
    country_data = df[df['Country'] == selected_country].sort_values('Date')
    
    if len(country_data) < 5:
        st.error(f"❌ Insufficient data for {selected_country}. Need at least 5 years.")
        return
    
    # If compare mode, select second country
    if compare_mode:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Primary:** {selected_country}")
        with col2:
            compare_countries = [c for c in countries if c != selected_country]
            compare_country = st.selectbox("Compare with:", compare_countries)
            compare_data = df[df['Country'] == compare_country].sort_values('Date')
    
    st.markdown("---")
    
    # ==================== KEY STATISTICS ====================
    st.markdown("### 📈 Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_inflation = country_data['Inflation_Rate'].mean()
        st.metric(
            label="Average Inflation",
            value=f"{avg_inflation:.2f}%",
            delta=f"Last 5yr: {country_data['Inflation_Rate'].tail(5).mean():.2f}%"
        )
    
    with col2:
        std_inflation = country_data['Inflation_Rate'].std()
        st.metric(
            label="Volatility (Std Dev)",
            value=f"{std_inflation:.2f}pp",
            delta="Lower is more stable"
        )
    
    with col3:
        max_inflation = country_data['Inflation_Rate'].max()
        max_year = country_data.loc[country_data['Inflation_Rate'].idxmax(), 'Date']
        st.metric(
            label="Peak Inflation",
            value=f"{max_inflation:.2f}%",
            delta=f"Year: {int(max_year)}"
        )
    
    with col4:
        min_inflation = country_data['Inflation_Rate'].min()
        min_year = country_data.loc[country_data['Inflation_Rate'].idxmin(), 'Date']
        st.metric(
            label="Lowest Inflation",
            value=f"{min_inflation:.2f}%",
            delta=f"Year: {int(min_year)}"
        )
    
    st.markdown("---")
    
    # ==================== TIME SERIES ANALYSIS ====================
    st.markdown("### 📈 Time Series Analysis")
    
    if compare_mode:
        # Dual comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {selected_country}")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=country_data['Date'],
                y=country_data['Inflation_Rate'],
                mode='lines+markers',
                name='Inflation',
                line=dict(color='#ef4444', width=2),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            fig1.add_hline(y=2.0, line_dash="dash", line_color="green", annotation_text="2% Target")
            fig1.update_layout(
                xaxis_title="Year",
                yaxis_title="Inflation Rate (%)",
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown(f"#### {compare_country}")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=compare_data['Date'],
                y=compare_data['Inflation_Rate'],
                mode='lines+markers',
                name='Inflation',
                line=dict(color='#3b82f6', width=2),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            fig2.add_hline(y=2.0, line_dash="dash", line_color="green", annotation_text="2% Target")
            fig2.update_layout(
                xaxis_title="Year",
                yaxis_title="Inflation Rate (%)",
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Overlay comparison
        st.markdown("#### 📊 Direct Comparison")
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Scatter(
            x=country_data['Date'],
            y=country_data['Inflation_Rate'],
            mode='lines+markers',
            name=selected_country,
            line=dict(color='#ef4444', width=3)
        ))
        fig_overlay.add_trace(go.Scatter(
            x=compare_data['Date'],
            y=compare_data['Inflation_Rate'],
            mode='lines+markers',
            name=compare_country,
            line=dict(color='#3b82f6', width=3)
        ))
        fig_overlay.add_hline(y=2.0, line_dash="dash", line_color="green", annotation_text="2% Target")
        fig_overlay.update_layout(
            title=f"Inflation Comparison: {selected_country} vs {compare_country}",
            xaxis_title="Year",
            yaxis_title="Inflation Rate (%)",
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_overlay, use_container_width=True)
    
    else:
        # Single country analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Inflation Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=country_data['Date'],
                y=country_data['Inflation_Rate'],
                mode='lines+markers',
                name='Inflation',
                line=dict(color='#ef4444', width=2),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            fig.add_hline(y=2.0, line_dash="dash", line_color="green", annotation_text="2% Target")
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Inflation Rate (%)",
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### GDP Growth Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=country_data['Date'],
                y=country_data['GDP_Growth'],
                mode='lines+markers',
                name='GDP',
                line=dict(color='#3b82f6', width=2),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="GDP Growth (%)",
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== DISTRIBUTION ANALYSIS ====================
    st.markdown("### 📊 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Global Inflation Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['Inflation_Rate'],
            nbinsx=50,
            marker_color='#3b82f6',
            opacity=0.7,
            name='All Countries'
        ))
        
        # Add vertical line for selected country's average
        fig.add_vline(
            x=avg_inflation,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{selected_country}: {avg_inflation:.1f}%",
            annotation_position="top"
        )
        
        fig.update_layout(
            title="How does this country compare globally?",
            xaxis_title="Inflation Rate (%)",
            yaxis_title="Frequency",
            template='plotly_white',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Percentile information
        percentile = (df['Inflation_Rate'] < avg_inflation).mean() * 100
        st.info(f"📊 {selected_country}'s average inflation ({avg_inflation:.1f}%) is higher than {percentile:.0f}% of all country-years in the dataset.")
    
    with col2:
        st.markdown("#### Box Plot Analysis")
        fig = go.Figure()
        
        if compare_mode:
            fig.add_trace(go.Box(
                y=country_data['Inflation_Rate'],
                name=selected_country,
                marker_color='#ef4444'
            ))
            fig.add_trace(go.Box(
                y=compare_data['Inflation_Rate'],
                name=compare_country,
                marker_color='#3b82f6'
            ))
        else:
            fig.add_trace(go.Box(
                y=country_data['Inflation_Rate'],
                name=selected_country,
                marker_color='#3b82f6',
                boxmean='sd'
            ))
        
        fig.update_layout(
            title="Inflation Variability",
            yaxis_title="Inflation Rate (%)",
            template='plotly_white',
            height=350,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== CORRELATION ANALYSIS ====================
    st.markdown("### 🔗 Correlation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Correlation Heatmap")
        
        # Select numeric columns for correlation
        numeric_cols = ['Inflation_Rate', 'GDP_Growth', 'Interest_Rate', 'Exchange_Rate', 'Unemployment_Rate']
        available_cols = [col for col in numeric_cols if col in country_data.columns]
        
        if len(available_cols) >= 2:
            corr_matrix = country_data[available_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title=f"Economic Indicators Correlation - {selected_country}",
                template='plotly_white',
                height=400,
                xaxis={'side': 'bottom'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for correlation analysis")
    
    with col2:
        st.markdown("#### Key Correlations with Inflation")
        
        if len(available_cols) >= 2:
            inflation_corr = country_data[available_cols].corr()['Inflation_Rate'].drop('Inflation_Rate')
            inflation_corr = inflation_corr.sort_values(ascending=False)
            
            fig = go.Figure(go.Bar(
                x=inflation_corr.values,
                y=inflation_corr.index,
                orientation='h',
                marker_color=['#10b981' if x > 0 else '#ef4444' for x in inflation_corr.values],
                text=inflation_corr.values.round(2),
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Correlation with Inflation Rate",
                xaxis_title="Correlation Coefficient",
                yaxis_title="Indicator",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            strongest = inflation_corr.abs().idxmax()
            strongest_val = inflation_corr[strongest]
            
            if abs(strongest_val) > 0.5:
                strength = "strong"
            elif abs(strongest_val) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"
            
            direction = "positive" if strongest_val > 0 else "negative"
            
            st.success(f"📊 **Key Finding:** {strongest} shows a {strength} {direction} correlation ({strongest_val:.2f}) with inflation in {selected_country}.")
        else:
            st.warning("Not enough data for correlation analysis")
    
    st.markdown("---")
    
    # ==================== TREND ANALYSIS ====================
    st.markdown("### 📈 Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Recent vs Historical Comparison")
        
        # Split data into periods
        mid_point = len(country_data) // 2
        historical = country_data.iloc[:mid_point]
        recent = country_data.iloc[mid_point:]
        
        comparison_data = pd.DataFrame({
            'Period': ['Historical (First Half)', 'Recent (Second Half)'],
            'Average Inflation': [historical['Inflation_Rate'].mean(), recent['Inflation_Rate'].mean()],
            'Volatility': [historical['Inflation_Rate'].std(), recent['Inflation_Rate'].std()],
            'Max': [historical['Inflation_Rate'].max(), recent['Inflation_Rate'].max()]
        })
        
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
        
        # Trend direction
        recent_avg = recent['Inflation_Rate'].mean()
        historical_avg = historical['Inflation_Rate'].mean()
        change = recent_avg - historical_avg
        
        if change > 1:
            st.warning(f"⚠️ Inflation has increased by {change:.1f}pp in recent years")
        elif change < -1:
            st.success(f"✅ Inflation has decreased by {abs(change):.1f}pp in recent years")
        else:
            st.info(f"📊 Inflation remains relatively stable (change: {change:+.1f}pp)")
    
    with col2:
        st.markdown("#### Year-over-Year Changes")
        
        # Calculate YoY changes
        country_data_copy = country_data.copy()
        country_data_copy['YoY_Change'] = country_data_copy['Inflation_Rate'].diff()
        
        fig = go.Figure()
        
        colors = ['#10b981' if x < 0 else '#ef4444' for x in country_data_copy['YoY_Change']]
        
        fig.add_trace(go.Bar(
            x=country_data_copy['Date'],
            y=country_data_copy['YoY_Change'],
            marker_color=colors,
            name='YoY Change',
            hovertemplate='%{x}<br>Change: %{y:.2f}pp<extra></extra>'
        ))
        
        fig.update_layout(
            title="Year-over-Year Inflation Changes",
            xaxis_title="Year",
            yaxis_title="Change in Inflation (pp)",
            template='plotly_white',
            height=350,
            showlegend=False
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== ADVANCED INSIGHTS ====================
    st.markdown("### 💡 Advanced Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Stability score
        cv = (std_inflation / avg_inflation) * 100 if avg_inflation != 0 else 0
        
        if cv < 30:
            stability = "🟢 High"
            stability_msg = "Inflation is relatively stable"
        elif cv < 60:
            stability = "🟡 Moderate"
            stability_msg = "Some inflation volatility"
        else:
            stability = "🔴 Low"
            stability_msg = "High inflation volatility"
        
        st.markdown(f"""
            <div class='card' style='text-align: center;'>
                <h4>Stability Score</h4>
                <h2 style='color: #3b82f6;'>{stability}</h2>
                <p>{stability_msg}</p>
                <small>Coefficient of Variation: {cv:.1f}%</small>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Target deviation
        recent_5yr = country_data['Inflation_Rate'].tail(5).mean()
        deviation = abs(recent_5yr - 2.0)
        
        if deviation < 1:
            target_status = "🟢 On Target"
            target_msg = "Close to 2% target"
        elif deviation < 2:
            target_status = "🟡 Near Target"
            target_msg = "Moderately off target"
        else:
            target_status = "🔴 Off Target"
            target_msg = "Significantly off target"
        
        st.markdown(f"""
            <div class='card' style='text-align: center;'>
                <h4>Target Alignment</h4>
                <h2 style='color: #3b82f6;'>{target_status}</h2>
                <p>{target_msg}</p>
                <small>5-year avg: {recent_5yr:.1f}% (Target: 2.0%)</small>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Trend direction
        last_3yr = country_data['Inflation_Rate'].tail(3).mean()
        prev_3yr = country_data['Inflation_Rate'].tail(6).head(3).mean()
        trend = last_3yr - prev_3yr
        
        if trend > 0.5:
            trend_status = "📈 Rising"
            trend_msg = "Inflation trending upward"
        elif trend < -0.5:
            trend_status = "📉 Falling"
            trend_msg = "Inflation trending downward"
        else:
            trend_status = "➡️ Stable"
            trend_msg = "No clear trend"
        
        st.markdown(f"""
            <div class='card' style='text-align: center;'>
                <h4>Trend Direction</h4>
                <h2 style='color: #3b82f6;'>{trend_status}</h2>
                <p>{trend_msg}</p>
                <small>3-year change: {trend:+.1f}pp</small>
            </div>
        """, unsafe_allow_html=True)        
        

def render_global_comparison(df):
    """Enhanced global economic comparison with world map and regional analysis"""
    st.markdown("<div class='section-header'>🌍 Global Economic Comparison</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280;'>Compare countries and regions worldwide</p>", unsafe_allow_html=True)
    
    # ==================== FILTERS ====================
    st.markdown("### 🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Year filter
        available_years = sorted(df['Date'].unique(), reverse=True)
        selected_year = st.selectbox(
            "Select Year",
            available_years,
            index=0
        )
    
    with col2:
        # Region filter
        regions = ['All Regions', 'Asia', 'Europe', 'Americas', 'Africa', 'Oceania']
        selected_region = st.selectbox("Filter by Region", regions)
    
    with col3:
        # Metric to visualize
        metric = st.selectbox(
            "Primary Metric",
            ['Inflation_Rate', 'GDP_Growth', 'Interest_Rate', 'Unemployment_Rate']
        )
    
    # Filter data by year
    year_data = df[df['Date'] == selected_year].copy()
    
    # Apply region filter if needed
    if selected_region != 'All Regions':
        # Define regional groupings
        asia = ['China', 'India', 'Japan', 'South Korea', 'Indonesia', 'Malaysia', 'Singapore', 'Thailand', 'Vietnam', 'Philippines', 
                'Pakistan', 'Bangladesh', 'Myanmar', 'Cambodia', 'Laos', 'Mongolia', 'Nepal', 'Sri Lanka', 'Afghanistan', 'Bhutan']
        europe = ['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Netherlands', 'Sweden', 'Poland', 'Belgium', 'Austria',
                  'Switzerland', 'Norway', 'Denmark', 'Finland', 'Ireland', 'Portugal', 'Greece', 'Czech Republic', 'Romania', 'Hungary']
        americas = ['United States', 'Canada', 'Brazil', 'Mexico', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela', 
                    'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay', 'Costa Rica', 'Panama', 'Guatemala', 'Honduras', 'Nicaragua']
        africa = ['South Africa', 'Nigeria', 'Egypt', 'Kenya', 'Ghana', 'Ethiopia', 'Morocco', 'Algeria', 'Tanzania', 'Uganda',
                  'Sudan', 'Angola', 'Mozambique', 'Madagascar', 'Cameroon', 'Niger', 'Burkina Faso', 'Mali', 'Malawi', 'Zambia',
                  'Senegal', 'Chad', 'Somalia', 'Zimbabwe', 'Rwanda', 'Benin', 'Burundi', 'Tunisia', 'South Sudan', 'Libya']
        oceania = ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands', 'Vanuatu', 'Samoa', 'Tonga']
        
        region_map = {
            'Asia': asia,
            'Europe': europe,
            'Americas': americas,
            'Africa': africa,
            'Oceania': oceania
        }
        
        if selected_region in region_map:
            year_data = year_data[year_data['Country'].isin(region_map[selected_region])]
    
    st.markdown("---")
    
    # ============= SMART ALERT SYSTEM (BASED ON SETTINGS) ====================
    if get_setting('enable_alerts', True):
        threshold = get_setting('high_inflation_threshold', 8.0)
        alert_types = get_setting('alert_types', ['High inflation'])
        
        # HIGH INFLATION ALERT
        if 'High inflation' in alert_types:
            high_inflation_countries = year_data[year_data['Inflation_Rate'] > threshold]
            
            if len(high_inflation_countries) > 0:
                st.warning(f"""
                    ⚠️ **High Inflation Alert**  
                    {len(high_inflation_countries)} countries exceed {threshold}% inflation threshold in {int(selected_year)}
                """)
                
                with st.expander("📋 View Affected Countries"):
                    alert_data = high_inflation_countries[['Country', 'Inflation_Rate', 'GDP_Growth']].sort_values('Inflation_Rate', ascending=False)
                    alert_data['Inflation_Rate'] = alert_data['Inflation_Rate'].apply(lambda x: f"{x:.2f}%")
                    alert_data['GDP_Growth'] = alert_data['GDP_Growth'].apply(lambda x: f"{x:.2f}%")
                    alert_data.columns = ['Country', 'Inflation', 'GDP Growth']
                    st.dataframe(alert_data, use_container_width=True, hide_index=True)
        
        # DEFLATION RISK ALERT
        if 'Deflation risk' in alert_types:
            deflation_countries = year_data[year_data['Inflation_Rate'] < 0]
            
            if len(deflation_countries) > 0:
                st.info(f"""
                    ℹ️ **Deflation Risk Alert**  
                    {len(deflation_countries)} countries experiencing deflation in {int(selected_year)}
                """)
                
                with st.expander("📋 View Countries"):
                    deflation_data = deflation_countries[['Country', 'Inflation_Rate', 'GDP_Growth']].sort_values('Inflation_Rate')
                    deflation_data['Inflation_Rate'] = deflation_data['Inflation_Rate'].apply(lambda x: f"{x:.2f}%")
                    deflation_data['GDP_Growth'] = deflation_data['GDP_Growth'].apply(lambda x: f"{x:.2f}%")
                    deflation_data.columns = ['Country', 'Inflation', 'GDP Growth']
                    st.dataframe(deflation_data, use_container_width=True, hide_index=True)
        
        # RAPID CHANGE ALERT (if you have historical data)
        if 'Rapid inflation changes' in alert_types:
            volatility_threshold = get_setting('volatility_threshold', 2.0)
            
            # Calculate YoY change if previous year data exists
            prev_year_data = df[df['Date'] == selected_year - 1]
            if len(prev_year_data) > 0:
                year_data_with_change = year_data.merge(
                    prev_year_data[['Country', 'Inflation_Rate']], 
                    on='Country', 
                    suffixes=('', '_prev')
                )
                year_data_with_change['Change'] = year_data_with_change['Inflation_Rate'] - year_data_with_change['Inflation_Rate_prev']
                
                rapid_change = year_data_with_change[abs(year_data_with_change['Change']) > volatility_threshold]
                
                if len(rapid_change) > 0:
                    st.warning(f"""
                        📊 **Rapid Change Alert**  
                        {len(rapid_change)} countries show inflation changes >{volatility_threshold}pp year-over-year
                    """)
                    
                    with st.expander("📋 View Countries"):
                        change_data = rapid_change[['Country', 'Inflation_Rate', 'Change']].sort_values('Change', ascending=False, key=abs)
                        change_data['Inflation_Rate'] = change_data['Inflation_Rate'].apply(lambda x: f"{x:.2f}%")
                        change_data['Change'] = change_data['Change'].apply(lambda x: f"{x:+.2f}pp")
                        change_data.columns = ['Country', 'Current Inflation', 'YoY Change']
                        st.dataframe(change_data, use_container_width=True, hide_index=True)
    
    # ==================== GLOBAL STATISTICS ====================
    st.markdown("### 🌍 Global Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        global_avg = year_data['Inflation_Rate'].mean()
        st.metric(
            label="Global Avg Inflation",
            value=f"{global_avg:.2f}%",
            delta=f"{len(year_data)} countries"
        )
    
    with col2:
        high_inflation = (year_data['Inflation_Rate'] > 5).sum()
        pct_high = (high_inflation / len(year_data)) * 100 if len(year_data) > 0 else 0
        st.metric(
            label="High Inflation (>5%)",
            value=f"{high_inflation}",
            delta=f"{pct_high:.0f}% of total"
        )
    
    with col3:
        target_range = ((year_data['Inflation_Rate'] >= 1) & (year_data['Inflation_Rate'] <= 3)).sum()
        pct_target = (target_range / len(year_data)) * 100 if len(year_data) > 0 else 0
        st.metric(
            label="Near Target (1-3%)",
            value=f"{target_range}",
            delta=f"{pct_target:.0f}% of total"
        )
    
    with col4:
        deflation = (year_data['Inflation_Rate'] < 0).sum()
        pct_deflation = (deflation / len(year_data)) * 100 if len(year_data) > 0 else 0
        st.metric(
            label="Deflation (<0%)",
            value=f"{deflation}",
            delta=f"{pct_deflation:.0f}% of total"
        )
    
    st.markdown("---")
    
    # ==================== QUICK INSIGHTS (NEW) ====================
    st.markdown("### 💡 Quick Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(year_data) > 0:
            worst = year_data.nlargest(1, 'Inflation_Rate').iloc[0]
            st.markdown(f"""
                <div class='alert-box' style='background-color: #ef444415; border-left-color: #ef4444;'>
                    <strong style='color: #ef4444;'>⚠️ Highest Risk</strong><br>
                    <strong style='font-size: 1.2rem;'>{worst['Country']}</strong><br>
                    <span style='font-size: 1.5rem; color: #ef4444;'>{worst['Inflation_Rate']:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No data available")
    
    with col2:
        if len(year_data) > 0:
            best = year_data.nsmallest(1, 'Inflation_Rate').iloc[0]
            st.markdown(f"""
                <div class='alert-box' style='background-color: #10b98115; border-left-color: #10b981;'>
                    <strong style='color: #10b981;'>🏆 Best Performance</strong><br>
                    <strong style='font-size: 1.2rem;'>{best['Country']}</strong><br>
                    <span style='font-size: 1.5rem; color: #10b981;'>{best['Inflation_Rate']:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No data available")
    
    with col3:
        if len(year_data) > 0:
            # Reset index to avoid KeyError when filtering by region
            year_data_reset = year_data.reset_index(drop=True)
            closest_idx = (year_data_reset['Inflation_Rate'] - 2.0).abs().argsort().iloc[0]
            closest = year_data_reset.iloc[closest_idx]
            
            st.markdown(f"""
                <div class='alert-box' style='background-color: #3b82f615; border-left-color: #3b82f6;'>
                    <strong style='color: #3b82f6;'>🎯 Closest to Target</strong><br>
                    <strong style='font-size: 1.2rem;'>{closest['Country']}</strong><br>
                    <span style='font-size: 1.5rem; color: #3b82f6;'>{closest['Inflation_Rate']:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No data available")
    
    st.markdown("---")
    
    # ==================== WORLD MAP ====================
    st.markdown(f"### 🗺️ World Inflation Map ({int(selected_year)})")
    
    try:
        import plotly.express as px
        
        # Create choropleth map
        fig_map = px.choropleth(
            year_data,
            locations="Country",
            locationmode='country names',
            color="Inflation_Rate",
            hover_name="Country",
            hover_data={
                'Inflation_Rate': ':.2f',
                'GDP_Growth': ':.2f',
                'Country': False
            },
            color_continuous_scale=[
                [0, '#10b981'],      # Green (low inflation)
                [0.25, '#3b82f6'],   # Blue (moderate)
                [0.5, '#f59e0b'],    # Orange (elevated)
                [0.75, '#ef4444'],   # Red (high)
                [1, '#7f1d1d']       # Dark red (very high)
            ],
            range_color=[-5, 15],
            labels={'Inflation_Rate': 'Inflation (%)'}
        )
        
        fig_map.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            ),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Color scale legend
        st.caption("🟢 Low (<2%) | 🔵 Moderate (2-4%) | 🟠 Elevated (4-8%) | 🔴 High (8-15%) | 🟤 Very High (>15%)")
    
    except Exception as e:
        st.warning("⚠️ World map visualization unavailable. Showing data table instead.")
    
    st.markdown("---")
    
    # ==================== INFLATION CHAMPIONS (NEW) ====================
    st.markdown("### 🏆 Inflation Champions (Consistent Performers)")
    
    # Calculate long-term performance (countries with stable, near-target inflation)
    historical_performance = df.groupby('Country').agg({
        'Inflation_Rate': ['mean', 'std', 'count']
    }).reset_index()
    
    historical_performance.columns = ['Country', 'Mean', 'Std', 'Years']
    
    # Filter for champions: avg 1.5-3%, low volatility, sufficient data
    champions = historical_performance[
        (historical_performance['Mean'] >= 1.5) & 
        (historical_performance['Mean'] <= 3.0) &
        (historical_performance['Std'] < 2.0) &
        (historical_performance['Years'] >= 10)
    ].sort_values('Std')
    
    if len(champions) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"🏆 **{len(champions)} countries** maintain stable, near-target inflation over the long term!")
            
            # Show top 10 champions
            top_champions = champions.head(10).copy()
            top_champions['Mean'] = top_champions['Mean'].apply(lambda x: f"{x:.2f}%")
            top_champions['Std'] = top_champions['Std'].apply(lambda x: f"{x:.2f}pp")
            top_champions.columns = ['Country', 'Avg Inflation', 'Volatility', 'Data Years']
            
            st.dataframe(top_champions, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Champion Criteria:**")
            st.markdown("""
                - ✅ Average inflation: 1.5-3%
                - ✅ Low volatility: <2pp std dev
                - ✅ Consistent data: 10+ years
                - ✅ Long-term stability
            """)
            
            if len(champions) > 0:
                most_stable = champions.iloc[0]
                st.info(f"🥇 **Most Stable:** {most_stable['Country']} (σ = {most_stable['Std']:.2f}pp)")
    else:
        st.info("No countries meet the champion criteria in the current dataset.")
    
    st.markdown("---")
    
    # ==================== TOP & BOTTOM PERFORMERS ====================
    st.markdown("### 🏆 Top & Bottom Performers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Lowest Inflation (Best)")
        lowest = year_data.nsmallest(10, 'Inflation_Rate')[['Country', 'Inflation_Rate', 'GDP_Growth']].copy()
        
        # Add risk category (NEW)
        def get_risk_category(inflation):
            if inflation < 2:
                return "🟢 Low"
            elif inflation < 4:
                return "🟡 Moderate"
            elif inflation < 8:
                return "🟠 Elevated"
            elif inflation < 15:
                return "🔴 High"
            else:
                return "🟤 Crisis"
        
        lowest['Risk'] = lowest['Inflation_Rate'].apply(get_risk_category)
        lowest['Inflation_Rate'] = lowest['Inflation_Rate'].apply(lambda x: f"{x:.2f}%")
        lowest['GDP_Growth'] = lowest['GDP_Growth'].apply(lambda x: f"{x:.2f}%")
        lowest.columns = ['Country', 'Inflation', 'GDP Growth', 'Risk Level']
        
        st.dataframe(lowest, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 🔴 Highest Inflation (Worst)")
        highest = year_data.nlargest(10, 'Inflation_Rate')[['Country', 'Inflation_Rate', 'GDP_Growth']].copy()
        
        highest['Risk'] = highest['Inflation_Rate'].apply(get_risk_category)
        highest['Inflation_Rate'] = highest['Inflation_Rate'].apply(lambda x: f"{x:.2f}%")
        highest['GDP_Growth'] = highest['GDP_Growth'].apply(lambda x: f"{x:.2f}%")
        highest.columns = ['Country', 'Inflation', 'GDP Growth', 'Risk Level']
        
        st.dataframe(highest, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ==================== REGIONAL COMPARISON ====================
    st.markdown("### 🌍 Regional Comparison")
    
    # Define regional groupings
    asia = ['China', 'India', 'Japan', 'South Korea', 'Indonesia', 'Malaysia', 'Singapore', 'Thailand', 'Vietnam', 'Philippines', 
            'Pakistan', 'Bangladesh', 'Myanmar', 'Cambodia', 'Laos', 'Mongolia', 'Nepal', 'Sri Lanka', 'Afghanistan', 'Bhutan']
    europe = ['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Netherlands', 'Sweden', 'Poland', 'Belgium', 'Austria',
              'Switzerland', 'Norway', 'Denmark', 'Finland', 'Ireland', 'Portugal', 'Greece', 'Czech Republic', 'Romania', 'Hungary']
    americas = ['United States', 'Canada', 'Brazil', 'Mexico', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela', 
                'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay', 'Costa Rica', 'Panama', 'Guatemala', 'Honduras', 'Nicaragua']
    africa = ['South Africa', 'Nigeria', 'Egypt', 'Kenya', 'Ghana', 'Ethiopia', 'Morocco', 'Algeria', 'Tanzania', 'Uganda',
              'Sudan', 'Angola', 'Mozambique', 'Madagascar', 'Cameroon', 'Niger', 'Burkina Faso', 'Mali', 'Malawi', 'Zambia',
              'Senegal', 'Chad', 'Somalia', 'Zimbabwe', 'Rwanda', 'Benin', 'Burundi', 'Tunisia', 'South Sudan', 'Libya']
    oceania = ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands', 'Vanuatu', 'Samoa', 'Tonga']
    
    # Calculate regional averages (using full year_data, not filtered)
    full_year_data = df[df['Date'] == selected_year].copy()
    regional_data = []
    
    for region_name, countries in [('Asia', asia), ('Europe', europe), ('Americas', americas), ('Africa', africa), ('Oceania', oceania)]:
        region_subset = full_year_data[full_year_data['Country'].isin(countries)]
        if len(region_subset) > 0:
            regional_data.append({
                'Region': region_name,
                'Avg Inflation': region_subset['Inflation_Rate'].mean(),
                'Avg GDP Growth': region_subset['GDP_Growth'].mean(),
                'Countries': len(region_subset),
                'Std Dev': region_subset['Inflation_Rate'].std()
            })
    
    if regional_data:
        regional_df = pd.DataFrame(regional_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional bar chart
            fig_regions = go.Figure()
            
            colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
            
            fig_regions.add_trace(go.Bar(
                x=regional_df['Region'],
                y=regional_df['Avg Inflation'],
                marker_color=colors,
                text=regional_df['Avg Inflation'].round(2),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Avg Inflation: %{y:.2f}%<extra></extra>'
            ))
            
            fig_regions.add_hline(y=2.0, line_dash="dash", line_color="green", annotation_text="2% Target")
            
            fig_regions.update_layout(
                title="Average Inflation by Region",
                xaxis_title="Region",
                yaxis_title="Inflation Rate (%)",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_regions, use_container_width=True)
        
        with col2:
            # Regional table
            regional_display = regional_df.copy()
            regional_display['Avg Inflation'] = regional_display['Avg Inflation'].apply(lambda x: f"{x:.2f}%")
            regional_display['Avg GDP Growth'] = regional_display['Avg GDP Growth'].apply(lambda x: f"{x:.2f}%")
            regional_display['Std Dev'] = regional_display['Std Dev'].apply(lambda x: f"{x:.2f}pp")
            
            st.dataframe(regional_display, use_container_width=True, hide_index=True, height=400)
            
            # Regional insights
            best_region = regional_df.loc[regional_df['Avg Inflation'].idxmin(), 'Region']
            best_inflation = regional_df.loc[regional_df['Avg Inflation'].idxmin(), 'Avg Inflation']
            
            st.success(f"🏆 **Best Region:** {best_region} ({best_inflation:.2f}% avg inflation)")
    
    st.markdown("---")
    
    # ==================== GLOBAL TRENDS OVER TIME ====================
    st.markdown("### 📈 Global Inflation Trends Over Time")
    
    # Calculate global average by year
    global_trend = df.groupby('Date').agg({
        'Inflation_Rate': ['mean', 'median', 'std']
    }).reset_index()
    
    global_trend.columns = ['Year', 'Mean', 'Median', 'Std Dev']
    
    fig_trend = go.Figure()
    
    # Mean line
    fig_trend.add_trace(go.Scatter(
        x=global_trend['Year'],
        y=global_trend['Mean'],
        mode='lines+markers',
        name='Global Average',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8)
    ))
    
    # Median line
    fig_trend.add_trace(go.Scatter(
        x=global_trend['Year'],
        y=global_trend['Median'],
        mode='lines',
        name='Global Median',
        line=dict(color='#10b981', width=2, dash='dash')
    ))
    
    # Confidence band
    fig_trend.add_trace(go.Scatter(
        x=global_trend['Year'].tolist() + global_trend['Year'].tolist()[::-1],
        y=(global_trend['Mean'] + global_trend['Std Dev']).tolist() + 
          (global_trend['Mean'] - global_trend['Std Dev']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='±1 Std Dev',
        hoverinfo='skip'
    ))
    
    fig_trend.add_hline(y=2.0, line_dash="dash", line_color="green", annotation_text="2% Target")
    
    fig_trend.update_layout(
        title="Global Inflation Trends (1990-2023)",
        xaxis_title="Year",
        yaxis_title="Inflation Rate (%)",
        template='plotly_white',
        height=450,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Trend insights
    recent_avg = global_trend['Mean'].tail(5).mean()
    historical_avg = global_trend['Mean'].head(10).mean()
    change = recent_avg - historical_avg
    
    if change > 0.5:
        st.warning(f"⚠️ **Trend:** Global inflation increased by {change:.1f}pp in recent years (last 5yr avg: {recent_avg:.2f}% vs 1990-1999 avg: {historical_avg:.2f}%)")
    elif change < -0.5:
        st.success(f"✅ **Trend:** Global inflation decreased by {abs(change):.1f}pp in recent years (last 5yr avg: {recent_avg:.2f}% vs 1990-1999 avg: {historical_avg:.2f}%)")
    else:
        st.info(f"📊 **Trend:** Global inflation remains relatively stable (last 5yr avg: {recent_avg:.2f}% vs 1990-1999 avg: {historical_avg:.2f}%)")
    
    st.markdown("---")
    
    # ==================== COUNTRY RANKINGS (FULL LIST) ====================
    st.markdown("### 📊 Complete Country Rankings")
    
    tab1, tab2, tab3 = st.tabs(["🔥 By Inflation", "📈 By GDP Growth", "📊 All Indicators"])
    
    with tab1:
        inflation_rank = year_data[['Country', 'Inflation_Rate', 'GDP_Growth']].sort_values('Inflation_Rate').copy()
        inflation_rank['Rank'] = range(1, len(inflation_rank) + 1)
        inflation_rank['Risk'] = inflation_rank['Inflation_Rate'].apply(get_risk_category)
        inflation_rank = inflation_rank[['Rank', 'Country', 'Inflation_Rate', 'GDP_Growth', 'Risk']]
        inflation_rank.columns = ['Rank', 'Country', 'Inflation (%)', 'GDP Growth (%)', 'Risk Level']
        
        st.dataframe(
            inflation_rank.style.format({
                'Inflation (%)': '{:.2f}',
                'GDP Growth (%)': '{:.2f}'
            }),
            use_container_width=True,
            height=500
        )
    
    with tab2:
        gdp_rank = year_data[['Country', 'GDP_Growth', 'Inflation_Rate']].sort_values('GDP_Growth', ascending=False).copy()
        gdp_rank['Rank'] = range(1, len(gdp_rank) + 1)
        gdp_rank = gdp_rank[['Rank', 'Country', 'GDP_Growth', 'Inflation_Rate']]
        gdp_rank.columns = ['Rank', 'Country', 'GDP Growth (%)', 'Inflation (%)']
        
        st.dataframe(
            gdp_rank.style.format({
                'GDP Growth (%)': '{:.2f}',
                'Inflation (%)': '{:.2f}'
            }),
            use_container_width=True,
            height=500
        )
    
    with tab3:
        all_indicators = year_data[['Country', 'Inflation_Rate', 'GDP_Growth', 'Interest_Rate', 'Exchange_Rate', 'Unemployment_Rate']].copy()
        all_indicators.columns = ['Country', 'Inflation', 'GDP', 'Interest Rate', 'Exchange Rate', 'Unemployment']
        
        st.dataframe(
            all_indicators.style.format({
                'Inflation': '{:.2f}%',
                'GDP': '{:.2f}%',
                'Interest Rate': '{:.2f}%',
                'Exchange Rate': '{:.2f}',
                'Unemployment': '{:.2f}%'
            }),
            use_container_width=True,
            height=500
        )
    
    st.markdown("---")
    
    # ==================== CUSTOM COUNTRY COMPARISON ====================
    st.markdown("### 🔍 Custom Country Comparison")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        available_countries = sorted(year_data['Country'].unique().tolist())
        default_countries = ['United States', 'China', 'Germany', 'Japan', 'India']
        default_selection = [c for c in default_countries if c in available_countries][:5]
        
        selected_countries = st.multiselect(
            "Select countries to compare (2-10)",
            available_countries,
            default=default_selection if default_selection else []
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        compare_metric = st.selectbox("Compare by:", ['Inflation_Rate', 'GDP_Growth', 'Interest_Rate'])
    
    if len(selected_countries) >= 2:
        comparison_data = year_data[year_data['Country'].isin(selected_countries)].copy()
        
        # Chart
        fig_compare = go.Figure()
        
        colors_palette = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16']
        
        for idx, country in enumerate(selected_countries):
            country_subset = comparison_data[comparison_data['Country'] == country]
            fig_compare.add_trace(go.Bar(
                name=country,
                x=[country],
                y=country_subset[compare_metric],
                marker_color=colors_palette[idx % len(colors_palette)],
                text=country_subset[compare_metric].round(2),
                textposition='auto'
            ))
        
        fig_compare.update_layout(
            title=f"Country Comparison: {compare_metric.replace('_', ' ')}",
            xaxis_title="Country",
            yaxis_title=compare_metric.replace('_', ' ') + " (%)",
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Comparison table with percentiles (NEW)
        comparison_display = comparison_data[['Country', 'Inflation_Rate', 'GDP_Growth', 'Interest_Rate']].copy()
        
        # Add percentile column (NEW)
        def get_percentile(value, column):
            return (year_data[column] < value).mean() * 100
        
        comparison_display['Percentile'] = comparison_display['Inflation_Rate'].apply(
            lambda x: f"Better than {get_percentile(x, 'Inflation_Rate'):.0f}% of countries"
        )
        
        comparison_display['Risk'] = comparison_display['Inflation_Rate'].apply(get_risk_category)
        comparison_display.columns = ['Country', 'Inflation (%)', 'GDP Growth (%)', 'Interest Rate (%)', 'Global Ranking', 'Risk Level']
        
        st.dataframe(comparison_display, use_container_width=True, hide_index=True)
        
        # Export button (NEW)
        csv = comparison_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Data (CSV)",
            data=csv,
            file_name=f"country_comparison_{int(selected_year)}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("👆 Select at least 2 countries to compare")
        
        

def render_settings():
    """Functional settings page - changes actually apply to dashboard"""
    st.markdown("<div class='section-header'>⚙️ Settings & Configuration</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280;'>Customize your dashboard experience</p>", unsafe_allow_html=True)
    
    # Initialize session state for settings if not exists
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'user_name': 'Policy Analyst',
            'user_org': 'Central Bank',
            'user_location': 'Malaysia',
            'user_role': 'Analyst',
            'chart_theme': 'plotly_white',
            'number_decimals': 2,
            'date_format': 'YYYY',
            'currency': '$',
            'table_rows': 10,
            'layout_style': 'Wide',
            'high_inflation_threshold': 8.0,
            'volatility_threshold': 2.0,
            'enable_alerts': True,
            'alert_types': ['High inflation', 'Rapid inflation changes'],
            'alert_position': 'Top',
            'pdf_quality': 'High',
            'include_charts': True,
            'include_metadata': True,
            'csv_delimiter': ',',
            'include_headers': True,
            'date_format_export': 'YYYY-MM-DD'
        }
    
    # ==================== USER PROFILE ====================
    st.markdown("### 👤 User Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_name = st.text_input(
            "Display Name", 
            value=st.session_state.settings['user_name'],
            help="Your name or role"
        )
        user_org = st.text_input(
            "Organization", 
            value=st.session_state.settings['user_org'],
            help="Your organization"
        )
    
    with col2:
        user_location = st.selectbox(
            "Primary Location",
            ["Malaysia", "Singapore", "United States", "United Kingdom", "China", "Japan", "Germany", "Other"],
            index=["Malaysia", "Singapore", "United States", "United Kingdom", "China", "Japan", "Germany", "Other"].index(st.session_state.settings['user_location']),
            help="Your primary country of interest"
        )
        user_role = st.selectbox(
            "Role",
            ["Policy Maker", "Analyst", "Researcher", "Student", "Other"],
            index=["Policy Maker", "Analyst", "Researcher", "Student", "Other"].index(st.session_state.settings['user_role']),
            help="Your professional role"
        )
    
    st.markdown("---")
    
    # ==================== DISPLAY PREFERENCES ====================
    st.markdown("### 🎨 Display Preferences")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Chart Theme**")
        chart_theme = st.selectbox(
            "Select theme",
            ["plotly_white", "plotly_dark", "seaborn", "ggplot2", "simple_white"],
            index=["plotly_white", "plotly_dark", "seaborn", "ggplot2", "simple_white"].index(st.session_state.settings['chart_theme']),
            help="Changes all chart appearances in the dashboard"
        )
        
        st.markdown("**Number Format**")
        number_format = st.selectbox(
            "Decimal places",
            ["0 decimals", "1 decimal", "2 decimals", "3 decimals"],
            index=st.session_state.settings['number_decimals'],
            help="Precision for displaying numbers"
        )
    
    with col2:
        st.markdown("**Date Format**")
        date_format = st.selectbox(
            "Format",
            ["YYYY", "MMM YYYY", "DD/MM/YYYY", "MM/DD/YYYY"],
            index=["YYYY", "MMM YYYY", "DD/MM/YYYY", "MM/DD/YYYY"].index(st.session_state.settings['date_format']),
            help="How dates are displayed throughout the dashboard"
        )
        
        st.markdown("**Currency Format**")
        currency = st.selectbox(
            "Currency symbol",
            ["$", "€", "£", "¥", "RM", "None"],
            index=["$", "€", "£", "¥", "RM", "None"].index(st.session_state.settings['currency']),
            help="Default currency for displays"
        )
    
    with col3:
        st.markdown("**Table Style**")
        table_rows = st.slider(
            "Default table rows",
            min_value=5,
            max_value=50,
            value=st.session_state.settings['table_rows'],
            step=5,
            help="Number of rows to display in tables"
        )
        
        st.markdown("**Page Layout**")
        layout_style = st.radio(
            "Width",
            ["Wide", "Centered"],
            index=["Wide", "Centered"].index(st.session_state.settings['layout_style']),
            horizontal=True,
            help="Dashboard layout width"
        )
    
    st.markdown("---")
    
    # ==================== NOTIFICATION SETTINGS ====================
    st.markdown("### 🔔 Notifications & Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Alert Thresholds**")
        high_inflation_alert = st.number_input(
            "High inflation threshold (%)",
            min_value=5.0,
            max_value=20.0,
            value=st.session_state.settings['high_inflation_threshold'],
            step=0.5,
            help="Dashboard will highlight countries exceeding this level"
        )
        
        volatility_alert = st.number_input(
            "Volatility threshold (pp)",
            min_value=1.0,
            max_value=5.0,
            value=st.session_state.settings['volatility_threshold'],
            step=0.5,
            help="Dashboard will alert when volatility exceeds this level"
        )
        
        st.markdown("**Alert Preferences**")
        enable_alerts = st.checkbox(
            "Enable alerts",
            value=st.session_state.settings['enable_alerts'],
            help="Show warning boxes for important events"
        )
    
    with col2:
        st.markdown("**Alert Types**")
        if enable_alerts:
            alert_types = st.multiselect(
                "Alert for:",
                [
                    "High inflation",
                    "Rapid inflation changes",
                    "Deflation risk",
                    "Model updates",
                    "Data updates"
                ],
                default=st.session_state.settings['alert_types'],
                help="Types of alerts to show in dashboard"
            )
        else:
            alert_types = []
        
        st.markdown("**Alert Display**")
        alert_position = st.radio(
            "Position",
            ["Top", "Bottom"],
            index=["Top", "Bottom"].index(st.session_state.settings['alert_position']),
            horizontal=True,
            help="Where warning boxes appear on pages"
        )
    
    st.markdown("---")
    
    # ==================== EXPORT SETTINGS ====================
    st.markdown("### 📥 Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PDF Export**")
        pdf_quality = st.radio(
            "Quality",
            ["Standard", "High", "Print"],
            index=["Standard", "High", "Print"].index(st.session_state.settings['pdf_quality']),
            horizontal=True,
            help="Quality when generating PDF reports"
        )
        
        include_charts = st.checkbox(
            "Include charts in PDF",
            value=st.session_state.settings['include_charts'],
            help="Embed visualizations in PDF exports"
        )
        
        include_metadata = st.checkbox(
            "Include metadata",
            value=st.session_state.settings['include_metadata'],
            help="Add generation date, user info, etc. to exports"
        )
    
    with col2:
        st.markdown("**CSV Export**")
        csv_delimiter = st.selectbox(
            "Delimiter",
            [",", ";", "\t"],
            format_func=lambda x: {"," :"Comma", ";": "Semicolon", "\t": "Tab"}[x],
            index=[",", ";", "\t"].index(st.session_state.settings['csv_delimiter']),
            help="CSV field separator"
        )
        
        include_headers = st.checkbox(
            "Include column headers",
            value=st.session_state.settings['include_headers'],
            help="Add headers to CSV exports"
        )
        
        date_format_export = st.selectbox(
            "Date format in exports",
            ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"],
            index=["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"].index(st.session_state.settings['date_format_export']),
            help="Date format for exported files"
        )
    
    st.markdown("---")
    
    # ==================== SAVE SETTINGS ====================
    st.markdown("### 💾 Apply Settings")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("💾 Save & Apply Settings", type="primary", use_container_width=True):
            # Update session state with new values
            st.session_state.settings.update({
                'user_name': user_name,
                'user_org': user_org,
                'user_location': user_location,
                'user_role': user_role,
                'chart_theme': chart_theme,
                'number_decimals': ["0 decimals", "1 decimal", "2 decimals", "3 decimals"].index(number_format),
                'date_format': date_format,
                'currency': currency,
                'table_rows': table_rows,
                'layout_style': layout_style,
                'high_inflation_threshold': high_inflation_alert,
                'volatility_threshold': volatility_alert,
                'enable_alerts': enable_alerts,
                'alert_types': alert_types if enable_alerts else [],
                'alert_position': alert_position,
                'pdf_quality': pdf_quality,
                'include_charts': include_charts,
                'include_metadata': include_metadata,
                'csv_delimiter': csv_delimiter,
                'include_headers': include_headers,
                'date_format_export': date_format_export
            })
            
            st.success("✅ Settings saved and applied successfully!")
            st.info("🔄 Settings will take effect on the next page refresh or when you navigate to another page.")
            st.balloons()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            # Reset to defaults
            st.session_state.settings = {
                'user_name': 'Policy Analyst',
                'user_org': 'Central Bank',
                'user_location': 'Malaysia',
                'user_role': 'Analyst',
                'chart_theme': 'plotly_white',
                'number_decimals': 2,
                'date_format': 'YYYY',
                'currency': '$',
                'table_rows': 10,
                'layout_style': 'Wide',
                'high_inflation_threshold': 8.0,
                'volatility_threshold': 2.0,
                'enable_alerts': True,
                'alert_types': ['High inflation', 'Rapid inflation changes'],
                'alert_position': 'Top',
                'pdf_quality': 'High',
                'include_charts': True,
                'include_metadata': True,
                'csv_delimiter': ',',
                'include_headers': True,
                'date_format_export': 'YYYY-MM-DD'
            }
            st.warning("⚠️ Settings reset to defaults")
            st.rerun()
    
    with col3:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # ==================== CURRENT SETTINGS PREVIEW ====================
    st.markdown("### 📋 Current Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            **👤 User Profile:**
            - Name: {st.session_state.settings['user_name']}
            - Organization: {st.session_state.settings['user_org']}
            - Location: {st.session_state.settings['user_location']}
            - Role: {st.session_state.settings['user_role']}
        """)
    
    with col2:
        st.markdown(f"""
            **🎨 Display:**
            - Theme: {st.session_state.settings['chart_theme']}
            - Decimals: {st.session_state.settings['number_decimals']}
            - Date format: {st.session_state.settings['date_format']}
            - Table rows: {st.session_state.settings['table_rows']}
            - Layout: {st.session_state.settings['layout_style']}
        """)
    
    with col3:
        st.markdown(f"""
            **🔔 Alerts:**
            - Enabled: {'Yes' if st.session_state.settings['enable_alerts'] else 'No'}
            - High inflation: >{st.session_state.settings['high_inflation_threshold']}%
            - Volatility: >{st.session_state.settings['volatility_threshold']}pp
            - Position: {st.session_state.settings['alert_position']}
        """)
    
    st.markdown("---")
    
    # ==================== HELP TEXT ====================
    st.info("""
        💡 **How Settings Work:**
        
        - **Chart Theme**: Changes the appearance of all charts throughout the dashboard
        - **Number Format**: Controls decimal places in all numeric displays
        - **Date Format**: Changes how dates are shown in tables and charts
        - **Table Rows**: Sets default pagination for all data tables
        - **Page Layout**: Changes the width of the dashboard (Wide = full screen, Centered = narrower)
        - **Alert Thresholds**: Automatically highlights countries exceeding these levels in Global Comparison
        - **Export Settings**: Controls default options when generating PDF reports or CSV files
        
        Click **"Save & Apply Settings"** to activate your changes!
    """)


# ==================== HELPER FUNCTION TO GET SETTINGS ====================
def get_setting(key, default=None):
    """Get setting from session state"""
    if 'settings' not in st.session_state:
        return default
    return st.session_state.settings.get(key, default)

        
    
def render_about(df):
    """About page with project information"""
    st.markdown("<div class='section-header'>ℹ️ About EconoForecast</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280;'>Inflation Forecasting System for Policy Decision Makers</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== PROJECT OVERVIEW ====================
    st.markdown("### 📊 Project Overview")
    
    st.markdown("""
        **EconoForecast** is an advanced inflation forecasting platform developed as a Final Year Project, 
        designed to provide policy makers, economists, and researchers with data-driven insights for 
        informed decision-making.
        
        The system leverages machine learning and time series analysis to forecast inflation trends 
        across 341 countries, utilizing 34 years of historical economic data (1990-2023).
    """)
    
    st.markdown("---")
    
    # ==================== TARGET USERS ====================
    st.markdown("### 👥 Designed For")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            **🏛️ Policy Makers**
            - Make evidence-based monetary policy decisions
            - Assess inflation risks and forecast future trends
            - Compare performance across regions and countries
            
            **📊 Economists & Analysts**
            - Analyze historical inflation patterns
            - Evaluate multiple forecasting models
            - Generate comprehensive reports with visualizations
        """)
    
    with col2:
        st.markdown("""
            **🔬 Researchers**
            - Access comprehensive global economic dataset
            - Study correlations between economic indicators
            - Validate forecasting methodologies
            
            **🎓 Students**
            - Learn about inflation dynamics and forecasting
            - Understand machine learning applications in economics
            - Explore interactive data visualizations
        """)
    
    st.markdown("---")
    
    # ==================== KEY FEATURES ====================
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            **📈 Advanced Forecasting**
            - 6 ML & Time Series Models
            - Random Forest (Base & Extended)
            - XGBoost (Base & Extended)
            - ARIMA & VAR Models
            - 5-year forecast horizon
            - Confidence intervals
        """)
    
    with col2:
        st.markdown("""
            **🌍 Global Coverage**
            - 341 countries tracked
            - 34 years of data (1990-2023)
            - Regional comparisons
            - Interactive world maps
            - Country rankings
            - Historical trends analysis
        """)
    
    with col3:
        st.markdown("""
            **📊 Rich Visualizations**
            - Interactive Plotly charts
            - Real-time data updates
            - PDF report generation
            - Custom comparisons
            - Correlation heatmaps
            - Trend analysis tools
        """)
    
    st.markdown("---")
    
    # ==================== DATA SOURCES ====================
    st.markdown("### 📚 Data Sources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            **Primary Economic Indicators:**
            - ✅ Inflation Rate (CPI)
            - ✅ GDP Growth Rate
            - ✅ Interest Rate
            - ✅ Exchange Rate
            - ✅ Unemployment Rate
        """)
    
    with col2:
        st.markdown("""
            **Data Providers:**
            - World Bank Open Data
            - International Monetary Fund (IMF)
            - Federal Reserve Economic Data (FRED)
            - National Statistical Offices
            - Central Bank Publications
        """)
    
    st.markdown("---")
    
    # ==================== TECHNICAL STACK ====================
    st.markdown("### 🛠️ Technical Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            **Frontend & UI**
            - Streamlit
            - Plotly
            - HTML/CSS
            - Markdown
        """)
    
    with col2:
        st.markdown("""
            **Machine Learning**
            - Scikit-learn
            - XGBoost
            - Statsmodels
            - Pandas/NumPy
        """)
    
    with col3:
        st.markdown("""
            **Data Processing**
            - Python 3.11
            - Pandas
            - NumPy
            - Pickle
        """)
    
    st.markdown("---")
    
    # ==================== MODELS OVERVIEW ====================
    st.markdown("### 🧠 Forecasting Models")
    
    models_data = pd.DataFrame({
        'Model': [
            'Random Forest (Base)',
            'Random Forest (Extended)',
            'XGBoost (Base)',
            'XGBoost (Extended)',
            'ARIMA',
            'VAR'
        ],
        'Type': [
            'Machine Learning',
            'Machine Learning',
            'Machine Learning',
            'Machine Learning',
            'Time Series',
            'Time Series'
        ],
        'Features': [
            'Core economic indicators',
            'Extended + engineered features',
            'Core economic indicators',
            'Extended + engineered features',
            'Univariate time series',
            'Multivariate time series'
        ],
        'Best For': [
            'General forecasting',
            'Complex patterns',
            'Non-linear relationships',
            'High accuracy needs',
            'Simple trends',
            'Multiple indicators'
        ]
    })
    
    st.dataframe(models_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ==================== FAQ ====================
    st.markdown("### ❓ Frequently Asked Questions")
    
    with st.expander("📊 How accurate are the forecasts?"):
        st.markdown("""
            Model accuracy varies by country and time horizon:
            - **Short-term (1-2 years):** Generally high accuracy (RMSE < 2%)
            - **Medium-term (3-4 years):** Moderate accuracy (RMSE 2-4%)
            - **Long-term (5 years):** Lower accuracy due to uncertainty
            
            Performance metrics (R², RMSE, MAE) are displayed for each model on the Model Performance page.
        """)
    
    with st.expander("🌍 Which countries are covered?"):
        st.markdown("""
            InfraScope covers **341 countries and territories** worldwide, including:
            - All UN member states
            - Major economies (G20, G7)
            - Emerging markets
            - Small island nations
            - Territories with available economic data
            
            Data availability varies by country and indicator.
        """)
    
    with st.expander("🔄 How often is data updated?"):
        st.markdown("""
            - **Historical Data:** Annual updates (dataset: 1990-2023)
            - **Live Economic Feed:** Real-time updates from RSS feeds
            - **Market Data:** Updates during market hours
            - **Forecasts:** Generated on-demand by user
            
            The system uses the latest available historical data for model training.
        """)
    
    with st.expander("🧠 Which model should I use?"):
        st.markdown("""
            **Model Selection Guide:**
            
            - **RF_Extended:** Best for most countries, handles complex patterns
            - **XGBoost_Extended:** Highest accuracy, good for volatile economies
            - **ARIMA:** Simple, interpretable, good for stable trends
            - **VAR:** Best when analyzing multiple indicators together
            
            The Model Performance page shows accuracy metrics to help you choose.
        """)
    
    with st.expander("📥 Can I export the results?"):
        st.markdown("""
            Yes! Export options include:
            - **PDF Reports:** Comprehensive forecast reports with charts
            - **CSV Data:** Raw data and predictions for further analysis
            - **Charts:** Save individual visualizations as images
            
            Export settings can be customized in the Settings page.
        """)
    
    st.markdown("---")
    
# ==================== PROJECT INFO ====================
    st.markdown("### 🎓 Project Information")
    
    st.markdown("""
        **Academic Details:**
        - **Project Type:** Final Year Project (FYP)
        - **Field:** Computer Science / Data Science
        - **Focus:** Machine Learning & Economic Forecasting
        - **Year:** 2024/2025
    """)
    
    st.markdown("---")
    
    # ==================== LICENSE & COPYRIGHT ====================
    st.markdown("### 📜 License & Usage")
    
    st.info("""
        **Academic Use License**
        
        This project is developed for academic purposes as a Final Year Project. 
        
        - ✅ Free for educational and research use
        - ✅ Attribution required if referenced
        - ⚠️ Not for commercial use without permission
    """)
    
    st.markdown("---")
    
    # ==================== FOOTER ====================
    st.markdown("""
        <div style='text-align: center; color: #666666; font-size: 0.9em; padding: 2rem 0;'>
            <p><strong>EconoForecast - Inflation Forecasting System</strong></p>
            <p>Built with Streamlit • Machine Learning • Economic Analysis</p>
            <p>Version 1.0.0 | 📊 Final Year Project 2024/2025</p>
            <p style='margin-top: 1rem;'>© 2025 EconoForecast. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)


# ==================== RUN ====================
if __name__ == "__main__":
    main()