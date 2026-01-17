# 🌍 EconoForecast - Inflation Forecasting System

## Overview
A comprehensive economic intelligence dashboard for inflation forecasting and policy analysis using machine learning and time series models. Built to analyze global economic trends and provide actionable insights for policymakers and analysts.

## Key Features
- **Multi-Model Forecasting:** 6 advanced models (Random Forest, XGBoost, ARIMA, VAR, LSTM, Prophet) for inflation prediction
- **Global Coverage:** Economic analysis across 341 countries worldwide
- **Visual Analytics:** Interactive correlation analysis and trend visualization
- **Live News Feed:** Real-time economic news integration
- **Export Capabilities:** PDF report generation for stakeholder presentations
- **Interactive Dashboards:** User-friendly interface built with Streamlit

## Dataset
- **Time Period:** 34 years of economic data (1990-2023)
- **Geographic Coverage:** 341 countries worldwide
- **Economic Indicators:** 
  - Inflation (CPI)
  - GDP Growth
  - Interest Rates
  - Exchange Rates
  - Unemployment Rates

## Tech Stack
- **Languages:** Python, SQL
- **ML Libraries:** scikit-learn, XGBoost, statsmodels, Prophet, TensorFlow/Keras
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly, Streamlit
- **Tools:** Jupyter Notebook, Git

## Models Implemented
1. **Random Forest Regressor** - Ensemble learning for robust predictions
2. **XGBoost** - Gradient boosting with hyperparameter optimization
3. **ARIMA** - Classical time series autoregressive model
4. **VAR (Vector Autoregression)** - Multivariate time series analysis
5. **LSTM** - Deep learning for sequential patterns
6. **Prophet** - Facebook's forecasting tool for trend and seasonality

## Key Results
- ✅ Analyzed **34 years** of economic data (1990-2023)
- ✅ Covered **341 countries** with multiple economic indicators
- ✅ Implemented **6 forecasting models** with comparative analysis
- ✅ Built interactive dashboard with real-time news integration
- ✅ Generated automated PDF reports for policy recommendations

## Project Structure
```
econoforecast/
├── data/                 # Economic datasets (1990-2023)
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code
│   ├── models/          # ML model implementations
│   ├── dashboard.py     # Streamlit application
│   └── utils/           # Helper functions
├── reports/             # Generated PDF reports
├── requirements.txt     # Python dependencies
└── README.md
```

## How to Run
```bash
# Clone the repository
git clone https://github.com/Abakar-Fatime/econoforecast.git
cd econoforecast

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run src/dashboard.py
```

## Dashboard Features
- 📊 **Forecasting Hub:** Compare predictions across 6 models
- 🌍 **Country Comparison:** Analyze economic trends across nations
- 📈 **Correlation Analysis:** Identify relationships between indicators
- 📰 **Live News Feed:** Stay updated with economic developments
- 📄 **PDF Export:** Generate professional reports

## Future Enhancements
- API integration for real-time data updates
- Cloud deployment (AWS/Azure)
- Advanced deep learning models (Transformer-based)
- Multi-language support for international users

## Author
**Abakar Sougui Fatime**  
Bachelor of Science (Honours) in Computer Science (Data Analytics)  
Asia Pacific University of Technology & Innovation  
📧 souguifatimeabakar@gmail.com | 💼 [LinkedIn](https://linkedin.com/in/abakar-sougui-fatime) | 💻 [GitHub](https://github.com/Abakar-Fatime)

## License
MIT License

---