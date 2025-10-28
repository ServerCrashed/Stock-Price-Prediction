# Stock Price Prediction Web Application

A sophisticated stock price prediction web application using LSTM/GRU deep learning models, built with Python and Streamlit. This application supports both US (NASDAQ) and Indian (NSE/BSE) stock markets.

## 🌟 Features

- **Real-time Data**: Fetches near real-time stock data from Yahoo Finance (15-minute delay)
- **Multi-Market Support**: Works with NASDAQ, NSE, and BSE stocks
- **Deep Learning Models**: Choose between LSTM and GRU models for prediction
- **7-Day Forecast**: Predicts stock prices for the next 7 days
- **Interactive Charts**: Beautiful, interactive visualizations using Plotly
- **Technical Indicators**: Includes MA(7), MA(21), MA(50), volatility, and returns
- **Minimal Elegant Design**: Clean, user-friendly interface

## 📁 Project Structure

```
stock-prediction-app/
│
├── stock_prediction_app.py          # Main Streamlit application
├── stock_prediction_training_notebook.txt  # Training notebook (convert to .ipynb)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── models/                           # Trained models (after training)
│   ├── AAPL_lstm_model.keras
│   ├── AAPL_gru_model.keras
│   ├── AAPL_scaler.pkl
│   └── AAPL_metadata.pkl
│
└── notebooks/                        # Jupyter notebooks
    └── model_training.ipynb
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU for faster training

### Installation

1. **Clone or download the project files**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run stock_prediction_app.py
```

4. **Access the app**
   - The app will open in your default browser at `http://localhost:8501`

## 📊 Data Sources

### Stock Ticker Format

| Market | Example Ticker | Format in App |
|--------|---------------|---------------|
| NASDAQ (US) | Apple | `AAPL` |
| NSE (India) | Reliance | `RELIANCE` (becomes `RELIANCE.NS`) |
| BSE (India) | TCS | `TCS` (becomes `TCS.BO`) |

### Popular Ticker Examples

**US Stocks (NASDAQ):**
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- TSLA (Tesla)
- AMZN (Amazon)

**Indian Stocks (NSE):**
- RELIANCE (Reliance Industries)
- TCS (Tata Consultancy Services)
- INFY (Infosys)
- HDFCBANK (HDFC Bank)
- ICICIBANK (ICICI Bank)

**Indian Stocks (BSE):**
- Same tickers as NSE, with `.BO` suffix

## 🧠 Model Training

### Setting Up Training Environment

1. **Open PyCharm and create a Jupyter Notebook**
   ```bash
   # In PyCharm terminal
   pip install jupyter notebook
   ```

2. **Copy the training code**
   - Open `stock_prediction_training_notebook.txt`
   - Create a new Jupyter notebook (.ipynb file)
   - Copy each section into separate cells

3. **Configure training parameters**
```python
# In the training notebook
TICKER = 'AAPL'  # or 'RELIANCE.NS' for NSE
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
LOOK_BACK = 60  # Use 60 days of history
EPOCHS = 50
BATCH_SIZE = 32
```

4. **Run the training**
   - Execute cells sequentially
   - Training will take 10-30 minutes depending on your hardware
   - Models and scalers will be saved automatically

### Training Output Files

After training, you'll get:
- `{TICKER}_lstm_model.keras` - Trained LSTM model
- `{TICKER}_gru_model.keras` - Trained GRU model
- `{TICKER}_scaler.pkl` - Data normalization scaler
- `{TICKER}_metadata.pkl` - Model metadata and metrics
- Various PNG visualization files

### Model Performance Metrics

The training process evaluates models using:
- **RMSE** (Root Mean Square Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **R²** (R-squared): Higher is better (max 1.0)
- **MAPE** (Mean Absolute Percentage Error): Lower is better

Expected performance:
- Good model: RMSE < 5% of stock price, R² > 0.90
- Excellent model: RMSE < 3% of stock price, R² > 0.95

## 💻 Using the Web App

### Step 1: Select Market
Choose from:
- US (NASDAQ)
- India (NSE)
- India (BSE)

### Step 2: Enter Ticker
Enter the stock symbol (without suffix). The app automatically adds the correct suffix based on your market selection.

### Step 3: Configure Prediction
- **Model Type**: Choose LSTM or GRU
- **Prediction Days**: Select 1-7 days ahead
- **Historical Data**: Choose lookback period (1Y, 2Y, 5Y, Max)

### Step 4: View Results
- Current price and metrics
- 7-day price predictions
- Interactive candlestick chart
- Technical indicators
- Company information

## 🔧 Advanced Configuration

### Customizing the App

Edit `stock_prediction_app.py`:

```python
# Change default ticker
ticker_input = st.text_input(
    "Enter Stock Ticker",
    value="YOUR_TICKER",  # Change this
    ...
)

# Modify prediction range
pred_days = st.slider(
    "Prediction Days",
    min_value=1,
    max_value=14,  # Increase maximum days
    value=7,
    ...
)
```

### Loading Your Trained Models

To use your own trained models in the app, modify the `predict_future_prices()` function:

```python
def predict_future_prices(data, model_type='LSTM', days=7):
    from tensorflow.keras.models import load_model
    import joblib

    # Load your trained model
    model = load_model(f'{ticker}_{model_type.lower()}_model.keras')
    scaler = joblib.load(f'{ticker}_scaler.pkl')

    # Prepare data and make predictions
    # ... (implement your prediction logic)
```

## 📈 Technical Details

### Model Architecture

**LSTM Model:**
```
Layer 1: LSTM(50 units, return_sequences=True) + Dropout(0.2)
Layer 2: LSTM(50 units, return_sequences=True) + Dropout(0.2)
Layer 3: LSTM(50 units, return_sequences=False) + Dropout(0.2)
Layer 4: Dense(25)
Layer 5: Dense(1)

Total parameters: ~35,000
```

**GRU Model:**
```
Layer 1: GRU(50 units, return_sequences=True) + Dropout(0.2)
Layer 2: GRU(50 units, return_sequences=True) + Dropout(0.2)
Layer 3: GRU(50 units, return_sequences=False) + Dropout(0.2)
Layer 4: Dense(25)
Layer 5: Dense(1)

Total parameters: ~26,000 (faster than LSTM)
```

### Features Used

The models use the following features:
1. **Close Price** (main target)
2. **Volume** (trading volume)
3. **MA(7)** (7-day moving average)
4. **MA(21)** (21-day moving average)
5. **MA(50)** (50-day moving average)
6. **Volatility** (21-day standard deviation)

### Data Preprocessing

1. **Normalization**: MinMaxScaler (0-1 range)
2. **Sequence Length**: 60 time steps
3. **Train/Test Split**: 80/20
4. **Missing Data**: Forward-filled

## 🐛 Troubleshooting

### Common Issues

**Issue 1: "Could not fetch data for ticker"**
- Solution: Verify ticker symbol is correct
- For NSE/BSE: Use base symbol only (e.g., "RELIANCE" not "RELIANCE.NS")
- Check internet connection

**Issue 2: "ModuleNotFoundError"**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Issue 3: TensorFlow installation issues**
```bash
# For CPU-only installation
pip install tensorflow-cpu==2.14.0

# For GPU support (requires CUDA)
pip install tensorflow-gpu==2.14.0
```

**Issue 4: Streamlit not opening**
```bash
# Clear cache and restart
streamlit cache clear
streamlit run stock_prediction_app.py
```

### Performance Issues

If the app is slow:
1. Reduce lookback period to 1-2 years
2. Use GRU instead of LSTM (faster)
3. Clear browser cache
4. Ensure yfinance cache is enabled (default)

## 📝 Notes

### Limitations

1. **Data Delay**: Yahoo Finance data has ~15 minute delay
2. **Market Hours**: Real-time data only during market hours
3. **Weekends**: No trading data available
4. **Holidays**: Market holidays may affect predictions
5. **Model Accuracy**: Past performance doesn't guarantee future results

### Best Practices

1. **Training Data**: Use at least 5 years of historical data
2. **Model Updates**: Retrain models quarterly for best accuracy
3. **Multiple Stocks**: Train separate models for each stock
4. **Validation**: Always validate predictions against actual prices
5. **Risk Management**: Use predictions as one input among many

## 🔐 Security & Privacy

- All data is fetched from public Yahoo Finance API
- No user data is stored
- Models run locally on your machine
- No external API keys required

## 📄 License

This project is for educational purposes only. Not financial advice.

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Yahoo Finance ticker format
3. Verify all dependencies are installed
4. Check Python version (3.8+)

## 🎓 Learning Resources

- **LSTM Networks**: Understanding Long Short-Term Memory Networks
- **Time Series Forecasting**: Applied Time Series Analysis
- **Streamlit Documentation**: https://docs.streamlit.io
- **yfinance Documentation**: https://pypi.org/project/yfinance/

## 🔮 Future Enhancements

Potential features to add:
- [ ] Multi-stock comparison
- [ ] Portfolio optimization
- [ ] Sentiment analysis integration
- [ ] Real-time notifications
- [ ] Export predictions to CSV
- [ ] Ensemble model predictions
- [ ] Backtesting functionality
- [ ] Custom time ranges for prediction
- [ ] Support for more exchanges

## ⚠️ Disclaimer

**This application is for educational and research purposes only.**

Stock market prediction is inherently uncertain. This tool should NOT be used as the sole basis for investment decisions. Always:
- Do your own research
- Consult financial advisors
- Understand the risks
- Never invest more than you can afford to lose

The developers assume no responsibility for financial losses incurred through use of this application.

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Python Version**: 3.8+
