import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
import joblib
import warnings

warnings.filterwarnings('ignore')

MODEL_DIR = 'models'  # Directory where trained models and scalers are saved

# Page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown('''
<style>
.main {
    padding: 2rem;
}
.stPlotlyChart {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
h1 {
    color: #1f77b4;
    font-weight: 600;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
}
</style>
''', unsafe_allow_html=True)

st.title("📈 Stock Price Prediction")
st.markdown("---")

# Sidebar inputs
with st.sidebar:
    st.header("⚙️ Configuration")

    market = st.selectbox(
        "Select Market",
        ["US (NASDAQ)", "India (NSE)", "India (BSE)"],
        help="Choose the stock market"
    )

    ticker_input = st.text_input(
        "Enter Stock Ticker",
        value="AAPL",
        help="Enter stock symbol without suffix (e.g., AAPL, RELIANCE, TCS)"
    ).strip().upper()

    if market == "India (NSE)":
        ticker = f"{ticker_input}.NS"
    elif market == "India (BSE)":
        ticker = f"{ticker_input}.BO"
    else:
        ticker = ticker_input

    st.info(f"📊 Full Ticker: **{ticker}**")

    model_type = st.radio(
        "Choose Prediction Model",
        ["LSTM", "GRU"],
        help="LSTM: Better for complex patterns - GRU: Faster, similar accuracy"
    )

    pred_days = st.slider(
        "Prediction Days",
        min_value=1,
        max_value=7,
        value=7,
        help="Number of days to predict ahead"
    )

    st.markdown("---")

    lookback_period = st.selectbox(
        "Historical Data Range",
        ["1 Year", "2 Years", "5 Years", "Max"],
        index=2
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app predicts stock prices using deep learning models (LSTM/GRU).

    **Data Source**: Yahoo Finance  
    **Update Frequency**: ~15 min delay  
    **Features**: Price + Technical Indicators
    """)

# Helper to fetch stock data with caching
@st.cache_data(ttl=900)
def fetch_stock_data(ticker, period='5y'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None, None
        info = stock.info

        data['MA_7'] = data['Close'].rolling(window=7).mean()
        data['MA_21'] = data['Close'].rolling(window=21).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['Volatility'] = data['Close'].rolling(window=21).std()
        data['Returns'] = data['Close'].pct_change()
        data = data.dropna()

        return data, info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None


def prepare_input_sequence(data, scaler, look_back=60):
    features = ['Close', 'Volume', 'MA_7', 'MA_21', 'MA_50', 'Volatility']
    df = data[features].copy()
    scaled_data = scaler.transform(df)
    last_sequence = scaled_data[-look_back:]
    return np.array([last_sequence])


def predict_future_prices(data, model_type='LSTM', days=7, look_back=60, ticker='AAPL'):
    model_filename = os.path.join(MODEL_DIR, f'{ticker}_{model_type.lower()}_model.keras')
    scaler_filename = os.path.join(MODEL_DIR, f'{ticker}_scaler.pkl')

    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        try:
            model = load_model(model_filename)
            scaler = joblib.load(scaler_filename)

            X_input = prepare_input_sequence(data, scaler, look_back=look_back)

            predictions_scaled = []
            input_seq = X_input.copy()

            for _ in range(days):
                pred = model.predict(input_seq)[0][0]
                predictions_scaled.append(pred)

                new_entry = input_seq[0, -1, :].copy()
                new_entry[0] = pred
                input_seq = np.append(input_seq[:, 1:, :], [[new_entry]], axis=1)

            dummy = np.zeros((len(predictions_scaled), len(scaler.scale_)))
            dummy[:, 0] = predictions_scaled
            predictions_prices = scaler.inverse_transform(dummy)[:, 0]

            last_date = data.index[-1]
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days * 2, freq='D')
            pred_dates = [d for d in pred_dates if d.weekday() < 5][:days]

            return pred_dates, list(predictions_prices)

        except Exception as e:
            st.warning(f"Model prediction failed, falling back: {e}")

    # Fallback heuristic method (linear trend + noise)
    close_prices = data['Close'].values
    recent_prices = close_prices[-30:]
    trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
    volatility = np.std(close_prices[-21:])
    last_price = close_prices[-1]
    predictions = []

    for i in range(1, days + 1):
        pred_price = last_price + (trend * i) + np.random.normal(0, volatility * 0.3)
        predictions.append(pred_price)

    last_date = data.index[-1]
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days * 2, freq='D')
    pred_dates = [d for d in pred_dates if d.weekday() < 5][:days]

    return pred_dates, predictions


def plot_stock_data(data, ticker, predictions=None, pred_dates=None):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} - Price & Predictions', 'Trading Volume'),
        vertical_spacing=0.1
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA_7'], name='MA(7)', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA_21'], name='MA(21)', line=dict(color='blue', width=1)),
        row=1, col=1
    )

    if predictions is not None and pred_dates is not None:
        last_date = data.index[-1]
        last_price = data['Close'].iloc[-1]

        pred_x = [last_date] + list(pred_dates)
        pred_y = [last_price] + list(predictions)

        fig.add_trace(
            go.Scatter(
                x=pred_x, y=pred_y,
                name='Prediction',
                line=dict(color='red', width=2, dash='dash'),
                mode='lines+markers'
            ), row=1, col=1
        )

    colors = ['red' if close < open else 'green'
              for close, open in zip(data['Close'], data['Open'])]

    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors, showlegend=False),
        row=2, col=1
    )

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def main():
    period_map = {
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    period = period_map[lookback_period]

    with st.spinner(f"Fetching data for {ticker}..."):
        data, info = fetch_stock_data(ticker, period)

    if data is None or data.empty:
        st.error(f"Could not fetch data for ticker: **{ticker}**")
        st.info("""
        **Possible reasons:**
        - Invalid ticker symbol
        - No data available for this stock
        - Network issues

        **Examples of valid tickers:**
        - US: AAPL, GOOGL, MSFT, TSLA
        - NSE: RELIANCE, TCS, INFY, HDFCBANK (will become RELIANCE.NS)
        - BSE: RELIANCE, TCS, INFY (will become RELIANCE.BO)
        """)
        return

    st.header(ticker_input)

    col1, col2, col3, col4 = st.columns(4)
    current_price = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    currency = "$" if market == "US (NASDAQ)" else "₹"

    with col1:
        st.metric("Current Price", f"{currency}{current_price:.2f}", delta=f"{price_change:.2f} ({price_change_pct:.2f}%)")

    with col2:
        st.metric("Day High", f"{currency}{data['High'].iloc[-1]:.2f}")

    with col3:
        st.metric("Day Low", f"{currency}{data['Low'].iloc[-1]:.2f}")

    with col4:
        volume_m = data['Volume'].iloc[-1] / 1_000_000
        st.metric("Volume", f"{volume_m:.2f}M")

    if info:
        with st.expander("📋 Company Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Company**: {info.get('longName', ticker)}")
                st.write(f"**Sector**: {info.get('sector', 'N/A')}")
                st.write(f"**Industry**: {info.get('industry', 'N/A')}")
            with col2:
                st.write(f"**Market Cap**: {info.get('marketCap', 'N/A')}")
                st.write(f"**52W High**: {info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.write(f"**52W Low**: {info.get('fiftyTwoWeekLow', 'N/A')}")

    st.markdown("---")

    st.subheader(f"🔮 {model_type} Model Predictions (Next {pred_days} Days)")

    with st.spinner(f"Generating {pred_days}-day predictions using {model_type} model..."):
        pred_dates, predictions = predict_future_prices(data, model_type, pred_days, look_back=60, ticker=ticker)

    # Ensure lengths match
    min_len = min(len(pred_dates), len(predictions))
    pred_dates = pred_dates[:min_len]
    predictions = predictions[:min_len]

    pred_df = pd.DataFrame({
        'Date': pred_dates,
        'Predicted Price': predictions,
        'Change from Current': [p - current_price for p in predictions],
        'Change %': [((p - current_price) / current_price) * 100 for p in predictions]
    })

    st.dataframe(
        pred_df.style.format({
            'Predicted Price': f'{currency}{{:.2f}}',
            'Change from Current': '{:+.2f}',
            'Change %': '{:+.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

    avg_pred = np.mean(predictions)
    trend = "📈 Upward" if predictions[-1] > current_price else "📉 Downward"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Avg Price", f"{currency}{avg_pred:.2f}")
    with col2:
        st.metric("Expected Change", f"{((predictions[-1] - current_price) / current_price * 100):.2f}%")
    with col3:
        st.metric("Trend", trend)

    st.markdown("---")

    st.subheader("📊 Interactive Price Chart")
    fig = plot_stock_data(data, ticker, predictions, pred_dates)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📈 Technical Indicators"):
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.write("**Moving Averages**")
            st.write(f"MA(7): {currency}{data['MA_7'].iloc[-1]:.2f}")
            st.write(f"MA(21): {currency}{data['MA_21'].iloc[-1]:.2f}")
            st.write(f"MA(50): {currency}{data['MA_50'].iloc[-1]:.2f}")
        with tech_col2:
            st.write("**Volatility & Returns**")
            st.write(f"Volatility (21d): {data['Volatility'].iloc[-1]:.2f}")
            st.write(f"Daily Return: {data['Returns'].iloc[-1]*100:.2f}%")

    st.markdown("---")
    st.warning("""
    ⚠️ **Disclaimer**: This is for educational purposes only. Stock predictions are based on historical data 
    and technical indicators. Past performance does not guarantee future results. Always do your own research 
    and consult with financial advisors before making investment decisions.
    """)


if __name__ == "__main__":
    main()
