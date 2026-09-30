import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

FEATURES = ["Open", "High", "Low", "Volume", "MA_7", "MA_30",
            "MA_50", "Volatility", "High_Low_Diff", "Day"]

POPULAR = {
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "SBI (SBIN.NS)": "SBIN.NS",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
    "Custom ticker...": "CUSTOM",
}


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return df
    # newer yfinance versions return MultiIndex columns -> flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Day"] = range(len(df))
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_30"] = df["Close"].rolling(30).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Price_Change"] = df["Close"].diff()
    df["Volatility"] = df["Close"].rolling(7).std()
    df["High_Low_Diff"] = df["High"] - df["Low"]
    df["Daily_Return"] = df["Close"].pct_change()
    return df.dropna().reset_index(drop=True)


def train_model(df: pd.DataFrame, test_size: float):
    X, y = df[FEATURES], df["Close"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression().fit(X_train_s, y_train)
    return model, scaler, X_train, X_test, y_train, y_test, \
        model.predict(X_train_s), model.predict(X_test_s)


def forecast(df, model, scaler, n_days):
    """Recursive forecast: each predicted close feeds the next day's features."""
    closes = list(df["Close"].values)
    last_volume = float(df["Volume"].iloc[-1])
    day = int(df["Day"].iloc[-1])
    preds = []
    for _ in range(n_days):
        s = pd.Series(closes)
        prev = closes[-1]
        day += 1
        row = {
            "Open": prev, "High": prev, "Low": prev,  # unknown for future days
            "Volume": last_volume,
            "MA_7": s.tail(7).mean(),
            "MA_30": s.tail(30).mean(),
            "MA_50": s.tail(50).mean(),
            "Volatility": s.tail(7).std(),
            "High_Low_Diff": 0.0,
            "Day": day,
        }
        X_new = pd.DataFrame([row])[FEATURES]
        p = float(model.predict(scaler.transform(X_new))[0])
        preds.append(p)
        closes.append(p)
    return preds


# ------------------------------------------------------------------ sidebar
st.title("📈 Stock Price Prediction")
st.caption("Linear Regression on Yahoo Finance data — a learning project, not financial advice.")

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Stock", list(POPULAR.keys()))
    if POPULAR[choice] == "CUSTOM":
        ticker = st.text_input("Ticker symbol", "RELIANCE.NS").upper().strip()
    else:
        ticker = POPULAR[choice]

    start_date = st.date_input("Start date", dt.date(2020, 1, 1))
    end_date = st.date_input("End date", dt.date.today())
    forecast_days = st.slider("Days to forecast", 1, 30, 7)
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    run = st.button("Run analysis", type="primary", use_container_width=True)

if not run:
    st.info("Pick a stock in the sidebar and click **Run analysis**.")
    st.stop()

# ------------------------------------------------------------------ pipeline
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner(f"Fetching data for {ticker}..."):
    raw = load_data(ticker, start_date, end_date)

if raw.empty:
    st.error(f"No data found for '{ticker}'. Check the symbol (NSE stocks need '.NS').")
    st.stop()

df = add_features(raw)
if len(df) < 100:
    st.error("Not enough data after feature engineering. Choose a longer date range.")
    st.stop()

(model, scaler, X_train, X_test, y_train, y_test,
 y_train_pred, y_test_pred) = train_model(df, test_size)

cur = "₹" if ticker.endswith((".NS", ".BO")) else "$"

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

current_price = float(df["Close"].iloc[-1])
previous_price = float(df["Close"].iloc[-2])
change = current_price - previous_price
change_pct = change / previous_price * 100
ma7, ma30, ma50 = (float(df[c].iloc[-1]) for c in ("MA_7", "MA_30", "MA_50"))

if current_price > ma7 > ma30:
    signal = "🟢 Bullish trend"
elif current_price < ma7 < ma30:
    signal = "🔴 Bearish trend"
else:
    signal = "🟡 Neutral / sideways"

future_preds = forecast(df, model, scaler, forecast_days)
future_dates = pd.bdate_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
future_change = (future_preds[-1] - current_price) / current_price * 100

# ------------------------------------------------------------------ overview
st.subheader(f"{ticker} — Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current price", f"{cur}{current_price:,.2f}", f"{change_pct:+.2f}%")
c2.metric("7-day MA", f"{cur}{ma7:,.2f}")
c3.metric("30-day MA", f"{cur}{ma30:,.2f}")
c4.metric("Trend signal", signal)

tab_hist, tab_model, tab_fc, tab_data = st.tabs(
    ["📊 Price history", "🤖 Model performance", "🔮 Forecast", "🗂 Data"]
)

# ------------------------------------------------------------------ tab 1
with tab_hist:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["Date"], df["Close"], label="Close", color="blue", linewidth=1.8)
    ax.plot(df["Date"], df["MA_7"], label="7-day MA", color="orange", linestyle="--")
    ax.plot(df["Date"], df["MA_30"], label="30-day MA", color="green", linestyle="--")
    ax.plot(df["Date"], df["MA_50"], label="50-day MA", color="red", linestyle="--")
    ax.set_ylabel(f"Price ({cur})")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ------------------------------------------------------------------ tab 2
with tab_model:
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Training set**")
        st.metric("RMSE", f"{cur}{train_rmse:,.2f}")
        st.metric("MAE", f"{cur}{train_mae:,.2f}")
        st.metric("R²", f"{train_r2:.4f}")
    with m2:
        st.markdown("**Test set**")
        st.metric("RMSE", f"{cur}{test_rmse:,.2f}")
        st.metric("MAE", f"{cur}{test_mae:,.2f}")
        st.metric("R²", f"{test_r2:.4f}")

    st.warning(
        "**Read this R² with care.** The model predicts a day's Close using that same day's "
        "Open/High/Low and moving averages that include the Close, so a high R² mostly reflects "
        "this overlap rather than real forecasting power. To predict genuinely *future* prices, "
        "use only past data (e.g. shift the target by one day)."
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    test_dates = df["Date"].iloc[len(X_train):]
    errors = y_test.values - y_test_pred

    axes[0, 0].plot(df["Date"].iloc[:len(X_train)], y_train.values, label="Actual", color="blue", alpha=0.7)
    axes[0, 0].plot(df["Date"].iloc[:len(X_train)], y_train_pred, label="Predicted", color="red", alpha=0.7)
    axes[0, 0].set_title("Training: actual vs predicted")
    axes[0, 0].legend()

    axes[0, 1].plot(test_dates, y_test.values, label="Actual", color="blue")
    axes[0, 1].plot(test_dates, y_test_pred, label="Predicted", color="red")
    axes[0, 1].set_title("Testing: actual vs predicted")
    axes[0, 1].legend()

    axes[1, 0].hist(errors, bins=30, color="teal", alpha=0.7, edgecolor="black")
    axes[1, 0].axvline(0, color="red", linestyle="--")
    axes[1, 0].set_title("Error distribution")

    imp = pd.DataFrame({"Feature": FEATURES, "Importance": np.abs(model.coef_)}) \
        .sort_values("Importance")
    axes[1, 1].barh(imp["Feature"], imp["Importance"], color="teal")
    axes[1, 1].set_title("Feature importance (|coefficient|)")

    for a in axes.flat:
        a.grid(alpha=0.3)
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------------ tab 3
with tab_fc:
    f1, f2, f3 = st.columns(3)
    f1.metric("Trend", "📈 Upward" if future_preds[-1] > current_price else "📉 Downward")
    f2.metric(f"Change ({forecast_days} days)", f"{future_change:+.2f}%")
    f3.metric("Range", f"{cur}{min(future_preds):,.0f} – {cur}{max(future_preds):,.0f}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["Date"].iloc[-60:], df["Close"].iloc[-60:], label="Historical", color="blue", linewidth=2)
    ax.plot(future_dates, future_preds, label="Forecast", color="red",
            marker="o", linestyle="--", linewidth=2)
    ax.axvline(df["Date"].iloc[-1], color="gray", linestyle=":", label="Last data point")
    ax.set_ylabel(f"Price ({cur})")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    fc_df = pd.DataFrame({
        "Date": future_dates.date,
        f"Predicted price ({cur})": np.round(future_preds, 2),
        "Change from current (%)": np.round((np.array(future_preds) - current_price) / current_price * 100, 2),
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)
    st.caption("Multi-day forecasts from a simple linear model get unreliable quickly. "
               "Treat them as a demo of the technique.")

# ------------------------------------------------------------------ tab 4
with tab_data:
    st.dataframe(df.tail(100), use_container_width=True)
    st.download_button(
        "Download full dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_data.csv",
        mime="text/csv",
    )

st.divider()
st.caption("⚠️ Educational project only. Not investment advice.")
