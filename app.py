import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 Investment Strategy Backtester")

# Sidebar Inputs
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "^GSPC")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-05-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2026-05-01"))

st.sidebar.subheader("Buy The Dip Settings")
dip_threshold = st.sidebar.slider("Initial Dip Threshold (%)", 1, 20, 5) / 100
investment_amount = st.sidebar.number_input("Initial Buy Amount ($)", 1000)
subsequent_dip_threshold = st.sidebar.slider("Subsequent Dip Threshold (%)", 1, 10, 2) / 100
subsequent_investment_amount = st.sidebar.number_input("Subsequent Buy Amount ($)", 2000)

st.sidebar.subheader("SIP Settings")
manual_sip_amount = st.sidebar.number_input("Monthly SIP Amount ($)", 500)

# Caching data for speed
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        df = data['Close'].iloc[:, 0].to_frame(name='Close')
    else:
        df = data[['Close']].copy()
    return df

df = load_data(ticker, start_date, end_date)

# Logic
rolling_high = df['Close'].cummax()
market_drawdown = (df['Close'] - rolling_high) / rolling_high

# BTD Loop
investment_triggers, shares_purchased = [], []
last_buy_price, has_triggered_initial_dip = None, False
for price, dd in zip(df['Close'], market_drawdown):
    inv, sh = 0, 0
    if dd == 0: has_triggered_initial_dip, last_buy_price = False, None
    if not has_triggered_initial_dip and dd <= -dip_threshold:
        has_triggered_initial_dip, last_buy_price = True, price
        inv, sh = investment_amount, investment_amount / price
    elif has_triggered_initial_dip and last_buy_price and price <= last_buy_price * (1 - subsequent_dip_threshold):
        last_buy_price, inv, sh = price, subsequent_investment_amount, subsequent_investment_amount / price
    investment_triggers.append(inv); shares_purchased.append(sh)

df['Btd_Invested_Amt'], df['Btd_Shares_Bought'] = investment_triggers, shares_purchased
df['Btd_Total_Invested'], df['Btd_Total_Shares'] = df['Btd_Invested_Amt'].cumsum(), df['Btd_Shares_Bought'].cumsum()
df['Btd_Portfolio_Value'] = df['Btd_Total_Shares'] * df['Close']

# SIP Logic
df['Year_Month'] = df.index.to_period('M')
is_first_day = df['Year_Month'] != df['Year_Month'].shift(1)
df['Sip_Invested_Amt'] = np.where(is_first_day, manual_sip_amount, 0.0)
df['Sip_Total_Invested'], df['Sip_Total_Shares'] = df['Sip_Invested_Amt'].cumsum(), (df['Sip_Invested_Amt'] / df['Close']).cumsum()
df['Sip_Portfolio_Value'] = df['Sip_Total_Shares'] * df['Close']

# Display
col1, col2 = st.columns(2)

def calculate_cagr(final_val, total_invested, start, end):
    years = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
    return (((final_val / total_invested) ** (1/years)) - 1) * 100 if years > 0 and total_invested > 0 else 0

with col1:
    st.subheader("Buy The Dip")
    st.write(f"Total Invested: ${df['Btd_Total_Invested'].iloc[-1]:,.2f}")
    st.write(f"Final Value: ${df['Btd_Portfolio_Value'].iloc[-1]:,.2f}")
    st.write(f"CAGR: {calculate_cagr(df['Btd_Portfolio_Value'].iloc[-1], df['Btd_Total_Invested'].iloc[-1], start_date, end_date):.2f}%")

with col2:
    st.subheader("Monthly SIP")
    st.write(f"Total Invested: ${df['Sip_Total_Invested'].iloc[-1]:,.2f}")
    st.write(f"Final Value: ${df['Sip_Portfolio_Value'].iloc[-1]:,.2f}")
    st.write(f"CAGR: {calculate_cagr(df['Sip_Portfolio_Value'].iloc[-1], df['Sip_Total_Invested'].iloc[-1], start_date, end_date):.2f}%")

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df['Close'], label='Price', alpha=0.3)
ax.scatter(df[df['Btd_Invested_Amt'] > 0].index, df[df['Btd_Invested_Amt'] > 0]['Close'], color='red', label='BTD Buy')
ax.legend()
st.pyplot(fig)
