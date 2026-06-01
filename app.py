import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 Investment Strategy Backtester")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "^GSPC")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-05-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2026-05-01"))
dip_threshold = st.sidebar.slider("Initial Dip Threshold (%)", 1, 20, 5) / 100
investment_amount = st.sidebar.number_input("Initial Buy Amount ($)", 1000)
subsequent_dip_threshold = st.sidebar.slider("Subsequent Dip Threshold (%)", 1, 10, 2) / 100
subsequent_investment_amount = st.sidebar.number_input("Subsequent Buy Amount ($)", 2000)
manual_sip_amount = st.sidebar.number_input("Monthly SIP Amount ($)", 500)

if st.sidebar.button("Run Backtest"):
    @st.cache_data
    def get_data(ticker, start, end):
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        return data['Close'].iloc[:, 0].to_frame(name='Close') if isinstance(data.columns, pd.MultiIndex) else data[['Close']]

    df = get_data(ticker, start_date, end_date)
    
    # Logic
    rolling_high = df['Close'].cummax()
    market_drawdown = (df['Close'] - rolling_high) / rolling_high
    
    # BTD
    inv_list, share_list = [], []
    last_buy_price, triggered = None, False
    for price, dd in zip(df['Close'], market_drawdown):
        inv, sh = 0, 0
        if dd == 0: triggered, last_buy_price = False, None
        if not triggered and dd <= -dip_threshold:
            triggered, last_buy_price = True, price
            inv, sh = investment_amount, investment_amount / price
        elif triggered and last_buy_price and price <= last_buy_price * (1 - subsequent_dip_threshold):
            last_buy_price, inv, sh = price, subsequent_investment_amount, subsequent_investment_amount / price
        inv_list.append(inv); share_list.append(sh)
    
    df['Btd_Invested_Amt'], df['Btd_Shares_Bought'] = inv_list, share_list
    df['Btd_Total_Invested'], df['Btd_Total_Shares'] = df['Btd_Invested_Amt'].cumsum(), df['Btd_Shares_Bought'].cumsum()
    df['Btd_Portfolio_Value'] = df['Btd_Total_Shares'] * df['Close']
    btd_max_dd = ((df['Btd_Portfolio_Value'] - df['Btd_Portfolio_Value'].cummax()) / df['Btd_Portfolio_Value'].cummax()).min() * 100

    # SIP
    df['Year_Month'] = df.index.to_period('M')
    is_first = df['Year_Month'] != df['Year_Month'].shift(1)
    df['Sip_Invested_Amt'] = np.where(is_first, manual_sip_amount, 0.0)
    df['Sip_Total_Invested'], df['Sip_Total_Shares'] = df['Sip_Invested_Amt'].cumsum(), (df['Sip_Invested_Amt'] / df['Close']).cumsum()
    df['Sip_Portfolio_Value'] = df['Sip_Total_Shares'] * df['Close']
    sip_max_dd = ((df['Sip_Portfolio_Value'] - df['Sip_Portfolio_Value'].cummax()) / df['Sip_Portfolio_Value'].cummax()).min() * 100

    # Metrics Function
    def get_metrics(cash, shares, val, max_dd, start, end):
        profit = val - cash
        ret = (profit / cash) * 100 if cash > 0 else 0
        years = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
        cagr = (((val / cash) ** (1/years)) - 1) * 100 if years > 0 and cash > 0 else 0
        return cash, val, profit, ret, max_dd, cagr

    btd = get_metrics(df['Btd_Total_Invested'].iloc[-1], df['Btd_Total_Shares'].iloc[-1], df['Btd_Portfolio_Value'].iloc[-1], btd_max_dd, start_date, end_date)
    sip = get_metrics(df['Sip_Total_Invested'].iloc[-1], df['Sip_Total_Shares'].iloc[-1], df['Sip_Portfolio_Value'].iloc[-1], sip_max_dd, start_date, end_date)

    # UI Display
    c1, c2 = st.columns(2)
    for i, title, data in [(c1, "Buy The Dip", btd), (c2, "Monthly SIP", sip)]:
        with i:
            st.subheader(title)
            st.write(f"Total Cash Deployed: ${data[0]:,.2f}")
            st.write(f"Final Portfolio Value: ${data[1]:,.2f}")
            st.write(f"Net Profit/Loss: ${data[2]:,.2f}")
            st.write(f"Strategy Return: {data[3]:.2f}%")
            st.write(f"Max Strategy Drawdown: {data[4]:.2f}%")
            st.write(f"Strategy CAGR: {data[5]:.2f}%")

    # Charting
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['Close'], label='Close Price', color='royalblue', alpha=0.4)
    buys = df[df['Btd_Invested_Amt'] > 0]
    ax.scatter(buys.index, buys['Close'], color='forestgreen', marker='^', label='BTD Buy')
    ax.legend()
    st.pyplot(fig)
