import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Configure Page
st.set_page_config(page_title="Global Stocks", layout="wide")
st.title("Global Stocks")

# Dictionaries for dropdowns
INDEX_MAP = {
    'S&P 500 (SPX)': '^GSPC',
    'NASDAQ 100 (NDQ)': '^NDX',
    'Dow Jones (DJI)': '^DJI',
    'Russell 2000 (RUT)': '^RUT',
    'Hang Seng (Hong Kong)': '^HSI',
    'Shanghai Composite (China)': '000001.SS',
    'Nikkei 225 (Japan)': '^N225',
    'KOSPI (Korea)': '^KS11',
    'ASX 200 (Australia)': '^AXJO',
    'EURO STOXX 50 (Europe)': '^STOXX50E',
    'APPLE (APPL)': 'APPL',
    'NVIDA (NVDA)': 'NVDA',
    'GOOGLE (GOOGL)': 'GOOGL',
    'AMAZON (AMZN)': 'AMZN',
    'META (META)': 'META',
    'MICROSOFT (MSFT)': 'MSFT',
    'TESLA (TSLA)': 'TSLA'
}

PERIOD_MAP = {
    '3 Months': 90,
    '6 Months': 180,
    '12 Months': 365,
    '2 Years': 730
}

col1, col2 = st.columns(2)
with col1:
    selected_index_name = st.selectbox("Select Index", list(INDEX_MAP.keys()))
with col2:
    selected_period_name = st.selectbox("Select Time Period", list(PERIOD_MAP.keys()), index=2)

ticker = INDEX_MAP[selected_index_name]
days = PERIOD_MAP[selected_period_name]

@st.cache_data(ttl=3600)
def load_data(ticker, days):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=days)
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

with st.spinner('Fetching Data...'):
    df = load_data(ticker, days)

if df.empty:
    st.error("Failed to fetch data for the selected ticker.")
else:
    # --- CALCULATION LOGIC ---
    df['Short_Up'] = df['High'].ewm(span=25, adjust=False).mean()
    df['Short_Dn'] = df['Low'].ewm(span=25, adjust=False).mean()
    df['Long_Up'] = df['High'].ewm(span=90, adjust=False).mean()
    df['Long_Dn'] = df['Low'].ewm(span=90, adjust=False).mean()

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = (df['EMA12'] - df['EMA26']) * 100
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    df['High_9_Count'] = 0
    df['Low_9_Count'] = 0
    for i in range(4, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-4]:
            df.iloc[i, df.columns.get_loc('High_9_Count')] = df['High_9_Count'].iloc[i-1] + 1
        if df['Close'].iloc[i] < df['Close'].iloc[i-4]:
            df.iloc[i, df.columns.get_loc('Low_9_Count')] = df['Low_9_Count'].iloc[i-1] + 1

    df['High_9_Valid'] = False
    df['Low_9_Valid'] = False
    for i in range(len(df)):
        if df['High_9_Count'].iloc[i] == 9:
            df.iloc[i-8:i+1, df.columns.get_loc('High_9_Valid')] = True
        if df['Low_9_Count'].iloc[i] == 9:
            df.iloc[i-8:i+1, df.columns.get_loc('Low_9_Valid')] = True

    df['Direct_Top_Pass'] = False
    df['Across_Top_Pass'] = False
    df['Top_Pass'] = False
    df['Top_Struct'] = False
    df['Direct_Bot_Pass'] = False
    df['Across_Bot_Pass'] = False
    df['Bot_Pass'] = False
    df['Bot_Struct'] = False

    gc_indices = []
    dc_indices = []

    def calc_mdif(dif, ref_dif):
        if pd.isna(ref_dif) or ref_dif == 0: return dif
        try:
            pdif = int(np.floor(np.log10(abs(ref_dif)))) - 1
            return int(dif / (10 ** pdif))
        except:
            return dif

    for i in range(1, len(df)):
        if df['DIF'].iloc[i] > df['DEA'].iloc[i] and df['DIF'].iloc[i-1] <= df['DEA'].iloc[i-1]:
            gc_indices.append(i)
        elif df['DIF'].iloc[i] < df['DEA'].iloc[i] and df['DIF'].iloc[i-1] >= df['DEA'].iloc[i-1]:
            dc_indices.append(i)

        if len(gc_indices) >= 2 and df['DIF'].iloc[i] > df['DEA'].iloc[i]:
            m1 = gc_indices[-1]
            m2 = gc_indices[-2]
            if m1 > m2:
                ch1 = df['Close'].iloc[m1 : i+1].max()
                ch2 = df['Close'].iloc[m2 : m1].max()
                difh2 = df['DIF'].iloc[m2 : m1].max()
                mdifh2 = calc_mdif(difh2, difh2)
                mdift2 = calc_mdif(df['DIF'].iloc[i], difh2)
                mdift2_prev = calc_mdif(df['DIF'].iloc[i-1], difh2)

                direct_top = (ch1 > ch2) and (mdift2 < mdifh2) and (df['MACD'].iloc[i] > 0) and (df['MACD'].iloc[i-1] > 0) and (mdift2 >= mdift2_prev)
                df.at[df.index[i], 'Direct_Top_Pass'] = direct_top

                across_top = False
                mdift3 = mdift3_prev = 0
                if len(gc_indices) >= 3:
                    m3 = gc_indices[-3]
                    if m2 > m3:
                        ch3 = df['Close'].iloc[m3 : m2].max()
                        difh3 = df['DIF'].iloc[m3 : m2].max()
                        mdifh3 = calc_mdif(difh3, difh3)
                        mdift3 = calc_mdif(df['DIF'].iloc[i], difh3)
                        mdift3_prev = calc_mdif(df['DIF'].iloc[i-1], difh3)
                        across_top = (ch1 > ch3) and (ch3 > ch2) and (mdift3 < mdifh3) and (df['MACD'].iloc[i] > 0) and (df['MACD'].iloc[i-1] > 0) and (mdift3 >= mdift3_prev)
                df.at[df.index[i], 'Across_Top_Pass'] = across_top
                df.at[df.index[i], 'Top_Pass'] = direct_top or across_top

                if df['Direct_Top_Pass'].iloc[i-1] and mdift2 < mdift2_prev:
                    df.at[df.index[i], 'Top_Struct'] = True
                elif df['Across_Top_Pass'].iloc[i-1] and mdift3 < mdift3_prev:
                    df.at[df.index[i], 'Top_Struct'] = True

        if len(dc_indices) >= 2 and df['DIF'].iloc[i] < df['DEA'].iloc[i]:
            n1 = dc_indices[-1]
            n2 = dc_indices[-2]
            if n1 > n2:
                cl1 = df['Close'].iloc[n1 : i+1].min()
                cl2 = df['Close'].iloc[n2 : n1].min()
                difl2 = df['DIF'].iloc[n2 : n1].min()
                mdifl2 = calc_mdif(difl2, difl2)
                mdifb2 = calc_mdif(df['DIF'].iloc[i], difl2)
                mdifb2_prev = calc_mdif(df['DIF'].iloc[i-1], difl2)

                direct_bot = (cl1 < cl2) and (mdifb2 > mdifl2) and (df['MACD'].iloc[i] < 0) and (df['MACD'].iloc[i-1] < 0) and (mdifb2 <= mdifb2_prev)
                df.at[df.index[i], 'Direct_Bot_Pass'] = direct_bot

                across_bot = False
                mdifb3 = mdifb3_prev = 0
                if len(dc_indices) >= 3:
                    n3 = dc_indices[-3]
                    if n2 > n3:
                        cl3 = df['Close'].iloc[n3 : n2].min()
                        difl3 = df['DIF'].iloc[n3 : n2].min()
                        mdifl3 = calc_mdif(difl3, difl3)
                        mdifb3 = calc_mdif(df['DIF'].iloc[i], difl3)
                        mdifb3_prev = calc_mdif(df['DIF'].iloc[i-1], difl3)
                        across_bot = (cl1 < cl3) and (cl3 < cl2) and (mdifb3 > mdifl3) and (df['MACD'].iloc[i] < 0) and (df['MACD'].iloc[i-1] < 0) and (mdifb3 <= mdifb3_prev)
                df.at[df.index[i], 'Across_Bot_Pass'] = across_bot
                df.at[df.index[i], 'Bot_Pass'] = direct_bot or across_bot

                if df['Direct_Bot_Pass'].iloc[i-1] and mdifb2 > mdifb2_prev:
                    df.at[df.index[i], 'Bot_Struct'] = True
                elif df['Across_Bot_Pass'].iloc[i-1] and mdifb3 > mdifb3_prev:
                    df.at[df.index[i], 'Bot_Struct'] = True

    # --- PLOTTING ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Short_Up'], line=dict(color='red', width=1), name='Short Up (25)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Short_Dn'], line=dict(color='green', width=1), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.1)', name='Short Dn (25)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Long_Up'], line=dict(color='magenta', width=1), name='Long Up (90)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Long_Dn'], line=dict(color='blue', width=1), name='Long Dn (90)'), row=1, col=1)

    colors = ['red' if m > 0 else 'green' for m in df['MACD']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD'], marker_color=colors, name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DIF'], line=dict(color='black', width=1), name='DIF'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DEA'], line=dict(color='blue', width=1), name='DEA'), row=2, col=1)

    for i in range(len(df)):
        if df['Top_Pass'].iloc[i]:
            fig.add_vrect(x0=df.index[i] - pd.Timedelta(days=0.5), x1=df.index[i] + pd.Timedelta(days=0.5),
                          fillcolor="green", opacity=0.3, line_width=0, row=2, col=1)
        if df['Top_Struct'].iloc[i]:
            fig.add_annotation(x=df.index[i], y=df['DIF'].iloc[i] * 1.1, text="▼ Top Struct",
                               font=dict(color='magenta', size=9), textangle=-90, showarrow=False, row=2, col=1)
        if df['Bot_Pass'].iloc[i]:
            fig.add_vrect(x0=df.index[i] - pd.Timedelta(days=0.5), x1=df.index[i] + pd.Timedelta(days=0.5),
                          fillcolor="red", opacity=0.3, line_width=0, row=2, col=1)
        if df['Bot_Struct'].iloc[i]:
            fig.add_annotation(x=df.index[i], y=df['DIF'].iloc[i] * 1.1, text="▲ Bot Struct",
                               font=dict(color='magenta', size=9), textangle=-90, showarrow=False, row=2, col=1)

    fig.update_layout(
        title=f"Quantitative Structure & 9-Turn System: {selected_index_name}",
        yaxis_title="Price",
        yaxis2_title="MACD",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=800
    )

    # Remove gaps for non-trading days (weekends and holidays)
    dt_all = pd.date_range(start=df.index.min(), end=df.index.max())
    dt_obs = [d.strftime("%Y-%m-%d") for d in df.index]
    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if d not in dt_obs]
    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

    st.plotly_chart(fig, use_container_width=True)
