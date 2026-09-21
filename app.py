import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import yfinance as yf

# =====================================================================
# 1. CORE TECHNICAL INDICATORS PIPELINE
# =====================================================================
def calculate_indicators(df):
    """Calculates all 5 strict mechanical checklist parameters precisely from live data."""
    df = df.sort_index().copy()

    # Index feeds can return zero or missing volume. Replace it with 1 so
    # session VWAP remains defined for index data such as ^NSEI/^NSEBANK.
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').replace(0, 1).fillna(1)
    else:
        df['volume'] = 1.0

    # A. Moving Averages
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()

    # B. Intraday Session VWAP
    df['date_only'] = df.index.date
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = typical_price * df['volume']

    df['cum_tp_vol'] = df.groupby('date_only')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    df['VWAP'] = df['cum_tp_vol'] / df['cum_vol']

    # C. Relative Strength Index (RSI 14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # D. Moving Average Convergence Divergence (MACD 12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']

    return df

# =====================================================================
# 2. STRATEGY CHECKLIST & RISK EVALUATOR MATRICES
# =====================================================================
def evaluate_entry_quality(df, symbol):
    """Evaluates the risk profile and flashes signals instantly on the screen."""
    if len(df) < 30 or df['VWAP'].isna().all():
        return {"status": "⏳ INITIALIZING STREAMS", "pct": "0.0%", "action": "Analyzing market blocks...", "price": 0, "vwap": 0}

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    is_crossover = (last_candle['SMA_7'] > last_candle['EMA_21']) and (prev_candle['SMA_7'] <= prev_candle['EMA_21'])
    above_vwap = last_candle['close'] > last_candle['VWAP']
    rsi_bullish = last_candle['RSI'] > 50
    macd_growing = last_candle['MACD_Histogram'] > prev_candle['MACD_Histogram']
    follow_thru = last_candle['close'] > prev_candle['high']
    close_price = last_candle['close']
    vwap_price = last_candle['VWAP']
    distance_pct = ((close_price - vwap_price) / vwap_price) * 100

    if is_crossover and above_vwap and rsi_bullish and macd_growing and follow_thru:
        if distance_pct > 10.0:
            return {"status": "⚠️ CHASING / POOR RISK", "pct": f"+{distance_pct:.2f}%", "action": "❌ SKIP ENTRY (Price too far away from VWAP support floor)", "price": close_price, "vwap": vwap_price}
        return {"status": "🟢 OPTIMAL ENTRY READY", "pct": f"+{distance_pct:.2f}%", "action": "✅ EXECUTE POSITION (Safe proximity boundaries cleared)", "price": close_price, "vwap": vwap_price}

    return {"status": "⏳ MONITORING TREND LINE", "pct": f"{distance_pct:.2f}%", "action": "No clean breakout alignment detected.", "price": close_price, "vwap": vwap_price}

# =====================================================================
# 3. STREAMLIT VISUAL APP PANEL LAYOUT
# =====================================================================
st.set_page_config(page_title="Automated Nifty Scanner", layout="wide")
st.title("📈 Live Nifty Index Strategy Dashboard")
st.caption("Observation-Only Engine. Pulling real-time market data across 5-minute candlestick intervals with Timezone Correction.")

st.sidebar.header("🎯 Target Selection Settings")
underlying = st.sidebar.selectbox("Select Index Asset", ["NIFTY", "BANKNIFTY"], index=0)
ticker_mapping = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
TARGET_TICKER = ticker_mapping.get(underlying, "^NSEI")
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Monitor Targets:")
st.sidebar.code(f"Live Source: {TARGET_TICKER}")

if "scanner_active" not in st.session_state:
    st.session_state.scanner_active = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Launch Live 5-Min Monitor Loop", type="primary"):
        st.session_state.scanner_active = True
with col2:
    if st.button("🛑 Terminate Dashboard Session"):
        st.session_state.scanner_active = False
        st.write("Scanner paused safely.")

if st.session_state.scanner_active:
    st.success(f"🚀 Live Observation Stream Online for {underlying}! Tracking index movements dynamically...")
    scorecard_placeholder = st.empty()
    data_tables_placeholder = st.empty()

    while st.session_state.scanner_active:
        current_time = datetime.now()
        if current_time.minute % 5 == 0 and current_time.second < 10:
            st.toast("⏰ Fetching fresh live market data frames from exchange...", icon="🔄")
            try:
                df_raw = yf.download(tickers=TARGET_TICKER, period="5d", interval="5m", progress=False)
                if not df_raw.empty:
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.get_level_values(0)
                    df_live = df_raw.copy()
                    if df_live.index.tz is None:
                        df_live.index = df_live.index.tz_localize('UTC')
                    df_live.index = df_live.index.tz_convert('Asia/Kolkata')
                    df_live.index.name = 'datetime'
                    df_live['ist_time'] = df_live.index.strftime('%Y-%m-%d %H:%M')
                    df_live.columns = [str(c).lower() for c in df_live.columns]

                    today_ist = pd.Timestamp.now(tz='Asia/Kolkata').date()
                    df_live = df_live[df_live.index.date == today_ist].sort_index()

                    if not df_live.empty:
                        df_processed = calculate_indicators(df_live)
                        eval_metrics = evaluate_entry_quality(df_processed, underlying)
                        scorecard_rows = [{
                            "Index Ticker": underlying,
                            "Strategy Status Flag": eval_metrics["status"],
                            "Distance Relative to VWAP": eval_metrics["pct"],
                            "Live Close Price": f"₹{eval_metrics['price']:.2f}",
                            "Session VWAP Level": f"₹{eval_metrics['vwap']:.2f}",
                            "Tactical System Action": eval_metrics["action"]
                        }]
                        with scorecard_placeholder.container():
                            st.markdown("### 📊 Live Strategy Entry Evaluation Board")
                            st.table(pd.DataFrame(scorecard_rows))
                        with data_tables_placeholder.container():
                            st.subheader("🗂️ Today's 5-Minute Candlestick History (IST)")
                            display_columns = ['ist_time', 'open', 'high', 'low', 'close', 'SMA_7', 'EMA_21', 'VWAP', 'RSI', 'MACD_Histogram']
                            st.dataframe(df_processed[display_columns], use_container_width=True, height=600)
                    else:
                        st.warning("No data rows found for today's session yet.")
                else:
                    st.error("Market feed failed to return data frames.")
            except Exception as e:
                st.error(f"Error during calculations loop: {e}")
        time.sleep(1)
