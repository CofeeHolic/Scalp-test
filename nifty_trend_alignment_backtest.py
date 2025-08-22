# -*- coding: utf-8 -*-
"""
Nifty-Stock Trend Alignment Backtesting Script (Enhanced)

This script backtests an intraday trading strategy that aligns stock trades
with the broader market trend, determined by the Nifty 50 index.

It has been enhanced with two additional filters for higher-probability trades:
1.  Higher-Timeframe (HTF) Confirmation: Aligns 5-min trades with the 1-hour trend.
2.  ATR Volatility Filter: Avoids trades during very low-volatility periods.

To run in Google Colab, first install the necessary libraries by running this cell:
!pip install yfinance pandas pandas_ta
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

# ==============================================================================
# 1. CONFIGURABLE PARAMETERS
# ==============================================================================
TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
NIFTY_TICKER = '^NSEI'
START_DATE = '2025-08-11' # Using a recent date for data availability
END_DATE = '2025-08-16'
INTERVAL_5M = '5m'
INTERVAL_60M = '60m'
ATR_VOLATILITY_THRESHOLD = 0.75 # Trade only if current ATR is > 75% of its average

# ==============================================================================
# 2. DATA PREPARATION (ENHANCED)
# ==============================================================================

def fetch_data(tickers, start, end, interval):
    """Fetches data for a list of tickers at a specific interval."""
    print(f"Fetching {interval} data for: {tickers}")
    data = yf.download(
        tickers=tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False
    )
    if data.empty:
        raise ValueError(f"No {interval} data fetched. Check tickers and date range.")
    # For single ticker fetches, yfinance might not create a MultiIndex.
    # We standardize to a MultiIndex for consistent processing.
    if len(tickers) == 1 and isinstance(data.columns, pd.Index):
         data.columns = pd.MultiIndex.from_product([data.columns, tickers])
    return data

def prepare_data(data_5m, data_60m, tickers, nifty_ticker):
    """
    Calculates indicators and merges Nifty/HTF signals into stock data.
    """
    prepared_data = {}

    # --- Prepare HTF Nifty Signal (60-min) ---
    nifty_60m_close = data_60m.loc[:, ('Close', nifty_ticker)]
    nifty_60m_ema = ta.ema(nifty_60m_close, length=50)
    nifty_htf_uptrend = (nifty_60m_close > nifty_60m_ema).rename('Nifty_HTF_Uptrend')

    for ticker in tickers:
        print(f"Preparing data for {ticker} with HTF and ATR filters...")

        # --- Select 5-min stock data and calculate base indicators ---
        stock_df = pd.DataFrame({
            'Open': data_5m[('Open', ticker)], 'High': data_5m[('High', ticker)],
            'Low': data_5m[('Low', ticker)], 'Close': data_5m[('Close', ticker)],
            'Volume': data_5m[('Volume', ticker)],
        }).copy()
        stock_df.dropna(inplace=True)
        if stock_df.empty:
            print(f"Warning: No 5m data for {ticker}.")
            continue

        stock_df['EMA_20'] = ta.ema(stock_df['Close'], length=20)
        stock_df['EMA_50'] = ta.ema(stock_df['Close'], length=50)
        stock_df['Volume_SMA_20'] = ta.sma(stock_df['Volume'], length=20)

        # --- Calculate Volatility Filter (ATR on 5-min data) ---
        stock_df['ATR_14'] = ta.atr(stock_df['High'], stock_df['Low'], stock_df['Close'], length=14)
        stock_df['ATR_SMA_50'] = ta.sma(stock_df['ATR_14'], length=50)

        # --- Prepare and align Stock HTF Signal (60-min) ---
        stock_60m_close = data_60m.loc[:, ('Close', ticker)]
        stock_60m_ema = ta.ema(stock_60m_close, length=50)
        stock_htf_uptrend = (stock_60m_close > stock_60m_ema).rename('Stock_HTF_Uptrend')

        # --- Prepare Nifty 5-min Trend Signal ---
        nifty_5m_close = data_5m.loc[:, ('Close', nifty_ticker)]
        nifty_5m_ema = ta.ema(nifty_5m_close, length=50)
        nifty_5m_uptrend = (nifty_5m_close > nifty_5m_ema).rename('Nifty_Uptrend_5m')

        # --- Merge all signals into the 5-min DataFrame ---
        # 1. Join the 5-min Nifty trend
        merged_df = stock_df.join(nifty_5m_uptrend)

        # 2. Join the aligned HTF signals by reindexing and forward-filling
        merged_df = merged_df.join(nifty_htf_uptrend.reindex(merged_df.index, method='ffill'))
        merged_df = merged_df.join(stock_htf_uptrend.reindex(merged_df.index, method='ffill'))

        # --- Final cleanup ---
        merged_df.dropna(inplace=True)
        merged_df.reset_index(inplace=True)

        prepared_data[ticker] = merged_df

    return prepared_data

# ==============================================================================
# 3. BACKTESTING ENGINE (ENHANCED)
# ==============================================================================
def run_backtest(data, ticker):
    """
    Runs the candle-by-candle backtest with enhanced filtering.
    """
    trades = []
    in_trade = False
    trade_type = None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_time = None

    print(f"\nRunning backtest for {ticker} with enhanced filters...")
    if len(data) < 1:
        print("Not enough data to backtest.")
        return []

    for i in range(len(data)):
        candle = data.iloc[i]

        # --- Trade Management: Check for exit on current candle's high/low ---
        if in_trade:
            exit_price = 0
            if trade_type == 'LONG':
                if candle['Low'] <= stop_loss: exit_price = stop_loss
                elif candle['High'] >= take_profit: exit_price = take_profit
            elif trade_type == 'SHORT':
                if candle['High'] >= stop_loss: exit_price = stop_loss
                elif candle['Low'] <= take_profit: exit_price = take_profit

            if exit_price != 0:
                pl = (exit_price - entry_price) if trade_type == 'LONG' else (entry_price - exit_price)
                trades.append({
                    'Ticker': ticker, 'Trade Type': trade_type,
                    'Entry Time': entry_time, 'Entry Price': entry_price,
                    'Exit Time': candle['Datetime'], 'Exit Price': exit_price, 'P/L': pl
                })
                in_trade = False
                trade_type = None

        # --- Entry Logic with Enhanced Filters ---
        if not in_trade:
            # --- Filter Conditions ---
            is_volatile_enough = candle['ATR_14'] > (candle['ATR_SMA_50'] * ATR_VOLATILITY_THRESHOLD)

            # --- Long Entry Conditions ---
            if (is_volatile_enough and
                candle['Nifty_Uptrend_5m'] and
                candle['Stock_HTF_Uptrend'] and
                candle['Close'] > candle['EMA_50'] and
                candle['Low'] <= candle['EMA_20'] and
                candle['Close'] > candle['Open'] and
                candle['Volume'] > candle['Volume_SMA_20']):

                in_trade = True
                trade_type = 'LONG'
                entry_price = candle['Close']
                entry_time = candle['Datetime']
                stop_loss = candle['Low']
                risk = entry_price - stop_loss
                if risk <= 0: # Ensure risk is positive
                    in_trade = False; continue
                take_profit = entry_price + (risk * 1.5)
                continue

            # --- Short Entry Conditions ---
            if (is_volatile_enough and
                not candle['Nifty_Uptrend_5m'] and
                not candle['Stock_HTF_Uptrend'] and
                candle['Close'] < candle['EMA_50'] and
                candle['High'] >= candle['EMA_20'] and
                candle['Close'] < candle['Open'] and
                candle['Volume'] > candle['Volume_SMA_20']):

                in_trade = True
                trade_type = 'SHORT'
                entry_price = candle['Close']
                entry_time = candle['Datetime']
                stop_loss = candle['High']
                risk = stop_loss - entry_price
                if risk <= 0: # Ensure risk is positive
                    in_trade = False; continue
                take_profit = entry_price - (risk * 1.5)

    print(f"Backtest complete for {ticker}. Found {len(trades)} trades.")
    return trades

# ==============================================================================
# 4. SUMMARY REPORT
# ==============================================================================
def generate_summary(trades_df):
    if trades_df.empty:
        print("\nNo trades were executed. Cannot generate summary.")
        return
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['P/L'] > 0]
    losing_trades = trades_df[trades_df['P/L'] <= 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_pl = trades_df['P/L'].sum()
    gross_profit = winning_trades['P/L'].sum()
    gross_loss = abs(losing_trades['P/L'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    print("\n--- Backtesting Summary Report ---")
    print(f"Total Number of Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total P/L: {total_pl:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print("----------------------------------\n")

# ==============================================================================
# 5. MAIN EXECUTION (ENHANCED)
# ==============================================================================
if __name__ == '__main__':
    try:
        all_tickers_list = TICKERS + [NIFTY_TICKER]

        # Step 1: Fetch data for both intervals
        data_5m = fetch_data(all_tickers_list, START_DATE, END_DATE, INTERVAL_5M)
        data_60m = fetch_data(all_tickers_list, START_DATE, END_DATE, INTERVAL_60M)

        # Step 2: Prepare data by combining 5m and 60m information
        all_stock_data = prepare_data(data_5m, data_60m, TICKERS, NIFTY_TICKER)

        all_trades = []
        # Step 3: Run backtest for each stock
        for ticker, stock_data in all_stock_data.items():
            trades = run_backtest(stock_data, ticker)
            all_trades.extend(trades)

        # Step 4: Aggregate results and generate report
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.sort_values(by='Entry Time', inplace=True)
            trades_df.reset_index(drop=True, inplace=True)

            print("\n--- All Simulated Trades ---")
            print(trades_df.to_string())

            generate_summary(trades_df)
        else:
            print("\nNo trades were executed across any of the tickers.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

    print("Script finished.")
