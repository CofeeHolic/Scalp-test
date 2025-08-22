# -*- coding: utf-8 -*-
"""
Nifty-Stock Trend Alignment Backtesting Script

This script backtests an intraday trading strategy that aligns stock trades
with the broader market trend, determined by the Nifty 50 index. It is
designed to run in a Google Colab environment.

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
START_DATE = '2025-08-11' # Using a recent date for 5-min data availability
END_DATE = '2025-08-16'
INTERVAL = '5m'
# Note: yfinance has limitations on fetching 5-min data for past dates.
# Please adjust dates to a recent period if you encounter data issues.

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================

def fetch_all_data(tickers, nifty_ticker, start, end, interval):
    """Fetches data for Nifty and all specified stock tickers."""
    all_tickers = tickers + [nifty_ticker]
    print(f"Fetching 5-minute data for: {all_tickers}")
    data = yf.download(
        tickers=all_tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    if data.empty:
        raise ValueError("No data fetched. Check tickers and date range.")
    return data

def prepare_data(data, tickers, nifty_ticker):
    """
    Calculates indicators and merges Nifty trend signal into stock data.

    Returns:
        dict: A dictionary of DataFrames, one for each stock, with all
              necessary indicators and the Nifty trend signal.
    """
    prepared_data = {}

    # 1. Prepare Nifty Data
    nifty_data = data['Close'][nifty_ticker].to_frame()
    nifty_data.columns = ['Nifty_Close']
    nifty_data['Nifty_50_EMA'] = ta.ema(nifty_data['Nifty_Close'], length=50)
    nifty_data['Nifty_Uptrend'] = nifty_data['Nifty_Close'] > nifty_data['Nifty_50_EMA']

    # 2. Prepare each stock's data
    for ticker in tickers:
        # Select the OHLCV data for the current ticker from the multi-level column DataFrame
        stock_df = pd.DataFrame({
            'Open': data[('Open', ticker)],
            'High': data[('High', ticker)],
            'Low': data[('Low', ticker)],
            'Close': data[('Close', ticker)],
            'Volume': data[('Volume', ticker)],
        })
        stock_df.dropna(inplace=True) # Drop rows where this stock had no data
        if stock_df.empty:
            print(f"Warning: No data for {ticker} after initial cleaning.")
            continue

        # Calculate stock indicators
        stock_df['EMA_20'] = ta.ema(stock_df['Close'], length=20)
        stock_df['EMA_50'] = ta.ema(stock_df['Close'], length=50)
        stock_df['Volume_SMA_20'] = ta.sma(stock_df['Volume'], length=20)

        # 3. Merge Nifty trend data into the stock DataFrame
        # This aligns each stock candle with the corresponding Nifty trend signal
        merged_df = stock_df.join(nifty_data[['Nifty_Uptrend']], how='inner')

        merged_df.dropna(inplace=True) # Clean NaNs from indicator calculations
        merged_df.reset_index(inplace=True) # Move Datetime from index to column

        prepared_data[ticker] = merged_df
        print(f"Data prepared for {ticker}.")

    return prepared_data

# ==============================================================================
# 3. BACKTESTING ENGINE
# ==============================================================================

def run_backtest(data, ticker):
    """Runs the candle-by-candle backtest for a single stock."""
    trades = []
    in_trade = False
    trade_type = None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_time = None

    print(f"\nRunning backtest for {ticker}...")
    if len(data) < 1:
        print("Not enough data to backtest.")
        return []

    for i in range(len(data)):
        candle = data.iloc[i]

        # --- Trade Management: Check for exit on current candle's high/low ---
        if in_trade:
            exit_price = 0
            if trade_type == 'LONG':
                if candle['Low'] <= stop_loss:
                    exit_price = stop_loss
                elif candle['High'] >= take_profit:
                    exit_price = take_profit
            elif trade_type == 'SHORT':
                if candle['High'] >= stop_loss:
                    exit_price = stop_loss
                elif candle['Low'] <= take_profit:
                    exit_price = take_profit

            if exit_price != 0:
                pl = (exit_price - entry_price) if trade_type == 'LONG' else (entry_price - exit_price)
                trades.append({
                    'Ticker': ticker,
                    'Trade Type': trade_type,
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time': candle['Datetime'],
                    'Exit Price': exit_price,
                    'P/L': pl
                })
                in_trade = False
                trade_type = None

        # --- Entry Logic: Check for new trade opportunities ---
        if not in_trade:
            # Long Entry Conditions
            is_nifty_up = candle['Nifty_Uptrend']
            is_stock_trend_up = candle['Close'] > candle['EMA_50']
            is_pullback_to_ema20 = candle['Low'] <= candle['EMA_20']
            is_bullish_candle = candle['Close'] > candle['Open']
            is_volume_spike = candle['Volume'] > candle['Volume_SMA_20']

            if is_nifty_up and is_stock_trend_up and is_pullback_to_ema20 and is_bullish_candle and is_volume_spike:
                in_trade = True
                trade_type = 'LONG'
                entry_price = candle['Close']
                entry_time = candle['Datetime']
                stop_loss = candle['Low']
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 1.5)
                continue

            # Short Entry Conditions
            is_nifty_down = not candle['Nifty_Uptrend']
            is_stock_trend_down = candle['Close'] < candle['EMA_50']
            is_pullback_to_ema20_short = candle['High'] >= candle['EMA_20']
            is_bearish_candle = candle['Close'] < candle['Open']
            # Volume spike condition is the same for both long and short

            if is_nifty_down and is_stock_trend_down and is_pullback_to_ema20_short and is_bearish_candle and is_volume_spike:
                in_trade = True
                trade_type = 'SHORT'
                entry_price = candle['Close']
                entry_time = candle['Datetime']
                stop_loss = candle['High']
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 1.5)

    print(f"Backtest complete for {ticker}. Found {len(trades)} trades.")
    return trades

# ==============================================================================
# 4. SUMMARY REPORT
# ==============================================================================

def generate_summary(trades_df):
    """Generates and prints a summary report of the backtest performance."""
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
# 5. MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    try:
        # Step 1: Fetch all data in one go
        raw_data = fetch_all_data(TICKERS, NIFTY_TICKER, START_DATE, END_DATE, INTERVAL)

        # Step 2: Calculate indicators and align data
        all_stock_data = prepare_data(raw_data, TICKERS, NIFTY_TICKER)

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
