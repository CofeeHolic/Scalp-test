# -*- coding: utf-8 -*-
"""
Filtered EMA Crossover Scalper Backtesting Script

This script backtests a sophisticated EMA crossover scalping strategy.
It is designed for a high win rate by using multiple filters to confirm
trade entries, including overall market trend, momentum, and volume.

Strategy Blueprint:
- Core Signal: 9/21 EMA crossover on the 5-minute chart.
- Filter 1: Nifty 50 trend on the 1-hour chart (Overall Market Trend).
- Filter 2: RSI on the 5-minute chart (Momentum Confirmation).
- Filter 3: Volume SMA on the 5-minute chart (Volume Confirmation).
- Risk Management: 1:1 Risk/Reward Ratio with stop-loss at the crossover candle's high/low.

To run in Google Colab, first install the necessary libraries:
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
START_DATE = '2025-08-01'
END_DATE = '2025-08-22'
INTERVAL_5M = '5m'
INTERVAL_60M = '60m'

# Strategy Parameters
FAST_EMA_PERIOD = 9
SLOW_EMA_PERIOD = 21
RSI_PERIOD = 14
RSI_LEVEL = 50
VOLUME_SMA_PERIOD = 20
NIFTY_HTF_EMA_PERIOD = 50
RISK_REWARD_RATIO = 2.0 # Target 1:2 R:R
ADX_THRESHOLD = 20 # ADX level to confirm trend strength

# ==============================================================================
# 2. DATA FETCHING & PREPARATION
# ==============================================================================

def fetch_data(tickers, start, end, interval):
    """Fetches data for a list of tickers at a specific interval."""
    print(f"Fetching {interval} data for: {tickers}")
    data = yf.download(
        tickers=tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, group_by='ticker'
    )
    if data.empty:
        raise ValueError(f"No {interval} data fetched. Check tickers and date range.")
    return data

def prepare_data(data_5m, data_60m, tickers, nifty_ticker):
    """
    Calculates all necessary indicators and aligns data for the strategy.
    """
    prepared_data = {}

    # --- Prepare Nifty HTF (1-hour) Trend Filter ---
    nifty_60m_close = data_60m[nifty_ticker]['Close']
    nifty_60m_ema = ta.ema(nifty_60m_close, length=NIFTY_HTF_EMA_PERIOD)
    nifty_htf_uptrend = (nifty_60m_close > nifty_60m_ema).rename('Nifty_HTF_Uptrend')

    for ticker in tickers:
        print(f"Preparing data for {ticker}...")

        stock_5m_df = data_5m[ticker].copy()
        stock_5m_df.dropna(inplace=True)
        if stock_5m_df.empty:
            print(f"Warning: No 5m data for {ticker}.")
            continue

        # --- Calculate 5-min Indicators for the Stock ---
        stock_5m_df['EMA_fast'] = ta.ema(stock_5m_df['Close'], length=FAST_EMA_PERIOD)
        stock_5m_df['EMA_slow'] = ta.ema(stock_5m_df['Close'], length=SLOW_EMA_PERIOD)
        stock_5m_df['RSI'] = ta.rsi(stock_5m_df['Close'], length=RSI_PERIOD)
        stock_5m_df['Volume_SMA'] = ta.sma(stock_5m_df['Volume'], length=VOLUME_SMA_PERIOD)
        stock_5m_df.ta.adx(length=14, append=True) # Adds ADX_14, DMP_14, DMN_14 columns

        # --- Align Nifty HTF trend with the 5-min stock data ---
        # Reindex the 1-hour Nifty trend to the 5-min index and forward-fill
        aligned_nifty_trend = nifty_htf_uptrend.reindex(stock_5m_df.index, method='ffill')
        merged_df = stock_5m_df.join(aligned_nifty_trend)

        # --- Final cleanup ---
        merged_df.dropna(inplace=True)
        merged_df.reset_index(inplace=True)

        prepared_data[ticker] = merged_df

    return prepared_data

# ==============================================================================
# 3. BACKTESTING ENGINE
# ==============================================================================

def run_backtest(data, ticker):
    """Runs the candle-by-candle backtest for the filtered crossover strategy."""
    trades = []
    in_trade = False
    trade_type = None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_time = None

    print(f"\nRunning backtest for {ticker}...")
    if len(data) < 2: # Need at least 2 rows to check for a crossover
        print("Not enough data to backtest.")
        return []

    # Iterate from the second candle to check previous candle for crossover
    for i in range(1, len(data)):
        prev_candle = data.iloc[i-1]
        candle = data.iloc[i]

        # --- Trade Management: Check for exit ---
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

        # --- Entry Logic ---
        if not in_trade:
            # --- Crossover Signals ---
            long_crossover = (prev_candle['EMA_fast'] < prev_candle['EMA_slow']) and \
                             (candle['EMA_fast'] > candle['EMA_slow'])
            short_crossover = (prev_candle['EMA_fast'] > prev_candle['EMA_slow']) and \
                              (candle['EMA_fast'] < candle['EMA_slow'])

            # --- Filter Conditions for LONG ---
            if (long_crossover and
                candle['ADX_14'] > ADX_THRESHOLD and
                candle['Nifty_HTF_Uptrend'] and
                candle['RSI'] > RSI_LEVEL and
                candle['Volume'] > candle['Volume_SMA']):

                entry_price = candle['Close']
                stop_loss = candle['Low']
                risk = entry_price - stop_loss

                if risk > 0:
                    in_trade = True
                    trade_type = 'LONG'
                    entry_time = candle['Datetime']
                    take_profit = entry_price + (risk * RISK_REWARD_RATIO)
                    continue

            # --- Filter Conditions for SHORT ---
            if (short_crossover and
                candle['ADX_14'] > ADX_THRESHOLD and
                not candle['Nifty_HTF_Uptrend'] and
                candle['RSI'] < RSI_LEVEL and
                candle['Volume'] > candle['Volume_SMA']):

                entry_price = candle['Close']
                stop_loss = candle['High']
                risk = stop_loss - entry_price

                if risk > 0:
                    in_trade = True
                    trade_type = 'SHORT'
                    entry_time = candle['Datetime']
                    take_profit = entry_price - (risk * RISK_REWARD_RATIO)

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
    total_pl = trades_df['P/L'].sum()
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = winning_trades['P/L'].sum()
    gross_loss = abs(trades_df[trades_df['P/L'] <= 0]['P/L'].sum())
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
        all_tickers_list = TICKERS + [NIFTY_TICKER]

        # Fetch data for both intervals
        data_5m = fetch_data(all_tickers_list, START_DATE, END_DATE, INTERVAL_5M)
        data_60m = fetch_data(all_tickers_list, START_DATE, END_DATE, INTERVAL_60M)

        # Prepare data by calculating all indicators and aligning timeframes
        all_stock_data = prepare_data(data_5m, data_60m, TICKERS, NIFTY_TICKER)

        all_trades = []
        # Run the backtest for each stock with the prepared data
        for ticker, stock_data in all_stock_data.items():
            trades = run_backtest(stock_data, ticker)
            all_trades.extend(trades)

        # Aggregate results and generate a final report
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
