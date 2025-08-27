# -*- coding: utf-8 -*-
"""
Final Standalone Script: Bollinger Band Mean Reversion

This script backtests a mean reversion strategy designed to perform well
in choppy or range-bound markets. It uses Bollinger Bands to identify
over-extended prices and RSI for confirmation.

This is a standalone script. To run it, simply execute:
python3 bollinger_mean_reversion.py

You can change the Tickers and Date Range in the configuration section below.
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

# ==============================================================================
# 1. CONFIGURABLE PARAMETERS
# ==============================================================================
TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
START_DATE = '2025-08-20'
END_DATE = '2025-08-23'
INTERVAL = '5m'

# Strategy Parameters
BBANDS_LENGTH = 20
BBANDS_STD_DEV = 2.0
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# ==============================================================================
# 2. DATA FETCHING & PREPARATION
# ==============================================================================

def fetch_data(tickers, start, end, interval):
    """Fetches data for a list of tickers, one by one for robustness."""
    all_data = {}
    for ticker in tickers:
        print(f"Fetching {interval} data for: {ticker}")
        data = yf.download(
            tickers=ticker, start=start, end=end, interval=interval,
            auto_adjust=True, progress=False
        )
        if not data.empty:
            all_data[ticker] = data
        else:
            print(f"Warning: No {interval} data for {ticker}.")
    return all_data

def prepare_data(data):
    """Calculates all necessary indicators for the mean reversion strategy."""
    prepared_data = {}
    for ticker, stock_df in data.items():
        print(f"Preparing data for {ticker}...")

        # Make a copy to avoid SettingWithCopyWarning
        stock_df = stock_df.copy()

        # Handle yfinance's inconsistent column structure
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.droplevel(1)

        # Calculate Indicators
        stock_df.ta.bbands(length=BBANDS_LENGTH, std=BBANDS_STD_DEV, append=True)
        stock_df['RSI'] = ta.rsi(stock_df['Close'], length=RSI_PERIOD)
        stock_df.ta.atr(length=14, append=True)
        stock_df.rename(columns={"ATRr_14": "ATR_14"}, inplace=True)

        stock_df.dropna(inplace=True)
        prepared_data[ticker] = stock_df.reset_index()

    return prepared_data

# ==============================================================================
# 3. BACKTESTING ENGINE
# ==============================================================================

def run_backtest(data, ticker):
    """Runs the candle-by-candle backtest for the mean reversion strategy."""
    trades = []
    in_trade = False
    trade = {}

    bbl_col = f'BBL_{BBANDS_LENGTH}_{BBANDS_STD_DEV:.1f}'
    bbm_col = f'BBM_{BBANDS_LENGTH}_{BBANDS_STD_DEV:.1f}'
    bbu_col = f'BBU_{BBANDS_LENGTH}_{BBANDS_STD_DEV:.1f}'

    print(f"\nRunning backtest for {ticker}...")
    if len(data) < 1: return []

    for i in range(len(data)):
        candle = data.iloc[i]

        if in_trade:
            exit_price = 0
            current_middle_band = candle[bbm_col]

            if trade['type'] == 'LONG':
                take_profit = current_middle_band
                if candle['Low'] <= trade['stop_loss']: exit_price = trade['stop_loss']
                elif candle['High'] >= take_profit: exit_price = take_profit
            elif trade['type'] == 'SHORT':
                take_profit = current_middle_band
                if candle['High'] >= trade['stop_loss']: exit_price = trade['stop_loss']
                elif candle['Low'] <= take_profit: exit_price = trade['take_profit']

            if exit_price != 0:
                pnl = (exit_price - trade['entry_price']) if trade['type'] == 'LONG' else (trade['entry_price'] - exit_price)
                trade.update({'exit_time': candle['Datetime'], 'exit_price': exit_price, 'pnl': pnl})
                trades.append(trade)
                in_trade = False

        if not in_trade:
            if (candle['Low'] <= candle[bbl_col] and candle['RSI'] < RSI_OVERSOLD):
                in_trade = True
                trade = {'ticker': ticker, 'type': 'LONG', 'entry_time': candle['Datetime'],
                         'entry_price': candle['Close'], 'stop_loss': candle['Low'] - candle['ATR_14'],
                         'take_profit': candle[bbm_col]}
            elif (candle['High'] >= candle[bbu_col] and candle['RSI'] > RSI_OVERBOUGHT):
                in_trade = True
                trade = {'ticker': ticker, 'type': 'SHORT', 'entry_time': candle['Datetime'],
                         'entry_price': candle['Close'], 'stop_loss': candle['High'] + candle['ATR_14'],
                         'take_profit': candle[bbm_col]}

    return trades

# ==============================================================================
# 4. SUMMARY REPORT
# ==============================================================================
def generate_summary(trades_df):
    if trades_df.empty:
        print("\nNo trades were executed.")
        return
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    total_pl = trades_df['pnl'].sum()
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = winning_trades['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    print("\n--- Backtesting Summary Report ---")
    print(f"Total Number of Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total P/L (points): {total_pl:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print("----------------------------------\n")

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    try:
        data = fetch_data(TICKERS, START_DATE, END_DATE, INTERVAL)
        all_stock_data = prepare_data(data)

        all_trades = []
        for ticker, stock_data in all_stock_data.items():
            trades = run_backtest(stock_data, ticker)
            all_trades.extend(trades)

        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.sort_values(by='entry_time', inplace=True)
            trades_df.reset_index(drop=True, inplace=True)
            print("\n--- All Simulated Trades ---")
            print(trades_df.to_string())
            generate_summary(trades_df)
        else:
            print("\nNo trades were executed.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("Script finished.")
