# -*- coding: utf-8 -*-
"""
Momentum Confluence Scalper Backtesting Script

This script is designed to run in a Google Colab environment to backtest a specific
stock trading strategy called the "Momentum Confluence Scalper". It uses yfinance
to fetch historical data, pandas for data manipulation, and pandas_ta for
technical indicator calculations.

To run in Google Colab, first install the necessary libraries by running this cell:
!pip install yfinance pandas pandas_ta
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime

# ==============================================================================
# 1. BACKTESTING PARAMETERS (Easily changeable)
# ==============================================================================
TICKERS = ['CANBK.NS', 'IRFC.NS', 'PNB.NS']
START_DATE = '2025-08-11' # Using a more recent, realistic date for 1-min data
END_DATE = '2025-08-16'   # yfinance has limitations on 1-min data for past dates
TIME_INTERVAL = '1m'
# Note: yfinance limits 1-min data requests to the last 7 days for free users.
# I have adjusted the dates to a recent period to ensure data is available.
# Please adjust these dates as needed, keeping the 7-day limit in mind.

# ==============================================================================
# 2. DATA FETCHING
# ==============================================================================

def fetch_data(tickers, start, end, interval):
    """
    Fetches historical stock data for multiple tickers from Yahoo Finance.

    Args:
        tickers (list): A list of stock ticker symbols.
        start (str): The start date for the data (YYYY-MM-DD).
        end (str): The end date for the data (YYYY-MM-DD).
        interval (str): The data interval (e.g., '1m', '1h', '1d').

    Returns:
        dict: A dictionary where keys are ticker symbols and values are
              pandas DataFrames with the corresponding historical data.
              Returns an empty dict if no data can be fetched.
    """
    print(f"Fetching data for {tickers} from {start} to {end} at {interval} interval...")
    all_data = {}
    for ticker in tickers:
        try:
            data = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,  # Automatically adjusts for splits and dividends
                progress=False
            )
            if data.empty:
                print(f"Warning: No data found for {ticker} in the given date range.")
                continue
            all_data[ticker] = data
            print(f"Successfully fetched data for {ticker}.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return all_data

# ==============================================================================
# 3. INDICATOR CALCULATION
# ==============================================================================

def calculate_indicators(df):
    """
    Calculates the required technical indicators for the strategy.

    Args:
        df (pd.DataFrame): The input DataFrame with OHLCV data.

    Returns:
        pd.DataFrame: The DataFrame with added indicator columns.
    """
    if df.empty:
        return df

    # Calculate VWAP using pandas_ta
    # Note: pandas_ta's vwap requires the column name to be 'Volume'
    df.ta.vwap(append=True) # Default name is VWAP_D

    # Calculate EMAs
    df.ta.ema(length=9, append=True) # Default name is EMA_9
    df.ta.ema(length=20, append=True) # Default name is EMA_20

    # Calculate ATR for risk management
    df.ta.atr(length=14, append=True) # Default name is ATRr_14

    # Calculate Volume SMA
    df.ta.sma(close='Volume', length=20, append=True) # Default name is SMA_20

    # Rename default columns to what the script expects
    df.rename(columns={
        "ATRr_14": "ATR_14",
        "SMA_20": "Volume_SMA_20"
    }, inplace=True)

    # Clean up NaN values created by indicators
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    return df

# ==============================================================================
# 4. BACKTESTING ENGINE
# ==============================================================================

def run_backtest(data, ticker):
    """
    Runs the candle-by-candle backtest for a single stock.

    Args:
        data (pd.DataFrame): The DataFrame with price data and indicators.
        ticker (str): The ticker symbol being backtested.

    Returns:
        list: A list of dictionaries, where each dictionary represents a trade.
    """
    trades = []
    in_trade = False
    trade_type = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_time = None

    print(f"\nRunning backtest for {ticker}...")
    if len(data) < 2:
        print("Not enough data to backtest.")
        return []

    # Iterate through each candle (row) in the DataFrame
    for i in range(1, len(data)):
        current_candle = data.iloc[i]
        prev_candle = data.iloc[i-1]

        # --- Trade Management: Check if an existing trade should be closed ---
        if in_trade:
            exit_price = 0
            # Check for Stop Loss or Take Profit hit
            if trade_type == 'LONG':
                if current_candle['Low'] <= stop_loss:
                    exit_price = stop_loss  # Assume SL executes at the set price
                elif current_candle['High'] >= take_profit:
                    exit_price = take_profit # Assume TP executes at the set price
            elif trade_type == 'SHORT':
                if current_candle['High'] >= stop_loss:
                    exit_price = stop_loss
                elif current_candle['Low'] <= take_profit:
                    exit_price = take_profit

            if exit_price != 0:
                pl = (exit_price - entry_price) if trade_type == 'LONG' else (entry_price - exit_price)
                trades.append({
                    'Ticker': ticker,
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time': current_candle['Datetime'],
                    'Exit Price': exit_price,
                    'P/L': pl,
                    'Type': trade_type
                })
                # Reset trade state
                in_trade = False
                trade_type = None

        # --- Entry Logic: Check for new trade opportunities ---
        if not in_trade:
            atr_value = current_candle['ATR_14']

            # Long Entry Conditions
            is_bullish_candle = current_candle['Close'] > current_candle['Open']
            is_above_vwap = current_candle['Close'] > current_candle['VWAP_D']
            is_volume_spike = current_candle['Volume'] > 1.5 * current_candle['Volume_SMA_20']
            # Fresh Crossover: EMA9 was below EMA20, now it's above
            is_fresh_long_cross = (prev_candle['EMA_9'] < prev_candle['EMA_20']) and \
                                  (current_candle['EMA_9'] > current_candle['EMA_20'])

            if is_above_vwap and is_fresh_long_cross and is_bullish_candle and is_volume_spike:
                in_trade = True
                trade_type = 'LONG'
                entry_price = current_candle['Close']
                entry_time = current_candle['Datetime']
                stop_loss = entry_price - (1.5 * atr_value)
                take_profit = entry_price + (2.0 * atr_value)
                continue # Skip to next candle after entering a trade

            # Short Entry Conditions
            is_bearish_candle = current_candle['Close'] < current_candle['Open']
            is_below_vwap = current_candle['Close'] < current_candle['VWAP_D']
            # Fresh Crossover: EMA9 was above EMA20, now it's below
            is_fresh_short_cross = (prev_candle['EMA_9'] > prev_candle['EMA_20']) and \
                                   (current_candle['EMA_9'] < current_candle['EMA_20'])

            if is_below_vwap and is_fresh_short_cross and is_bearish_candle and is_volume_spike:
                in_trade = True
                trade_type = 'SHORT'
                entry_price = current_candle['Close']
                entry_time = current_candle['Datetime']
                stop_loss = entry_price + (1.5 * atr_value)
                take_profit = entry_price - (2.0 * atr_value)

    print(f"Backtest complete for {ticker}. Found {len(trades)} trades.")
    return trades

# ==============================================================================
# 5. SUMMARY REPORT
# ==============================================================================

def generate_summary(trades_df):
    """
    Generates and prints a summary report of the backtest performance.

    Args:
        trades_df (pd.DataFrame): The DataFrame containing all completed trades.
    """
    if trades_df.empty:
        print("\nNo trades were executed. Cannot generate summary.")
        return

    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['P/L'] > 0]
    losing_trades = trades_df[trades_df['P/L'] <= 0]

    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_pl = trades_df['P/L'].sum()

    avg_profit = winning_trades['P/L'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['P/L'].mean() if len(losing_trades) > 0 else 0

    gross_profit = winning_trades['P/L'].sum()
    gross_loss = abs(losing_trades['P/L'].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print("\n--- Backtesting Summary Report ---")
    print(f"Total Number of Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit/Loss: {total_pl:.2f}")
    print(f"Average Profit per Winning Trade: {avg_profit:.2f}")
    print(f"Average Loss per Losing Trade: {avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print("----------------------------------\n")


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # Fetch data for all tickers
    stock_data = fetch_data(TICKERS, START_DATE, END_DATE, TIME_INTERVAL)

    all_trades = []

    # Process each ticker
    for ticker, data in stock_data.items():
        # Fix for yfinance returning a MultiIndex even for single tickers
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1) # Keep the first level ('Open', 'High', etc.)

        # Calculate indicators for the current ticker's data
        data_with_indicators = calculate_indicators(data)

        # Run the backtest
        trades = run_backtest(data_with_indicators, ticker)
        all_trades.extend(trades)

    # Create the final DataFrame of all trades
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        # Sort by entry time for a chronological view
        trades_df.sort_values(by='Entry Time', inplace=True)
        trades_df.reset_index(drop=True, inplace=True)

        # --- Print Final Results ---
        print("\n--- All Simulated Trades ---")
        print(trades_df.to_string())

        generate_summary(trades_df)
    else:
        print("\nNo trades were executed across any of the tickers.")

    print("Script finished.")
