# -*- coding: utf-8 -*-
"""
Bollinger Band Mean Reversion Backtesting Script

This script backtests a mean reversion strategy designed to perform well
in choppy or range-bound markets. It uses Bollinger Bands to identify
over-extended prices and RSI for confirmation.

Strategy Blueprint:
- Core Signal: Price touching or crossing the outer Bollinger Bands.
- Confirmation: RSI indicating overbought or oversold conditions.
- Target: The middle Bollinger Band (20-period SMA).
- Stop-Loss: Based on the entry candle's high/low.

To run in Google Colab, first install the necessary libraries:
!pip install yfinance pandas pandas_ta
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta

# ==============================================================================
# 1. CONFIGURABLE PARAMETERS
# ==============================================================================
TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
START_DATE = '2025-08-01' # Using the same period where the trend strategy failed
END_DATE = '2025-08-22'   # yfinance end_date is exclusive
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
    """Fetches data for a list of tickers at a specific interval."""
    print(f"Fetching {interval} data for: {tickers}")
    data = yf.download(
        tickers=tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, group_by='ticker'
    )
    if data.empty:
        raise ValueError(f"No {interval} data fetched. Check tickers and date range.")
    return data

def prepare_data(data, tickers):
    """Calculates all necessary indicators for the mean reversion strategy."""
    prepared_data = {}
    for ticker in tickers:
        print(f"Preparing data for {ticker}...")

        stock_df = data[ticker].copy()
        stock_df.dropna(inplace=True)
        if stock_df.empty:
            print(f"Warning: No data for {ticker}.")
            continue

        # --- Calculate Indicators ---
        # Bollinger Bands (BBL_20_2.0, BBM_20_2.0, BBU_20_2.0)
        stock_df.ta.bbands(length=BBANDS_LENGTH, std=BBANDS_STD_DEV, append=True)
        # RSI
        stock_df['RSI'] = ta.rsi(stock_df['Close'], length=RSI_PERIOD)

        # --- Final cleanup ---
        stock_df.dropna(inplace=True)
        stock_df.reset_index(inplace=True)

        prepared_data[ticker] = stock_df

    return prepared_data

# ==============================================================================
# 3. BACKTESTING ENGINE
# ==============================================================================

def run_backtest(data, ticker):
    """Runs the candle-by-candle backtest for the mean reversion strategy."""
    trades = []
    in_trade = False
    trade_type = None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_time = None

    # Dynamically get the names of the Bollinger Band columns
    bbl_col = f'BBL_{BBANDS_LENGTH}_{BBANDS_STD_DEV:.1f}'
    bbm_col = f'BBM_{BBANDS_LENGTH}_{BBANDS_STD_DEV:.1f}'
    bbu_col = f'BBU_{BBANDS_LENGTH}_{BBANDS_STD_DEV:.1f}'

    print(f"\nRunning backtest for {ticker}...")
    if len(data) < 1:
        print("Not enough data to backtest.")
        return []

    for i in range(len(data)):
        candle = data.iloc[i]

        # --- Trade Management: Check for exit ---
        if in_trade:
            exit_price = 0
            # TP target is the middle band, which moves, so we update it on each candle
            current_middle_band = candle[bbm_col]

            if trade_type == 'LONG':
                take_profit = current_middle_band
                if candle['Low'] <= stop_loss: exit_price = stop_loss
                elif candle['High'] >= take_profit: exit_price = take_profit
            elif trade_type == 'SHORT':
                take_profit = current_middle_band
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
            # --- Entry Conditions for LONG ---
            if (candle['Low'] <= candle[bbl_col] and candle['RSI'] < RSI_OVERSOLD):
                in_trade = True
                trade_type = 'LONG'
                entry_price = candle['Close']
                entry_time = candle['Datetime']
                stop_loss = candle['Low'] - (candle['High'] - candle['Low']) # Simple SL below the low
                take_profit = candle[bbm_col]
                continue

            # --- Entry Conditions for SHORT ---
            if (candle['High'] >= candle[bbu_col] and candle['RSI'] > RSI_OVERBOUGHT):
                in_trade = True
                trade_type = 'SHORT'
                entry_price = candle['Close']
                entry_time = candle['Datetime']
                stop_loss = candle['High'] + (candle['High'] - candle['Low']) # Simple SL above the high
                take_profit = candle[bbm_col]

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
        data = fetch_data(TICKERS, START_DATE, END_DATE, INTERVAL)
        all_stock_data = prepare_data(data, TICKERS)

        all_trades = []
        for ticker, stock_data in all_stock_data.items():
            trades = run_backtest(stock_data, ticker)
            all_trades.extend(trades)

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
