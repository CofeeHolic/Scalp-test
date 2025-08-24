# -*- coding: utf-8 -*-
"""
Final Standalone Script: Bullish Trend Follower

This script backtests a trend-following strategy specialized for BULLISH trends.
It will ONLY take LONG trades when the overall market trend is bullish.

This is one of the three final specialized strategy scripts.

To run, simply execute:
python3 bullish_trend_follower.py
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

# CONFIGURABLE PARAMETERS
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
ADX_PERIOD = 14
ADX_THRESHOLD = 25 # Stricter threshold for trends
RISK_REWARD_RATIO = 2.0

def fetch_data(tickers, start, end, interval):
    """Fetches data for multiple tickers, one by one for robustness."""
    all_data = {}
    for ticker in tickers:
        data = yf.download(tickers=ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if not data.empty:
            all_data[ticker] = data
        else:
            print(f"Warning: No {interval} data for {ticker}.")
    return all_data

def prepare_data(data_5m, data_60m, tickers, nifty_ticker):
    """Prepares data for the bullish trend-following strategy."""
    prepared_data = {}
    if nifty_ticker not in data_60m:
        raise ValueError("Nifty 60m data is required for trend analysis but was not found.")

    nifty_60m_close = data_60m[nifty_ticker]['Close']
    nifty_60m_ema = ta.ema(nifty_60m_close, length=50) # 50-period EMA for overall trend direction
    nifty_htf_uptrend = (nifty_60m_close > nifty_60m_ema).rename('Nifty_HTF_Uptrend')

    for ticker in tickers:
        if ticker not in data_5m or ticker not in data_60m: continue

        stock_5m_df = data_5m[ticker].copy().dropna()
        if stock_5m_df.empty: continue

        stock_5m_df['EMA_fast'] = ta.ema(stock_5m_df['Close'], length=FAST_EMA_PERIOD)
        stock_5m_df['EMA_slow'] = ta.ema(stock_5m_df['Close'], length=SLOW_EMA_PERIOD)
        stock_5m_df['RSI'] = ta.rsi(stock_5m_df['Close'], length=RSI_PERIOD)
        stock_5m_df.ta.adx(length=ADX_PERIOD, append=True)

        merged_df = stock_5m_df.join(nifty_htf_uptrend.reindex(stock_5m_df.index, method='ffill'))
        merged_df.dropna(inplace=True)
        prepared_data[ticker] = merged_df.reset_index()
    return prepared_data

def run_backtest(data, ticker):
    """Runs the backtest, looking ONLY for LONG trades."""
    trades = []
    in_trade = False

    print(f"\nRunning BULLISH backtest for {ticker}...")
    if len(data) < 2: return []

    for i in range(1, len(data)):
        prev_candle = data.iloc[i-1]
        candle = data.iloc[i]

        if in_trade:
            if candle['Low'] <= trade['stop_loss'] or candle['High'] >= trade['take_profit']:
                exit_price = trade['stop_loss'] if candle['Low'] <= trade['stop_loss'] else trade['take_profit']
                pnl = exit_price - trade['entry_price']
                trade.update({'exit_time': candle['Datetime'], 'exit_price': exit_price, 'pnl': pnl})
                trades.append(trade)
                in_trade = False

        if not in_trade:
            long_crossover = (prev_candle['EMA_fast'] < prev_candle['EMA_slow']) and (candle['EMA_fast'] > candle['EMA_slow'])
            adx_col_name = f'ADX_{ADX_PERIOD}'

            # Bullish Entry Condition: Only take LONG trades if Nifty trend is also UP.
            if (long_crossover and candle[adx_col_name] > ADX_THRESHOLD and candle['Nifty_HTF_Uptrend'] and candle['RSI'] > RSI_LEVEL):
                entry_price = candle['Close']
                stop_loss = candle['Low']
                risk = abs(entry_price - stop_loss)
                if risk > 0:
                    in_trade = True
                    take_profit = entry_price + (risk * RISK_REWARD_RATIO)
                    trade = {'ticker': ticker, 'type': 'LONG', 'entry_time': candle['Datetime'], 'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit}
    return trades

def generate_summary(trades_df):
    """Generates a summary report."""
    if trades_df.empty:
        print("\nNo trades were executed.")
        return
    # Identical summary logic...
    total_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).sum() / total_trades * 100
    profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) > 0 else np.inf
    print("\n--- Backtesting Summary Report ---")
    print(f"Total Trades: {total_trades}, Win Rate: {win_rate:.2f}%, Profit Factor: {profit_factor:.2f}")

if __name__ == '__main__':
    try:
        all_tickers_list = TICKERS + [NIFTY_TICKER]
        data_5m = fetch_data(all_tickers_list, START_DATE, END_DATE, INTERVAL_5M)
        data_60m = fetch_data(all_tickers_list, START_DATE, END_DATE, INTERVAL_60M)
        all_stock_data = prepare_data(data_5m, data_60m, TICKERS, NIFTY_TICKER)

        all_trades = []
        for ticker, stock_data in all_stock_data.items():
            trades = run_backtest(stock_data, ticker)
            all_trades.extend(trades)

        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.sort_values(by='entry_time', inplace=True)
            print("\n--- All Simulated Trades ---")
            print(trades_df.to_string())
            generate_summary(trades_df)
        else:
            print("\nNo trades were executed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    print("\nScript finished.")
