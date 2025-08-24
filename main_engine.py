# -*- coding: utf-8 -*-
"""
Professional Backtesting Engine

This script serves as a modular backtesting engine that can run various
trading strategies defined in external JSON configuration files.

It separates the core backtesting logic (data fetching, execution, reporting)
from the strategy-specific parameters and rules.

To run, use the command line:
python main_engine.py --strategy <strategy_name>

Example:
python main_engine.py --strategy trend_follower
python main_engine.py --strategy mean_reversion
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import argparse
from pathlib import Path

# ==============================================================================
# 1. CORE ENGINE COMPONENTS
# ==============================================================================

def load_strategy_config(strategy_name):
    """Loads strategy configuration from a JSON file."""
    config_path = Path('strategies') / f'{strategy_name}.json'
    if not config_path.is_file():
        raise FileNotFoundError(f"Strategy configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"--- Loaded Strategy: {config['strategy_name']} ---")
    return config

def fetch_data(tickers, start, end, intervals):
    """Fetches data for multiple tickers and intervals."""
    data_store = {}
    for interval in intervals:
        print(f"Fetching {interval} data for: {tickers}")
        data = yf.download(
            tickers=tickers, start=start, end=end, interval=interval,
            auto_adjust=True, progress=False, group_by='ticker'
        )
        if data.empty:
            raise ValueError(f"No {interval} data fetched for {tickers}.")
        data_store[interval] = data
    return data_store

def prepare_data(data_store, config, tickers, nifty_ticker):
    """
    Calculates indicators based on the loaded strategy configuration.
    """
    strategy_type = config['strategy_type']
    params = config['params']
    prepared_data = {}

    print(f"Preparing data for '{strategy_type}' strategy...")

    if strategy_type == 'trend_follower':
        data_5m = data_store['5m']
        data_60m = data_store['60m']

        # Prepare Nifty HTF (1-hour) Trend Filter
        nifty_60m_close = data_60m[nifty_ticker]['Close']
        nifty_60m_ema = ta.ema(nifty_60m_close, length=params['nifty_htf_ema_period'])
        nifty_htf_uptrend = (nifty_60m_close > nifty_60m_ema).rename('Nifty_HTF_Uptrend')

        for ticker in tickers:
            stock_5m_df = data_5m[ticker].copy().dropna()
            if stock_5m_df.empty: continue

            # Calculate 5-min Indicators
            stock_5m_df['EMA_fast'] = ta.ema(stock_5m_df['Close'], length=params['fast_ema_period'])
            stock_5m_df['EMA_slow'] = ta.ema(stock_5m_df['Close'], length=params['slow_ema_period'])
            stock_5m_df['RSI'] = ta.rsi(stock_5m_df['Close'], length=params['rsi_period'])
            stock_5m_df['Volume_SMA'] = ta.sma(stock_5m_df['Volume'], length=params['volume_sma_period'])
            stock_5m_df.ta.adx(length=params['adx_period'], append=True)

            # Align and merge
            aligned_nifty_trend = nifty_htf_uptrend.reindex(stock_5m_df.index, method='ffill')
            merged_df = stock_5m_df.join(aligned_nifty_trend)
            merged_df.dropna(inplace=True)
            prepared_data[ticker] = merged_df.reset_index()

    elif strategy_type == 'mean_reversion':
        data_5m = data_store['5m']
        for ticker in tickers:
            stock_df = data_5m[ticker].copy().dropna()
            if stock_df.empty: continue

            # Calculate Indicators
            stock_df.ta.bbands(length=params['bbands_length'], std=params['bbands_std_dev'], append=True)
            stock_df['RSI'] = ta.rsi(stock_df['Close'], length=params['rsi_period'])
            stock_df.ta.atr(length=14, append=True) # For robust stop-loss
            stock_df.rename(columns={"ATRr_14": "ATR_14"}, inplace=True)

            stock_df.dropna(inplace=True)
            prepared_data[ticker] = stock_df.reset_index()

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return prepared_data

def run_backtest(data, ticker, config):
    """
    Runs the backtest with logic dictated by the strategy configuration.
    """
    strategy_type = config['strategy_type']
    params = config['params']
    trades = []
    in_trade = False
    trade_type = None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_time = None

    print(f"\nRunning backtest for {ticker} using '{strategy_type}' logic...")
    if len(data) < 2: return []

    for i in range(1, len(data)):
        prev_candle = data.iloc[i-1]
        candle = data.iloc[i]

        # --- Trade Management ---
        if in_trade:
            exit_price = 0
            # For mean reversion, TP is dynamic (the moving middle band)
            if strategy_type == 'mean_reversion':
                take_profit = candle[f"BBM_{params['bbands_length']}_{params['bbands_std_dev']:.1f}"]

            if trade_type == 'LONG':
                if candle['Low'] <= stop_loss: exit_price = stop_loss
                elif candle['High'] >= take_profit: exit_price = take_profit
            elif trade_type == 'SHORT':
                if candle['High'] >= stop_loss: exit_price = stop_loss
                elif candle['Low'] <= take_profit: exit_price = take_profit

            if exit_price != 0:
                pl = (exit_price - entry_price) if trade_type == 'LONG' else (entry_price - exit_price)
                trades.append({'Ticker': ticker, 'Trade Type': trade_type, 'Entry Time': entry_time,
                               'Entry Price': entry_price, 'Exit Time': candle['Datetime'],
                               'Exit Price': exit_price, 'P/L': pl})
                in_trade = False

        # --- Entry Logic ---
        if not in_trade:
            if strategy_type == 'trend_follower':
                long_crossover = (prev_candle['EMA_fast'] < prev_candle['EMA_slow']) and (candle['EMA_fast'] > candle['EMA_slow'])
                short_crossover = (prev_candle['EMA_fast'] > prev_candle['EMA_slow']) and (candle['EMA_fast'] < candle['EMA_slow'])

                if (long_crossover and candle[f"ADX_{params['adx_period']}"] > params['adx_threshold'] and
                    candle['Nifty_HTF_Uptrend'] and candle['RSI'] > params['rsi_level'] and
                    candle['Volume'] > candle['Volume_SMA']):

                    entry_price = candle['Close']
                    stop_loss = candle['Low']
                    risk = entry_price - stop_loss
                    if risk > 0:
                        in_trade, trade_type, entry_time = True, 'LONG', candle['Datetime']
                        take_profit = entry_price + (risk * params['risk_reward_ratio'])

                elif (short_crossover and candle[f"ADX_{params['adx_period']}"] > params['adx_threshold'] and
                      not candle['Nifty_HTF_Uptrend'] and candle['RSI'] < params['rsi_level'] and
                      candle['Volume'] > candle['Volume_SMA']):

                    entry_price = candle['Close']
                    stop_loss = candle['High']
                    risk = stop_loss - entry_price
                    if risk > 0:
                        in_trade, trade_type, entry_time = True, 'SHORT', candle['Datetime']
                        take_profit = entry_price - (risk * params['risk_reward_ratio'])

            elif strategy_type == 'mean_reversion':
                if (candle['Low'] <= candle[f"BBL_{params['bbands_length']}_{params['bbands_std_dev']:.1f}"] and
                    candle['RSI'] < params['rsi_oversold']):

                    in_trade, trade_type, entry_price, entry_time = True, 'LONG', candle['Close'], candle['Datetime']
                    stop_loss = candle['Low'] - candle['ATR_14']
                    take_profit = candle[f"BBM_{params['bbands_length']}_{params['bbands_std_dev']:.1f}"]

                elif (candle['High'] >= candle[f"BBU_{params['bbands_length']}_{params['bbands_std_dev']:.1f}"] and
                      candle['RSI'] > params['rsi_overbought']):

                    in_trade, trade_type, entry_price, entry_time = True, 'SHORT', candle['Close'], candle['Datetime']
                    stop_loss = candle['High'] + candle['ATR_14']
                    take_profit = candle[f"BBM_{params['bbands_length']}_{params['bbands_std_dev']:.1f}"]

    print(f"Backtest complete for {ticker}. Found {len(trades)} trades.")
    return trades

def generate_summary(trades_df):
    """Generates and prints a summary report of the backtest performance."""
    if trades_df.empty:
        print("\nNo trades were executed. Cannot generate summary.")
        return
    total_trades = len(trades_df); win_rate = (trades_df['P/L'] > 0).sum() / total_trades * 100
    total_pl = trades_df['P/L'].sum(); profit_factor = trades_df[trades_df['P/L'] > 0]['P/L'].sum() / abs(trades_df[trades_df['P/L'] <= 0]['P/L'].sum())
    print("\n--- Backtesting Summary Report ---")
    print(f"Total Number of Trades: {total_trades}"); print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total P/L: {total_pl:.2f}"); print(f"Profit Factor: {profit_factor:.2f}")
    print("----------------------------------\n")

# ==============================================================================
# 2. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modular Backtesting Engine.')
    parser.add_argument('--strategy', type=str, required=True, help='Name of the strategy config file (without .json extension).')
    args = parser.parse_args()

    try:
        # Load strategy config
        config = load_strategy_config(args.strategy)

        # General parameters
        TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
        NIFTY_TICKER = '^NSEI'
        START_DATE = '2025-07-01'
        END_DATE = '2025-08-16'

        # Fetch data based on required intervals
        all_tickers_list = list(set(TICKERS + [NIFTY_TICKER]))
        data_store = fetch_data(all_tickers_list, START_DATE, END_DATE, config['intervals'])

        # Prepare data (calculate indicators)
        all_stock_data = prepare_data(data_store, config, TICKERS, NIFTY_TICKER)

        # Run backtest for each stock
        all_trades = []
        for ticker, stock_data in all_stock_data.items():
            trades = run_backtest(stock_data, ticker, config)
            all_trades.extend(trades)

        # Aggregate results and generate report
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.sort_values(by='Entry Time', inplace=True)
            trades_df.reset_index(drop=True, inplace=True)
            print("\n--- All Simulated Trades ---")
            print(trades_df.to_string())
            generate_summary(trades_df)
        else:
            print("\nNo trades were executed across any of the tickers.")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("Script finished.")
