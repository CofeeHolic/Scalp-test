# -*- coding: utf-8 -*-
"""
Professional Backtesting Engine

This script serves as a modular backtesting engine that can run various
trading strategies defined in external JSON configuration files.

It has been upgraded to include capital and position sizing logic for
more realistic backtest results.

To run, use the command line:
python main_engine.py --strategy <strategy_name>
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
    """Calculates indicators based on the loaded strategy configuration."""
    strategy_type = config['strategy_type']
    params = config['params']
    prepared_data = {}

    print(f"Preparing data for '{strategy_type}' strategy...")

    if strategy_type == 'trend_follower':
        data_5m = data_store['5m']
        data_60m = data_store['60m']
        nifty_60m_close = data_60m[nifty_ticker]['Close']
        nifty_60m_ema = ta.ema(nifty_60m_close, length=params['nifty_htf_ema_period'])
        nifty_htf_uptrend = (nifty_60m_close > nifty_60m_ema).rename('Nifty_HTF_Uptrend')

        for ticker in tickers:
            stock_5m_df = data_5m[ticker].copy().dropna()
            if stock_5m_df.empty: continue
            stock_5m_df['EMA_fast'] = ta.ema(stock_5m_df['Close'], length=params['fast_ema_period'])
            stock_5m_df['EMA_slow'] = ta.ema(stock_5m_df['Close'], length=params['slow_ema_period'])
            stock_5m_df['RSI'] = ta.rsi(stock_5m_df['Close'], length=params['rsi_period'])
            stock_5m_df['Volume_SMA'] = ta.sma(stock_5m_df['Volume'], length=params['volume_sma_period'])
            stock_5m_df.ta.adx(length=params['adx_period'], append=True)
            aligned_nifty_trend = nifty_htf_uptrend.reindex(stock_5m_df.index, method='ffill')
            merged_df = stock_5m_df.join(aligned_nifty_trend)
            merged_df.dropna(inplace=True)
            prepared_data[ticker] = merged_df.reset_index()

    elif strategy_type == 'mean_reversion':
        data_5m = data_store['5m']
        for ticker in tickers:
            stock_df = data_5m[ticker].copy().dropna()
            if stock_df.empty: continue
            stock_df.ta.bbands(length=params['bbands_length'], std=params['bbands_std_dev'], append=True)
            stock_df['RSI'] = ta.rsi(stock_df['Close'], length=params['rsi_period'])
            stock_df.ta.atr(length=14, append=True)
            stock_df.rename(columns={"ATRr_14": "ATR_14"}, inplace=True)
            stock_df.dropna(inplace=True)
            prepared_data[ticker] = stock_df.reset_index()

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return prepared_data

def run_backtest(data, ticker, config, capital, risk_perc):
    """
    Runs the backtest with CAPITAL and POSITION SIZING logic.
    """
    strategy_type = config['strategy_type']
    params = config['params']
    trades = []
    in_trade = False

    print(f"\nRunning backtest for {ticker} using '{strategy_type}' logic...")
    if len(data) < 2: return []

    for i in range(1, len(data)):
        prev_candle = data.iloc[i-1]
        candle = data.iloc[i]

        # --- Trade Management ---
        if in_trade:
            exit_price = 0
            if strategy_type == 'mean_reversion':
                trade['take_profit'] = data.iloc[i][f"BBM_{params['bbands_length']}_{params['bbands_std_dev']:.1f}"]

            if trade['type'] == 'LONG':
                if data.iloc[i]['Low'] <= trade['stop_loss']: exit_price = trade['stop_loss']
                elif data.iloc[i]['High'] >= trade['take_profit']: exit_price = trade['take_profit']
            elif trade['type'] == 'SHORT':
                if data.iloc[i]['High'] >= trade['stop_loss']: exit_price = trade['stop_loss']
                elif data.iloc[i]['Low'] <= trade['take_profit']: exit_price = trade['take_profit']

            if exit_price != 0:
                pnl = (exit_price - trade['entry_price']) * trade['shares'] if trade['type'] == 'LONG' else (trade['entry_price'] - exit_price) * trade['shares']
                capital += pnl
                trade.update({'exit_time': data.iloc[i]['Datetime'], 'exit_price': exit_price, 'pnl': pnl, 'capital_after': capital})
                trades.append(trade)
                in_trade = False

        # --- Entry Logic ---
        if not in_trade:
            entry_price, stop_loss, trade_type = 0, 0, None

            if strategy_type == 'trend_follower':
                long_crossover = (prev_candle['EMA_fast'] < prev_candle['EMA_slow']) and (candle['EMA_fast'] > candle['EMA_slow'])
                short_crossover = (prev_candle['EMA_fast'] > prev_candle['EMA_slow']) and (candle['EMA_fast'] < candle['EMA_slow'])

                if (long_crossover and candle[f"ADX_{params['adx_period']}"] > params['adx_threshold'] and
                    candle['Nifty_HTF_Uptrend'] and candle['RSI'] > params['rsi_level'] and
                    candle['Volume'] > candle['Volume_SMA']):

                    entry_price, stop_loss, trade_type = candle['Close'], candle['Low'], 'LONG'

                elif (short_crossover and candle[f"ADX_{params['adx_period']}"] > params['adx_threshold'] and
                      not candle['Nifty_HTF_Uptrend'] and candle['RSI'] < params['rsi_level'] and
                      candle['Volume'] > candle['Volume_SMA']):

                    entry_price, stop_loss, trade_type = candle['Close'], candle['High'], 'SHORT'

            elif strategy_type == 'mean_reversion':
                if (data.iloc[i]['Low'] <= data.iloc[i][f"BBL_{params['bbands_length']}_{params['bbands_std_dev']:.1f}"] and
                    data.iloc[i]['RSI'] < params['rsi_oversold']):
                    entry_price, stop_loss, trade_type = data.iloc[i]['Close'], data.iloc[i]['Low'] - data.iloc[i]['ATR_14'], 'LONG'
                elif (data.iloc[i]['High'] >= data.iloc[i][f"BBU_{params['bbands_length']}_{params['bbands_std_dev']:.1f}"] and
                      data.iloc[i]['RSI'] > params['rsi_overbought']):
                    entry_price, stop_loss, trade_type = data.iloc[i]['Close'], data.iloc[i]['High'] + data.iloc[i]['ATR_14'], 'SHORT'

            if entry_price > 0:
                risk_per_share = abs(entry_price - stop_loss)
                if risk_per_share > 0:
                    capital_to_risk = capital * risk_perc
                    shares = round(capital_to_risk / risk_per_share)
                    if shares > 0:
                        in_trade = True
                        # --- Calculate Take Profit based on Strategy ---
                        if strategy_type == 'trend_follower':
                            take_profit = entry_price + (risk_per_share * params['risk_reward_ratio']) if trade_type == 'LONG' else entry_price - (risk_per_share * params['risk_reward_ratio'])
                        elif strategy_type == 'mean_reversion':
                            take_profit = data.iloc[i][f"BBM_{params['bbands_length']}_{params['bbands_std_dev']:.1f}"]

                        trade = {'ticker': ticker, 'type': trade_type, 'entry_time': data.iloc[i]['Datetime'],
                                 'entry_price': entry_price, 'shares': shares, 'stop_loss': stop_loss,
                                 'take_profit': take_profit, 'capital_before': capital}

    return trades, capital

def generate_summary(trades_df, capital_at_start):
    """Generates a summary report with capital-based metrics."""
    if trades_df.empty:
        print("\nNo trades were executed.")
        return

    capital_at_end = trades_df['capital_after'].iloc[-1]
    total_return_pct = (capital_at_end / capital_at_start - 1) * 100

    print("\n--- Backtesting Summary Report ---")
    print(f"Starting Capital: {capital_at_start:,.2f}")
    print(f"Ending Capital:   {capital_at_end:,.2f}")
    print(f"Total Return:     {total_return_pct:.2f}%")
    print(f"Total P/L:        {trades_df['pnl'].sum():,.2f}")
    print(f"Total Number of Trades: {len(trades_df)}")
    print(f"Win Rate:         {(trades_df['pnl'] > 0).sum() / len(trades_df) * 100:.2f}%")
    profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    print(f"Profit Factor:    {profit_factor:.2f}")
    print("----------------------------------\n")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modular Backtesting Engine.')
    parser.add_argument('--strategy', type=str, required=True, help='Name of the strategy config file.')
    args = parser.parse_args()

    try:
        config = load_strategy_config(args.strategy)

        # --- Portfolio and Run Parameters ---
        TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
        NIFTY_TICKER = '^NSEI'
        START_DATE = '2025-08-01'
        END_DATE = '2025-08-22'
        STARTING_CAPITAL = 20000.0
        RISK_PERCENTAGE = 0.02 # Risk 2% of capital per trade

        all_tickers_list = list(set(TICKERS + [NIFTY_TICKER]))
        data_store = fetch_data(all_tickers_list, START_DATE, END_DATE, config['intervals'])
        all_stock_data = prepare_data(data_store, config, TICKERS, NIFTY_TICKER)

        all_trades = []
        current_capital = STARTING_CAPITAL

        # Note: This is a simplified sequential backtest. A more advanced engine
        # would process all tickers simultaneously candle-by-candle.
        for ticker in TICKERS:
            if ticker in all_stock_data:
                trades, current_capital = run_backtest(all_stock_data[ticker], ticker, config, current_capital, RISK_PERCENTAGE)
                all_trades.extend(trades)

        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.sort_values(by='entry_time', inplace=True)
            trades_df.reset_index(drop=True, inplace=True)
            print("\n--- All Simulated Trades ---")
            print(trades_df.to_string())
            generate_summary(trades_df, STARTING_CAPITAL)
        else:
            print("\nNo trades were executed across any of the tickers.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("Script finished.")
