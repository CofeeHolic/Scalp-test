# -*- coding: utf-8 -*-
"""
Master Trading Bot Controller (Upgraded)

This script acts as the "brain" for the algorithmic trading system.
Its purpose is to automatically analyze market conditions and then
execute the appropriate trading strategy using the main_engine.py.

UPGRADE: This version now detects the DIRECTION of the trend (Bullish/Bearish)
and instructs the engine to only take trades in that direction.

This creates a highly specialized system:
- Choppy Market -> Mean Reversion Strategy
- Bullish Trend -> Trend Following Strategy (Longs Only)
- Bearish Trend -> Trend Following Strategy (Shorts Only)
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import subprocess
import sys

# --- Configuration ---
NIFTY_TICKER = '^NSEI'
ADX_THRESHOLD_TRENDING = 25
ADX_THRESHOLD_RANGING = 20
TREND_EMA_PERIOD = 50 # EMA period on 1-hour chart to determine trend direction

def analyze_market_regime():
    """
    Analyzes market regime and trend direction.

    Returns:
        A tuple: (regime, direction)
        - regime (str): 'trending', 'ranging', or 'indecisive'.
        - direction (str): 'bullish', 'bearish', or 'none'.
    """
    print("--- Analyzing Market Regime and Direction ---")
    try:
        nifty_data = yf.download(
            tickers=NIFTY_TICKER, period='30d', interval='60m',
            auto_adjust=True, progress=False
        )
        if nifty_data.empty:
            print("Warning: Could not fetch Nifty data.")
            return 'indecisive', 'none'

        if isinstance(nifty_data.columns, pd.MultiIndex):
            nifty_data.columns = nifty_data.columns.droplevel(1)

        # --- Calculate Indicators for Analysis ---
        nifty_data.ta.adx(length=14, append=True)
        nifty_data['EMA_trend'] = ta.ema(nifty_data['Close'], length=TREND_EMA_PERIOD)

        # Drop rows with NaN values that are created by the indicators
        nifty_data.dropna(inplace=True)
        if nifty_data.empty:
            print("Warning: Not enough data for analysis after indicator calculation.")
            return 'indecisive', 'none'

        latest_adx = nifty_data['ADX_14'].iloc[-1]
        latest_close = nifty_data['Close'].iloc[-1]
        latest_ema = nifty_data['EMA_trend'].iloc[-1]

        print(f"Latest Nifty ADX (1-hour): {latest_adx:.2f}")
        print(f"Latest Nifty Close: {latest_close:.2f}, EMA({TREND_EMA_PERIOD}): {latest_ema:.2f}")

        # --- Determine Regime and Direction ---
        regime = 'indecisive'
        direction = 'none'

        if latest_adx > ADX_THRESHOLD_TRENDING:
            regime = 'trending'
            if latest_close > latest_ema:
                direction = 'bullish'
                print("Conclusion: Market is in a BULLISH TREND.")
            else:
                direction = 'bearish'
                print("Conclusion: Market is in a BEARISH TREND.")
        elif latest_adx < ADX_THRESHOLD_RANGING:
            regime = 'ranging'
            print("Conclusion: Market is RANGING/CHOPPY.")
        else:
            print("Conclusion: Market is INDECISIVE.")

        return regime, direction

    except Exception as e:
        print(f"An error occurred during market analysis: {e}")
        return 'indecisive', 'none'

def run_strategy(strategy_name, direction=None):
    """
    Executes the main_engine.py with the specified strategy and optional direction.
    """
    command = [
        sys.executable,
        'main_engine.py',
        '--strategy',
        strategy_name
    ]
    # Add the direction argument only if it's specified (for trend_follower)
    if direction:
        command.extend(['--direction', direction])

    print(f"\n--- Executing Strategy: {strategy_name} (Direction: {direction or 'N/A'}) ---")
    try:
        process = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        print("\n--- Main Engine Output ---")
        print(process.stdout)
        if process.stderr:
            print("\n--- Main Engine Errors ---")
            print(process.stderr)
        print("--------------------------")

    except Exception as e:
        print(f"An unexpected error occurred while running the strategy: {e}")


if __name__ == '__main__':
    # Step 1: Analyze the market
    regime, direction = analyze_market_regime()

    # Step 2: Select and run the appropriate strategy
    if regime == 'trending':
        run_strategy('trend_follower', direction=direction)
    elif regime == 'ranging':
        run_strategy('mean_reversion')
    else:
        print("\n--- No Action Taken: Market regime is indecisive. ---")

    print("\nMaster controller script finished.")
