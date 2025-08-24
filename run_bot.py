# -*- coding: utf-8 -*-
"""
Master Trading Bot Controller

This script acts as the "brain" for the algorithmic trading system.
Its purpose is to automatically analyze the market conditions and then
execute the appropriate trading strategy using the main_engine.py.

This script performs the following steps:
1.  Fetches the latest 1-hour data for the Nifty 50 index.
2.  Calculates the ADX (Average Directional Index) to measure trend strength.
3.  Based on the ADX value, it decides if the market is TRENDING, RANGING, or INDECISIVE.
4.  It then calls the `main_engine.py` script as a subprocess, passing in the
    correct strategy name (`trend_follower` or `mean_reversion`) based on its analysis.

This creates a fully automated system that adapts to changing market conditions.
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import subprocess
import sys

# --- Configuration ---
NIFTY_TICKER = '^NSEI'
ADX_THRESHOLD_TRENDING = 25  # ADX value above which we consider the market to be trending
ADX_THRESHOLD_RANGING = 20   # ADX value below which we consider the market to be ranging

def analyze_market_regime():
    """
    Analyzes the current market regime by calculating the ADX on Nifty's 1-hour chart.

    Returns:
        str: The determined market regime ('trending', 'ranging', or 'indecisive').
    """
    print("--- Analyzing Market Regime ---")
    try:
        # Fetch the last ~100 hours of data to ensure enough data for ADX calculation
        nifty_data = yf.download(
            tickers=NIFTY_TICKER,
            period='5d',  # 5 days should give enough 1-hour candles
            interval='60m',
            auto_adjust=True,
            progress=False
        )
        if nifty_data.empty:
            print("Warning: Could not fetch Nifty data for analysis.")
            return 'indecisive'

        # Standardize columns to handle potential MultiIndex from yfinance
        if isinstance(nifty_data.columns, pd.MultiIndex):
            nifty_data.columns = nifty_data.columns.droplevel(1)

        # Calculate ADX
        nifty_data.ta.adx(length=14, append=True)

        # Get the latest ADX value
        latest_adx = nifty_data['ADX_14'].iloc[-1]
        print(f"Latest Nifty ADX (1-hour): {latest_adx:.2f}")

        # Determine the regime
        if latest_adx > ADX_THRESHOLD_TRENDING:
            print("Conclusion: Market is TRENDING.")
            return 'trending'
        elif latest_adx < ADX_THRESHOLD_RANGING:
            print("Conclusion: Market is RANGING/CHOPPY.")
            return 'ranging'
        else:
            print("Conclusion: Market is INDECISIVE.")
            return 'indecisive'

    except Exception as e:
        print(f"An error occurred during market analysis: {e}")
        return 'indecisive'

def run_strategy(strategy_name):
    """
    Executes the main_engine.py with the specified strategy using a subprocess.
    """
    command = [
        sys.executable,  # Use the same python executable that is running this script
        'main_engine.py',
        '--strategy',
        strategy_name
    ]

    print(f"\n--- Executing Strategy: {strategy_name} ---")
    try:
        # We use subprocess.run to execute the command and stream the output
        process = subprocess.run(
            command,
            check=True,       # Raise an exception if the command returns a non-zero exit code
            capture_output=True, # Capture stdout and stderr
            text=True         # Decode stdout/stderr as text
        )
        print("\n--- Main Engine Output ---")
        print(process.stdout)
        if process.stderr:
            print("\n--- Main Engine Errors ---")
            print(process.stderr)
        print("--------------------------")

    except FileNotFoundError:
        print(f"Error: 'main_engine.py' not found. Make sure it's in the same directory.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing main_engine.py for strategy '{strategy_name}':")
        print(e.stdout)
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while running the strategy: {e}")


if __name__ == '__main__':
    # Step 1: Analyze the market
    regime = analyze_market_regime()

    # Step 2: Select and run the appropriate strategy
    if regime == 'trending':
        run_strategy('trend_follower')
    elif regime == 'ranging':
        run_strategy('mean_reversion')
    else:
        print("\n--- No Action Taken: Market regime is indecisive. ---")

    print("\nMaster controller script finished.")
